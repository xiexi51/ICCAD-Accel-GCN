#pragma once
#include "preprocess.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <unistd.h>

inline void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
inline void preprocess_check(int status) {
    if (status) throw std::runtime_error(ag_error());
}

template<class T> struct DeviceBuffer {
    T *data = nullptr;
    explicit DeviceBuffer(size_t count) {
        cuda_check(cudaMalloc(&data, std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const T *source, size_t count) {
        if (count) cuda_check(cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T *target, size_t count) const {
        if (count) cuda_check(cudaMemcpy(target, data, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

inline uint64_t fingerprint(const void *data, size_t bytes) {
    uint64_t hash = 14695981039346656037ULL;
    const auto *p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) { hash ^= p[i]; hash *= 1099511628211ULL; }
    return hash;
}

// Cache format: eight uint64 header fields, then int32 perm, sorted_ptr, block4.
// Bump version whenever metadata semantics change. Native little-endian format.
// Metadata depends only on row degrees: changing columns/weights is safe.
class Metadata {
    static constexpr uint64_t magic = 0x4147434e4d455441ULL, version = 1;
    int n, nnz;
    size_t capacity;
    std::vector<int> payload() const {
        std::vector<int> host(size_t(2) * n + 1 + size_t(4) * blocks);
        perm.download(host.data(), n);
        sorted_ptr.download(host.data() + n, size_t(n) + 1);
        records.download(reinterpret_cast<int4*>(host.data() + size_t(2) * n + 1), blocks);
        return host;
    }
    bool load(const std::filesystem::path &file, uint64_t key) {
        std::ifstream input(file, std::ios::binary);
        if (!input) return false;
        std::array<uint64_t, 8> h{};
        if (!input.read(reinterpret_cast<char*>(h.data()), sizeof(h))) return false;
        if (h[0] != magic || h[1] != version || h[2] != uint64_t(n) ||
            h[3] != uint64_t(nnz) || h[4] != key || h[5] > capacity) return false;
        size_t words = size_t(2) * n + 1 + size_t(4) * h[5];
        if (h[6] != words || std::filesystem::file_size(file) != sizeof(h) + words * sizeof(int))
            return false;
        std::vector<int> host(words);
        if (!input.read(reinterpret_cast<char*>(host.data()), words * sizeof(int)) ||
            fingerprint(host.data(), words * sizeof(int)) != h[7]) return false;
        blocks = static_cast<int>(h[5]);
        perm.upload(host.data(), n);
        sorted_ptr.upload(host.data() + n, size_t(n) + 1);
        records.upload(reinterpret_cast<const int4*>(host.data() + size_t(2) * n + 1), blocks);
        return true;
    }
    void save(const std::filesystem::path &file, uint64_t key) const {
        auto host = payload();
        std::array<uint64_t, 8> h{magic, version, uint64_t(n), uint64_t(nnz), key,
            uint64_t(blocks), host.size(), fingerprint(host.data(), host.size() * sizeof(int))};
        auto temporary = file;
        temporary += ".tmp." + std::to_string(getpid());
        try {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            output.exceptions(std::ios::failbit | std::ios::badbit);
            output.write(reinterpret_cast<const char*>(h.data()), sizeof(h));
            output.write(reinterpret_cast<const char*>(host.data()), host.size() * sizeof(int));
            output.close();
            std::filesystem::rename(temporary, file);
        } catch (...) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            throw;
        }
    }
public:
    DeviceBuffer<int> perm, sorted_ptr;
    DeviceBuffer<int4> records;
    int blocks = 0;
    bool cache_hit = false;
    double setup_ms = 0;
    Metadata(const std::vector<int> &ptr, int edges, const int *device_ptr,
             const std::string &graph, const std::string &cache_dir)
        : n(static_cast<int>(ptr.size()) - 1), nnz(edges),
          capacity(std::max(size_t(1), size_t(n) + size_t(nnz) / 384)),
          perm(n), sorted_ptr(size_t(n) + 1), records(capacity) {
        auto start = std::chrono::steady_clock::now();
        uint64_t key = 0;
        std::filesystem::path file;
        if (!cache_dir.empty()) {
            key = fingerprint(ptr.data(), ptr.size() * sizeof(int));
            std::filesystem::create_directories(cache_dir);
            file = std::filesystem::path(cache_dir) / (graph + ".agmeta");
            cache_hit = load(file, key);
        }
        if (!cache_hit) {
            std::unique_ptr<void, decltype(&ag_destroy)> workspace(ag_create(n), ag_destroy);
            if (!workspace) throw std::runtime_error(ag_error());
            DeviceBuffer<int> count(1);
            preprocess_check(ag_mapping(workspace.get(), device_ptr, perm.data, sorted_ptr.data, nullptr));
            preprocess_check(ag_partition(workspace.get(), sorted_ptr.data, records.data, count.data, nullptr));
            count.download(&blocks, 1);
            if (!cache_dir.empty()) save(file, key);
        }
        cuda_check(cudaDeviceSynchronize());
        setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};
