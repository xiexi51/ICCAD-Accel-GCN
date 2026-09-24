#include "cuda_preprocess/metadata.h"
#include "spmm_accel.h"
#include <cmath>
#include <climits>
#include <iostream>
#include <random>
#include <string>

namespace fs = std::filesystem;

std::vector<int> read_array(const fs::path &file) {
    std::ifstream input(file, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open " + file.string());
    auto bytes = input.tellg();
    if (bytes < 0 || bytes % sizeof(int) || uint64_t(bytes) / sizeof(int) > INT_MAX)
        throw std::runtime_error("Invalid int32 array: " + file.string());
    std::vector<int> values(size_t(bytes) / sizeof(int));
    input.seekg(0);
    if (bytes && !input.read(reinterpret_cast<char*>(values.data()), bytes))
        throw std::runtime_error("Cannot read " + file.string());
    return values;
}

struct Options {
    std::string graph, graphs = "../graphs", cache;
    int cols = 0, warmup = 20, iterations = 100;
    bool check = true;
};

int positive(const std::string &s) {
    size_t used = 0;
    int value = std::stoi(s, &used);
    if (used != s.size() || value <= 0) throw std::runtime_error("Expected a positive integer: " + s);
    return value;
}

// CUDA event time per call, after explicit warmup. Allocation and validation excluded.
template<class F> float time_spmm(F run, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) run();
    cuda_check(cudaDeviceSynchronize());
    cudaEvent_t begin, end;
    cuda_check(cudaEventCreate(&begin));
    cuda_check(cudaEventCreate(&end));
    cuda_check(cudaEventRecord(begin));
    for (int i = 0; i < iterations; ++i) run();
    cuda_check(cudaEventRecord(end));
    cuda_check(cudaEventSynchronize(end));
    float ms;
    cuda_check(cudaEventElapsedTime(&ms, begin, end));
    cuda_check(cudaEventDestroy(begin));
    cuda_check(cudaEventDestroy(end));
    return ms / iterations;
}

void test_graph(const std::string &graph, const Options &options) {
    auto ptr = read_array(fs::path(options.graphs) / (graph + ".graph.ptrdump"));
    auto idx = read_array(fs::path(options.graphs) / (graph + ".graph.edgedump"));
    int n = static_cast<int>(ptr.size()) - 1, nnz = static_cast<int>(idx.size());
    if (n < 0 || ptr.front() != 0 || ptr.back() != nnz || !std::is_sorted(ptr.begin(), ptr.end()))
        throw std::runtime_error("Invalid CSR row offsets: " + graph);
    for (int col : idx) if (col < 0 || col >= n) throw std::runtime_error("Invalid CSR column: " + graph);
    DeviceBuffer<int> dptr(ptr.size()), didx(idx.size());
    DeviceBuffer<float> values(nnz);
    dptr.upload(ptr.data(), ptr.size());
    didx.upload(idx.data(), idx.size());
    std::vector<float> ones(nnz, 1.0f);
    values.upload(ones.data(), ones.size());
    ones.clear(); ones.shrink_to_fit();

    // This is the only preprocessing path. No Python or reordered edge files.
    Metadata metadata(ptr, nnz, dptr.data, graph, options.cache);
    std::cout << graph << " nodes=" << n << " edges=" << nnz
              << " metadata=" << (metadata.cache_hit ? "cache" : "cuda")
              << " blocks=" << metadata.blocks << " metadata_setup_ms=" << metadata.setup_ms << '\n';
    if (!n) return;
    int first = options.cols ? options.cols : 16;
    int last = options.cols ? options.cols : 128;
    for (int cols = first; cols <= last; ++cols) {
        // Kernel address arithmetic uses signed int32 element offsets.
        size_t elements = size_t(n) * cols;
        if (elements > INT_MAX) throw std::runtime_error("Dense matrix exceeds int32 addressing limit");
        std::vector<float> input(elements);
        std::mt19937 engine(123);
        std::uniform_real_distribution<float> random(-1, 1);
        for (auto &value : input) value = random(engine);
        DeviceBuffer<float> x(elements), y(elements), reference(elements);
        x.upload(input.data(), elements);
        auto accel = [&] {
            cuda_check(static_cast<cudaError_t>(ag_spmm_mapped(
                reinterpret_cast<int*>(metadata.records.data), metadata.blocks, metadata.perm.data,
                dptr.data, metadata.sorted_ptr.data, didx.data, values.data,
                x.data, y.data, n, cols, 1, nullptr)));
        };
        std::unique_ptr<void, decltype(&ag_cusparse_destroy)> sparse(
            ag_cusparse_create(n, nnz, cols, dptr.data, didx.data, values.data, x.data, reference.data),
            ag_cusparse_destroy);
        if (!sparse) throw std::runtime_error(ag_cusparse_error());
        auto baseline = [&] {
            if (ag_cusparse_run(sparse.get(), x.data, reference.data, nullptr))
                throw std::runtime_error(ag_cusparse_error());
        };
        if (options.check) {
            accel(); baseline();
            std::vector<float> actual(elements), expected(elements);
            y.download(actual.data(), elements);
            reference.download(expected.data(), elements);
            for (size_t i = 0; i < elements; ++i)
                if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
                    std::abs(actual[i] - expected[i]) > 0.005f + 0.0005f * std::abs(expected[i]))
                    throw std::runtime_error("SpMM validation failed: " + graph + " element " + std::to_string(i));
        }
        float accel_ms = time_spmm(accel, options.warmup, options.iterations);
        float sparse_ms = time_spmm(baseline, options.warmup, options.iterations);
        std::cout << graph << " cols=" << cols << " accel_ms=" << accel_ms
                  << " cusparse_ms=" << sparse_ms << " warmup=" << options.warmup
                  << " iterations=" << options.iterations
                  << " validation=" << (options.check ? "pass" : "disabled") << '\n';
    }
}

int main(int argc, char **argv) {
    try {
        Options options;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::runtime_error("Missing value for " + arg);
                return argv[i];
            };
            if (arg == "--help") {
                std::cout << "Usage: spmm_test [GRAPH [COLS]] [--graphs-dir DIR]\n"
                    "  --metadata-cache DIR  Reuse metadata; generate and save on a miss\n"
                    "  --metadata-generate   Generate with CUDA every run (default)\n"
                    "  --cols N              Feature width (default: sweep 16..128)\n"
                    "  --warmup N            Warmup calls (default: 20)\n"
                    "  --iterations N        Timed calls, averaged (default: 100)\n"
                    "  --no-check            Skip comparison with cuSPARSE\n"
                    "Without GRAPH, process every *.graph.ptrdump in the input directory.\n";
                return 0;
            } else if (arg == "--graphs-dir") options.graphs = value();
            else if (arg == "--metadata-cache") {
                options.cache = value();
                if (options.cache.empty()) throw std::runtime_error("Cache directory must not be empty");
            } else if (arg == "--metadata-generate") options.cache.clear();
            else if (arg == "--cols") options.cols = positive(value());
            else if (arg == "--warmup") options.warmup = positive(value());
            else if (arg == "--iterations") options.iterations = positive(value());
            else if (arg == "--no-check") options.check = false;
            else if (arg.rfind("--", 0) == 0) throw std::runtime_error("Unknown option: " + arg);
            else if (options.graph.empty()) options.graph = arg;
            else if (!options.cols) options.cols = positive(arg);
            else throw std::runtime_error("Unexpected argument: " + arg);
        }
        cuda_check(cudaFree(nullptr));
        std::vector<std::string> graphs;
        if (!options.graph.empty()) {
            if (fs::path(options.graph).filename().string() != options.graph)
                throw std::runtime_error("GRAPH must be a name, not a path; use --graphs-dir");
            graphs.push_back(options.graph);
        } else {
            const std::string suffix = ".graph.ptrdump";
            for (const auto &entry : fs::directory_iterator(options.graphs)) {
                auto name = entry.path().filename().string();
                if (entry.is_regular_file() && name.size() > suffix.size() &&
                    name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
                    graphs.push_back(name.substr(0, name.size() - suffix.size()));
            }
            std::sort(graphs.begin(), graphs.end());
        }
        if (graphs.empty()) throw std::runtime_error("No CSR graphs found");
        for (const auto &graph : graphs) test_graph(graph, options);
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
