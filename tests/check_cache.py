"""Integration test for the native executable; no Python preprocessing or packages."""
import pathlib
import struct
import subprocess
import sys
import tempfile

executable = str(pathlib.Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="accel-cache-test-") as tmp:
    root = pathlib.Path(tmp)
    cache = root / "cache"

    def write_graph(ptr, indices):
        for suffix, values in [("ptrdump", ptr), ("edgedump", indices)]:
            (root / ("test.graph." + suffix)).write_bytes(struct.pack(f"<{len(values)}i", *values))

    def run(mode, expected, cols=41):
        result = subprocess.run(
            [executable, "test", str(cols), "--graphs-dir", str(root),
             "--warmup", "1", "--iterations", "2", *mode],
            text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "metadata=" + expected in result.stdout, result.stdout
        assert "validation=pass" in result.stdout, result.stdout
        return result.stdout

    # Includes isolated rows, stable degree ties, and a row split into two blocks.
    ptr = [0, 0, 1, 2, 387] + [387] * 396
    indices = [1, 2] + [0, 1, 2, 3] * 96 + [0]
    write_graph(ptr, indices)
    run([], "cuda")
    assert not cache.exists()
    mode = ["--metadata-cache", str(cache)]
    run(mode, "cuda")
    saved = cache / "test.agmeta"
    original = saved.read_bytes()
    run(mode, "cache", 128)
    assert saved.read_bytes() == original
    run([*mode, "--metadata-generate"], "cuda")
    assert saved.read_bytes() == original

    # Same n and nnz, changed degrees must invalidate the cache.
    write_graph([0, 1, 1, 2, 387] + [387] * 396, indices)
    run(mode, "cuda")
    run(mode, "cache")
    # Column IDs do not affect metadata and should not invalidate it.
    write_graph([0, 1, 1, 2, 387] + [387] * 396, [3 - col for col in indices])
    run(mode, "cache")

    for damage in (b"truncated", bytes(64)):
        saved.write_bytes(damage)
        run(mode, "cuda")
    damaged = bytearray(saved.read_bytes())
    damaged[-1] ^= 1
    saved.write_bytes(damaged)
    run(mode, "cuda")
    run(mode, "cache")
    write_graph([0, 0, 0], [])
    run(mode, "cuda")
    run(mode, "cache")
    write_graph([0, 2, 1], [0])
    result = subprocess.run([executable, "test", "32", "--graphs-dir", str(root)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "Invalid CSR" in result.stderr
print("PASS: CUDA generation, cache reuse/invalidation/corruption, original-order SpMM, invalid CSR")
