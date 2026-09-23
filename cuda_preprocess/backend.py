"""ctypes adapter; operations enqueue on PyTorch's current CUDA stream."""
import ctypes as C
from pathlib import Path
import torch

lib = C.CDLL(str(Path(__file__).with_name('libaccel_preprocess.so')))
P, I = C.c_void_p, C.c_int
lib.ag_error.restype = C.c_char_p
lib.ag_create.argtypes = [I]
lib.ag_create.restype = P
lib.ag_destroy.argtypes = [P]
lib.ag_partition.argtypes = [P, P, P, P, P]
lib.ag_reorder.argtypes = [P] * 9
lib.ag_spmm.argtypes = [P, I, P, P, P, P, I, I, I, P]
lib.ag_spmm_mapped.argtypes = [P, I, P, P, P, P, P, P, P, I, I, I, P]
lib.ag_cusparse_create.argtypes = [I, I, I, P, P, P, P, P]
lib.ag_cusparse_create.restype = P
lib.ag_cusparse_run.argtypes = [P, P, P, P]
lib.ag_cusparse_destroy.argtypes = [P]

def check(code):
    if code:
        raise RuntimeError(lib.ag_error().decode())

def ptr(t):
    return t.data_ptr() if t is not None else None

def stream():
    return torch.cuda.current_stream().cuda_stream

class Preprocessor:
    """Reusable workspace. Input must be valid contiguous CUDA int32 CSR.

    Output metadata capacity n + floor(nnz/384) is an upper bound. The valid
    prefix length is stored in count[0] on device. No host synchronization in
    partition/reorder; call count.item() once if a host launch grid is needed.
    Full reorder only permutes rows, retaining the original column numbering.
    """
    def __init__(self, n, nnz, mapping_only=False):
        self.n, self.nnz = n, nnz
        self.handle = lib.ag_create(n)
        if not self.handle:
            raise RuntimeError(lib.ag_error().decode())
        self.meta = torch.empty((max(1, n + nnz // 384), 4), device='cuda', dtype=torch.int32)
        self.count = torch.empty(1, device='cuda', dtype=torch.int32)
        self.perm = torch.empty(n, device='cuda', dtype=torch.int32)
        self.rowptr = torch.empty(n + 1, device='cuda', dtype=torch.int32)
        self.indices = None if mapping_only else torch.empty(nnz, device='cuda', dtype=torch.int32)

    def partition(self, rowptr):
        check(lib.ag_partition(self.handle, ptr(rowptr), ptr(self.meta), ptr(self.count), stream()))
        return self.meta, self.count

    def reorder(self, rowptr, indices, values=None, out_values=None):
        if self.indices is None:
            raise ValueError('mapping_only workspace: use mapping(rowptr)')
        check(lib.ag_reorder(self.handle, ptr(rowptr), ptr(indices), ptr(values),
                            ptr(self.perm), ptr(self.rowptr), ptr(self.indices), ptr(out_values), stream()))
        return self.rowptr, self.indices, self.perm

    def mapping(self, rowptr):
        """Stable degree sort + virtual rowptr + exact metadata; no edge I/O."""
        check(lib.ag_reorder(self.handle, ptr(rowptr), None, None,
                            ptr(self.perm), ptr(self.rowptr), None, None, stream()))
        return self.partition(self.rowptr)

    def full(self, rowptr, indices, values=None, out_values=None):
        self.reorder(rowptr, indices, values, out_values)
        return self.partition(self.rowptr)

    def close(self):
        if self.handle:
            torch.cuda.synchronize()
            lib.ag_destroy(self.handle)
            self.handle = None

def spmm(meta, blocks, indices, values, x, out):
    check(lib.ag_spmm(ptr(meta), blocks, ptr(indices), ptr(values), ptr(x), ptr(out),
                      x.shape[0], indices.numel(), x.shape[1], stream()))
    return out

def spmm_mapped(meta, blocks, perm, original_ptr, sorted_ptr, indices, values,
                x, out, original_output=False):
    check(lib.ag_spmm_mapped(ptr(meta), blocks, ptr(perm), ptr(original_ptr),
                            ptr(sorted_ptr), ptr(indices), ptr(values), ptr(x), ptr(out),
                            x.shape[0], x.shape[1], int(original_output), stream()))
    return out

class CuSparse:
    def __init__(self, rowptr, indices, values, x, out):
        self.refs = rowptr, indices, values
        self.handle = lib.ag_cusparse_create(x.shape[0], indices.numel(), x.shape[1],
                                             ptr(rowptr), ptr(indices), ptr(values), ptr(x), ptr(out))
        if not self.handle:
            raise RuntimeError(lib.ag_error().decode())

    def run(self, x, out):
        check(lib.ag_cusparse_run(self.handle, ptr(x), ptr(out), stream()))
        return out

    def close(self):
        if self.handle:
            torch.cuda.synchronize()
            lib.ag_cusparse_destroy(self.handle)
            self.handle = None
