import json
import numpy as np
import torch
from scipy.sparse import csr_matrix
from backend import Preprocessor, spmm, spmm_mapped
from reference import partition

def check_case(deg, rng):
    n = len(deg)
    ptr = np.r_[0, np.cumsum(deg)].astype(np.int32)
    idx = rng.integers(0, max(1, n), int(ptr[-1]), dtype=np.int32)
    p, i = torch.from_numpy(ptr).cuda(), torch.from_numpy(idx).cuda()
    w = Preprocessor(n, len(idx))
    w.partition(p)
    np.testing.assert_array_equal(w.meta[:w.count.item()].cpu().numpy(), partition(ptr))
    w.full(p, i)
    perm = np.argsort(deg, kind='stable').astype(np.int32)
    expected_ptr = np.r_[0, np.cumsum(np.asarray(deg)[perm])].astype(np.int32)
    expected_idx = np.concatenate([idx[ptr[r]:ptr[r+1]] for r in perm]) if n else idx
    np.testing.assert_array_equal(w.perm.cpu().numpy(), perm)
    np.testing.assert_array_equal(w.rowptr.cpu().numpy(), expected_ptr)
    np.testing.assert_array_equal(w.indices.cpu().numpy(), expected_idx)
    count = w.count.item()
    np.testing.assert_array_equal(w.meta[:count].cpu().numpy(), partition(expected_ptr))
    mapped = Preprocessor(n, len(idx), mapping_only=True)
    mapped.mapping(p)
    assert mapped.indices is None
    assert mapped.count.item() == count
    np.testing.assert_array_equal(mapped.perm.cpu().numpy(), perm)
    np.testing.assert_array_equal(mapped.rowptr.cpu().numpy(), expected_ptr)
    np.testing.assert_array_equal(mapped.meta[:count].cpu().numpy(), partition(expected_ptr))
    if n:
        values = torch.rand(len(idx), device='cuda')
        sorted_values = torch.empty_like(values)
        w.full(p, i, values, sorted_values)
        host_values = values.cpu().numpy()
        expected_values = np.concatenate([host_values[ptr[r]:ptr[r+1]] for r in perm])
        np.testing.assert_array_equal(sorted_values.cpu().numpy(), expected_values)
        csr = csr_matrix((sorted_values.cpu().numpy(), expected_idx, expected_ptr), shape=(n, n))
        for cols in (32, 41, 128):
            x = torch.randn(n, cols, device='cuda')
            out = torch.empty_like(x)
            spmm(w.meta, count, w.indices, sorted_values, x, out)
            expected = torch.from_numpy(csr @ x.cpu().numpy()).cuda()
            torch.testing.assert_close(out, expected, rtol=3e-4, atol=3e-3)
            spmm_mapped(mapped.meta,count,mapped.perm,p,mapped.rowptr,i,values,x,out)
            torch.testing.assert_close(out, expected, rtol=3e-4, atol=3e-3)
            spmm_mapped(mapped.meta,count,mapped.perm,p,mapped.rowptr,i,values,x,out,original_output=True)
            original_expected=expected[torch.from_numpy(np.argsort(perm)).cuda()]
            torch.testing.assert_close(out, original_expected, rtol=3e-4, atol=3e-3)
    mapped.close()
    w.close()

def main():
    rng = np.random.default_rng(123)
    cases = [[], [0], [0]*19, [1]*25, [384]*7, [385, 768, 769, 4096],
             np.repeat([0,1,31,32,33,63,64,65,95,96,97,127,128,129,190,191,192,193,383,384,385,767,768,769], 25)]
    cases += [rng.integers(0, 2000, size=257) for _ in range(6)]
    for deg in cases:
        check_case(deg, rng)
    print(json.dumps({'cases_passed': len(cases), 'metadata_exact': True,
                      'stable_reorder_exact': True, 'mapping_only_exact': True,
                      'mapped_spmm_sorted_and_original_output': True, 'spmm_widths': [32, 41, 128]}))

if __name__ == '__main__':
    main()
