from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import os

# generate partition patterns
deg_bound = 12 * 32
num_warps = 12
warp_nz = [0]
d_block_rows = [0]
warp_max_nz = deg_bound // num_warps
factor = [1, 2, 3, 4, 6, 12]
jf = 0
i = 1
while i < deg_bound // 2:
    if factor[jf] * warp_max_nz >= i:
        warp_nz.append((i + factor[jf] - 1) // factor[jf])
        d_block_rows.append(num_warps // factor[jf])
        i += 1
    else:
        jf += 1

if not os.path.exists('./block_level_meta/'):
    os.makedirs('./block_level_meta/')

base_path = './graphs/'
fileset = Path(base_path).glob('*.config')

for file in fileset:
    print(file.stem)
    new_indptr = np.fromfile(base_path + file.stem + ".new_indptr", dtype=np.int32)
    new_indices = np.fromfile(base_path + file.stem + ".new_indices", dtype=np.int32)
    v_num = len(new_indptr) - 1
    e_num = len(new_indices)
    vals = np.ones(e_num)
    new_csr = csr_matrix((vals, new_indices, new_indptr))
    print(v_num, e_num)
    sorted_deg = np.ediff1d(new_csr.indptr)
    cur_row = 0
    cur_loc = 0
    cur_degree = 0
    block_degree = []
    block_row_begin = []
    block_loc_begin = []
    block_info = []

    # block-level partitioning
    while True:
        if sorted_deg[cur_row] != cur_degree:
            cur_degree = sorted_deg[cur_row]

        if cur_degree == 0:
            cur_row += 1

        elif cur_degree >= 1 and cur_degree <= deg_bound:
            if cur_degree >= len(warp_nz):
                w_nz = deg_bound // num_warps
            else:
                w_nz = warp_nz[cur_degree]
            if cur_degree >= len(d_block_rows):
                b_row = 1
            else:
                b_row = d_block_rows[cur_degree]

            block_row_begin.append(cur_row)
            block_loc_begin.append(cur_loc)

            j = 0
            while sorted_deg[cur_row] == cur_degree:
                cur_row += 1
                j += 1
                if j == b_row:
                    break
                if cur_row == len(new_indptr) - 1:
                    break
            cur_loc += j * cur_degree
            block_degree.append(cur_degree)
            block_info.append((w_nz << 16) + j)

        elif cur_degree > deg_bound:
            tmp_loc = 0
            while True:
                block_degree.append(cur_degree)
                block_row_begin.append(cur_row)
                block_loc_begin.append(cur_loc)
                if tmp_loc + deg_bound > cur_degree:
                    block_info.append(cur_degree - tmp_loc)
                    cur_loc += cur_degree - tmp_loc
                    tmp_loc = cur_degree
                else:
                    block_info.append(deg_bound)
                    tmp_loc += deg_bound
                    cur_loc += deg_bound
                if tmp_loc == cur_degree:
                    break
            cur_row += 1

        else:
            print("cur_degree number is wrong")
            break

        if cur_row == len(new_indptr) - 1:
            break

    block_4 = np.dstack([block_degree, block_row_begin, block_loc_begin, block_info]).flatten()
    block_4.astype(np.int32).tofile('./block_level_meta/' + file.stem + '.block4')