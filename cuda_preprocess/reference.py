"""CPU reference preserving block_level_partition.py's exact partition rules."""
import numpy as np

def partition(ptr):
    degree = np.diff(ptr)
    factors = [1, 2, 3, 4, 6, 12]
    warp_nz, block_rows = [0], [0]
    f = 0
    for d in range(1, 192):
        while factors[f] * 32 < d:
            f += 1
        warp_nz.append((d + factors[f] - 1) // factors[f])
        block_rows.append(12 // factors[f])
    out = []
    row = loc = 0
    while row < len(degree):
        d = int(degree[row])
        if not d:
            row += 1
        elif d <= 384:
            w = warp_nz[d] if d < 192 else 32
            b = block_rows[d] if d < 192 else 1
            begin = row
            row += 1
            while row < len(degree) and row - begin < b and degree[row] == d:
                row += 1
            count = row - begin
            out.append((d, begin, loc, (w << 16) + count))
            loc += count * d
        else:
            for start in range(0, d, 384):
                count = min(384, d - start)
                out.append((d, row, loc, count))
                loc += count
            row += 1
    return np.asarray(out, dtype=np.int32).reshape(-1, 4)
