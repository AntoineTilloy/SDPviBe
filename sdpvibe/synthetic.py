"""Synthetic SDP shapes with the block structure of the stress-tensor system (Chang et al. 2024), scaled
from its real nmax=6 instance (N=742; 113 blocks: dims {1:45, 2:53, 3:1, 4:1, 5:13}; K <= 57):
N ~ nmax^2, number of blocks ~ nmax, points per block ~ nmax."""
import numpy as np


def st3d_like(nmax, scale_blocks=1.0, seed=0):
    s = nmax / 6
    N = int(round(742 * s * s))
    Kmax = int(round(57 * s))
    rng = np.random.default_rng(seed)
    blocks = []
    for dim, cnt in {1: 45, 2: 53, 3: 1, 4: 1, 5: 13}.items():
        n = max(1, int(round(cnt * s * scale_blocks)))
        for i in range(n):
            K = 1 if (dim == 1 and i < 3) else int(rng.integers(int(0.7 * Kmax), Kmax + 1))
            blocks.append(dict(dim=dim, K=K, he=(K - 1) // 2 + 1, ho=(K - 2) // 2 + 1 if K > 1 else 0))
    return blocks, N
