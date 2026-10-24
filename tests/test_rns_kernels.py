"""Exact integer references for residue conversion and fused CRT reconstruction.

Run on a CUDA host with: python -m pytest -q tests/test_rns_kernels.py
"""
import random

import pytest
import torch
import triton

from sdpvibe.mpmat import from_ints
from sdpvibe.limbops import bitlen_words
from sdpvibe.rns import RNS, _crt_dot_kernel


def test_bitlen_at_powers_of_two():
    # floor(log2(float64(word))) can be one too small even at exact powers of two.
    values = [v for bit in range(1, 95) for v in
              ((1 << bit) - 1, 1 << bit, (1 << bit) + 1,
               -(1 << bit) + 1, -(1 << bit), -(1 << bit) - 1)]
    values += [0, 1, -1]
    words = [[((v >> (32 * k)) + (1 << 31)) % (1 << 32) - (1 << 31) for v in values]
             for k in range(3)]
    got = bitlen_words(torch.tensor(words, dtype=torch.int32, device="cuda")).cpu().tolist()
    assert got == [(v if v >= 0 else ~v).bit_length() for v in values]


@pytest.mark.parametrize("L", [1, 10, 28, 34])
@pytest.mark.parametrize("K", [17, 65])
def test_residues_against_integers(L, K):
    rng = random.Random(123)
    values = [[rng.randrange(-(1 << (32 * L - 1)), 1 << (32 * L - 1)) for _ in range(K)]
              for _ in range(3)]
    values[0][3:8] = [0, 1, -1, -(1 << (32 * L - 1)) + 1, (1 << (32 * L - 1)) - 1]
    X = from_ints(values, 0, L).limbs
    # from_ints deliberately excludes the most-negative signed integer;
    # construct that representable bit pattern directly for the kernel test.
    values[0][6] -= 1
    X[:, 0, 6] = 0
    X[-1, 0, 6] = -(1 << 31)
    R = RNS(L)
    # An offset slice, an unaligned tail, and nonzero modulus offsets exercise
    # both the wider tile and the final group of sixteen moduli.
    for g0, G in ((0, R.n), (R.n - 16, 16)):
        hi, lo = R.residues(X, 3, K - 5, g0, G)
        got = (128 * hi.to(torch.int32) + lo).cpu().tolist()
        for gi, m in enumerate(R.mods[g0:g0 + G]):
            for i, row in enumerate(values):
                expected = [v % m for v in row[3:K - 2]]
                expected = [r - m if r > m // 2 else r for r in expected]
                expected += [0] * (hi.shape[-1] - len(expected))
                assert got[gi][i] == expected


@pytest.mark.parametrize("L", [1, 10, 28, 34, 250])  # 250 exercises the wide, two-product fallback
@pytest.mark.parametrize("width", [1, 5, 11])
def test_crt_against_integers(L, width):
    R = RNS(L)
    rng = random.Random(41)
    rows, N, start = 5, 13, 2
    limit = R.M // 4
    values = [[rng.randrange(-limit + 1, limit) for _ in range(N)] for _ in range(rows)]
    values[0][2:7] = [0, 1, -1, limit - 1, -limit + 1]
    values[1][2:7] = [1 << b for b in (0, 31, 32, 63, 64)]
    values[2][2:7] = [-(1 << b) for b in (0, 31, 32, 63, 64)]
    acc = torch.tensor([[[v % p for v in row] for row in values] for p in R.mods],
                       dtype=torch.int16, device="cuda")
    words, bitlen, count = R.crt_block(acc, N, start, width, rows)
    assert count == rows * width
    words = words[:, :count].cpu().tolist()
    bitlen = bitlen[:count].cpu().tolist()
    expected = [v for row in values for v in row[start:start + width]]
    for j, value in enumerate(expected):
        got = sum((word[j] & 0xffffffff) << (32 * i) for i, word in enumerate(words))
        if words[-1][j] < 0:
            got -= 1 << (32 * len(words))
        assert got == value
        # The existing fixed-point bit length uses the complemented magnitude
        # for negative numbers (one less bit at exact negative powers of two).
        assert bitlen[j] == (value if value >= 0 else ~value).bit_length()


@pytest.mark.parametrize("n", [16, 48, 128, 160, 1024])
@pytest.mark.parametrize("entries", [1, 33, 65])
def test_crt_digit_sum_extremes(n, entries):
    rng = random.Random(11)
    E = 39                                           # exercise the final partial digit tile
    c = torch.tensor([[rng.randrange(-128, 128) for _ in range(n)] for _ in range(E)], dtype=torch.int8)
    y = torch.tensor([[rng.choice([-16319, 16319, 0, -1, 1]) for _ in range(n)]
                      for _ in range(entries)], dtype=torch.int32)
    c[0] = -128
    y[0] = -16319                                    # near the int32 bound at n=1024
    lo = ((y + 64) & 127) - 64
    hi = (y - lo) >> 7
    expected = c.to(torch.int64) @ y.to(torch.int64).T
    out = torch.empty((E, entries), dtype=torch.int32, device="cuda")
    _crt_dot_kernel[(triton.cdiv(E, 32), triton.cdiv(entries, 64))](
        c.cuda(), hi.to(torch.int8).cuda(), lo.to(torch.int8).cuda(), out,
        E, entries, n, BE=32, BB=64, BK=32, num_warps=4, num_stages=3)
    assert torch.equal(out.cpu().to(torch.int64), expected)
