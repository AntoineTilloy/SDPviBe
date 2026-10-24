"""Fused alignment/addition preserves each operand's independent rounding."""
import random

import pytest
import torch

from sdpvibe.batched import BMP, badd


@pytest.mark.parametrize("L", [1, 4, 28, 34])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_alignment_against_integers(L, sign, uniform, strided):
    rng = random.Random(19)
    nb, rows, cols = 2, 3, 5
    limit = 1 << (32 * L - 1)
    edge = [0, 1, -1, limit - 1, -limit]
    a = [rng.randrange(-limit, limit) for _ in range(nb * rows * cols)]
    b = list(reversed(a))
    a[:5], b[:5] = edge, list(reversed(edge))
    # The sum still wraps like add_same when rounding consumes the extra bit.
    a[10:14] = [limit - 1, -limit, limit - 1, -limit]
    b[10:14] = [-limit, limit - 1, limit - 1, -limit]
    ea = [-100] * (nb * cols)
    eb = [-100] * (nb * cols) if uniform else [-100 + d for d in (0, 1, 31, 32, 33, 63, 64, 65, -10000, 10000)]

    def bmp(values, exps):
        words = [[((v >> (32 * k)) + (1 << 31)) % (1 << 32) - (1 << 31) for v in values]
                 for k in range(L)]
        limbs = torch.tensor(words, dtype=torch.int32, device="cuda").view(L, nb, rows, cols)
        exp = exps[0] if uniform else torch.tensor(exps, dtype=torch.int64, device="cuda").view(nb, cols)
        if strided:
            limbs = limbs.transpose(-1, -2).contiguous().transpose(-1, -2)
            if not uniform: exp = exp.T.contiguous().T
        return BMP(limbs, exp)

    out = badd(bmp(a, ea), bmp(b, eb), sign, out_exp="align")
    words, exps = out.flat().cpu().tolist(), out.exp.reshape(-1).cpu().tolist()
    for i, (av, bv) in enumerate(zip(a, b)):
        j = (i // (rows * cols)) * cols + i % cols
        ex = max(ea[j], eb[j]) + 1
        rounded = lambda v, shift: (v + (1 << (shift - 1))) >> shift
        expected = rounded(av, ex - ea[j]) + sign * rounded(bv, ex - eb[j])
        expected = (expected + limit) % (2 * limit) - limit
        actual = sum((words[k][i] & 0xffffffff) << (32 * k) for k in range(L))
        actual -= 2 * limit if words[-1][i] < 0 else 0
        assert actual == expected
        assert exps[j] == ex
