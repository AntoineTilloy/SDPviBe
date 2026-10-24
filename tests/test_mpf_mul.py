"""Product scanning against Python integers, including signs, carries and rounding."""
import random

import pytest
from sdpvibe import mpf as F


def mantissas(x):
    words = x.m.reshape(x.L, -1).cpu().tolist()
    return [sum((words[k][j] & 0xffffffff) << (32 * k) for k in range(x.L))
            - ((1 << (32 * x.L)) if words[-1][j] < 0 else 0)
            for j in range(x.e.numel())]


@pytest.mark.parametrize("L", [1, 2, 10, 26, 28, 34, 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_mul_exact_rounding(L, scalar):
    rng = random.Random(71)
    bits = F.NBITS(L)
    edge = [0, 1, -1, (1 << bits) - 1, -((1 << bits) - 1),
            1 << (bits - 1), -(1 << (bits - 1)), (1 << (bits - 1)) + 1]
    values = edge + [rng.randrange(-(1 << bits), 1 << bits) for _ in range(129)]
    a = F.from_ints(values, [rng.randrange(-2000, 2000) for _ in values], L, (len(values),))
    bvals = [(1 << bits) - 1] if scalar else list(reversed(values))
    b = F.from_ints(bvals, [rng.randrange(-2000, 2000) for _ in bvals], L, (len(bvals),))
    out = F.mul(a, b)
    am, bm, om = mantissas(a), mantissas(b), mantissas(out)
    ae, be, oe = (x.e.cpu().tolist() for x in (a, b, out))
    for i, actual in enumerate(om):
        j = 0 if scalar else i
        product = am[i] * bm[j]
        if not product:
            assert actual == 0 and oe[i] == F.EZ
            continue
        shift = abs(product).bit_length() - bits
        mag = abs(product)
        expected = ((mag + (1 << (shift - 1))) >> shift) if shift > 0 else mag << -shift
        assert actual == (expected if product > 0 else -expected)
        assert oe[i] == ae[i] + be[j] + shift
