"""Step-length eigenvalues (solver.min_eig) against exact values.

X = V V^T and dM = V diag(lam) V^T for a random integer V, built exactly with Python ints: the eigenvalues
of L^-1 dM L^-T (X = L L^T) are exactly lam.  Spectra with large positive eigenvalues next to the
smallest one defeat the float64 copy; the checked path (gamma given) must still match the exact step
length to EIG_TOL.
  python tests/test_step_length.py
"""
import sys, time
from fractions import Fraction
import numpy as np, mpmath as mp, torch
sys.path.insert(0, '.')
from sdpvibe import mpf as F
from sdpvibe.solver import min_eig, step_from_eig, EIG_TOL, _mpf_from_float

L = 40                                                               # 1278 bits: room for 2^700 spreads
mp.mp.prec = 32 * L
GAMMA = mp.mpf("0.7")


def mpf_exact(Q, K):
    """matrix of Python ints Q, value Q * 2^-K -> MPF"""
    n = len(Q)
    ints = [int(v) for row in Q for v in row]
    return F.from_ints(ints, [-K if v else F.EZ for v in ints], L, (n, n))


def case(n, lam, seed):
    """(X, dM) as batched-ready MPF (n, n); lam: Fractions"""
    rng = np.random.default_rng(seed)
    V = rng.integers(-3, 4, size=(n, n)).astype(object)
    while abs(np.linalg.det(V.astype(float))) < 1: V = rng.integers(-3, 4, size=(n, n)).astype(object)
    K = max(0, max(-int(np.floor(np.log2(abs(float(l))))) for l in lam if l) + 8)
    den = max(l.denominator for l in lam)
    assert all((den % l.denominator) == 0 for l in lam) and (den & (den - 1)) == 0
    K = max(K, den.bit_length() - 1)
    D = [int(l * (1 << K)) for l in lam]                             # exact
    dM = (V * np.array(D, dtype=object)[None, :]).dot(V.T)
    X = V.dot(V.T)
    return mpf_exact(X, 0), mpf_exact(dM, K)


def run(name, blocks, expect_refined=None):
    """blocks: list of lam lists (same n); returns ok"""
    n = len(blocks[0])
    Xs, Ds = zip(*[case(n, lam, seed) for seed, lam in enumerate(blocks)])
    X = F.MPF(torch.stack([x.m for x in Xs], 1), torch.stack([x.e for x in Xs]))
    D = F.MPF(torch.stack([d.m for d in Ds], 1), torch.stack([d.e for d in Ds]))
    ch = F.Chol(X)
    exact = min(mp.mpf(l.numerator) / l.denominator for lam in blocks for l in lam)
    a_ex = step_from_eig(exact, GAMMA)
    l64 = min_eig([ch], [D])
    st = {}
    torch.cuda.synchronize(); t0 = time.time()
    lck = min_eig([ch], [D], gamma=GAMMA, stats=st)
    torch.cuda.synchronize(); dt = time.time() - t0
    a64, ack = step_from_eig(l64, GAMMA), step_from_eig(lck, GAMMA)
    e64, eck = abs(a64 / a_ex - 1), abs(ack / a_ex - 1)
    ref = st.get("eig_refined", 0)
    ok = eck <= EIG_TOL and (expect_refined is None or ref == expect_refined)
    print(f"{'ok  ' if ok else 'FAIL'} {name:34s} n={n:3d} blocks={len(blocks)} lam_min={mp.nstr(exact, 6):>12}  "
          f"step error: float64 {mp.nstr(e64, 2):>8}  checked {mp.nstr(eck, 2):>8}  refined {ref}  {dt:6.3f} s")
    return ok


def spectrum(n, big, small, lo, seed=0):
    """big: exponents of large positive eigenvalues; small: count in [0.1, 3]; lo: the smallest (Fraction)"""
    rng = np.random.default_rng(100 + seed)
    lam = [Fraction(int(rng.integers(1, 8))) * (Fraction(2) ** k) for k in big]
    lam += [Fraction(int(rng.integers(1, 25)), 8) for _ in range(small)]
    lam += [lo]
    lam += [Fraction(int(rng.integers(1, 25)), 8) for _ in range(n - len(lam))]
    rng.shuffle(lam)
    return lam


ok = True
V = np.random.default_rng(7).standard_normal((3, 5, 5)) * 2.0 ** np.random.default_rng(8).integers(-900, 900, (3, 5, 5))
V[0, 0, 0] = 0
got = _mpf_from_float(V, L).to_mp().reshape(-1)
conv = all(mp.mpf(float(x)) == y for x, y in zip(V.reshape(-1), got))
print(f"{'ok  ' if conv else 'FAIL'} float64 -> MPF conversion exact")
ok &= conv
for n in (40, 144):
    ok &= run("ordinary spectrum", [spectrum(n, [], n - 1, Fraction(-11, 8))], 0)
    ok &= run("big positive, lam_min -1.375", [spectrum(n, [100] * 5 + [60] * 3, 4, Fraction(-11, 8))], 1)
    ok &= run("big positive, lam_min -0.5 (step 1)", [spectrum(n, [100] * 5, 4, Fraction(-1, 2))])
    ok &= run("multi-scale 2^700..2^30", [spectrum(n, [700, 500, 300, 200, 120, 60, 30], 4, Fraction(-5, 2))], 1)
    ok &= run("cluster at lam_min", [spectrum(n, [100] * 5, 0, Fraction(-11, 8)) [:-3]
                                     + [Fraction(-11, 8) - Fraction(1, 2 ** 40), Fraction(-11, 8) + Fraction(1, 2 ** 40),
                                        Fraction(-11, 8) + Fraction(1, 2 ** 20)]])
    ok &= run("big negative lam_min -2^100", [spectrum(n, [90] * 3, 4, -Fraction(2) ** 100)], 0)
    ok &= run("3 of 4 blocks need refinement", [spectrum(n, [100] * 5, 4, Fraction(-11, 8), 0),
                                                spectrum(n, [700, 400, 100], 4, Fraction(-5, 2), 1),
                                                spectrum(n, [], n - 1, Fraction(-3, 4), 2),
                                                spectrum(n, [300] * 2, 4, Fraction(-1, 2), 3)], 3)
    ok &= run("4 blocks, one needs refinement", [spectrum(n, [], n - 1, Fraction(-3, 4), s) for s in range(3)]
              + [spectrum(n, [200] * 4, 4, Fraction(-9, 4), 3)], 1)
print("all ok" if ok else "FAILURES")
sys.exit(0 if ok else 1)
