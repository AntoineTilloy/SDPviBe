"""End-to-end test + timing of GPU multiprecision Cholesky/inverse and TRSM.

Test matrix: A = V D V^T with V a random integer matrix and D = diag(2^-e_k) spanning a chosen
dynamic range (=> condition number ~ 2^range), built exactly with Python ints, then equilibrated
by powers of two (exact) so the diagonal lies in [1/4, 1].
Checks (all in multiprecision on GPU): ||L L^T - A||_max, ||X L - I||_max, and for small n the
error against an mpmath reference.
"""
import argparse, sys, time, random
import numpy as np, torch
sys.path.insert(0, '.')
from sdpvibe.mpmat import MP, from_ints, to_ints, gemm_nt, add, bitlen_limbs
from sdpvibe.linalg import chol_inv, trsm, fx_exp, chol, trsm_left

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--nrhs", type=int, default=0)
ap.add_argument("--range", type=int, default=400, help="log2 of the eigenvalue spread")
ap.add_argument("--L", type=int, default=34)
ap.add_argument("--ref", type=int, default=0, help="also compare with mpmath (slow, small n)")
ap.add_argument("--method", default="stable", choices=["stable", "inverse"])
ap.add_argument("--panel", default="subst")
ap.add_argument("--refine", type=int, default=0)
args = ap.parse_args()
n, L = args.n, args.L
FX = -fx_exp(L)
dev = "cuda"


def build(n, rng_bits, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.integers(-8, 9, size=(n, n))
    ek = np.linspace(0, rng_bits, n).astype(int)                 # D_k = 2^-ek
    emax = int(ek.max())
    # A * 2^emax = V diag(2^(emax-ek)) V^T  (exact integers, via object arrays in column chunks)
    W = (V.astype(object) * np.array([1 << int(emax - e) for e in ek], dtype=object)[None, :])
    A = W.dot(V.T.astype(object))                                # n x n Python ints
    # equilibrate by powers of two: s_i = 2^-round(log2(A_ii)/2)
    sh = [ (int(A[i, i]).bit_length() - 1) // 2 for i in range(n)]
    # value a_ij * 2^(-sh_i - sh_j) in fixed point with FX fractional bits
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = FX - sh[i] - sh[j] - 1                           # extra /2 => diag in [1/4, 1)
            v = int(A[i, j])
            out[i][j] = v << s if s >= 0 else (v + (1 << (-s - 1))) >> (-s)
    return out


t = time.time()
Aint = build(n, args.range)
print(f"built A (n={n}, eigen-spread 2^{args.range}) in {time.time()-t:.1f}s")
A = from_ints(Aint, -FX, L)
e = -FX

fact = (lambda M: chol(M, tile_inverses=True, panel_leaf=args.panel, refine=args.refine)) if args.method == "stable" else (lambda M: chol_inv(M))
fact(MP(A.limbs[:, :64, :64].contiguous(), e))                       # warm-up / compile
torch.cuda.synchronize(); t = time.time()
Lf, X = fact(A)
torch.cuda.synchronize(); tchol = time.time() - t
print(f"GPU {args.method} Cholesky n={n}: {tchol:.2f} s")


def maxabs_log2(M):
    bl = bitlen_limbs(M.limbs).max()
    return int(bl) + int(M.exp.max())


# backward error  L L^T - A
LLt = gemm_nt(Lf, Lf, out_exp=e)
R = add(LLt, A, -1, out_exp=e)
print(f"  max|L L^T - A| ~ 2^{maxabs_log2(R)}   (unit roundoff 2^-{FX})")
if args.method == "inverse":
    XL = gemm_nt(X, Lf.t(), out_exp=e)
    I = from_ints([[(1 << FX) if i == j else 0 for j in range(n)] for i in range(n)], e, L)
    R2 = add(XL, I, -1, out_exp=e)
    print(f"  max|X L - I|   ~ 2^{maxabs_log2(R2)},   max|X| ~ 2^{maxabs_log2(X)}")

if args.nrhs:
    rng = random.Random(3)
    Bint = [[rng.getrandbits(32 * L - 45) * rng.choice([-1, 1]) for _ in range(args.nrhs)] for _ in range(n)]
    B = from_ints(Bint, -FX, L)
    solve = (lambda M: trsm_left(Lf, M, X)) if args.method == "stable" else (lambda M: trsm(X, M))
    solve(MP(B.limbs[:, :, :64].contiguous(), B.exp[:64]))                # warm-up
    torch.cuda.synchronize(); t = time.time()
    P = solve(B); torch.cuda.synchronize(); ttr = time.time() - t
    print(f"GPU {args.method} trsm n={n} nrhs={args.nrhs}: {ttr:.2f} s")
    # residual L P - B, relative to max|B| = 2^(32L-45-FX)
    LP = gemm_nt(Lf, MP(P.limbs.transpose(1, 2).contiguous(), 0), out_exp=e, bt_rowexp=P.exp)
    R3 = add(LP, B, -1, out_exp=e)
    print(f"  max|L P - B| / max|B| ~ 2^{maxabs_log2(R3) - (32 * L - 45 - FX)},  max|P| ~ 2^{maxabs_log2(P)}")

if args.ref:
    import mpmath as mm
    mm.mp.prec = 2 * FX + 200
    Ar = mm.matrix([[mm.mpf(v) / mm.mpf(2) ** FX for v in row] for row in Aint])
    t = time.time(); Lr = mm.cholesky(Ar); tref = time.time() - t
    Li, le = to_ints(Lf)
    err = max(abs(mm.mpf(Li[i][j]) * mm.mpf(2) ** le[j] - Lr[i, j]) for i in range(n) for j in range(i + 1))
    print(f"  vs mpmath ({mm.mp.prec}-bit, {tref:.0f}s): max|L - L_ref| ~ 2^{float(mm.log(err, 2)):.0f}")
    # same-precision floating-point Cholesky (what SDPB's BigFloat path would do)
    mm.mp.prec = FX
    t = time.time(); Lf_ = mm.cholesky(Ar); tf = time.time() - t
    mm.mp.prec = 2 * FX + 200
    errf = max(abs(Lf_[i, j] - Lr[i, j]) for i in range(n) for j in range(i + 1))
    LLf = Lf_ * Lf_.T
    bwf = max(abs(LLf[i, j] - Ar[i, j]) for i in range(n) for j in range(n))
    print(f"  mpmath {FX}-bit floating Cholesky: max|L - L_ref| ~ 2^{float(mm.log(errf, 2)):.0f}, "
          f"max|L L^T - A| ~ 2^{float(mm.log(bwf, 2)):.0f}")
