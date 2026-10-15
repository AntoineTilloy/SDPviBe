"""GPU benchmark of multiprecision GEMM, Cholesky and TRSM at larger sizes (matrices built on GPU).

A = W W^T with W = V diag(2^-e_k/2), V random small integers: eigenvalue spread ~2^range,
equilibrated by exact power-of-two scaling. Checks: ||L L^T - A||, ||L P - B||.
"""
import argparse, sys, time
import torch
sys.path.insert(0, '.')
from sdpvibe.mpmat import MP, gemm_nt, add, rescale, bitlen_limbs
from sdpvibe.limbops import shift_round
from sdpvibe.linalg import chol, trsm_left, fx_exp

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--nrhs", type=int, default=1024)
ap.add_argument("--range", type=int, default=400)
ap.add_argument("--L", type=int, default=34)
ap.add_argument("--skip_check", type=int, default=0)
ap.add_argument("--gemm", type=int, default=1)
args = ap.parse_args()
n, L = args.n, args.L
e = fx_exp(L); FX = -e
dev = "cuda"
g = torch.Generator(device=dev); g.manual_seed(0)


def small_int_mp(V, exps):
    """MP from a small-integer matrix V (int64) with per-column exponents"""
    limbs = torch.zeros((L, *V.shape), dtype=torch.int32, device=dev)
    limbs[0] = V.to(torch.int32)
    limbs[1:] = torch.where(V < 0, -1, 0).to(torch.int32)[None]
    return MP(limbs, exps)


def build(n, rng_bits):
    V = torch.randint(-8, 9, (n, n), device=dev, generator=g)
    half = torch.linspace(0, rng_bits // 2, n, device=dev).long()           # D_k^(1/2) = 2^-half_k
    W = small_int_mp(V, -half)
    W = rescale(W, int(-half.max()))                                     # exact (left shifts)
    A = gemm_nt(W, W, out_exp="mat")
    # exact power-of-two equilibration into fixed point
    d = torch.diagonal(A.limbs, 0, 1, 2).contiguous()                    # (L, n)
    sh = torch.div(bitlen_limbs(d) - 1 + A.exp[0], 2, rounding_mode="floor")
    s = (e - A.exp[0] + sh[:, None] + sh[None, :] + 1).reshape(-1)
    Aeq = shift_round(A.limbs.view(L, n * n), s, L).view(L, n, n)
    return MP(Aeq, e)


def maxabs_log2(M):
    return int(bitlen_limbs(M.limbs).max()) + int(M.exp.max())


def timed(f, *a, **k):
    f(*a, **k)                                   # first call compiles kernels for new shapes
    torch.cuda.synchronize(); t = time.time(); r = f(*a, **k); torch.cuda.synchronize()
    return r, time.time() - t


A = build(n, args.range)
B = MP(torch.randint(-2**31, 2**31, (L, n, args.nrhs), dtype=torch.int64, device=dev, generator=g).to(torch.int32), e)
B.limbs[-1] &= 0xFF                                                        # |B| < ~1 in fixed point
B.limbs[-2:] = torch.where(torch.rand((1, n, args.nrhs), device=dev, generator=g) < 0.5, 0, B.limbs[-2:])

# warm-up (compiles kernels for these shapes)
chol(MP(A.limbs[:, :256, :256].contiguous(), e))
gemm_nt(MP(A.limbs[:, :, :256].contiguous(), e), MP(A.limbs[:, :, :256].contiguous(), e), out_exp="mat")

if args.gemm:
    C, tg = timed(gemm_nt, A, MP(B.limbs.transpose(1, 2).contiguous(), e), out_exp="col"); del C
    print(f"GEMM  {n}x{n} @ {n}x{args.nrhs}: {tg:.2f} s  ({n*n*args.nrhs/tg/1e9:.1f} G mp-MAC/s)")
for pl, rf in (("gemm", 1),):
    (Lf, invs), tc = timed(chol, A, tile_inverses=True, panel_leaf=pl, refine=rf)
    print(f"Cholesky n={n} (panel leaves {pl}, refine={rf}): {tc:.2f} s  ({n**3/6/tc/1e9:.2f} G mp-MAC/s equiv)")
(Lf, invs) = chol(A, tile_inverses=True, panel_leaf="gemm", refine=1)
P, tt = timed(trsm_left, Lf, B, invs, leaf="gemm")
print(f"TRSM  n={n} nrhs={args.nrhs} (tensor-core leaves): {tt:.2f} s  ({n*n*args.nrhs/2/tt/1e9:.2f} G mp-MAC/s equiv)")
if not args.skip_check:
    R = add(gemm_nt(Lf, Lf, out_exp=e), A, -1, out_exp=e)
    print(f"  max|L L^T - A| ~ 2^{maxabs_log2(R)}  (unit 2^-{FX})")
    LP = gemm_nt(Lf, MP(P.limbs.transpose(1, 2).contiguous(), 0), out_exp=e, bt_rowexp=P.exp)
    R3 = add(LP, B, -1, out_exp=e)
    print(f"  max|L P - B| ~ 2^{maxabs_log2(R3)},  max|P| ~ 2^{maxabs_log2(P)}")
print(f"peak GPU memory {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
