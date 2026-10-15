"""Batched Cholesky + TRSM over many equal-sized blocks (SDPB's Schur blocks S_j), vs per-block.
Checks batched results against the single-block path and reports residuals."""
import argparse, sys, time, torch
sys.path.insert(0, '.')
from sdpvibe.mpmat import MP, gemm_nt, add, bitlen_limbs
from sdpvibe.limbops import shift_round
from sdpvibe.linalg import chol, trsm_left, fx_exp
from sdpvibe.batched import BMP, bchol, btrsm_left, bgemm_nt, badd

ap = argparse.ArgumentParser()
ap.add_argument("--nb", type=int, default=32)
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--nrhs", type=int, default=256)
ap.add_argument("--range", type=int, default=300)
ap.add_argument("--L", type=int, default=34)
args = ap.parse_args()
nb, n, L = args.nb, args.n, args.L
e = fx_exp(L); FX = -e; dev = "cuda"
g = torch.Generator(device=dev); g.manual_seed(0)

def small_int_mp(V, exps):
    limbs = torch.zeros((L, *V.shape), dtype=torch.int32, device=dev)
    limbs[0] = V.to(torch.int32); limbs[1:] = torch.where(V < 0, -1, 0).to(torch.int32)[None]
    return MP(limbs, exps)

from sdpvibe.mpmat import rescale
def build(n, rng_bits):
    V = torch.randint(-8, 9, (n, n), device=dev, generator=g)
    half = torch.linspace(0, rng_bits // 2, n, device=dev).long()
    W = rescale(small_int_mp(V, -half), int(-half.max()))
    A = gemm_nt(W, W, out_exp="mat")
    d = torch.diagonal(A.limbs, 0, 1, 2).contiguous()
    sh = torch.div(bitlen_limbs(d) - 1 + A.exp[0], 2, rounding_mode="floor")
    s = (e - A.exp[0] + sh[:, None] + sh[None, :] + 1).reshape(-1)
    return shift_round(A.limbs.view(L, n * n), s, L).view(L, n, n)

blocks = [build(n, args.range) for _ in range(nb)]
A = BMP(torch.stack(blocks, 1).contiguous(), e)                     # (L, nb, n, n)
Bl = torch.randint(-2**31, 2**31, (L, nb, n, args.nrhs), dtype=torch.int64, device=dev, generator=g).to(torch.int32)
Bl[-1] &= 0xFF; Bl[-2:] = torch.where(torch.rand((1, nb, n, args.nrhs), device=dev, generator=g) < 0.5, 0, Bl[-2:])
B = BMP(Bl, e)

def timed(f, *a, **k):
    f(*a, **k); torch.cuda.synchronize(); t = time.time(); r = f(*a, **k); torch.cuda.synchronize()
    return r, time.time() - t

(Lb, tiles), tc = timed(bchol, A)
P, tt = timed(btrsm_left, Lb, B, tiles)
print(f"batched  nb={nb} n={n} nrhs={args.nrhs}: chol {tc:.2f} s, trsm {tt:.2f} s   "
      f"(per block: {tc/nb*1e3:.1f} ms + {tt/nb*1e3:.1f} ms)")
# compare with single-block path on block 0 and a couple of others
def single(i):
    Ai = MP(A.limbs[:, i].contiguous(), e); Bi = MP(B.limbs[:, i].contiguous(), e)
    Li, ti = chol(Ai, tile_inverses=True, panel_leaf="gemm", refine=1)
    return Li, trsm_left(Li, Bi, ti)
single(0); torch.cuda.synchronize(); t = time.time()
for i in range(min(nb, 4)): Li, Pi = single(i)
torch.cuda.synchronize(); ts = (time.time() - t) / min(nb, 4)
print(f"single-block path: {ts*1e3:.0f} ms per block  -> batched speedup x{ts*nb/(tc+tt):.1f}")
Li, Pi = single(nb - 1)
dL = add(MP(Lb.limbs[:, nb - 1].contiguous(), e), Li, -1, out_exp=e)
print(f"  batched vs single, last block: max|dL| ~ 2^{int(bitlen_limbs(dL.limbs).max()) - FX}")
LLt = bgemm_nt(Lb, Lb, out_exp=e); R = badd(LLt, A, -1, out_exp=e)
print(f"  max_b |L L^T - A| ~ 2^{int(bitlen_limbs(R.limbs).max()) - FX}")
