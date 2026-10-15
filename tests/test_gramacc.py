"""GramAcc (Q = sum_c P_c^T P_c in RNS, one CRT) vs one GEMM over all rows."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, mpmath as mp
from sdpvibe import mpf as F
from sdpvibe.solver import _rand_mpf

L = 18; N = 96
mp.mp.prec = F.NBITS(L) + 64
g = torch.Generator(device="cuda"); g.manual_seed(1)
for scales in ([0, 0, 0], [0, 10, -40], [0, 60, -5]):                 # chunk scale jumps (bits); 60 > HEAD -> flush
    chunks = [_rand_mpf((r, N), L, g).shift(sc) for r, sc in zip((300, 17, 129), scales)]
    full = F.MPF(torch.cat([c.m for c in chunks], 1), torch.cat([c.e for c in chunks], 0))
    ref = F.gemm(full, full, transA=True)
    acc = F.GramAcc(N, L)
    for c in chunks: acc.add(c)
    Q = acc.result()
    low = torch.tril(torch.ones(N, N, dtype=torch.bool, device="cuda"))   # lower triangle is the result
    Q = F.MPF(torch.where(low, Q.m, ref.m), torch.where(low, Q.e, ref.e))
    d = F.add(Q, ref, -1)
    rel = F.max_abs(d) / F.max_abs(ref)
    print(f"scales {scales}: flushes {acc.flushes}, max |Q - Q_ref| / max |Q_ref| = {mp.nstr(rel, 3)} "
          f"(2^-NB = {mp.nstr(mp.mpf(2) ** -F.NBITS(L), 3)})")
