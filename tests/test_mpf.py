"""MPF (per-entry floating point) vs mpmath: mul, add, sum, gemm, Cholesky solves."""
import sys, random
import mpmath as mp
import numpy as np
import torch
sys.path.insert(0, '.')
from sdpvibe import mpf as F

L = 26
NB = F.NBITS(L)
mp.mp.prec = 3 * NB
random.seed(1)


def rnd(n, spread=200):
    return [mp.mpf(random.choice([-1, 1])) * mp.rand() * mp.mpf(2) ** random.randint(-spread, spread)
            for _ in range(n)]


def relerr(got, ref):
    got, ref = np.asarray(got, dtype=object).reshape(-1), np.asarray(ref, dtype=object).reshape(-1)
    worst = mp.mpf(0)
    for g, r in zip(got, ref):
        d = abs(g - r) / (abs(r) if r != 0 else 1)
        worst = max(worst, d)
    return float(mp.log(worst, 2)) if worst > 0 else -9999


n = 500
a, b = rnd(n), rnd(n)
b[3] = mp.mpf(0); a[7] = -b[7]                                    # zero, exact cancellation
A, B = F.from_mp(a, L), F.from_mp(b, L)
ar, br = A.to_mp(), B.to_mp()                                       # rounded inputs
print("roundtrip  log2 rel err:", relerr(ar, a))
print("mul        log2 rel err:", relerr(F.mul(A, B).to_mp(), [x * y for x, y in zip(ar, br)]))
print("add        log2 rel err:", relerr(F.add(A, B).to_mp(), [x + y for x, y in zip(ar, br)]))
print("sub        log2 rel err:", relerr(F.add(A, B, -1).to_mp(), [x - y for x, y in zip(ar, br)]))
c = rnd(n, 10)
C = F.from_mp(c, L); cr = C.to_mp()
s = F.total(C); print("sum        log2 rel err:", relerr([s], [mp.fsum(cr)]))
print("dot        log2 rel err:", relerr([F.dot(C, C)], [mp.fsum([x * x for x in cr])]))

# gemm with rows/cols of different scales
M, K, N = 37, 45, 29
Am = [[(mp.rand() - mp.mpf(0.5)) * mp.mpf(2) ** (i % 7 * 30) for j in range(K)] for i in range(M)]
Bm = [[(mp.rand() - mp.mpf(0.5)) * mp.mpf(2) ** (-(j % 5) * 50) for j in range(N)] for i in range(K)]
GA = F.from_mp([x for r in Am for x in r], L, (M, K)); GB = F.from_mp([x for r in Bm for x in r], L, (K, N))
Ar, Br = GA.to_mp(), GB.to_mp()
Cref = mp.matrix(Ar.tolist()) * mp.matrix(Br.tolist())
Cg = F.gemm(GA, GB).to_mp()
print("gemm       log2 rel err (entrywise; normwise bound applies):", relerr(Cg, [Cref[i, j] for i in range(M) for j in range(N)]))
Cg2 = F.gemm(GB, GA, transA=True, transB=True).to_mp()
print("gemm^T     log2 rel err:", relerr(Cg2, [Cref[j, i] for i in range(N) for j in range(M)]))

# Cholesky of an ill-conditioned SPD matrix with wildly scaled rows (like SDPB's X, S)
nn = 70
mp.mp.prec = 4 * NB
Q = mp.randmatrix(nn) - mp.mpf(0.5)
D = mp.diag([mp.mpf(2) ** random.randint(-300, 300) for _ in range(nn)])
W = mp.diag([mp.mpf(10) ** (-mp.mpf(80) * i / nn) for i in range(nn)])            # cond ~1e80
Aspd = D * Q * W * Q.T * D
GS = F.from_mp([Aspd[i, j] for i in range(nn) for j in range(nn)], L, (nn, nn))
Ar = mp.matrix(GS.to_mp().tolist())
ch = F.Chol(GS)
Bv = mp.randmatrix(nn, 3) - mp.mpf(0.5)
GBv = F.from_mp([Bv[i, j] for i in range(nn) for j in range(3)], L, (nn, 3))
Lref = mp.cholesky(Ar)
x1 = ch.solve_lower(GBv).to_mp()
r1 = mp.lu_solve(Lref, Bv) if False else None
y = Lref ** -1 * Bv
print("L^-1 B     log2 normwise err:", float(mp.log(mp.mnorm(mp.matrix(x1.tolist()) - y, 1) / mp.mnorm(y, 1), 2)))
x2 = ch.solve_upper(GBv).to_mp()
y2 = Lref.T ** -1 * Bv
print("L^-T B     log2 normwise err:", float(mp.log(mp.mnorm(mp.matrix(x2.tolist()) - y2, 1) / mp.mnorm(y2, 1), 2)))
x3 = ch.solve(GBv).to_mp()
y3 = Ar ** -1 * Bv
print("A^-1 B     log2 normwise err:", float(mp.log(mp.mnorm(mp.matrix(x3.tolist()) - y3, 1) / mp.mnorm(y3, 1), 2)),
      f"(cond ~ 2^{float(mp.log(mp.mnorm(Ar,1)*mp.mnorm(Ar**-1,1),2)):.0f}, NB={NB})")
