import sys, time, random, torch, mpmath as mm, numpy as np
sys.path.insert(0, '.')
from sdpvibe.mpmat import from_ints, to_ints
from sdpvibe.basecase import chol_inv_tile

def spd(n, logcond, seed=0):
    rng = np.random.default_rng(seed)
    Qd, _ = np.linalg.qr(rng.standard_normal((n, n)))
    mm.mp.prec = 2400
    Q = mm.matrix(Qd.tolist())
    D = mm.diag([mm.mpf(10) ** (-logcond * k / (n - 1)) for k in range(n)])
    A = Q * D * Q.T
    d = [1 / mm.sqrt(A[i, i]) for i in range(n)]
    return mm.matrix(n, n, lambda i, j: A[i, j] * d[i] * d[j]) if False else mm.matrix([[A[i, j] * d[i] * d[j] for j in range(n)] for i in range(n)])

L = 34; FX = 32 * L - 40
for logc in (10, 60, 200):
    n = 32
    A = spd(n, logc)
    Aint = [[int(mm.nint(A[i, j] * mm.mpf(2) ** FX)) for j in range(n)] for i in range(n)]
    Am = from_ints(Aint, -FX, L)
    chol_inv_tile(Am)                         # warm-up / compile
    torch.cuda.synchronize(); t = time.time()
    Lg, Xg = chol_inv_tile(Am); torch.cuda.synchronize(); dt = time.time() - t
    # reference from the *same rounded input*
    Ar = mm.matrix([[mm.mpf(Aint[i][j]) / mm.mpf(2) ** FX for j in range(n)] for i in range(n)])
    Lr = mm.cholesky(Ar); Xr = mm.inverse(Lr)
    Li, le = to_ints(Lg); Xi, xe = to_ints(Xg)
    errL = max(abs(mm.mpf(Li[i][j]) * mm.mpf(2) ** le[j] - Lr[i, j]) for i in range(n) for j in range(n))
    nX = max(abs(Xr[i, j]) for i in range(n) for j in range(n))
    errX = max(abs(mm.mpf(Xi[i][j]) * mm.mpf(2) ** xe[j] - Xr[i, j]) for i in range(n) for j in range(n)) / nX
    print(f"cond 1e{logc}: max|L err| = 2^{float(mm.log(errL, 2)):.0f}, rel X err = 2^{float(mm.log(errX, 2)):.0f}, "
          f"max|X| = 2^{float(mm.log(nX, 2)):.0f}, GPU time {dt*1e3:.1f} ms")
