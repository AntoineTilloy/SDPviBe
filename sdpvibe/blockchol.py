"""Tiled Cholesky factorisation of the (large) N x N matrix Q, with the interface of mpf.Chol.

The monolithic factorisation (mpf.Chol on the whole Q) needs ~9x the memory of Q (equilibrated fixed-point
copy, padding, recursion temporaries).  Here Q is cut into b x b tiles (lower triangle only), and the
right-looking blocked algorithm

    for k:  L_kk = chol(A_kk);  L_ik = A_ik L_kk^-T (i > k);  A_ij -= L_ik L_jk^T (i >= j > k)

overwrites the tiles in place, so the peak is ~(1/2 + O(b/N)) Q plus a few tiles.  Every step uses the
validated primitives (Chol on a tile, triangular solves, exact GEMM, add); the result agrees with the
monolithic factorisation to working precision (different rounding order).
"""
import torch
import mpmath as mp
from .mpf import MPF, Chol, gemm, add


def _tile(Q, r0, r1, c0, c1):
    return MPF(Q.m[:, r0:r1, c0:c1].contiguous(), Q.e[r0:r1, c0:c1].contiguous())


def _sym_lower(A):
    """full symmetric tile from its lower triangle"""
    n = A.shape[-1]
    iu = torch.triu_indices(n, n, 1, device=A.e.device)
    m, e = A.m.clone(), A.e.clone()
    m[:, iu[0], iu[1]] = A.m[:, iu[1], iu[0]]
    e[iu[0], iu[1]] = A.e[iu[1], iu[0]]
    return MPF(m, e)


class BlockChol:
    def __init__(self, holder, b=2048):
        """holder: list [Q] (MPF (N, N), lower triangle valid); Q is released as tiles are formed"""
        Q = holder.pop()
        N = Q.shape[0]
        self.N, self.b, self.L = N, b, Q.L
        cuts = list(range(0, N, b)) + [N]
        self.cuts = cuts
        T = self.T = len(cuts) - 1
        A = {}
        for i in range(T):
            for j in range(i + 1):
                A[i, j] = _tile(Q, cuts[i], cuts[i + 1], cuts[j], cuts[j + 1])
        del Q
        self.D = []
        for k in range(T):
            ck = Chol(_sym_lower(A[k, k])); del A[k, k]
            self.D.append(ck)
            for i in range(k + 1, T):                                   # L_ik = A_ik L_kk^-T = (L_kk^-1 A_ik^T)^T
                A[i, k] = ck.solve_lower(A[i, k].T).T
            for i in range(k + 1, T):
                for j in range(k + 1, i + 1):
                    A[i, j] = add(A[i, j], gemm(A[i, k], A[j, k], transB=True), -1)
        self.Lo = A                                                     # off-diagonal factor tiles (i > k)

    def _rows(self, B, k):
        return MPF(B.m[:, self.cuts[k]:self.cuts[k + 1]].contiguous(), B.e[self.cuts[k]:self.cuts[k + 1]].contiguous())

    @staticmethod
    def _cat(parts):
        return MPF(torch.cat([p.m for p in parts], 1), torch.cat([p.e for p in parts], 0))

    def solve_lower(self, B):
        """L^-1 B, B: MPF (N, m)"""
        y = []
        for k in range(self.T):
            r = self._rows(B, k)
            for j in range(k):
                r = add(r, gemm(self.Lo[k, j], y[j]), -1)
            y.append(self.D[k].solve_lower(r))
        return self._cat(y)

    def solve_upper(self, Y):
        """L^-T Y"""
        x = [None] * self.T
        for k in reversed(range(self.T)):
            r = self._rows(Y, k)
            for i in range(k + 1, self.T):
                r = add(r, gemm(self.Lo[i, k], x[i], transA=True), -1)
            x[k] = self.D[k].solve_upper(r)
        return self._cat(x)

    def solve(self, B):
        return self.solve_upper(self.solve_lower(B))

    def log2_cond(self):
        """SDPB's Cholesky condition number estimate (max/min diag of L)^2, log2"""
        from .solver import chol_log2_cond_parts
        hi = lo = None
        for ck in self.D:
            h, l = chol_log2_cond_parts(ck)
            hi = h if hi is None else max(hi, h); lo = l if lo is None else min(lo, l)
        return 2 * (hi - lo)
