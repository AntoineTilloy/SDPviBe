"""Multiprecision dense linear algebra on GPU built from exact RNS GEMMs + CUDA-core base cases.

chol_inv(A): A SPD, equilibrated (|a_ij| <= 1), fixed point with exponent fx_exp(L).
   Returns L (fixed point, same exponent) and X = L^{-1} (block floating point), via
     (L11, X11) = chol_inv(A11)
     L21 = A21 X11^T                     [gemm]
     S   = A22 - L21 L21^T               [gemm, exact subtraction]
     (L22, X22) = chol_inv(S)
     X21 = -X22 (L21 X11)                [2 gemms]
   Only the bt x bt diagonal tiles need division / square roots (basecase.py).
trsm(X, B): P = L^{-1} B = X B           [one gemm, per-column exponents]
"""
import torch
from .mpmat import MP, gemm_nt, add, rescale, from_ints, headroom_bits, neg
from .basecase import chol_inv_tile
from .rns import dev


def fx_exp(L): return -(32 * L - 40)


def _cat(blocks, L):
    """2x2 block matrix [[B11, B12], [B21, B22]] (None = zero block) with a common exponent."""
    e = max(b.uniform_exp() for row in blocks for b in row if b is not None)
    n = [next(b for b in blocks[i] if b is not None).shape[0] for i in (0, 1)]
    m = [next(blocks[i][j] for i in (0, 1) if blocks[i][j] is not None).shape[1] for j in (0, 1)]
    out = torch.zeros((L, n[0] + n[1], m[0] + m[1]), dtype=torch.int32, device=dev)
    for i, r0 in ((0, 0), (1, n[0])):
        for j, c0 in ((0, 0), (1, m[0])):
            b = blocks[i][j]
            if b is None: continue
            if b.uniform_exp() != e: b = rescale(b, e)
            out[:, r0:r0 + n[i], c0:c0 + m[j]] = b.limbs
    return MP(out, e)


def chol_inv(A, bt=32, stats=None):
    n = A.shape[0]; L = A.L; e = fx_exp(L)
    assert A.uniform_exp() == e
    if n <= bt:
        if n < bt:                                     # pad with identity
            P = torch.zeros((L, bt, bt), dtype=torch.int32, device=dev)
            P[:, :n, :n] = A.limbs
            one = from_ints([[1 << (32 * L - 40)]], e, L).limbs[:, 0, 0]
            for i in range(n, bt): P[:, i, i] = one
            Lt, Xt = chol_inv_tile(MP(P, e))
            return Lt[:n, :n], Xt[:n, :n]
        return chol_inv_tile(A)
    n1 = (n // 2 + bt - 1) // bt * bt
    A11, A21, A22 = A[:n1, :n1], A[n1:, :n1], A[n1:, n1:]
    L11, X11 = chol_inv(A11, bt, stats)
    L21 = gemm_nt(A21, X11, out_exp=e)                 # A21 X11^T
    S = add(A22, gemm_nt(L21, L21, out_exp=e), -1, out_exp=e)
    L22, X22 = chol_inv(S, bt, stats)
    T = gemm_nt(L21, X11.t(), out_exp="mat")           # L21 X11
    X21 = gemm_nt(X22, T.t(), out_exp="mat")           # X22 L21 X11
    X21 = neg(X21)
    Lf = _cat([[L11, None], [L21, L22]], L)
    Xf = _cat([[X11, None], [X21, X22]], L)
    return Lf, Xf


def trsm(X, B, kc=4096):
    """P = X B (X = L^{-1}); B may have per-column exponents; P gets per-column exponents."""
    Bt = MP(B.limbs.transpose(1, 2).contiguous(), 0)
    return gemm_nt(X, Bt, out_exp="col", kc=kc, bt_rowexp=B.exp)


# ---------------------------------------------------------------------------------------------
# Stable variant: recursive Cholesky with recursive triangular solves. Leaves (bt x bt diagonal
# tiles) are factored and applied by forward substitution on CUDA cores (basecase.trsv_tile);
# everything above the leaves is exact RNS GEMMs on tensor cores.
from .basecase import chol_tile, trsv_tile, apply_tile_inverse


def _split(n, bt): return (n // 2 + bt - 1) // bt * bt


def chol(A, bt=32, tile_inverses=False, panel_leaf="subst", refine=0):
    """A (n x n) SPD equilibrated, fixed point exponent fx_exp(L) -> (L, tiles)"""
    n = A.shape[0]; L = A.L; e = fx_exp(L)
    if n <= bt:
        assert n == bt, "pad n to a multiple of the tile size"
        r = chol_tile(A, want_inverse=tile_inverses)
        return r[0], [r[1]]
    n1 = _split(n, bt)
    L11, t1 = chol(A[:n1, :n1], bt, tile_inverses, panel_leaf, refine)
    L21 = trsm_right(A[n1:, :n1], L11, t1, bt, panel_leaf, refine)
    S = add(A[n1:, n1:], gemm_nt(L21, L21, out_exp=e), -1, out_exp=e)
    L22, t2 = chol(S, bt, tile_inverses, panel_leaf, refine)
    return _cat([[L11, None], [L21, L22]], L), t1 + t2


def trsm_right(B, Lf, tiles, bt=32, leaf="subst", refine=0):
    """X with X L^T = B (B: m x n fixed point; X fixed point, e.g. a Cholesky panel)"""
    n = Lf.shape[0]; e = fx_exp(Lf.L)
    if n <= bt and leaf == "gemm" and tiles[0].X is not None:
        t = tiles[0]
        X = gemm_nt(B, t.X, out_exp=e)                           # B X_t^T
        for _ in range(refine):                                  # X += (B - X L^T) X_t^T
            Rr = add(B, gemm_nt(X, t.Lmp, out_exp=e), -1, out_exp=e)
            X = add(X, gemm_nt(Rr, t.X, out_exp=e), 1, out_exp=e)
        return X
    if n <= bt:
        m = B.shape[0]
        out, _ = trsv_tile(tiles[0], B.limbs, torch.full((m,), e, dtype=torch.int64, device=dev), out_exp=e)
        return MP(out, e)
    n1 = _split(n, bt); k1 = _ntiles(n1, bt)
    X1 = trsm_right(B[:, :n1], Lf[:n1, :n1], tiles[:k1], bt, leaf, refine)
    B2 = add(B[:, n1:], gemm_nt(X1, Lf[n1:, :n1], out_exp=e), -1, out_exp=e)
    X2 = trsm_right(B2, Lf[n1:, n1:], tiles[k1:], bt, leaf, refine)
    return MP(torch.cat([X1.limbs, X2.limbs], 2), e)


def trsm_left(Lf, B, tiles, bt=32, leaf="gemm", refine=0):
    """P = L^{-1} B, B (n x m) with per-column exponents; P with per-column exponents.
    leaf: 'subst' (CUDA-core substitution) or 'gemm' (tensor cores, explicit tile inverses)"""
    n = Lf.shape[0]
    if n <= bt:
        if leaf == "gemm" and tiles[0].X is not None:
            return apply_tile_inverse(tiles[0], B, refine)
        out, ex = trsv_tile(tiles[0], B.limbs.transpose(1, 2).contiguous(), B.exp)
        return MP(out.transpose(1, 2).contiguous(), ex)
    n1 = _split(n, bt); k1 = _ntiles(n1, bt)
    P1 = trsm_left(Lf[:n1, :n1], B[:n1, :], tiles[:k1], bt, leaf, refine)
    U = gemm_nt(Lf[n1:, :n1], MP(P1.limbs.transpose(1, 2).contiguous(), 0), out_exp="col", bt_rowexp=P1.exp)
    B2 = add(B[n1:, :], U, -1, out_exp="align")
    P2 = trsm_left(Lf[n1:, n1:], B2, tiles[k1:], bt, leaf, refine)
    e = torch.maximum(P1.exp, P2.exp)                          # common per-column exponent
    P1, P2 = rescale(P1, e), rescale(P2, e)
    return MP(torch.cat([P1.limbs, P2.limbs], 1), e)


def _ntiles(n, bt):
    """number of diagonal tiles produced by chol() on an n x n matrix"""
    if n <= bt: return 1
    n1 = _split(n, bt)
    return _ntiles(n1, bt) + _ntiles(n - n1, bt)
