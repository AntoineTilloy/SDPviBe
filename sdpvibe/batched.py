"""Batched multiprecision linear algebra: many independent equal-sized blocks per kernel launch.

SDPB factors hundreds of Schur-complement blocks S_j per iteration; one launch per block would
leave the GPU idle. A BMP holds a stack of nb matrices:
    limbs: int32 (L, nb, R, C) limb planes;  exp: Python int (uniform) or (nb, C) int64 tensor.
Blocks of different sizes are handled by grouping blocks of equal size (or padding).
"""
import torch
import triton
import triton.language as tl
from .rns import dev, _fmod, gemm_config
from .mpmat import ctx, headroom_bits
from .limbops import shift_round, add_same, bitlen_words, _shifted_word


class BMP:
    __slots__ = ("limbs", "exp")

    def __init__(self, limbs, exp):
        self.limbs, self.exp = limbs, exp            # exp: int or (nb, C) tensor

    @property
    def shape(self): return tuple(self.limbs.shape[1:])
    @property
    def L(self): return self.limbs.shape[0]

    def expt(self):
        nb, R, C = self.shape
        if isinstance(self.exp, int):
            return torch.full((nb, C), self.exp, dtype=torch.int64, device=dev)
        return self.exp

    def __getitem__(self, idx):
        r, c = idx
        e = self.exp if isinstance(self.exp, int) else self.exp[:, c].contiguous()
        return BMP(self.limbs[:, :, r, c].contiguous(), e)

    def t(self):
        assert isinstance(self.exp, int)
        return BMP(self.limbs.transpose(2, 3).contiguous(), self.exp)

    def flat(self): return self.limbs.reshape(self.L, -1)


def brescale(A, new_exp, Lout=None):
    nb, R, C = A.shape; Lo = Lout or A.L
    if isinstance(new_exp, int) and isinstance(A.exp, int):
        s = torch.full((nb * R * C,), new_exp - A.exp, dtype=torch.int64, device=dev)
        return BMP(shift_round(A.flat(), s, Lo).view(Lo, nb, R, C), new_exp)
    ne = new_exp if torch.is_tensor(new_exp) else torch.full((nb, C), int(new_exp), dtype=torch.int64, device=dev)
    s = (ne - A.expt())[:, None, :].expand(nb, R, C).reshape(-1)
    return BMP(shift_round(A.flat(), s, Lo).view(Lo, nb, R, C), ne.clone())


@triton.jit
def _badd_align_kernel(A, B, EA, EB, OUT, EO, entries, rows, cols, sign: tl.constexpr, L: tl.constexpr, BB: tl.constexpr):
    # Both operands shift right by at least one bit, so their rounding cannot overflow.
    e = tl.program_id(0) * BB + tl.arange(0, BB)
    ok = e < entries
    stride = entries.to(tl.int64)
    col = (e // (rows * cols)) * cols + e % cols
    ea = tl.load(EA + col, mask=ok, other=0); eb = tl.load(EB + col, mask=ok, other=0)
    ex = tl.maximum(ea, eb) + 1
    sa = ex - ea; sb = ex - eb
    qa = sa >> 5; ra = sa & 31
    qb = sb >> 5; rb = sb & 31
    ax = tl.where(tl.load(A + (L - 1) * stride + e, mask=ok, other=0) < 0, 0xffffffff, 0).to(tl.int64)
    bx = tl.where(tl.load(B + (L - 1) * stride + e, mask=ok, other=0) < 0, 0xffffffff, 0).to(tl.int64)
    ca = _shifted_word(A, e, stride, (sa - 1) >> 5, (sa - 1) & 31, ax, 0, L, ok) & 1
    cb = _shifted_word(B, e, stride, (sb - 1) >> 5, (sb - 1) & 31, bx, 0, L, ok) & 1
    carry = tl.zeros((BB,), tl.int64)
    for k in range(L):
        a = _shifted_word(A, e, stride, qa, ra, ax, k, L, ok) + ca
        b = _shifted_word(B, e, stride, qb, rb, bx, k, L, ok) + cb
        ca = a >> 32; cb = b >> 32
        total = (a & 0xffffffff) + sign * (b & 0xffffffff) + carry
        carry = total >> 32
        tl.store(OUT + k * stride + e, total.to(tl.int32), mask=ok)
    tl.store(EO + col, ex, mask=ok & ((e // cols) % rows == 0))


def badd(A, B, alpha=1, out_exp=None):
    nb, R, C = A.shape; L = A.L
    if isinstance(out_exp, int) and A.exp == out_exp and B.exp == out_exp:
        return BMP(add_same(A.flat(), B.flat(), alpha).view(L, nb, R, C), out_exp)
    assert out_exp == "align"
    out = torch.empty((L, nb, R, C), dtype=torch.int32, device=dev)
    e = torch.empty((nb, C), dtype=torch.int64, device=dev)
    _badd_align_kernel[(triton.cdiv(nb * R * C, 128),)](
        A.flat().contiguous(), B.flat().contiguous(), A.expt().contiguous(), B.expt().contiguous(),
        out, e, nb * R * C, R, C, alpha, L, 128, num_warps=4)
    return BMP(out, e)


# ------------------------------------------------------------------ batched RNS GEMM kernel
@triton.jit
def _bgemm_kernel(Ahi, Alo, Bhi, Blo, ACC, mods, minv, c14s, g0, nb, M, N, KC,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """residues A*: (G, nb*M, KC), B*: (G, nb*N, KC); ACC: (n, nb*M, N) int16"""
    pm = tl.program_id(0); pn = tl.program_id(1); z = tl.program_id(2)
    gid = z // nb; b = z % nb
    ri = pm * BM + tl.arange(0, BM)
    rj = pn * BN + tl.arange(0, BN)
    ba = (gid.to(tl.int64) * nb + b) * M * KC
    bb = (gid.to(tl.int64) * nb + b) * N * KC
    hh = tl.zeros((BM, BN), dtype=tl.int32)
    hx = tl.zeros((BM, BN), dtype=tl.int32)
    ll = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, KC, BK):
        ks = k0 + tl.arange(0, BK)
        ma = (ri[:, None] < M) & (ks[None, :] < KC)
        mb = (rj[None, :] < N) & (ks[:, None] < KC)
        pa = ba + ri[:, None] * KC + ks[None, :]
        pb = bb + rj[None, :] * KC + ks[:, None]
        ah = tl.load(Ahi + pa, mask=ma, other=0); al = tl.load(Alo + pa, mask=ma, other=0)
        bh = tl.load(Bhi + pb, mask=mb, other=0); bl = tl.load(Blo + pb, mask=mb, other=0)
        hh = tl.dot(ah, bh, hh, out_dtype=tl.int32)
        hx = tl.dot(ah, bl, hx, out_dtype=tl.int32)
        hx = tl.dot(al, bh, hx, out_dtype=tl.int32)
        ll = tl.dot(al, bl, ll, out_dtype=tl.int32)
    g = g0 + gid
    m = tl.load(mods + g); mi = tl.load(minv + g); c = tl.load(c14s + g)
    o = (g.to(tl.int64) * nb + b) * M * N + ri[:, None] * N + rj[None, :]
    msk = (ri[:, None] < M) & (rj[None, :] < N)
    old = tl.load(ACC + o, mask=msk, other=0).to(tl.int32)
    v = _fmod(hh, m, mi) * c + _fmod(hx, m, mi) * 128 + _fmod(ll, m, mi) + old
    tl.store(ACC + o, _fmod(v, m, mi).to(tl.int16), mask=msk)


@triton.jit
def _syrk_lower_kernel(Ahi, Alo, ACC, mods, minv, c14s, g0, N, KC,
                       BM: tl.constexpr, BK: tl.constexpr):
    """ACC[g] (N x N, int16) += residues of A A^T mod m_g, lower-triangle tiles only (pn <= pm).
    A*: (G, N, KC) int8 hi/lo residues."""
    pm = tl.program_id(0); pn = tl.program_id(1); gid = tl.program_id(2)
    if pn > pm:
        return
    ri = pm * BM + tl.arange(0, BM)
    rj = pn * BM + tl.arange(0, BM)
    ba = gid.to(tl.int64) * N * KC
    hh = tl.zeros((BM, BM), dtype=tl.int32)
    hx = tl.zeros((BM, BM), dtype=tl.int32)
    ll = tl.zeros((BM, BM), dtype=tl.int32)
    for k0 in range(0, KC, BK):
        ks = k0 + tl.arange(0, BK)
        ma = (ri[:, None] < N) & (ks[None, :] < KC)
        mb = (rj[None, :] < N) & (ks[:, None] < KC)
        pa = ba + ri[:, None] * KC + ks[None, :]
        pb = ba + rj[None, :] * KC + ks[:, None]
        ah = tl.load(Ahi + pa, mask=ma, other=0); al = tl.load(Alo + pa, mask=ma, other=0)
        bh = tl.load(Ahi + pb, mask=mb, other=0); bl = tl.load(Alo + pb, mask=mb, other=0)
        hh = tl.dot(ah, bh, hh, out_dtype=tl.int32)
        hx = tl.dot(ah, bl, hx, out_dtype=tl.int32)
        hx = tl.dot(al, bh, hx, out_dtype=tl.int32)
        ll = tl.dot(al, bl, ll, out_dtype=tl.int32)
    g = g0 + gid
    m = tl.load(mods + g); mi = tl.load(minv + g); c = tl.load(c14s + g)
    o = g.to(tl.int64) * N * N + ri[:, None] * N + rj[None, :]
    msk = (ri[:, None] < N) & (rj[None, :] < N)
    old = tl.load(ACC + o, mask=msk, other=0).to(tl.int32)
    v = _fmod(hh, m, mi) * c + _fmod(hx, m, mi) * 128 + _fmod(ll, m, mi) + old
    tl.store(ACC + o, _fmod(v, m, mi).to(tl.int16), mask=msk)


def bgemm_nt(A, Bt, out_exp="col", kc=4096, bt_rowexp=None):
    """C[b] = A[b] @ Bt[b]^T for all b. A: uniform int exponent. Bt: int exponent or per-row
    exponents bt_rowexp (nb, N). out_exp: 'col' (dynamic per (block, column)), 'mat' (dynamic per
    block), or int (fixed)."""
    nb, M, K = A.shape; nb2, N, K2 = Bt.shape; assert nb == nb2 and K == K2
    L = A.L; R = ctx(L)
    eA = A.exp if isinstance(A.exp, int) else A.exp[:, 0]          # per-block exponent of A
    acc = torch.zeros((R.n, nb * M, N), dtype=torch.int16, device=dev)
    GR = R.n if R.n * nb * (M + N) * min(kc, K) * 2 <= (256 << 20) else max(R.G, R.G * 2)
    same = A.limbs.data_ptr() == Bt.limbs.data_ptr()
    Av = A.limbs.view(L, nb * M, K); Bv = Bt.limbs.view(L, nb * N, K)
    for k0 in range(0, K, kc):
        kk = min(kc, K - k0)
        for g0 in range(0, R.n, GR):
            G = min(GR, R.n - g0)
            Ah, Al = R.residues(Av, k0, kk, g0, G)
            Bh, Bl = (Ah, Al) if same else R.residues(Bv, k0, kk, g0, G)
            kp = Ah.shape[-1]                                        # kk padded to a multiple of 16
            BM, BN, BK, warps = gemm_config(M, N, kp)
            _bgemm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN), G * nb)](
                Ah, Al, Bh, Bl, acc, R.m32, R.mi32, R.c14, g0, nb, M, N, kp,
                BM=BM, BN=BN, BK=BK, num_warps=warps, num_stages=3)
            del Ah, Al, Bh, Bl
    # exponent of the exact product, per (block, column)
    eB = Bt.exp if bt_rowexp is None else bt_rowexp
    if isinstance(eA, int) and isinstance(eB, int):
        e_in = eA + eB
    else:
        eA_t = eA if torch.is_tensor(eA) else torch.full((nb,), eA, dtype=torch.int64, device=dev)
        eB_t = eB if torch.is_tensor(eB) else torch.full((nb, N), eB, dtype=torch.int64, device=dev)
        if eB_t.dim() == 1: eB_t = eB_t[:, None].expand(nb, N)   # per-block uniform
        e_in = eA_t[:, None] + eB_t                                # (nb, N)
    return _bcrt_round(acc, nb, M, N, e_in, out_exp, L)


def _bcrt_round(acc, nb, M, N, e_in, out_exp, L):
    R = ctx(L)
    out = torch.empty((L, nb, M, N), dtype=torch.int32, device=dev)
    fixed = isinstance(e_in, int) and isinstance(out_exp, int)
    if not fixed:
        e_in = e_in if torch.is_tensor(e_in) else torch.full((nb, N), e_in, dtype=torch.int64, device=dev)
        exps = torch.empty((nb, N), dtype=torch.int64, device=dev)
    rows = nb * M
    colblock = max(1, min(N, 65536 // max(rows, 1)))
    for j0 in range(0, N, colblock):
        w = min(colblock, N - j0)
        Wd, bl, Bq = R.crt_block(acc, N, j0, w, rows)            # entries (b, i, jj) -> (b*M + i)*w + jj
        Bp = Wd.shape[1]
        if fixed:
            s = torch.full((Bp,), out_exp - e_in, dtype=torch.int64, device=dev)
        else:
            if isinstance(out_exp, str):
                sj = bl[:Bq].view(nb, M, w).amax(1) - headroom_bits(L)      # (nb, w)
                ej = e_in[:, j0:j0 + w] + sj
            else:
                ej = torch.full((nb, w), int(out_exp), dtype=torch.int64, device=dev)
                sj = ej - e_in[:, j0:j0 + w]
            s = sj[:, None, :].expand(nb, M, w).reshape(-1)
            if Bp != Bq: s = torch.cat([s, s.new_zeros(Bp - Bq)])
            exps[:, j0:j0 + w] = ej
        out[:, :, :, j0:j0 + w] = shift_round(Wd, s, L)[:, :Bq].view(L, nb, M, w)
    if fixed or isinstance(out_exp, int):
        return BMP(out, out_exp)
    C = BMP(out, exps)
    if out_exp == "mat":
        C = brescale(C, exps.amax(1, keepdim=True).expand(nb, N).contiguous())
    return C


# ------------------------------------------------------------------ batched Cholesky / TRSM
import cupy as cp
from .basecase import _module, _soa_to_aos, _aos_to_soa


class BTile:
    __slots__ = ("X", "Lmp")


def bchol_tile(A, LI=24, threads=256):
    """A: BMP (nb, b, b) fixed point -> (L BMP, BTile with inverse X (per-block exponents))"""
    nb, b, _ = A.shape; L = A.L; e = A.exp
    LF = L; WL = L + LI; F = 32 * LF
    Ws = brescale(A, -F, WL).limbs                                     # (WL, nb, b, b)
    W = _soa_to_aos(Ws).view(nb * b * b, WL)
    RINV = torch.zeros((nb * b, WL), dtype=torch.int32, device=dev)
    mod = _module(WL, LF, b)
    mod.get_function("chol_tiles")((nb,), (threads,), (cp.from_dlpack(W), cp.from_dlpack(RINV)))
    tri = torch.tril(torch.ones(b, b, dtype=torch.bool, device=dev))
    Lw = W.view(nb, b, b, WL); Lw[:, ~tri] = 0
    X = torch.zeros_like(W); SCR = torch.zeros_like(W)
    mod.get_function("trinv_tiles")((nb,), (threads,), (cp.from_dlpack(Lw), cp.from_dlpack(RINV), cp.from_dlpack(X), cp.from_dlpack(SCR)))
    Lout = brescale(BMP(_aos_to_soa(Lw), -F), e, L)
    Xs = _aos_to_soa(X.view(nb, b, b, WL))                              # (WL, nb, b, b)
    top = bitlen_words(Xs.reshape(WL, -1)).view(nb, b * b).amax(1) - F  # per block
    ex = (top - headroom_bits(L))[:, None].expand(nb, b).contiguous()
    t = BTile(); t.Lmp = Lout
    t.X = brescale(BMP(Xs, torch.full((nb, b), -F, dtype=torch.int64, device=dev)), ex, L)
    return Lout, t


def _split(n, bt): return (n // 2 + bt - 1) // bt * bt


def _ntiles(n, bt):
    if n <= bt: return 1
    n1 = _split(n, bt)
    return _ntiles(n1, bt) + _ntiles(n - n1, bt)


def bchol(A, bt=32, refine=1):
    """batched Cholesky of nb SPD equilibrated fixed-point blocks (nb, n, n) -> (L, tiles)"""
    nb, n, _ = A.shape; L = A.L; e = A.exp
    if n <= bt:
        assert n == bt
        Lt, t = bchol_tile(A)
        return Lt, [t]
    n1 = _split(n, bt)
    L11, t1 = bchol(A[:n1, :n1], bt, refine)
    L21 = btrsm_right(A[n1:, :n1], L11, t1, bt, refine)
    S = badd(A[n1:, n1:], bgemm_nt(L21, L21, out_exp=e), -1, out_exp=e)
    L22, t2 = bchol(S, bt, refine)
    out = torch.zeros((L, nb, n, n), dtype=torch.int32, device=dev)
    out[:, :, :n1, :n1] = L11.limbs; out[:, :, n1:, :n1] = L21.limbs; out[:, :, n1:, n1:] = L22.limbs
    return BMP(out, e), t1 + t2


def btrsm_right(B, Lf, tiles, bt=32, refine=1):
    """X with X L^T = B, batched; fixed point"""
    nb, m, n = B.shape; e = Lf.exp
    if n <= bt:
        t = tiles[0]
        X = bgemm_nt(B, t.X, out_exp=e, bt_rowexp=t.X.exp)
        for _ in range(refine):
            Rr = badd(B, bgemm_nt(X, t.Lmp, out_exp=e), -1, out_exp=e)
            X = badd(X, bgemm_nt(Rr, t.X, out_exp=e, bt_rowexp=t.X.exp), 1, out_exp=e)
        return X
    n1 = _split(n, bt); k1 = _ntiles(n1, bt)
    X1 = btrsm_right(B[:, :n1], Lf[:n1, :n1], tiles[:k1], bt, refine)
    B2 = badd(B[:, n1:], bgemm_nt(X1, Lf[n1:, :n1], out_exp=e), -1, out_exp=e)
    X2 = btrsm_right(B2, Lf[n1:, n1:], tiles[k1:], bt, refine)
    return BMP(torch.cat([X1.limbs, X2.limbs], 3), e)


def _tr(B):
    """limbs of B^T per block, for use as the 'Bt' operand (row exponents passed separately)"""
    return BMP(B.limbs.transpose(2, 3).contiguous(), 0)


def _bleaf(X, Lm, B, refine):
    """P = Lm^{-1} B with the explicit tile inverse X (+ refinement steps)"""
    P = bgemm_nt(X, _tr(B), out_exp="col", bt_rowexp=B.expt())
    for _ in range(refine):
        Rr = badd(B, bgemm_nt(Lm, _tr(P), out_exp="col", bt_rowexp=P.expt()), -1, out_exp="align")
        P = badd(P, bgemm_nt(X, _tr(Rr), out_exp="col", bt_rowexp=Rr.expt()), 1, out_exp="align")
    return P


def btrsm_left(Lf, B, tiles, bt=32, refine=0):
    """P = L^{-1} B batched; B (nb, n, m) with per-(block, column) exponents"""
    nb, n, m = B.shape
    if n <= bt:
        t = tiles[0]
        return _bleaf(t.X, t.Lmp, B, refine)
    n1 = _split(n, bt); k1 = _ntiles(n1, bt)
    P1 = btrsm_left(Lf[:n1, :n1], B[:n1, :], tiles[:k1], bt, refine)
    U = bgemm_nt(Lf[n1:, :n1], _tr(P1), out_exp="col", bt_rowexp=P1.expt())
    B2 = badd(B[n1:, :], U, -1, out_exp="align")
    P2 = btrsm_left(Lf[n1:, n1:], B2, tiles[k1:], bt, refine)
    ee = torch.maximum(P1.expt(), P2.expt())
    P1, P2 = brescale(P1, ee), brescale(P2, ee)
    return BMP(torch.cat([P1.limbs, P2.limbs], 2), ee)


def btrsm_left_T(Lf, B, tiles, bt=32, refine=1):
    """P = L^{-T} B batched (L lower, fixed point from bchol); B per-(block, column) exponents"""
    nb, n, m = B.shape
    if n <= bt:
        t = tiles[0]
        XT = BMP(t.X.limbs.transpose(2, 3).contiguous(), t.X.exp)      # per-block uniform exponent
        return _bleaf(XT, t.Lmp.t(), B, refine)
    n1 = _split(n, bt); k1 = _ntiles(n1, bt)
    P2 = btrsm_left_T(Lf[n1:, n1:], B[n1:, :], tiles[k1:], bt, refine)
    U = bgemm_nt(Lf[n1:, :n1].t(), _tr(P2), out_exp="col", bt_rowexp=P2.expt())
    B1 = badd(B[:n1, :], U, -1, out_exp="align")
    P1 = btrsm_left_T(Lf[:n1, :n1], B1, tiles[:k1], bt, refine)
    ee = torch.maximum(P1.expt(), P2.expt())
    P1, P2 = brescale(P1, ee), brescale(P2, ee)
    return BMP(torch.cat([P1.limbs, P2.limbs], 2), ee)
