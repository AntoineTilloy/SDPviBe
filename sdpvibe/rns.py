"""Exact multiprecision GEMM on INT8 tensor cores via residue number system (RNS) + CRT.

Number format (see mpmat.py): two's-complement integers of L 32-bit limbs (little endian),
stored as int32 tensors of shape (..., L). A matrix carries one exponent (or one per column).

gemm_nt(A, Bt) computes the exact integer product C = A @ Bt^T and rounds it (round-half-up)
to L limbs with a chosen exponent:
   1. residues of A, Bt modulo n 15-bit primes: limb bytes (as int8) @ table  [int8 tensor cores]
   2. residues split r = 128*hi + lo; per modulus 4 int8 products, fused mod-m accumulation
   3. CRT: (entries x n) @ (n x bytes of M/m_i) [int8 tensor cores] + carry normalisation
   4. rounding shift to the output exponent, conversion back to limbs
"""
import math
import numpy as np
import torch
import triton
import triton.language as tl

dev = "cuda"


def _primes_desc(hi, lo):
    s = np.ones(hi + 1, bool); s[:2] = False
    for p in range(2, int(hi ** 0.5) + 1):
        if s[p]: s[p * p::p] = False
    return [int(p) for p in range(hi, lo, -1) if s[p]]


def _sdigits(x, base, L):
    """signed digits in [-base/2, base/2) of a Python int"""
    h = base // 2; out = []
    for _ in range(L):
        r = ((x + h) % base) - h; out.append(r); x = (x - r) // base
    assert x == 0
    return out


# ------------------------------------------------------------------------------- kernels
@triton.jit
def _fmod(x, m, mi):
    q = tl.floor(x.to(tl.float32) * mi).to(tl.int32)
    r = x - q * m
    r = tl.where(r < 0, r + m, r)
    return tl.where(r >= m, r - m, r)


@triton.jit
def _residue_kernel(X, Tlo, Thi, C1, C2, mods, minv, Rhi, Rlo,
                    rows, kc, kp, ld, k0, plane, NB: tl.constexpr, G: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr):
    """X: int8 view of limb planes (L, rows, ld) int32: byte t of limb l of entry (i, k) is at
    4*((l*rows + i)*ld + k0 + k) + t. Output Rhi/Rlo: (G, rows, kp) int8, r = 128 hi + lo; columns
    kc <= k < kp are zero padding (residues of 0)."""
    pid = tl.program_id(0)
    grp = tl.program_id(1)
    Tlo += grp * G * NB; Thi += grp * G * NB; C1 += grp * G; C2 += grp * G; mods += grp * G; minv += grp * G
    Rhi += grp.to(tl.int64) * G * rows * kp; Rlo += grp.to(tl.int64) * G * rows * kp
    e = pid * BR + tl.arange(0, BR)                      # entry index in [0, rows*kp)
    valid = e < rows * kp
    i = e // kp; k = e % kp
    ld_ok = valid & (k < kc)                             # padding: zero bytes, residue 0
    base = (i.to(tl.int64) * ld + k0 + k) * 4
    gs = tl.arange(0, G)
    acc_lo = tl.zeros((BR, G), dtype=tl.int32)
    acc_hi = tl.zeros((BR, G), dtype=tl.int32)
    for b0 in tl.static_range(0, NB, BK):
        bs = b0 + tl.arange(0, BK)
        x = tl.load(X + base[:, None] + (bs // 4)[None, :].to(tl.int64) * plane + (bs % 4)[None, :],
                    mask=ld_ok[:, None] & (bs[None, :] < NB), other=0)
        x = x ^ -128                                       # unsigned byte u -> u - 128 (signed)
        t_l = tl.load(Tlo + gs[None, :] * NB + bs[:, None], mask=bs[:, None] < NB, other=0)
        t_h = tl.load(Thi + gs[None, :] * NB + bs[:, None], mask=bs[:, None] < NB, other=0)
        acc_lo = tl.dot(x, t_l, acc_lo, out_dtype=tl.int32)
        acc_hi = tl.dot(x, t_h, acc_hi, out_dtype=tl.int32)
    top = tl.load(X + base + (NB // 4 - 1) * plane.to(tl.int64) + 3, mask=ld_ok, other=0)
    neg = (top < 0).to(tl.int32)[:, None]
    m = tl.load(mods + gs)[None, :]; mi = tl.load(minv + gs)[None, :]
    c1 = tl.load(C1 + gs)[None, :]; c2 = tl.load(C2 + gs)[None, :]
    r = _fmod(acc_hi * 128 + acc_lo + c1 - neg * c2, m, mi)
    r = tl.where(r > m // 2, r - m, r)
    lo = ((r + 64) & 127) - 64
    hi = (r - lo) >> 7
    off = gs[None, :].to(tl.int64) * (rows * kp) + e[:, None]
    tl.store(Rhi + off, hi.to(tl.int8), mask=valid[:, None])
    tl.store(Rlo + off, lo.to(tl.int8), mask=valid[:, None])


@triton.jit
def _gemm_kernel(Ahi, Alo, Bhi, Blo, ACC, mods, minv, c14s, g0, M, N, KC,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """ACC[g] (M x N, int16) <- (ACC + 2^14 AhBh^T + 2^7 (AhBl^T + AlBh^T) + AlBl^T) mod m_g.
    A*: (G, M, KC) int8, B*: (G, N, KC) int8."""
    pm = tl.program_id(0); pn = tl.program_id(1); gid = tl.program_id(2)
    ri = pm * BM + tl.arange(0, BM)
    rj = pn * BN + tl.arange(0, BN)
    ba = gid.to(tl.int64) * M * KC
    bb = gid.to(tl.int64) * N * KC
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
    o = g.to(tl.int64) * M * N + ri[:, None] * N + rj[None, :]
    msk = (ri[:, None] < M) & (rj[None, :] < N)
    old = tl.load(ACC + o, mask=msk, other=0).to(tl.int32)
    v = _fmod(hh, m, mi) * c + _fmod(hx, m, mi) * 128 + _fmod(ll, m, mi) + old
    tl.store(ACC + o, _fmod(v, m, mi).to(tl.int16), mask=msk)


# ------------------------------------------------------------------------------- context
class RNS:
    """Moduli and tables for products of L-limb integers with inner dimension <= kmax."""

    def __init__(self, L, kmax=1 << 17, G=16):
        self.L, self.G, self.NB = L, G, 4 * L
        bound = kmax * (1 << (32 * L - 1)) ** 2              # |C| < bound
        mods, M = [], 1
        for p in _primes_desc(32639, 2 ** 14):
            if M > 4 * bound and len(mods) % G == 0: break
            mods.append(p); M *= p
        self.mods, self.M, self.n = mods, M, len(mods)
        n, NB = self.n, self.NB
        self.m32 = torch.tensor(mods, dtype=torch.int32, device=dev)
        self.mi32 = (1.0 / self.m32.double()).float()
        self.c14 = torch.tensor([pow(2, 14, m) for m in mods], dtype=torch.int32, device=dev)
        cen = lambda v, m: v - m if v > m // 2 else v
        T = np.array([[cen(pow(256, j, m), m) for j in range(NB)] for m in mods], np.int64)
        T_lo = ((T + 64) % 128) - 64; T_hi = (T - T_lo) // 128
        self.Tlo = torch.tensor(T_lo, dtype=torch.int8, device=dev).contiguous()
        self.Thi = torch.tensor(T_hi, dtype=torch.int8, device=dev).contiguous()
        self.C1 = torch.tensor([(128 * int(T[i].sum())) % m for i, m in enumerate(mods)], dtype=torch.int32, device=dev)
        self.C2 = torch.tensor([pow(2, 32 * L, m) for m in mods], dtype=torch.int32, device=dev)
        # CRT in base 256
        self.E = (math.ceil(M.bit_length() / 8) + 2 + 7) // 8 * 8
        self.inv = torch.tensor([pow(M // m % m, -1, m) for m in mods], dtype=torch.int32, device=dev)[:, None]
        self.Ct = torch.tensor([_sdigits(M // m, 256, self.E) for m in mods], dtype=torch.int8, device=dev).t().contiguous()   # (E, n)
        self.Md = torch.tensor(_sdigits(M, 256, self.E), dtype=torch.int32, device=dev)
        self.minv_d = (1.0 / self.m32.double())[:, None]
        self.inv32 = self.inv[:, 0].contiguous()

    # ---- step 1: residues of rows x kc entries (limbs) -> (n, rows, kp) hi, lo
    KPAD = 16

    def residues(self, X, k0, kc, g0=0, G=None):
        """X: (L, rows, K) int32 limb planes (contiguous). Residues of X[:, :, k0:k0+kc], moduli g0..g0+G,
        as (G, rows, kp) int8 hi / lo with the inner dimension zero-padded to kp = kc rounded up to a
        multiple of 16: GEMM kernels take kp (Rhi.shape[-1]) as their inner size.  Triton specialises
        integer arguments divisible by 16; for any other inner size the kernels' int8 tile loads are
        not vectorised and the residue GEMM runs ~16x slower (zero residues leave the products exact)."""
        L, rows, K = X.shape
        G = G or self.n
        kp = -(-kc // self.KPAD) * self.KPAD
        Xb = X.view(torch.int8)
        Rhi = torch.empty((G, rows, kp), dtype=torch.int8, device=dev); Rlo = torch.empty_like(Rhi)
        BR = 128; GG = self.G
        _residue_kernel[(triton.cdiv(rows * kp, BR), G // GG)](
            Xb, self.Tlo[g0:], self.Thi[g0:], self.C1[g0:], self.C2[g0:],
            self.m32[g0:], self.mi32[g0:], Rhi, Rlo,
            rows, kc, kp, K, k0, rows * K * 4, NB=self.NB, G=GG, BR=BR, BK=32, num_warps=4)
        return Rhi, Rlo

    # ---- step 2: accumulate residues of A @ Bt^T
    def accumulate(self, Ahi, Alo, Bhi, Blo, acc, g0=0):
        """acc[g0:g0+G] += residues of A @ B^T for the G moduli whose residues are given"""
        G, M, KC = Ahi.shape; N = Bhi.shape[1]
        BM = BN = 128 if min(M, N) >= 128 else 64
        warps = 8 if BM == 128 else 4
        _gemm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN), G)](
            Ahi, Alo, Bhi, Blo, acc, self.m32, self.mi32, self.c14, g0, M, N, KC, BM=BM, BN=BN, BK=64,
            num_warps=warps, num_stages=3)

    # ---- step 3: CRT of a block of entries -> words (E/4, B) word-major + bit lengths
    def crt_block(self, acc, N, j0, w, rows):
        """acc: (n, rows_total, N) int16; block = all rows (< rows), columns j0..j0+w"""
        from .limbops import crt_pre, crt_carry
        Yh, Yl, k = crt_pre(acc, self.inv32, self.m32, self.mi32, acc.stride(0), N, j0, w, rows)
        Bq = Yh.shape[0]; Bp = max(32, (Bq + 31) // 32 * 32)          # cuBLASLt int8: multiples of 32
        if Bp != Bq:
            Yh = torch.cat([Yh, Yh.new_zeros(Bp - Bq, Yh.shape[1])]); Yl = torch.cat([Yl, Yl.new_zeros(Bp - Bq, Yl.shape[1])])
            k = torch.cat([k, k.new_zeros(Bp - Bq)])
        ZhT = torch._int_mm(self.Ct, Yh.t()); ZlT = torch._int_mm(self.Ct, Yl.t())      # (E, Bp)
        W, bl = crt_carry(ZhT, ZlT, k, self.Md)
        return W, bl, Bq
