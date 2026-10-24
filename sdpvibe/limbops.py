"""Triton kernels for limb-level bookkeeping, in limb-plane ("structure of arrays") layout.

A batch of B multiword integers is an int32 tensor X of shape (L, B): X[k, e] is the k-th 32-bit
limb (little endian, two's complement) of entry e. Loops run over limbs inside the kernels and are
vectorised over entries, so every load/store is coalesced; no host synchronisation.

  crt_pre / crt_carry : CRT reconstruction (with int8 tensor-core GEMMs in between, see rns.py)
  shift_round(X, s, Lout)  -> round_half_up(X / 2^s) as Lout limbs (s per entry, may be < 0)
  add_same(A, B, sign)     -> A + sign*B (same exponent, same width, wraps)
  bitlen_words(X)          -> bit length of |X| per entry
"""
import torch
import triton
import triton.language as tl

dev = "cuda"
BB = 256


# ------------------------------------------------------------------ rounding shift
@triton.jit
def _shifted_word(X, e, B, q, r, ext, k, L: tl.constexpr, ok):
    """One word of an arithmetic right shift by 32*q+r (q may be negative)."""
    i = q + k
    lo = tl.load(X + i * B + e, mask=ok & (i >= 0) & (i < L), other=0).to(tl.int64) & 0xffffffff
    hi = tl.load(X + (i + 1) * B + e, mask=ok & (i + 1 >= 0) & (i + 1 < L), other=0).to(tl.int64) & 0xffffffff
    lo = tl.where(i >= L, ext, lo); hi = tl.where(i + 1 >= L, ext, hi)
    return ((lo | (hi << 32)) >> r) & 0xffffffff


@triton.jit
def _shift_round_kernel(X, S, OUT, FLAG, B, LIN: tl.constexpr, LOUT: tl.constexpr, BB: tl.constexpr,
                        PERENTRY: tl.constexpr):
    pid = tl.program_id(0)
    B64 = tl.cast(B, tl.int64)                               # offsets k*B can exceed 2^31
    e = pid * BB + tl.arange(0, BB)
    ok = e < B
    s = tl.load(S + e, mask=ok, other=0)
    q = s >> 5
    r = (s - q * 32).to(tl.int64)
    topw = tl.load(X + (LIN - 1) * B64 + e, mask=ok, other=0)
    sext = tl.where(topw < 0, 4294967295, 0).to(tl.int64)
    sb = s - 1
    qb = sb >> 5; rb = (sb - qb * 32).to(tl.int64)
    wb = tl.load(X + qb * B64 + e, mask=ok & (qb >= 0) & (qb < LIN), other=0).to(tl.int64) & 4294967295
    wb = tl.where(qb >= LIN, sext, wb)
    carry = tl.where(sb >= 0, (wb >> rb) & 1, 0)
    prev = tl.zeros((BB,), dtype=tl.int64)
    for k in range(LOUT + 1):
        v = _shifted_word(X, e, B64, q, r, sext, k, LIN, ok) + carry
        carry = v >> 32
        v = v & 4294967295
        if k < LOUT:
            tl.store(OUT + k * B64 + e, tl.where(v >= 2147483648, v - 4294967296, v).to(tl.int32), mask=ok)
            prev = v
        else:
            want = tl.where(prev >= 2147483648, 4294967295, 0)
            bad = ((v != want) & ok).to(tl.int32)
            if PERENTRY:
                tl.store(FLAG + e, bad, mask=ok)
            else:
                tl.atomic_max(FLAG, tl.max(bad, axis=0))          # one device-wide error flag


@triton.jit
def _add_kernel(A, Bm, OUT, sgn, B, L: tl.constexpr, BB: tl.constexpr):
    pid = tl.program_id(0)
    B64 = tl.cast(B, tl.int64)                               # offsets k*B can exceed 2^31
    e = pid * BB + tl.arange(0, BB)
    ok = e < B
    carry = tl.zeros((BB,), dtype=tl.int64)
    for k in range(L):
        a = tl.load(A + k * B64 + e, mask=ok, other=0).to(tl.int64) & 4294967295
        b = tl.load(Bm + k * B64 + e, mask=ok, other=0).to(tl.int64) & 4294967295
        if k == L - 1:
            a = tl.where(a >= 2147483648, a - 4294967296, a)
            b = tl.where(b >= 2147483648, b - 4294967296, b)
        v = a + sgn * b + carry
        carry = v >> 32
        v = v & 4294967295
        tl.store(OUT + k * B64 + e, tl.where(v >= 2147483648, v - 4294967296, v).to(tl.int32), mask=ok)


# ------------------------------------------------------------------ CRT
@triton.jit
def _crt_pre_kernel(ACC, INV, MODS, MINV, YH, YL, KOUT, B, n, ldg, N, j0, w,
                    NP: tl.constexpr, BB: tl.constexpr):
    """ACC: residues int16, element (g, i, j) at g*ldg + i*N + j. Block entries e = i*w + jj
    map to (i, j0 + jj). Writes YH/YL (B, n) int8 and k (B,)."""
    pid = tl.program_id(0)
    B64 = tl.cast(B, tl.int64)                               # offsets k*B can exceed 2^31
    e = pid * BB + tl.arange(0, BB)
    g = tl.arange(0, NP)
    okc = e < B
    i = e // w; jj = e % w
    off = i.to(tl.int64) * N + j0 + jj
    mk = (g[:, None] < n) & okc[None, :]
    a = tl.load(ACC + g[:, None].to(tl.int64) * ldg + off[None, :], mask=mk, other=0).to(tl.int32)
    inv = tl.load(INV + g, mask=g < n, other=0)[:, None]
    m = tl.load(MODS + g, mask=g < n, other=1)[:, None]
    mi = tl.load(MINV + g, mask=g < n, other=0.0)[:, None]
    x = a * inv
    q = tl.floor(x.to(tl.float32) * mi).to(tl.int32)
    y = x - q * m
    y = tl.where(y < 0, y + m, y); y = tl.where(y >= m, y - m, y)
    y = tl.where(y > m // 2, y - m, y)
    y = tl.where(mk, y, 0)
    ks = tl.sum(y.to(tl.float32) * mi, axis=0)                 # |frac| < 1/4 guaranteed
    k = tl.floor(ks + 0.5).to(tl.int32)
    lo = ((y + 64) & 127) - 64
    hi = (y - lo) >> 7
    o = e[None, :].to(tl.int64) * n + g[:, None]
    tl.store(YH + o, hi.to(tl.int8), mask=mk)
    tl.store(YL + o, lo.to(tl.int8), mask=mk)
    tl.store(KOUT + e, k, mask=okc)


@triton.jit
def _crt_carry_kernel(ZHT, ZLT, Kk, Md, OUTW, BITLEN, B, W: tl.constexpr, BB: tl.constexpr,
                      COMBINED: tl.constexpr):
    """ZHT/ZLT: (4W, B) int32 digit-major. OUTW: (W, B) int32 limbs (two's complement)."""
    pid = tl.program_id(0)
    B64 = tl.cast(B, tl.int64)                               # offsets k*B can exceed 2^31
    e = pid * BB + tl.arange(0, BB)
    ok = e < B
    kk = tl.load(Kk + e, mask=ok, other=0).to(tl.int64)
    carry = tl.zeros((BB,), dtype=tl.int64)
    last0 = tl.full((BB,), -1, tl.int32); last1 = tl.full((BB,), -1, tl.int32)
    w0 = tl.zeros((BB,), dtype=tl.int64); w1 = tl.zeros((BB,), dtype=tl.int64)
    lastw = tl.zeros((BB,), dtype=tl.int64)
    for wi in range(W):
        v = carry
        for t in tl.static_range(4):
            j = 4 * wi + t
            z = tl.load(ZHT + j * B64 + e, mask=ok, other=0).to(tl.int64)
            if not COMBINED:
                zl = tl.load(ZLT + j * B64 + e, mask=ok, other=0).to(tl.int64)
                z = 128 * z + zl
            md = tl.load(Md + j).to(tl.int64)
            v += (z - kk * md) << (8 * t)
        word = v & 4294967295
        carry = v >> 32
        tl.store(OUTW + wi * B64 + e, tl.where(word >= 2147483648, word - 4294967296, word).to(tl.int32), mask=ok)
        nz0 = word != 0; nz1 = word != 4294967295
        last0 = tl.where(nz0, wi, last0); w0 = tl.where(nz0, word, w0)
        last1 = tl.where(nz1, wi, last1); w1 = tl.where(nz1, word ^ 4294967295, w1)
        lastw = word
    neg = lastw >= 2147483648
    last = tl.where(neg, last1, last0); top = tl.where(neg, w1, w0)
    tb = 32 - tl.extra.cuda.libdevice.clz(top.to(tl.int32))
    bl = tl.where(last >= 0, 32 * last.to(tl.int64) + tb, 0)
    tl.store(BITLEN + e, bl, mask=ok)


# ------------------------------------------------------------------ error flags
ERR = {}          # device -> overflow flag (int32 (1,), set by kernels; checked by check_errors())


def _errbuf():
    d = torch.cuda.current_device()
    if d not in ERR: ERR[d] = torch.zeros((1,), dtype=torch.int32, device=dev)
    return ERR[d]


def check_errors():
    e = ERR.get(torch.cuda.current_device())
    if e is not None and int(e[0]):
        e.zero_()
        raise OverflowError("a rounding step overflowed its limb budget")


# ------------------------------------------------------------------ wrappers
def shift_round(X, s, Lout, check=False):
    """X: (Lin, B) int32 limbs; s: (B,) int64 -> (Lout, B)"""
    Lin, B = X.shape
    out = torch.empty((Lout, B), dtype=torch.int32, device=dev)
    if check:
        flag = torch.empty((B,), dtype=torch.int32, device=dev)
    else:
        flag = _errbuf()
    _shift_round_kernel[(triton.cdiv(B, BB),)](X.contiguous(), s.contiguous(), out, flag, B,
                                               LIN=Lin, LOUT=Lout, BB=BB, PERENTRY=check, num_warps=4)
    if check and bool(flag.any()):
        raise OverflowError("result does not fit in the requested number of limbs")
    return out


def add_same(A, Bm, sgn=1):
    """A + sgn*B for (L, B) int32 limbs with identical exponents (exact, wraps mod 2^(32L))"""
    L, Bn = A.shape
    out = torch.empty_like(A)
    _add_kernel[(triton.cdiv(Bn, BB),)](A.contiguous(), Bm.contiguous(), out, sgn, Bn, L=L, BB=BB, num_warps=4)
    return out


@triton.jit
def _bitlen_kernel(X, OUT, B, L: tl.constexpr, BB: tl.constexpr):
    pid = tl.program_id(0)
    B64 = tl.cast(B, tl.int64)                               # offsets k*B can exceed 2^31
    e = pid * BB + tl.arange(0, BB)
    ok = e < B
    topw = tl.load(X + (L - 1) * B64 + e, mask=ok, other=0)
    ext = tl.where(topw < 0, 4294967295, 0).to(tl.int64)
    last = tl.full((BB,), -1, tl.int64); wv = tl.zeros((BB,), dtype=tl.int64)
    for k in range(L):
        x = (tl.load(X + k * B64 + e, mask=ok, other=0).to(tl.int64) & 4294967295) ^ ext
        nz = x != 0
        last = tl.where(nz, k, last); wv = tl.where(nz, x, wv)
    tb = 32 - tl.extra.cuda.libdevice.clz(wv.to(tl.int32))
    tl.store(OUT + e, tl.where(last >= 0, 32 * last + tb, 0), mask=ok)


def bitlen_words(X):
    """(L, B) int32 limbs -> (B,) int64 bit length of |v| (two's complement, approx for negatives)"""
    L, B = X.shape
    out = torch.empty((B,), dtype=torch.int64, device=dev)
    _bitlen_kernel[(triton.cdiv(B, BB),)](X.contiguous(), out, B, L=L, BB=BB, num_warps=4)
    return out


def crt_pre(acc, inv, mods, minv, ldg, N, j0, w, rows):
    n = mods.numel(); B = rows * w
    YH = torch.empty((B, n), dtype=torch.int8, device=dev); YL = torch.empty_like(YH)
    K = torch.empty((B,), dtype=torch.int32, device=dev)
    NP = triton.next_power_of_2(n); BBp = 32
    _crt_pre_kernel[(triton.cdiv(B, BBp),)](acc, inv, mods, minv, YH, YL, K, B, n, ldg, N, j0, w,
                                            NP=NP, BB=BBp, num_warps=8)
    return YH, YL, K


def crt_carry(ZhT, ZlT, k, Md):
    """Carry CRT digit sums; ZlT=None means ZhT already contains 128*hi+lo."""
    E, B = ZhT.shape
    W = E // 4
    out = torch.empty((W, B), dtype=torch.int32, device=dev)
    bl = torch.empty((B,), dtype=torch.int64, device=dev)
    _crt_carry_kernel[(triton.cdiv(B, 128),)](ZhT, ZlT if ZlT is not None else ZhT, k, Md, out, bl, B,
                                            W=W, BB=128, COMBINED=ZlT is None, num_warps=4)
    return out, bl
