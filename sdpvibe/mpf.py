"""Multiprecision floating point arrays on GPU: one exponent per entry (like SDPB's BigFloat).

MPF: m int32 (L, *shape) limb planes (little endian, two's complement), e int64 (*shape);
value = m * 2^e. Normalised: |m| has exactly NB = 32L-2 bits (zero: m = 0, e = EZ).

Elementwise ops are CUDA-core kernels (one thread per entry, schoolbook limb products);
matrix products go through the exact RNS/INT8 tensor-core GEMM of rns.py with operands aligned
per row (A) / per column (B) -- the usual normwise floating-point error -- and every output entry
rounded once.  Cholesky works on a power-of-two equilibrated, fixed-point copy (linalg.chol).
"""
import os
import cupy as cp
import mpmath as mp
import numpy as np
import torch
from .limbops import shift_round, bitlen_words
from .mpmat import MP, ctx, gemm_nt, add as mp_add
from .linalg import chol as _chol, trsm_left, fx_exp, _split, _ntiles
from .rns import dev

EZ = -(1 << 40)                                      # exponent of zero

_SRC = r"""
typedef unsigned int u32; typedef unsigned long long u64; typedef long long i64;
#define L __L__
#define NB (32 * L - 2)
#define EZ (-(1LL << 40))

__device__ __forceinline__ void negate(u32* a, int n) {
    u64 c = 1; for (int k = 0; k < n; k++) { u64 t = (u64)(~a[k]) + c; a[k] = (u32)t; c = t >> 32; }
}
__device__ __forceinline__ int bitlen(const u32* a, int n) {
    for (int k = n - 1; k >= 0; k--) if (a[k]) return 32 * k + 32 - __clz(a[k]);
    return 0;
}
// dst[0..nd) = round_half_up(src / 2^s) for magnitudes (s > 0), or src * 2^-s (s <= 0)
__device__ void shift_mag(const u32* src, int ns, int s, u32* dst, int nd) {
    if (s > 0) {
        int q = s >> 5, r = s & 31;
        for (int k = 0; k < nd; k++) {
            int i0 = q + k, i1 = q + k + 1;
            u32 lo = i0 < ns ? src[i0] : 0, hi = i1 < ns ? src[i1] : 0;
            dst[k] = r ? ((lo >> r) | (hi << (32 - r))) : lo;
        }
        int sb = s - 1, qb = sb >> 5, rb = sb & 31;
        u64 c = (qb < ns) ? ((src[qb] >> rb) & 1u) : 0;
        for (int k = 0; k < nd && c; k++) { u64 t = (u64)dst[k] + c; dst[k] = (u32)t; c = t >> 32; }
    } else {
        int t = -s, q = t >> 5, r = t & 31;
        for (int k = 0; k < nd; k++) {
            int i0 = k - q, i1 = k - q - 1;
            u32 lo = (i0 >= 0 && i0 < ns) ? src[i0] : 0, lo2 = (i1 >= 0 && i1 < ns) ? src[i1] : 0;
            dst[k] = r ? ((lo << r) | (lo2 >> (32 - r))) : lo;
        }
    }
}

extern "C" __global__ void mpf_mul(const int* A, const i64* eA, const int* B, const i64* eB,
                                   int* O, i64* eO, i64 n, int bscalar) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
    i64 ib = bscalar ? 0 : i, nb = bscalar ? 1 : n;
    u32 a[L], b[L], p[2 * L], r[L + 1];
    for (int k = 0; k < L; k++) { a[k] = (u32)A[k * n + i]; b[k] = (u32)B[k * nb + ib]; }
    bool na = a[L - 1] >> 31, nbg = b[L - 1] >> 31;
    if (na) negate(a, L);
    if (nbg) negate(b, L);
    int la = bitlen(a, L), lb = bitlen(b, L);
    if (la == 0 || lb == 0) { for (int k = 0; k < L; k++) O[k * n + i] = 0; eO[i] = EZ; return; }
    for (int k = 0; k < 2 * L; k++) p[k] = 0;
    for (int x = 0; x < L; x++) {
        u32 ax = a[x]; if (!ax) continue;
        u64 c = 0;
        for (int y = 0; y < L; y++) { u64 t = (u64)ax * b[y] + p[x + y] + c; p[x + y] = (u32)t; c = t >> 32; }
        p[x + L] = (u32)c;
    }
    int s = bitlen(p, 2 * L) - NB;
    shift_mag(p, 2 * L, s, r, L);
    if (na != nbg) negate(r, L);
    for (int k = 0; k < L; k++) O[k * n + i] = (int)r[k];
    eO[i] = eA[i] + eB[ib] + s;
}

// O = A + sgn*B (entrywise, both normalised), result normalised
extern "C" __global__ void mpf_add(const int* A, const i64* eA, const int* B, const i64* eB,
                                   int* O, i64* eO, i64 n, int sgn) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
    u32 a[L], b[L], wa[L + 2], wb[L + 2], r[L + 1];
    for (int k = 0; k < L; k++) { a[k] = (u32)A[k * n + i]; b[k] = (u32)B[k * n + i]; }
    bool na = a[L - 1] >> 31, nbg = (b[L - 1] >> 31) ^ (sgn < 0);
    if (a[L - 1] >> 31) negate(a, L);
    if (b[L - 1] >> 31) negate(b, L);
    i64 ea = eA[i], eb = eB[i];
    int la = bitlen(a, L), lb = bitlen(b, L);
    if (la == 0) ea = EZ;
    if (lb == 0) eb = EZ;
    if (ea == EZ && eb == EZ) { for (int k = 0; k < L; k++) O[k * n + i] = 0; eO[i] = EZ; return; }
    i64 t = (ea > eb ? ea : eb) - 32;
    i64 da = ea - t, db = eb - t;                       // left shifts (<= 32) or right shifts
    int sa = da < -(i64)(32 * (L + 3)) ? 32 * (L + 3) : (int)(-da);
    int sb2 = db < -(i64)(32 * (L + 3)) ? 32 * (L + 3) : (int)(-db);
    if (ea == EZ) { for (int k = 0; k < L + 2; k++) wa[k] = 0; } else shift_mag(a, L, sa, wa, L + 2);
    if (eb == EZ) { for (int k = 0; k < L + 2; k++) wb[k] = 0; } else shift_mag(b, L, sb2, wb, L + 2);
    if (na) negate(wa, L + 2);
    if (nbg) negate(wb, L + 2);
    u64 c = 0;
    for (int k = 0; k < L + 2; k++) { u64 v = (u64)wa[k] + wb[k] + c; wa[k] = (u32)v; c = v >> 32; }
    bool ns = wa[L + 1] >> 31;
    if (ns) negate(wa, L + 2);
    int bl = bitlen(wa, L + 2);
    if (bl == 0) { for (int k = 0; k < L; k++) O[k * n + i] = 0; eO[i] = EZ; return; }
    int s = bl - NB;
    shift_mag(wa, L + 2, s, r, L);
    if (ns) negate(r, L);
    for (int k = 0; k < L; k++) O[k * n + i] = (int)r[k];
    eO[i] = t + s;
}

// normalise LI-limb two's complement integers (limb planes, n entries) with exponents eI
extern "C" __global__ void mpf_norm(const int* X, const i64* eI, int LI, int* O, i64* eO, i64 n) {
    i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (i >= n) return;
    u32 w[__LMAX__], r[L + 1];
    for (int k = 0; k < LI; k++) w[k] = (u32)X[k * n + i];
    bool ng = w[LI - 1] >> 31;
    if (ng) negate(w, LI);
    int bl = bitlen(w, LI);
    if (bl == 0) { for (int k = 0; k < L; k++) O[k * n + i] = 0; eO[i] = EZ; return; }
    int s = bl - NB;
    shift_mag(w, LI, s, r, L);
    if (ng) negate(r, L);
    for (int k = 0; k < L; k++) O[k * n + i] = (int)r[k];
    eO[i] = eI[i] + s;
}
"""
_mods = {}
LMAX_WIDE = 160


def _mod(L):
    if L not in _mods:
        src = _SRC.replace("__L__", str(L)).replace("__LMAX__", str(LMAX_WIDE))
        _mods[L] = cp.RawModule(code=src, options=("-std=c++17",))
    return _mods[L]


def _launch(L, name, n, *args):
    if n == 0: return
    T = 128
    _mod(L).get_function(name)(((n + T - 1) // T,), (T,), args)


def _c(t): return cp.from_dlpack(t.contiguous())


class MPF:
    __slots__ = ("m", "e")

    def __init__(self, m, e):
        self.m, self.e = m, e

    @property
    def L(self): return self.m.shape[0]
    @property
    def shape(self): return tuple(self.e.shape)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple): idx = (idx,)
        return MPF(self.m[(slice(None),) + idx].contiguous(), self.e[idx].contiguous())

    @property
    def T(self):
        """transpose of the last two dimensions"""
        return MPF(self.m.transpose(-1, -2).contiguous(), self.e.transpose(-1, -2).contiguous())

    def reshape(self, *s):
        return MPF(self.m.reshape(self.L, *s), self.e.reshape(*s))

    def clone(self): return MPF(self.m.clone(), self.e.clone())

    def __neg__(self):
        return add(zeros(self.shape, self.L), self, -1)

    def __add__(self, o): return add(self, o, 1)
    def __sub__(self, o): return add(self, o, -1)

    def shift(self, k):
        """multiply by 2^k (k: int or tensor broadcastable to shape) -- exact"""
        e = self.e + k
        return MPF(self.m, torch.where(self.e == EZ, EZ, e))

    def to_mp(self):
        """-> numpy object array of mpmath mpf (host; tests / scalars)"""
        L = self.L
        a = self.m.reshape(L, -1).t().contiguous().cpu().numpy().astype(np.uint32)
        ex = self.e.reshape(-1).cpu().numpy()
        out = np.empty(a.shape[0], dtype=object)
        for i in range(a.shape[0]):
            v = int.from_bytes(a[i].tobytes(), "little", signed=True)
            out[i] = mp.mpf(0) if v == 0 else mp.mpf((v, int(ex[i])))
        return out.reshape(self.shape)

    def to_float(self):
        """approximate float64 values (with exponent clamping); host numpy"""
        top = self.m[-1].double() * 2.0 ** 32 + (self.m[-2].double() % 2.0 ** 32) if self.L > 1 else self.m[-1].double()
        sc = (self.e + 32 * (self.L - 2)).clamp(-1100, 1100).double()
        v = torch.where(self.e == EZ, torch.zeros_like(top), top * torch.pow(2.0, sc))
        return v.cpu().numpy()


def zeros(shape, L):
    return MPF(torch.zeros((L, *shape), dtype=torch.int32, device=dev),
               torch.full(shape, EZ, dtype=torch.int64, device=dev))


def NBITS(L): return 32 * L - 2


def from_ints(ints, exps, L, shape):
    """Python ints (mantissas) and exponents (flat lists) -> normalised MPF"""
    n = len(ints)
    nb = 4 * L
    buf = bytearray(nb * n)
    for i, v in enumerate(ints):
        buf[i * nb:(i + 1) * nb] = v.to_bytes(nb, "little", signed=True)
    m = torch.from_numpy(np.frombuffer(bytes(buf), dtype=np.int32).reshape(n, L).T.copy()).to(dev)
    e = torch.tensor(exps, dtype=torch.int64, device=dev)
    return normalize(m, e, L).reshape(*shape)


def _dec_to_bin(t, NBb):
    """decimal string -> (signed mantissa with <= NBb bits, binary exponent), correctly rounded-ish"""
    t = t.strip()
    neg = t.startswith("-")
    if t[0] in "+-": t = t[1:]
    e10 = 0
    if "e" in t or "E" in t:
        t, ex = t.replace("E", "e").split("e"); e10 = int(ex)
    if "." in t:
        ip, fp = t.split("."); e10 -= len(fp); t = ip + fp
    man = int(t) if t else 0
    if man == 0: return 0, EZ
    if e10 >= 0:
        v = man * 10 ** e10; bl = v.bit_length()
        if bl <= NBb: return (-v if neg else v), 0
        sh = bl - NBb; v = (v + (1 << (sh - 1))) >> sh
        return (-v if neg else v), sh
    den = 10 ** (-e10)
    sh = NBb + den.bit_length() - man.bit_length() + 1     # result has ~NBb+1 bits
    q = ((man << sh) + den // 2) // den
    ex = -sh
    while q.bit_length() > NBb:                            # renormalise
        q = (q + 1) >> 1; ex += 1
    return (-q if neg else q), ex


def from_strings(strs, L, shape=None):
    """decimal strings -> MPF (fast path for SDP input)"""
    NBb = NBITS(L)
    ints, exps = [], []
    for t in strs:
        m, e = _dec_to_bin(t, NBb); ints.append(m); exps.append(e)
    return from_ints(ints, exps, L, shape or (len(ints),))


def strings_to_host(strs, L):
    """decimal strings -> (limbs int32 (n, L) entry-major numpy, exps int64 numpy); CPU only (worker-safe)"""
    NBb = NBITS(L); nb = 4 * L
    n = len(strs)
    buf = bytearray(nb * n); exps = np.empty(n, dtype=np.int64)
    for i, t in enumerate(strs):
        m, e = _dec_to_bin(t, NBb)
        buf[i * nb:(i + 1) * nb] = m.to_bytes(nb, "little", signed=True); exps[i] = e
    return np.frombuffer(bytes(buf), dtype=np.int32).reshape(n, L), exps


def host_to_mpf(limbs, exps, L, shape):
    """inverse of strings_to_host, on the GPU (normalised)"""
    m = torch.from_numpy(np.ascontiguousarray(limbs.T)).to(dev)
    e = torch.from_numpy(exps).to(dev)
    return normalize(m, e, L).reshape(*shape)


def from_mp(vals, L, shape=None):
    """iterable of mpmath numbers / strings -> MPF (rounded to NB bits)"""
    vals = list(vals)
    NBb = NBITS(L)
    ints, exps = [], []
    for v in vals:
        v = mp.mpf(v)
        if v == 0:
            ints.append(0); exps.append(EZ); continue
        sign, man, ex, bc = v._mpf_
        if bc > NBb:                                  # round to NB bits (half up on magnitude)
            sh = bc - NBb; man = (man + (1 << (sh - 1))) >> sh; ex += sh
        ints.append(-man if sign else man); exps.append(ex)
    return from_ints(ints, exps, L, shape or (len(vals),))


def normalize(Xw, ein, L):
    """Xw: (LI, n) int32 wide two's complement, ein (n,) -> normalised MPF (n,)"""
    LI, n = Xw.shape
    assert LI <= LMAX_WIDE
    O = torch.empty((L, n), dtype=torch.int32, device=dev); eO = torch.empty((n,), dtype=torch.int64, device=dev)
    _launch(L, "mpf_norm", n, _c(Xw), _c(ein), np.int32(LI), _c(O), _c(eO), np.int64(n))
    return MPF(O, eO)


def mul(A, B):
    """entrywise product; B may be a single entry (broadcast)"""
    L = A.L; n = A.e.numel()
    bs = B.e.numel() == 1
    assert bs or B.shape == A.shape
    O = torch.empty_like(A.m); eO = torch.empty_like(A.e)
    _launch(L, "mpf_mul", n, _c(A.m), _c(A.e), _c(B.m), _c(B.e), _c(O), _c(eO), np.int64(n), np.int32(bs))
    return MPF(O, eO)


def add(A, B, sgn=1):
    assert A.shape == B.shape, (A.shape, B.shape)
    L = A.L; n = A.e.numel()
    O = torch.empty_like(A.m); eO = torch.empty_like(A.e)
    _launch(L, "mpf_add", n, _c(A.m), _c(A.e), _c(B.m), _c(B.e), _c(O), _c(eO), np.int64(n), np.int32(sgn))
    return MPF(O, eO)


def scalar(v, L):
    return from_mp([v], L, (1,))


def eye(n, L, val=1):
    A = zeros((n, n), L)
    d = from_mp([val], L, (1,))
    idx = torch.arange(n, device=dev)
    A.m[:, idx, idx] = d.m[:, :1]
    A.e[idx, idx] = d.e[0]
    return A


# ------------------------------------------------------------------ reductions (host results)
def _top(A):
    """per-entry e + bitlen (log2 of magnitude, roughly); -inf for zeros"""
    bl = bitlen_words(A.m.reshape(A.L, -1)).view(A.shape)
    return torch.where(A.e == EZ, torch.full_like(A.e, EZ), A.e + bl)


def total(A):
    """exact-ish sum of all entries -> mpmath mpf (error ~ 2^-NB * n * max)"""
    L = A.L; n = A.e.numel()
    if n == 0: return mp.mpf(0)
    t = int(A.e.max()) - 32
    if t < EZ // 2: return mp.mpf(0)
    s = (t - A.e.reshape(-1))
    W = shift_round(A.m.reshape(L, -1), s, L + 2)                    # (L+2, n), exact for the largest
    S = (W.to(torch.int64) & 0xFFFFFFFF).sum(1)                      # unsigned limb sums, int64
    top = W[-1].to(torch.int64).sum()                                # top limb signed
    S = S.cpu().tolist(); top = int(top)
    v = sum(int(S[k]) << (32 * k) for k in range(L + 1)) + (top << (32 * (L + 1)))
    return mp.mpf((v, t)) if v else mp.mpf(0)


def dot(A, B): return total(mul(A, B))


def max_abs(A):
    if A.e.numel() == 0: return mp.mpf(0)
    tp = _top(A).reshape(-1)
    t = int(tp.max())
    if t <= EZ // 2: return mp.mpf(0)
    cand = torch.nonzero(tp >= t - 1).reshape(-1)                    # same top bit position +-1
    Af = A.reshape(-1)
    vals = MPF(Af.m[:, cand], Af.e[cand]).shift(-t).to_float()
    i = int(cand[int(np.argmax(np.abs(vals)))])
    return abs(Af[i:i + 1].to_mp()[0])


def trace(A):
    n = A.shape[0]; idx = torch.arange(n, device=dev)
    return total(MPF(A.m[:, idx, idx], A.e[idx, idx]))


# ------------------------------------------------------------------ GEMM
def _align_rows(A):
    """A (..., R, K) -> (mantissas (L, ..., R, K) with a common exponent per row, row exps (..., R))"""
    L = A.L
    tp = _top(A)
    E = tp.amax(-1) - NBITS(L)
    E = torch.where(E < EZ // 2, torch.zeros_like(E), E)
    s = (E[..., None] - A.e).reshape(-1)
    s = torch.where(A.e.reshape(-1) == EZ, torch.full_like(s, 1 << 30), s)
    return shift_round(A.m.reshape(L, -1), s, L).view(A.m.shape), E


def gemm(A, B, transA=False, transB=False):
    """C = op(A) op(B) (2-d, or batched 3-d (nb, ., .)); exact products, rounded once per entry
    (operands aligned per row of op(A) / column of op(B): normwise floating-point error)"""
    if A.e.dim() == 2:
        A1 = MPF(A.m[:, None], A.e[None]); B1 = MPF(B.m[:, None], B.e[None])
        C = gemm(A1, B1, transA, transB)
        return MPF(C.m[:, 0], C.e[0])
    from .batched import _bgemm_kernel
    import triton
    if transA: A = A.T
    if not transB: B = B.T                                           # B as (nb, N, K)
    nb, M, K = A.shape; nb2, N, K2 = B.shape; assert K == K2 and nb == nb2, (A.shape, B.shape)
    L = A.L
    Am, rA = _align_rows(A)                                          # (L, nb, M, K), (nb, M)
    Bm, cB = _align_rows(B)                                          # (L, nb, N, K), (nb, N)
    R = ctx(L)
    out_m = torch.empty((L, nb, M, N), dtype=torch.int32, device=dev)
    out_e = torch.empty((nb, M, N), dtype=torch.int64, device=dev)
    ACC = 400 << 20
    w = N if R.n * nb * M * N * 2 <= ACC else max(32, ACC // (R.n * nb * M * 2) // 32 * 32)
    kc = 4096
    BMk = 128 if min(M, N) >= 128 else (64 if min(M, N) >= 64 else 32)
    for j0 in range(0, N, w):
        jw = min(w, N - j0)
        Bj = Bm[:, :, j0:j0 + jw].contiguous().view(L, nb * jw, K)
        Av = Am.view(L, nb * M, K)
        acc = torch.zeros((R.n, nb * M, jw), dtype=torch.int16, device=dev)
        GR = R.n if R.n * nb * (M + jw) * min(kc, K) * 2 <= (256 << 20) else R.G * 2
        for k0 in range(0, K, kc):
            kk = min(kc, K - k0)
            for g0 in range(0, R.n, GR):
                G = min(GR, R.n - g0)
                Ah, Al = R.residues(Av, k0, kk, g0, G)
                Bh, Bl = R.residues(Bj, k0, kk, g0, G)
                _bgemm_kernel[(triton.cdiv(M, BMk), triton.cdiv(jw, BMk), G * nb)](
                    Ah, Al, Bh, Bl, acc, R.m32, R.mi32, R.c14, g0, nb, M, jw, kk,
                    BM=BMk, BN=BMk, BK=32 if kk < 64 else 64, num_warps=8 if BMk == 128 else 4, num_stages=3)
                del Ah, Al, Bh, Bl
        rows = nb * M
        cb = max(1, min(jw, 65536 // max(rows, 1)))
        for c0 in range(0, jw, cb):
            cw = min(cb, jw - c0)
            Wd, bl, Bq = R.crt_block(acc, jw, c0, cw, rows)              # entries (b*M + i)*cw + jj
            ein = (rA[:, :, None] + cB[:, None, j0 + c0:j0 + c0 + cw]).reshape(-1)
            if Wd.shape[1] != Bq:
                ein = torch.cat([ein, ein.new_zeros(Wd.shape[1] - Bq)])
            r = normalize(Wd, ein, L)
            out_m[:, :, :, j0 + c0:j0 + c0 + cw] = r.m[:, :Bq].view(L, nb, M, cw)
            out_e[:, :, j0 + c0:j0 + c0 + cw] = r.e[:Bq].view(nb, M, cw)
        del acc
    return MPF(out_m, out_e)


def symmetrize(A):
    """(A + A^T)/2"""
    return add(A, A.T).shift(-1)


# ------------------------------------------------------------------ Cholesky (equilibrated, batched)
class Chol:
    """A = L L^T per block, with L = D^-1 Lt, D = diag(2^-k): Lt fixed point, k integer.
    A: MPF (n, n) or batched (nb, n, n); padded internally to a multiple of bt with identity."""

    SOLVE_REFINE = int(os.environ.get("SDPB_GPU_SOLVE_REFINE", "0"))

    def __init__(self, A, bt=32, refine=1):
        from .batched import BMP, bchol
        self.two_d = A.e.dim() == 2
        if self.two_d: A = MPF(A.m[:, None], A.e[None])
        nb, n, _ = A.shape; L = A.L
        self.nb, self.n, self.L, self.bt, self.refine = nb, n, L, bt, refine
        idx = torch.arange(n, device=dev)
        tpd = _top(A)[:, idx, idx]                                   # (nb, n)
        if bool((A.m[-1][:, idx, idx] < 0).any()) or bool((tpd <= EZ // 2).any()):
            raise ValueError("matrix is not positive definite (diagonal)")
        k = torch.div(tpd + 1, 2, rounding_mode="floor")              # A_ii < 2^(2k)
        self.k = k
        npad = max(bt, (n + bt - 1) // bt * bt)
        self.npad = npad
        efx = fx_exp(L)
        s = (efx - (A.e - k[:, :, None] - k[:, None, :])).reshape(-1)
        s = torch.where(A.e.reshape(-1) == EZ, torch.full_like(s, 1 << 30), s)
        Af = shift_round(A.m.reshape(L, -1), s, L).view(L, nb, n, n)
        P = torch.zeros((L, nb, npad, npad), dtype=torch.int32, device=dev)
        P[:, :, :n, :n] = Af
        if npad > n:
            one = from_mp([1], L, (1,))
            onefx = shift_round(one.m, torch.tensor([efx - int(one.e[0])], device=dev), L)[:, 0]
            ii = torch.arange(n, npad, device=dev)
            P[:, :, ii, ii] = onefx[:, None, None]
        self.Lt, self.tiles = bchol(BMP(P, efx), bt, refine)

    def _to_bmp(self, B):
        from .batched import BMP
        L = self.L
        Bm, cE = _align_rows(B.T)                                     # per column of B: (L, nb, m, n)
        P = torch.zeros((L, self.nb, self.npad, B.shape[-1]), dtype=torch.int32, device=dev)
        P[:, :, :self.n] = Bm.transpose(2, 3)
        return BMP(P, cE)

    def _from_bmp(self, P):
        L = self.L; m = P.shape[2]
        X = P.limbs[:, :, :self.n].reshape(L, -1)
        ein = P.expt()[:, None, :].expand(self.nb, self.n, m).reshape(-1)
        return normalize(X.contiguous(), ein.contiguous(), L).reshape(self.nb, self.n, m)

    def _io(self, B, f):
        if self.two_d:
            r = f(MPF(B.m[:, None], B.e[None]))
            return MPF(r.m[:, 0], r.e[0])
        return f(B)

    def solve_lower(self, B):
        """L^{-1} B = Lt^{-1} (D B)"""
        from .batched import btrsm_left
        def f(B):
            P = btrsm_left(self.Lt, self._to_bmp(B.shift(-self.k[:, :, None])), self.tiles, self.bt, self.SOLVE_REFINE)
            return self._from_bmp(P)
        return self._io(B, f)

    def solve_upper(self, B):
        """L^{-T} B = D (Lt^{-T} B)"""
        from .batched import btrsm_left_T
        def f(B):
            P = btrsm_left_T(self.Lt, self._to_bmp(B), self.tiles, self.bt, self.SOLVE_REFINE)
            return self._from_bmp(P).shift(-self.k[:, :, None])
        return self._io(B, f)

    def solve(self, B):
        """A^{-1} B"""
        return self.solve_upper(self.solve_lower(B))


# ------------------------------------------------------------------ Gram accumulation (Q = sum_c P_c^T P_c)
class GramAcc:
    """Q = sum over chunks of P_c^T P_c, accumulated exactly in RNS residues and reconstructed (CRT)
    once: a GEMM's CRT costs ~N^2 regardless of the inner size, so a CRT per chunk dominates when
    there are many chunks.  Columns are aligned to running reference exponents E_j (max column top
    seen so far); entries up to 2^HEAD above E_j fit in one extra limb.  A column that outgrows its
    headroom triggers a flush (CRT of the partial sum, added in MPF).  Precision: never worse than
    one GEMM over all rows (normwise per column)."""
    HEAD = 28

    def __init__(self, N, L):
        from .mpmat import ctx
        self.N, self.L = N, L
        self.R = ctx(L + 1)
        self.E = torch.full((N,), EZ, dtype=torch.int64, device=dev)
        self.acc = None
        self.part = None                                             # flushed partial sums (MPF)
        self.flushes = 0

    def add(self, P):
        """P: MPF (rows, N).  Only the lower triangle of the result is accumulated (Q is symmetric;
        callers mirror it, e.g. copy_lower_to_upper before the Cholesky factorisation)."""
        from .batched import _syrk_lower_kernel
        import triton
        L, N, R = self.L, self.N, self.R
        tp = _top(P).amax(0)                                         # (N,) column tops
        live = tp > EZ // 2
        known = self.E > EZ // 2
        if bool((live & known & (tp > self.E + self.HEAD)).any()):
            self._flush()                                            # new references: max of old and new tops
            self.E = torch.where(live & (tp > self.E), tp, self.E)
            known = self.E > EZ // 2
        self.E = torch.where(live & ~known, tp, self.E)                # reference fixed once data accumulated
        PT = P.T                                                     # (N, rows) contiguous
        rows = PT.shape[1]
        s = ((self.E - NBITS(L))[:, None] - PT.e).reshape(-1)
        s = torch.where(PT.e.reshape(-1) == EZ, torch.full_like(s, 1 << 30), s)
        X = shift_round(PT.m.reshape(L, -1), s, L + 1).view(L + 1, 1, N, rows)
        del PT, s
        if self.acc is None:
            self.acc = torch.zeros((R.n, N, N), dtype=torch.int16, device=dev)
        kc = 4096
        BM = 128 if N >= 128 else (64 if N >= 64 else 32)
        for k0 in range(0, rows, kc):
            kk = min(kc, rows - k0)
            GR = R.n if R.n * 2 * N * kk * 2 <= (256 << 20) else R.G * 2
            Xv = X.view(L + 1, N, rows)
            for g0 in range(0, R.n, GR):
                G = min(GR, R.n - g0)
                Ah, Al = R.residues(Xv, k0, kk, g0, G)
                _syrk_lower_kernel[(triton.cdiv(N, BM), triton.cdiv(N, BM), G)](
                    Ah, Al, self.acc, R.m32, R.mi32, R.c14, g0, N, kk,
                    BM=BM, BK=32 if kk < 64 else 64, num_warps=8 if BM == 128 else 4, num_stages=3)
                del Ah, Al

    def _crt(self):
        L, N, R = self.L, self.N, self.R
        out_m = torch.empty((L, N, N), dtype=torch.int32, device=dev)
        out_e = torch.empty((N, N), dtype=torch.int64, device=dev)
        rE = torch.where(self.E > EZ // 2, self.E - NBITS(L), torch.zeros_like(self.E))
        cb = max(1, min(N, 65536 // N))
        for c0 in range(0, N, cb):
            cw = min(cb, N - c0)
            Wd, bl, Bq = R.crt_block(self.acc, N, c0, cw, N)
            ein = (rE[:, None] + rE[None, c0:c0 + cw]).reshape(-1)
            if Wd.shape[1] != Bq:
                ein = torch.cat([ein, ein.new_zeros(Wd.shape[1] - Bq)])
            r = normalize(Wd, ein, L)
            out_m[:, :, c0:c0 + cw] = r.m[:, :Bq].view(L, N, cw)
            out_e[:, c0:c0 + cw] = r.e[:Bq].view(N, cw)
        return MPF(out_m, out_e)

    def _flush(self):
        if self.acc is None: return
        q = self._crt()
        self.part = q if self.part is None else add(self.part, q)
        self.acc = None; self.flushes += 1

    def result(self):
        if self.acc is None: return self.part
        q = self._crt(); self.acc = None
        return q if self.part is None else add(self.part, q)
