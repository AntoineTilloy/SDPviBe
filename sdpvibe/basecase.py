"""Base case on CUDA cores: batched multiprecision Cholesky + triangular inverse of small tiles.

Wide fixed-point format inside the kernel: WL 32-bit limbs, two's complement, F = 32*LF
fractional bits (so LI = WL - LF integer limbs; values up to ~2^(32 LI - 1)).
Input tiles are SPD and (globally) equilibrated, so |a_ij| <= 1 and |l_ij| <= 1; pivots may be
tiny, and their reciprocals (and the inverse X = L^-1) may be huge -> integer limbs.
"""
import cupy as cp
import torch
from .mpmat import MP, rescale, bitlen_limbs, headroom_bits
from .rns import dev

_SRC = r"""
#define WL __WL__
#define LF __LF__
#define BT __BT__
typedef unsigned int u32; typedef unsigned long long u64; typedef long long i64;

__device__ __forceinline__ void cpy(u32* r, const u32* a) { for (int i = 0; i < WL; i++) r[i] = a[i]; }
__device__ __forceinline__ void zero(u32* r) { for (int i = 0; i < WL; i++) r[i] = 0; }
__device__ __forceinline__ bool isneg(const u32* a) { return a[WL - 1] >> 31; }
__device__ void negate(u32* a) { u64 c = 1; for (int i = 0; i < WL; i++) { u64 t = (u64)(~a[i]) + c; a[i] = (u32)t; c = t >> 32; } }
__device__ void addto(u32* r, const u32* a) { u64 c = 0; for (int i = 0; i < WL; i++) { u64 t = (u64)r[i] + a[i] + c; r[i] = (u32)t; c = t >> 32; } }
__device__ void subfrom(u32* r, const u32* a) {
    i64 br = 0; for (int i = 0; i < WL; i++) { i64 t = (i64)r[i] - (i64)a[i] + br; r[i] = (u32)t; br = t >> 32; } }
__device__ __forceinline__ int topl(const u32* a) { for (int i = WL - 1; i > 0; i--) if (a[i]) return i; return 0; }
__device__ void shr1(u32* a) {  // arithmetic shift right by 1
    for (int i = 0; i < WL - 1; i++) a[i] = (a[i] >> 1) | (a[i + 1] << 31);
    a[WL - 1] = (u32)(((int)a[WL - 1]) >> 1); }

// r = (a*b) >> F, signed, truncated; t > 0: use only the top t limbs of |a| and |b|
// product scanning: each output column accumulated in registers (no stores in the inner loop)
__device__ void mulfx(u32* r, const u32* a, const u32* b, int t) {
    u32 x[WL], y[WL]; cpy(x, a); cpy(y, b);
    bool sa = isneg(x), sb = isneg(y); if (sa) negate(x); if (sb) negate(y);
    int ta = topl(x), tb = topl(y);
    int la = t ? max(0, ta - t + 1) : 0, lb = t ? max(0, tb - t + 1) : 0;
    int kmin = max(LF - 2, la + lb), kmax = ta + tb;
    for (int j = 0; j < WL; j++) r[j] = 0;
    u64 carry = 0, hprev = 0;
    for (int k = kmin; k <= kmax + 2 && k < LF + WL; k++) {
        u64 lo = 0, hi = 0;
        if (k <= kmax) {
            int i0 = max(la, k - tb), i1 = min(ta, k - lb);
            #pragma unroll 4
            for (int i = i0; i <= i1; i++) { u64 p = (u64)x[i] * y[k - i]; lo += (u32)p; hi += p >> 32; }
        }
        u64 s = lo + hprev + carry;
        if (k >= LF) r[k - LF] = (u32)s;
        carry = s >> 32; hprev = hi;
    }
    if (sa ^ sb) negate(r);
}

// small x small and small x general products: product-scanning mulfx (bounds from actual sizes)
__device__ __forceinline__ void mul_ss(u32* r, const u32* a, const u32* b) { mulfx(r, a, b, 0); }
__device__ __forceinline__ void mul_sg(u32* r, const u32* a, const u32* b) { mulfx(r, a, b, 0); }

__device__ double to_double(const u32* a) {
    u32 x[WL]; cpy(x, a); bool s = isneg(x); if (s) negate(x);
    int t = topl(x);
    double v = (double)x[t];
    if (t >= 1) v = v * 4294967296.0 + (double)x[t - 1];
    if (t >= 2) v = v * 4294967296.0 + (double)x[t - 2];
    int e = 32 * (max(t - 2, 0)) - 32 * LF;
    v = ldexp(v, e);
    return s ? -v : v;
}
__device__ void from_double(u32* r, double v) {   // v > 0
    zero(r);
    int e; double m = frexp(v, &e);              // v = m 2^e, m in [0.5,1)
    u64 mi = (u64)ldexp(m, 53);                  // 53-bit integer, v = mi 2^(e-53)
    int pos = e - 53 + 32 * LF;                  // bit position of mi's lsb
    if (pos < 0) { mi >>= (-pos); pos = 0; }
    int li = pos / 32, sh = pos % 32;
    unsigned __int128 w = ((unsigned __int128)mi) << sh;
    for (int k = 0; k < 4 && li + k < WL; k++) { r[li + k] = (u32)(w & 0xFFFFFFFFu); w >>= 32; }
}

// r = d^{-1/2}, precision-doubling Newton: r <- r + r (1 - d r^2) / 2
__device__ void rsqrt_fx(u32* r, const u32* d) {
    double dd = to_double(d);
    from_double(r, 1.0 / sqrt(dd));
    u32 t1[WL], t2[WL], one[WL];
    zero(one); one[LF] = 1;
    for (int bits = 100; ; bits *= 2) {
        int t = bits / 32 + 2; if (t > WL) t = 0;
        mulfx(t1, r, r, t); mulfx(t2, t1, d, t);         // d r^2
        cpy(t1, one); subfrom(t1, t2);                   // e = 1 - d r^2
        mulfx(t2, r, t1, t); shr1(t2); addto(r, t2);
        if (t == 0) break;
    }
}

// ---------------- warp-cooperative arithmetic (all 32 lanes of one warp; operands in shared memory)
__device__ __forceinline__ void wcpy(u32* d, const u32* a) { for (int i = threadIdx.x & 31; i < WL; i += 32) d[i] = a[i]; __syncwarp(); }
__device__ void wneg(u32* a) {
    for (int i = threadIdx.x & 31; i < WL; i += 32) a[i] = ~a[i];
    __syncwarp();
    if ((threadIdx.x & 31) == 0) { for (int i = 0; i < WL; i++) { a[i] += 1u; if (a[i]) break; } }
    __syncwarp();
}
// r = (a*b) >> F, signed, truncated, top-t limbs (t=0: all). sx, sy: WL u32 scratch, col: 2*(2*WL+4) u64 scratch
__device__ void wmul(u32* r, const u32* a, const u32* b, int t, u32* sx, u32* sy, u64* col) {
    int lane = threadIdx.x & 31;
    wcpy(sx, a); wcpy(sy, b);
    bool sa = isneg(sx), sb = isneg(sy);
    __syncwarp();
    if (sa) wneg(sx);
    if (sb) wneg(sy);
    int ta = topl(sx), tb = topl(sy);
    int la = t ? max(0, ta - t + 1) : 0, lb = t ? max(0, tb - t + 1) : 0;
    int kmin = max(LF - 2, la + lb), kmax = ta + tb;
    for (int k = kmin + lane; k <= kmax; k += 32) {
        u64 lo = 0, hi = 0;
        int i0 = max(la, k - tb), i1 = min(ta, k - lb);
        for (int i = i0; i <= i1; i++) { u64 p = (u64)sx[i] * sy[k - i]; lo += (u32)p; hi += p >> 32; }
        col[2 * (k - kmin)] = lo; col[2 * (k - kmin) + 1] = hi;
    }
    for (int j = lane; j < WL; j += 32) r[j] = 0;
    __syncwarp();
    if (lane == 0) {
        u64 carry = 0, hprev = 0;
        for (int k = kmin; k <= kmax + 2 && k < LF + WL; k++) {
            u64 lo = 0, hi = 0;
            if (k <= kmax) { lo = col[2 * (k - kmin)]; hi = col[2 * (k - kmin) + 1]; }
            u64 s = lo + hprev + carry;
            if (k >= LF) r[k - LF] = (u32)s;
            carry = s >> 32; hprev = hi;
        }
    }
    __syncwarp();
    if (sa ^ sb) wneg(r);
}
__device__ void wsub(u32* r, const u32* a) { if ((threadIdx.x & 31) == 0) subfrom(r, a); __syncwarp(); }
__device__ void wadd(u32* r, const u32* a) { if ((threadIdx.x & 31) == 0) addto(r, a); __syncwarp(); }
__device__ void wshr1(u32* a) {
    int lane = threadIdx.x & 31; u32 v[(WL + 31) / 32]; int c = 0;
    for (int i = lane; i < WL; i += 32, c++) v[c] = (i < WL - 1) ? ((a[i] >> 1) | (a[i + 1] << 31)) : (u32)(((int)a[i]) >> 1);
    __syncwarp(); c = 0;
    for (int i = lane; i < WL; i += 32, c++) a[i] = v[c];
    __syncwarp();
}
// warp-cooperative r = d^{-1/2} (r, d in shared memory)
__device__ void wrsqrt(u32* r, const u32* d, u32* t1, u32* t2, u32* one, u32* sx, u32* sy, u64* col) {
    int lane = threadIdx.x & 31;
    if (lane == 0) { from_double(r, 1.0 / sqrt(to_double(d))); zero(one); one[LF] = 1; }
    __syncwarp();
    for (int bits = 100; ; bits *= 2) {
        int t = bits / 32 + 2; if (t >= WL) t = 0;
        wmul(t1, r, r, t, sx, sy, col); wmul(t2, t1, d, t, sx, sy, col);
        wcpy(t1, one); wsub(t1, t2);
        wmul(t2, r, t1, t, sx, sy, col); wshr1(t2); wadd(r, t2);
        if (t == 0) break;
    }
}

// ---- batched Cholesky of BT x BT tiles (in place, lower), plus reciprocals of the pivots
extern "C" __global__ void chol_tiles(u32* A, u32* RINV) {
    int tile = blockIdx.x;
    u32* T = A + (size_t)tile * BT * BT * WL;
    u32* RI = RINV + (size_t)tile * BT * WL;
    __shared__ u32 sd[WL], sr[WL], st1[WL], st2[WL], sone[WL], sx[WL], sy[WL];
    __shared__ u64 scol[2 * (2 * WL + 4)];
    for (int k = 0; k < BT; k++) {
        if (threadIdx.x < 32) {
            u32* akk = T + (size_t)(k * BT + k) * WL;
            wcpy(sd, akk);
            wrsqrt(sr, sd, st1, st2, sone, sx, sy, scol);
            wmul(st1, sd, sr, 0, sx, sy, scol);
            wcpy(akk, st1); wcpy(RI + (size_t)k * WL, sr);
        }
        __syncthreads();
        const u32* rk = RI + (size_t)k * WL;
        for (int i = k + 1 + threadIdx.x; i < BT; i += blockDim.x) {
            u32* aik = T + (size_t)(i * BT + k) * WL; u32 v[WL];
            mul_sg(v, aik, rk); cpy(aik, v);
        }
        __syncthreads();
        int m = BT - k - 1, npairs = m * (m + 1) / 2;
        for (int p = threadIdx.x; p < npairs; p += blockDim.x) {
            int ii = (int)((sqrtf(8.0f * p + 1.0f) - 1.0f) * 0.5f);
            while ((ii + 1) * (ii + 2) / 2 <= p) ii++;
            while (ii * (ii + 1) / 2 > p) ii--;
            int jj = p - ii * (ii + 1) / 2;
            int i = k + 1 + ii, j = k + 1 + jj;          // j <= i
            u32 v[WL];
            mul_ss(v, T + (size_t)(i * BT + k) * WL, T + (size_t)(j * BT + k) * WL);
            subfrom(T + (size_t)(i * BT + j) * WL, v);
        }
        __syncthreads();
    }
}

// ---- batched inverse X = L^{-1} (lower) using the pivot reciprocals
// row by row: x_ii = 1/l_ii ; x_ik = -(1/l_ii) * sum_{j=k}^{i-1} l_ij x_jk
extern "C" __global__ void trinv_tiles(const u32* Lt, const u32* RINV, u32* X, u32* SCR) {
    int tile = blockIdx.x;
    const u32* T = Lt + (size_t)tile * BT * BT * WL;
    const u32* RI = RINV + (size_t)tile * BT * WL;
    u32* XT = X + (size_t)tile * BT * BT * WL;
    u32* S = SCR + (size_t)tile * BT * BT * WL;           // S[k*BT + j] = l_ij x_jk
    __shared__ u32 wsx[8 * WL], wsy[8 * WL];
    __shared__ u64 wcol[8 * 2 * (2 * WL + 4)];
    for (int e = threadIdx.x; e < BT * BT; e += blockDim.x) zero(XT + (size_t)e * WL);
    __syncthreads();
    for (int i = 0; i < BT; i++) {
        int np = i * (i + 1) / 2;                        // pairs k <= j < i
        for (int p = threadIdx.x; p < np; p += blockDim.x) {
            int jj = (int)((sqrtf(8.0f * p + 1.0f) - 1.0f) * 0.5f);
            while ((jj + 1) * (jj + 2) / 2 <= p) jj++;
            while (jj * (jj + 1) / 2 > p) jj--;
            int k = p - jj * (jj + 1) / 2, j = jj;       // k <= j < i
            mul_sg(S + (size_t)(k * BT + j) * WL, T + (size_t)(i * BT + j) * WL, XT + (size_t)(j * BT + k) * WL);
        }
        __syncthreads();
        for (int k = threadIdx.x; k <= i; k += blockDim.x) {       // x_ik = -rinv_i * sum_j S[k][j]
            u32* xik = XT + (size_t)(i * BT + k) * WL;
            if (k == i) { cpy(xik, RI + (size_t)i * WL); continue; }
            u32 acc[WL]; zero(acc);
            for (int j = k; j < i; j++) addto(acc, S + (size_t)(k * BT + j) * WL);
            u32 v[WL]; mulfx(v, acc, RI + (size_t)i * WL, 0); negate(v); cpy(xik, v);
        }
        __syncthreads();
    }
}

// ---- batched triangular solve L y = v (L: BT x BT lower tile, RINV = 1/diag), one warp per vector.
// V: (nvec, BT, WL) wide fixed point, overwritten by y. Lane i owns v_i.
extern "C" __global__ void trsv_warp(const u32* Lt, const u32* RINV, u32* V, int nvec) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= nvec) return;                          // whole warp exits together
    __shared__ u32 sp[8][WL];                       // per-warp broadcast slot (blockDim <= 256)
    int ws = (threadIdx.x >> 5);
    u32* vi = V + ((size_t)w * BT + lane) * WL;
    u32 b[WL]; if (lane < BT) cpy(b, vi);
    for (int j = 0; j < BT; j++) {
        if (lane == j) { u32 p[WL]; mulfx(p, b, RINV + (size_t)j * WL, 0); cpy(b, p); cpy(sp[ws], p); }
        __syncwarp();
        if (lane > j && lane < BT) { u32 t[WL]; mulfx(t, Lt + (size_t)(lane * BT + j) * WL, sp[ws], 0); subfrom(b, t); }
        __syncwarp();
    }
    if (lane < BT) cpy(vi, b);
}
"""

_mods = {}


def _module(WL, LF, BT):
    key = (WL, LF, BT, torch.cuda.current_device())
    if key not in _mods:
        src = _SRC.replace("__WL__", str(WL)).replace("__LF__", str(LF)).replace("__BT__", str(BT))
        _mods[key] = cp.RawModule(code=src, options=("-std=c++17", "--device-int128"))
    return _mods[key]


class Tile:
    """Cholesky factor of a diagonal tile in the kernel's wide, entry-major format (+ pivot reciprocals)."""
    __slots__ = ("Lw", "RINV", "b", "WL", "LF", "X", "Lmp")


def _soa_to_aos(X): return X.permute(*range(1, X.dim()), 0).contiguous()
def _aos_to_soa(X): return X.permute(X.dim() - 1, *range(0, X.dim() - 1)).contiguous()


def chol_tile(A, LI=24, threads=256, want_inverse=False):
    """A MP (b x b) fixed point -> (Lfac MP with A's exponent, Tile, [X = Lfac^-1 MP])"""
    b = A.shape[0]; L = A.L; e = A.uniform_exp()
    LF = L; WL = L + LI; F = 32 * LF
    W = _soa_to_aos(rescale(A, -F, WL).limbs).view(b * b, WL)          # exact left shift into wide format
    RINV = torch.zeros((b, WL), dtype=torch.int32, device=dev)
    mod = _module(WL, LF, b)
    mod.get_function("chol_tiles")((1,), (threads,), (cp.from_dlpack(W), cp.from_dlpack(RINV)))
    tri = torch.tril(torch.ones(b, b, dtype=torch.bool, device=dev))
    Lw = W.view(b, b, WL); Lw[~tri] = 0
    Lout = rescale(MP(_aos_to_soa(Lw), -F), e, L)
    t = Tile(); t.Lw, t.RINV, t.b, t.WL, t.LF = Lw.contiguous(), RINV, b, WL, LF
    t.X = None; t.Lmp = Lout
    if not want_inverse:
        return Lout, t
    X = torch.zeros_like(W); SCR = torch.zeros_like(W)
    mod.get_function("trinv_tiles")((1,), (threads,), (cp.from_dlpack(W), cp.from_dlpack(RINV), cp.from_dlpack(X), cp.from_dlpack(SCR)))
    Xs = _aos_to_soa(X.view(b, b, WL))
    top = int(bitlen_limbs(Xs).max()) - F
    t.X = rescale(MP(Xs, -F), top - headroom_bits(L), L)
    return Lout, t, t.X


def chol_inv_tile(A, LI=24, threads=256):
    Lout, _, X = chol_tile(A, LI, threads, want_inverse=True)
    return Lout, X


def apply_tile_inverse(t, B, refine=0):
    """P = L_tile^{-1} B by tensor-core GEMM with the explicit tile inverse (+ refinement steps).
    B: MP (b x m) with per-column exponents."""
    from .mpmat import gemm_nt, add
    tr = lambda M: MP(M.limbs.transpose(1, 2).contiguous(), 0)
    P = gemm_nt(t.X, tr(B), out_exp="col", bt_rowexp=B.exp)
    for _ in range(refine):
        LP = gemm_nt(t.Lmp, tr(P), out_exp="col", bt_rowexp=P.exp)
        Rr = add(B, LP, -1, out_exp="col")
        D = gemm_nt(t.X, tr(Rr), out_exp="col", bt_rowexp=Rr.exp)
        P = add(P, D, 1, out_exp="align")
    return P


def trsv_tile(t, Vlimbs, vexp, out_exp=None):
    """Solve L y = v for many vectors with a factored tile (CUDA-core substitution).
    Vlimbs: (L, nvec, b) limb planes; vexp: (nvec,) exponents (value = mantissa 2^vexp).
    out_exp: None -> dynamic per vector; int -> fixed (e.g. fixed point for Cholesky panels).
    Returns (limbs (L, nvec, b), exps (nvec,))."""
    L, nvec, b = Vlimbs.shape
    F = 32 * t.LF; FX = 32 * L - 40
    from .limbops import shift_round, bitlen_words
    # treat each vector as fixed point with exponent -FX (its own scale), widen exactly
    Wsoa = rescale(MP(Vlimbs.reshape(L, 1, nvec * b), -FX), -F, t.WL).limbs        # (WL, 1, nvec*b)
    W = _soa_to_aos(Wsoa.reshape(t.WL, nvec, b))                                    # (nvec, b, WL)
    mod = _module(t.WL, t.LF, t.b)
    threads = 256
    mod.get_function("trsv_warp")(((nvec * 32 + threads - 1) // threads,), (threads,),
                                 (cp.from_dlpack(t.Lw), cp.from_dlpack(t.RINV), cp.from_dlpack(W), cp.int32(nvec)))
    Ws = _aos_to_soa(W).view(t.WL, nvec * b)
    base = vexp + FX - F                                              # exponent of the wide mantissas
    if out_exp is None:
        ne = bitlen_words(Ws).view(nvec, b).amax(1) + base - headroom_bits(L)
    else:
        ne = torch.full((nvec,), int(out_exp), dtype=torch.int64, device=dev)
    s = (ne - base)[:, None].expand(nvec, b).reshape(-1)
    return shift_round(Ws, s, L).view(L, nvec, b), ne
