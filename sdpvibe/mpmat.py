"""Multiprecision matrices on GPU, limb-plane layout.

An MP matrix holds limbs: int32 tensor of shape (L, rows, cols) -- limb plane k contains the k-th
32-bit limb (little endian, two's complement) of every entry -- and exponents:
    value[i, j] = int(limbs[:, i, j]) * 2**exp[j]       (exp: (cols,) int64, or a Python int e0)
Most matrices use a single exponent (e0 cached as a Python int to avoid device syncs).
All limb bookkeeping runs in Triton kernels (limbops.py).
"""
import torch
from .rns import RNS, dev
from .limbops import shift_round, add_same, bitlen_words


class MP:
    """e0: Python int when the exponent is uniform (avoids device syncs), else None"""
    __slots__ = ("limbs", "_exp", "e0")

    def __init__(self, limbs, exp):
        self.limbs = limbs
        if isinstance(exp, int):
            self.e0, self._exp = exp, None
        elif torch.is_tensor(exp) and exp.dim() == 0:
            self.e0, self._exp = None, exp.expand(limbs.shape[2]).clone()
        else:
            self.e0, self._exp = None, exp

    @property
    def exp(self):
        if self._exp is None:
            self._exp = torch.full((self.limbs.shape[2],), self.e0, dtype=torch.int64, device=dev)
        return self._exp

    @property
    def shape(self): return tuple(self.limbs.shape[1:])
    @property
    def L(self): return self.limbs.shape[0]

    def t(self):
        """transpose; requires a single exponent"""
        return MP(self.limbs.transpose(1, 2).contiguous(), self.uniform_exp())

    def uniform_exp(self):
        if self.e0 is not None: return self.e0
        e = int(self._exp[0])
        assert bool((self._exp == e).all()), "matrix has per-column exponents"
        self.e0 = e
        return e

    def __getitem__(self, idx):
        r, c = idx
        if self.e0 is not None: return MP(self.limbs[:, r, c].contiguous(), self.e0)
        return MP(self.limbs[:, r, c].contiguous(), self._exp[c].contiguous())

    def flat(self):
        """(L, rows*cols) view"""
        return self.limbs.reshape(self.L, -1)


def transpose_rows(B):
    """B (n x m) with per-column exponents -> limbs of B^T (m x n) and the row exponents (m,)"""
    return MP(B.limbs.transpose(1, 2).contiguous(), 0), B.exp


def headroom_bits(L): return 32 * L - 32          # target mantissa size for dynamic rounding


# ------------------------------------------------------------------ host conversions (tests)
def from_ints(rows, exp, L):
    """rows: list of lists of Python ints (mantissas) -> MP"""
    R, C = len(rows), len(rows[0])
    mask = (1 << 32) - 1
    flat = []
    for row in rows:
        for v in row:
            assert abs(v) < 1 << (32 * L - 1)
            u = v % (1 << (32 * L))
            flat.append([((u >> (32 * k)) & mask) for k in range(L)])
    arr = torch.tensor(flat, dtype=torch.int64).view(R, C, L).permute(2, 0, 1).contiguous()
    arr = torch.where(arr >= 1 << 31, arr - (1 << 32), arr).to(torch.int32)
    return MP(arr.to(dev), exp)


def to_ints(A):
    """MP -> list of lists of Python int mantissas, list of column exponents"""
    L = A.L
    a = (A.limbs.cpu().to(torch.int64) % (1 << 32)).permute(1, 2, 0)
    R, C = A.shape
    out = []
    for i in range(R):
        row = []
        for j in range(C):
            v = 0
            for k in reversed(range(L)):
                v = (v << 32) | int(a[i, j, k])
            if v >= 1 << (32 * L - 1): v -= 1 << (32 * L)
            row.append(v)
        out.append(row)
    return out, A.exp.cpu().tolist()


# ------------------------------------------------------------------ re-rounding
def _exp_vec(e, C):
    if torch.is_tensor(e):
        return e if e.dim() == 1 else e.expand(C)
    return torch.full((C,), int(e), dtype=torch.int64, device=dev)


def rescale(A, new_exp, Lout=None):
    """re-round A to exponent(s) new_exp (int, 0-d or (cols,) tensor), optionally to Lout limbs"""
    R, C = A.shape
    Lo = Lout or A.L
    if isinstance(new_exp, int) and A.e0 is not None:
        s = torch.full((R * C,), new_exp - A.e0, dtype=torch.int64, device=dev)
        return MP(shift_round(A.flat(), s, Lo).view(Lo, R, C), new_exp)
    ne = _exp_vec(new_exp, C)
    s = (ne - A.exp)[None, :].expand(R, C).reshape(-1)
    return MP(shift_round(A.flat(), s, Lo).view(Lo, R, C), ne.clone())


def bitlen_limbs(X): return bitlen_words(X.reshape(X.shape[0], -1))


def normalize(A, mode="mat"):
    R, C = A.shape
    top = bitlen_limbs(A.limbs).view(R, C) + A.exp[None, :]
    t = top.max() if mode == "mat" else top.amax(0)
    return rescale(A, t - headroom_bits(A.L))


def neg(A):
    R, C = A.shape
    return MP(add_same(torch.zeros_like(A.flat()), A.flat(), -1).view(A.L, R, C), A.e0 if A.e0 is not None else A.exp.clone())


def add(A, B, alpha=1, out_exp=None):
    """A + alpha*B (alpha = +-1). Exact when exponents agree and out_exp equals them.
    out_exp='align': floating-point style (round both to the larger exponent per column)."""
    assert A.shape == B.shape and A.L == B.L
    R, C = A.shape; L = A.L
    if isinstance(out_exp, int) and A.e0 == out_exp and B.e0 == out_exp:
        return MP(add_same(A.flat(), B.flat(), alpha).view(L, R, C), out_exp)
    if out_exp == "align":
        e = torch.maximum(A.exp, B.exp) + 1
        a = rescale(A, e); b = rescale(B, e)
        return MP(add_same(a.flat(), b.flat(), alpha).view(L, R, C), e)
    # general: align exactly in a wider format, add, round
    e = torch.minimum(A.exp, B.exp)
    d = int(max((A.exp - e).max(), (B.exp - e).max()))
    Lw = L + (d + 31) // 32 + 2
    a = rescale(A, e, Lw); b = rescale(B, e, Lw)
    W = MP(add_same(a.flat(), b.flat(), alpha).view(Lw, R, C), e.clone())
    if out_exp is None or out_exp in ("mat", "col"):
        top = bitlen_limbs(W.limbs).view(R, C) + W.exp[None, :]
        top = top.amax(0) if out_exp == "col" else top.max()
        out_exp = top - headroom_bits(L)
    return rescale(W, out_exp, L)


# ------------------------------------------------------------------ GEMM
_ctx = {}


def ctx(L):
    key = (L, torch.cuda.current_device())                      # tensors live on the device (multi-GPU)
    if key not in _ctx: _ctx[key] = RNS(L)
    return _ctx[key]


def gemm_nt(A, Bt, out_exp="col", kc=4096, colblock=None, bt_rowexp=None):
    """C = A @ Bt^T, exact, then rounded.  A: (M, K) single exponent; Bt: (N, K) single exponent,
    or per-row exponents bt_rowexp (N,).  out_exp: 'col' (dynamic per column), 'mat' (dynamic
    common), or an int / (N,) tensor of target exponents."""
    M, K = A.shape; N, K2 = Bt.shape; assert K == K2
    L = A.L; R = ctx(L)
    ACC_BUDGET = 400 << 20                                        # residue accumulator bytes
    if R.n * M * N * 2 > ACC_BUDGET and N > 64:                   # split output into column panels
        w = max(64, ACC_BUDGET // (R.n * M * 2) // 64 * 64)
        parts = []
        for j0 in range(0, N, w):
            sl = slice(j0, min(N, j0 + w))
            oe = out_exp[sl] if torch.is_tensor(out_exp) and out_exp.dim() == 1 else out_exp
            parts.append(gemm_nt(A, Bt[sl, :], "col" if out_exp == "mat" else oe, kc, colblock,
                                 None if bt_rowexp is None else bt_rowexp[sl]))
        if all(p.e0 is not None for p in parts) and len({p.e0 for p in parts}) == 1:
            return MP(torch.cat([p.limbs for p in parts], 2), parts[0].e0)
        C = MP(torch.cat([p.limbs for p in parts], 2), torch.cat([p.exp for p in parts]))
        return rescale(C, C.exp.max()) if out_exp == "mat" else C
    eA = A.uniform_exp()
    e_in = (eA + bt_rowexp) if bt_rowexp is not None else eA + Bt.uniform_exp()   # tensor or int
    acc = torch.zeros((R.n, M, N), dtype=torch.int16, device=dev)
    # moduli per residue batch: all at once if the residues fit in ~256 MB
    GR = R.n if R.n * (M + N) * min(kc, K) * 2 <= (256 << 20) else R.G * 2
    same = A.limbs.data_ptr() == Bt.limbs.data_ptr()
    for k0 in range(0, K, kc):
        kk = min(kc, K - k0)
        for g0 in range(0, R.n, GR):
            G = min(GR, R.n - g0)
            Ah, Al = R.residues(A.limbs, k0, kk, g0, G)
            Bh, Bl = (Ah, Al) if same else R.residues(Bt.limbs, k0, kk, g0, G)
            R.accumulate(Ah, Al, Bh, Bl, acc, g0)
            del Ah, Al, Bh, Bl
    return _crt_round(acc, e_in, out_exp, L, colblock)


def _crt_round(acc, e_in, out_exp, L, colblock=None):
    R = ctx(L)
    n, M, N = acc.shape
    if colblock is None: colblock = max(1, min(N, 65536 // max(M, 1)))
    out = torch.empty((L, M, N), dtype=torch.int32, device=dev)
    fixed = isinstance(e_in, int) and isinstance(out_exp, int)
    if not fixed:
        e_in = _exp_vec(e_in, N)
        exps = torch.empty((N,), dtype=torch.int64, device=dev)
    for j0 in range(0, N, colblock):
        w = min(colblock, N - j0)
        Wd, bl, Bq = R.crt_block(acc, N, j0, w, M)                 # limbs (E/4, Bp), entries i*w + jj
        Bp = Wd.shape[1]
        if fixed:
            s = torch.full((Bp,), out_exp - e_in, dtype=torch.int64, device=dev)
        else:
            if isinstance(out_exp, str):
                sj = bl[:Bq].view(M, w).amax(0) - headroom_bits(L)
                ej = e_in[j0:j0 + w] + sj
            else:
                ej = _exp_vec(out_exp, N)[j0:j0 + w]
                sj = ej - e_in[j0:j0 + w]
            s = sj[None, :].expand(M, w).reshape(-1)
            if Bp != Bq: s = torch.cat([s, s.new_zeros(Bp - Bq)])
            exps[j0:j0 + w] = ej
        out[:, :, j0:j0 + w] = shift_round(Wd, s, L)[:, :Bq].view(L, M, w)
    if fixed:
        return MP(out, out_exp)
    C = MP(out, exps)
    if out_exp == "mat":                                           # common exponent (one extra rounding)
        C = rescale(C, exps.max())
    return C
