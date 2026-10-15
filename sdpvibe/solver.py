"""SDPB's primal-dual interior point algorithm (SDPB 3.x, src/sdp_solve) on GPU multiprecision.

Reads the SDP produced by `pmp2sdp --outputFormat json` and follows SDPB's iteration step by step
(same residues, Schur complement, Mehrotra predictor-corrector, step lengths, termination), so
iterations can be compared one-to-one with SDPB's iterations.json.

  primal: minimise c.x + const  s.t.  X = sum_p A_p x_p >= 0,  B^T x = b
  dual  : maximise b.y + const  s.t.  Tr(A_p Y) + (B y)_p = c_p,  Y >= 0

Blocks are processed in batched groups: all blocks of a group are zero-padded to common sizes
(points K, bilinear basis sizes h).  Padding is inert: padded rows/cols of X, Y are kept at the
identity, padded Schur rows at the identity, and residues / R / dY are masked to zero there.
"""
import json, os, pickle, time
import mpmath as mp
import numpy as np
import torch
from . import mpf as F
from .mpf import MPF, mul, add, gemm, symmetrize, total, dot, max_abs, Chol, EZ
from .rns import dev


# ------------------------------------------------------------------ helpers (batched MPF, (nb, R, C))
def put(dst, r0, c0, src):
    h, w = src.shape[-2:]
    dst.m[..., r0:r0 + h, c0:c0 + w] = src.m
    dst.e[..., r0:r0 + h, c0:c0 + w] = src.e


def sub(A, r0, r1, c0, c1):
    return MPF(A.m[..., r0:r1, c0:c1].contiguous(), A.e[..., r0:r1, c0:c1].contiguous())


def masked(A, M):
    """zero the entries where M (broadcastable bool) is False"""
    M = M.expand(A.shape)
    return MPF(torch.where(M[None], A.m, 0), torch.where(M, A.e, EZ))


def copy_lower_to_upper(A):
    n = A.shape[-1]
    iu = torch.triu_indices(n, n, 1, device=dev)
    m, e = A.m.clone(), A.e.clone()
    m[..., iu[0], iu[1]] = A.m[..., iu[1], iu[0]]
    e[..., iu[0], iu[1]] = A.e[..., iu[1], iu[0]]
    return MPF(m, e)


def diag_of(A):
    n = A.shape[-1]; idx = torch.arange(n, device=dev)
    return MPF(A.m[..., idx, idx].contiguous(), A.e[..., idx, idx].contiguous()).reshape(*A.shape[:-2], n, 1)


def add_diag(A, s, dmask=None):
    """A + s on the diagonal (where dmask (nb, n) is True); s: MPF scalar"""
    n = A.shape[-1]; idx = torch.arange(n, device=dev)
    d = MPF(A.m[..., idx, idx].contiguous(), A.e[..., idx, idx].contiguous())
    sv = MPF(s.m.reshape(-1, *([1] * d.e.dim())).expand(d.m.shape).contiguous(), s.e.reshape([1] * d.e.dim()).expand(d.e.shape).contiguous())
    if dmask is not None: sv = masked(sv, dmask)
    d = add(d, sv)
    m, e = A.m.clone(), A.e.clone()
    m[..., idx, idx] = d.m; e[..., idx, idx] = d.e
    return MPF(m, e)


def set_diag_where(A, val, dmask):
    """set diagonal entries to val (MPF scalar) where dmask (nb, n) is True"""
    n = A.shape[-1]; idx = torch.arange(n, device=dev)
    m, e = A.m.clone(), A.e.clone()
    dm = m[..., idx, idx]; de = e[..., idx, idx]
    dm = torch.where(dmask[None], val.m.reshape(-1, 1, 1), dm); de = torch.where(dmask, val.e.reshape(1, 1), de)
    m[..., idx, idx] = dm; e[..., idx, idx] = de
    return MPF(m, e)


def col_scale(V, v):
    """V (nb, h, K) with column k multiplied by v[:, k] (v: (nb, K, 1))"""
    nb, h, K = V.shape
    vv = MPF(v.m.reshape(v.L, nb, 1, K).expand(-1, -1, h, -1).contiguous(), v.e.reshape(nb, 1, K).expand(-1, h, -1).contiguous())
    return mul(V, vv)


def bexpand(v, nb):
    """(R, C) -> (nb, R, C) copies"""
    return MPF(v.m[:, None].expand(-1, nb, -1, -1).contiguous(), v.e[None].expand(nb, -1, -1).contiguous())


def dup(a):
    """(nb, R, C) -> (2 nb, R, C): same vector for both parity batches"""
    return MPF(torch.cat([a.m, a.m], 1), torch.cat([a.e, a.e], 0))


def fold(A, nb):
    """sum of the two parity halves (2 nb, ...) -> (nb, ...)"""
    return add(A[:nb], A[nb:])


def flat_rows(A):
    """(nb, R, C) -> (nb*R, C)"""
    nb, R, C = A.shape
    return A.reshape(nb * R, C)


class Timers:
    """per-phase GPU timing (enabled with SDPB_GPU_TIMING=1; synchronises the device)"""
    def __init__(self):
        self.on = os.environ.get("SDPB_GPU_TIMING") == "1"
        self.acc = {}; self.peak = {}; self.t = None; self.first = 1

    def restart(self, it):
        """drop the accumulated times (first iteration: kernel compilation)"""
        self.acc = {}; self.first = it

    def __call__(self, name):
        if not self.on: return
        torch.cuda.current_stream().synchronize(); now = time.time()        # not the copy streams
        if self.t is not None:
            self.acc[self.cur] = self.acc.get(self.cur, 0.0) + now - self.t
            self.peak[self.cur] = max(self.peak.get(self.cur, 0), torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        self.t, self.cur = now, name

    def report(self):
        if not self.on: return
        self("end"); tot = sum(self.acc.values())
        print(f"phase timings (iterations {self.first}+): " + ", ".join(f"{k} {v:.2f}s ({100 * v / tot:.0f}%)" for k, v in
                                            sorted(self.acc.items(), key=lambda kv: -kv[1])))
        print("phase peak GPU memory: " + ", ".join(f"{k} {v / 2**30:.2f} GB" for k, v in self.peak.items() if k != "end"))


# ------------------------------------------------------------------ SDP input
class Group:
    pass


_PARSE = None


def _parse_block(j):
    """worker: read block j of the SDP directory and convert all numbers on the CPU"""
    path, L = _PARSE
    rd = lambda f: json.load(open(os.path.join(path, f)))
    info = rd(f"block_info_{j}.json"); data = rd(f"block_data_{j}.json")
    out = {"dim": info["dim"], "K": info["num_points"]}
    for key in ("bilinear_bases_even", "bilinear_bases_odd"):
        rows = data[key]
        out[key] = (len(rows), F.strings_to_host([x for r in rows for x in r], L) if rows else None)
    out["c"] = F.strings_to_host(data["c"], L)
    out["B"] = F.strings_to_host([x for r in data["B"] for x in r], L)
    return out


def _gb(x): return x / 2 ** 30


def load_raw(path, L, workers=None, nblocks=None):
    """parse all blocks of a pmp2sdp JSON directory on the CPU (parallel), cached as
    <path>/sdpvibe_raw_L<L>.pkl (python -m sdpvibe.prepare builds the cache ahead of time)"""
    t0 = time.time()
    if nblocks is None:
        nblocks = json.load(open(os.path.join(path, "control.json")))["num_blocks"]
    cache = os.path.join(path, f"sdpvibe_raw_L{L}.pkl")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(os.path.join(path, "control.json")):
        with open(cache, "rb") as f: raw = pickle.load(f)
        return raw, f"raw cache {time.time() - t0:.1f}s"
    global _PARSE
    _PARSE = (path, L)
    import multiprocessing
    nw = workers or min(nblocks, os.cpu_count() or 1)
    if nw > 1:
        with multiprocessing.get_context("fork").Pool(nw) as pool:
            raw = pool.map(_parse_block, range(nblocks), chunksize=1)
    else:
        raw = [_parse_block(j) for j in range(nblocks)]
    info = f"parse {time.time() - t0:.1f}s ({nw} workers)"
    try:
        with open(cache, "wb") as f: pickle.dump(raw, f, protocol=5)
    except OSError:
        pass
    return raw, info


class Layout:
    """Memory plan: blocks of equal dim are batched in chunks; each chunk's large arrays
    (B, Pm = L_S^{-1} B, Cholesky of S, pairings A_X_inv / A_Y) are GPU-resident or streamed
    from host memory.  Without memory pressure: one chunk per dim, everything resident.
    When chunked, Pm is not stored ("lean"): the search direction uses B and the Cholesky factor of S
    instead (P^T w = B^T L^-T w, P dy = L^-1 B dy), halving memory and host traffic.
    Working-set coefficients (bytes per MPF entry of S, B, pairings) from synthetic SDPs
    (tests/bench_stream.py with SDPB_GPU_TIMING=1)."""
    A_S, A_B, A_A, A_Q, A_X = 9.0, 3.0, 5.0, 7.0, 6.0
    A_QS, A_QC = 2.6, 9.0      # Q-related memory (x N^2 MPF entries): while chunks run (RNS accumulator /
                               # cholQ), and during chol(Q) itself (measured ~8.7x; nothing else live)
    PRIO = {"B": 5, "V": 5, "xs": 5, "cholS": 4, "ch": 4, "res": 3, "dir": 2, "dp": 2, "dc": 2,
            "AXI": 0.5, "AY": 0.5}                                      # ~ exposed transfers / iteration

    def __init__(self, blocks, N, L, budget=None, chunk=None, stream=None):
        E = self.E = 4 * L + 8
        self.N, self.L = N, L
        if budget is None:
            free, total = torch.cuda.mem_get_info()
            budget = free + torch.cuda.memory_reserved()
        cap = float(os.environ.get("SDPB_GPU_MEM_GB", 0)) * 2 ** 30
        if cap:                                  # emulate a smaller GPU: plan for it and enforce it
            budget = cap
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap / torch.cuda.get_device_properties(0).total_memory))
        chunk = float(os.environ.get("SDPB_GPU_CHUNK_GB", 0)) * 2 ** 30 or chunk
        stream = os.environ.get("SDPB_GPU_STREAM", stream or "auto")      # auto | all | none
        self.budget = budget
        bydim = {}
        for j, b in enumerate(blocks): bydim.setdefault(b["dim"], []).append(j)
        for d in bydim: bydim[d].sort(key=lambda j: (blocks[j]["K"], max(blocks[j]["he"], blocks[j]["ho"])))
        fixed = self.A_Q * N * N * E + 1.2 * 2 ** 30                     # Q, cholQ; context, GEMM workspace
        whole = [bydim[d] for d in sorted(bydim)]
        tot = lambda ch: sum(self.persistent(blocks, c) for c in ch) + max(self.ws(blocks, c) for c in ch)
        self.keep_pm = True
        self.panels = int(os.environ.get("SDPB_GPU_PANELS", 4))           # column panels for L^-1 B (chunked mode)
        if stream == "none" or (stream == "auto" and chunk is None and fixed + tot(whole) <= budget):
            self.chunks, self.host_keys = whole, set()
        else:
            self.keep_pm = False
            avail = budget - self.A_QS * N * N * E - 1.2 * 2 ** 30          # chunked: Q in RNS form (GramAcc)
            # streaming costs little (copies overlap compute) but small batches do (Cholesky/trsm tile
            # latency, per-chunk launches): use the largest chunks that fit, then fill what is left
            # with resident arrays
            W = chunk or avail
            while True:
                self.chunks = self._split(blocks, bydim, W)
                slot = max(sum(self.items(blocks, c).values()) for c in self.chunks)     # prefetched next chunk
                need = max(self.ws(blocks, c) for c in self.chunks) + slot
                # single blocks that are too large cannot be split further: only shrink while it helps
                if chunk or W < 2 ** 26 or all(len(c) == 1 or self.ws(blocks, c) + slot <= 0.9 * avail
                                                for c in self.chunks): break
                W *= 0.85
            if self.A_QC * N * N * E + 1.2 * 2 ** 30 > budget:
                print(f"warning: chol(Q) needs ~{_gb(self.A_QC * N * N * E):.1f} GB (N={N}) > GPU budget (estimate)")
            if need > 0.9 * avail:
                print(f"warning: largest chunk needs ~{_gb(need):.1f} GB, available after fixed allocations "
                      f"~{_gb(avail):.1f} GB (estimates; may not fit)")
            items = []
            for gi, c in enumerate(self.chunks):
                for name, nbt in self.items(blocks, c).items():
                    items.append((-self.PRIO[name], gi, name, nbt))
            if stream == "all":
                self.host_keys = {(n, gi) for _, gi, n, _ in items}
            else:
                left = min(0.9 * avail - need, 0.9 * budget - self.A_QC * N * N * E - 1.2 * 2 ** 30)
                self.host_keys = set()
                for _, gi, name, nbt in sorted(items):
                    if nbt <= left: left -= nbt
                    else: self.host_keys.add((name, gi))
        self.blocks = blocks

    def _split(self, blocks, bydim, W):
        chunks = []
        for d in sorted(bydim):
            cur = []
            for j in bydim[d]:
                if cur and self.ws(blocks, cur + [j]) > W:
                    chunks.append(cur); cur = []
                cur.append(j)
            chunks.append(cur)
        return chunks

    @staticmethod
    def shape(blocks, c):
        b0 = blocks[c[0]]; dim = b0["dim"]
        K = max(blocks[j]["K"] for j in c); h = max(max(blocks[j]["he"], blocks[j]["ho"]) for j in c)
        return len(c), dim, K, h, K * dim * (dim + 1) // 2

    def items(self, blocks, c):
        nb, dim, K, h, P = self.shape(blocks, c); E = self.E
        pad = lambda m: max(32, (m + 31) // 32 * 32)
        n, nbp, npn = dim * h, 2 * nb, pad(dim * h)
        it = {"B": nb * P * self.N * E, "cholS": nb * pad(P) ** 2 * 4 * self.L * (1 + 64 / pad(P)),
              "AXI": nbp * (dim * K) ** 2 * E, "AY": nbp * (dim * K) ** 2 * E,
              "V": nbp * (h * K + n * dim * K) * E, "xs": (2 * nbp * n * n + nb * P) * E,
              "ch": 2 * nbp * npn * npn * 4 * self.L * (1 + 64 / npn), "res": (2 * nbp * n * n + nb * P) * E,
              "dir": (nbp * n * n + nb * P) * E, "dp": 2 * nbp * n * n * E, "dc": (2 * nbp * n * n + nb * P) * E}
        if getattr(self, "keep_pm", True): it["Pm"] = it["B"]
        return it

    def persistent(self, blocks, c): return sum(self.items(blocks, c).values())

    def ws(self, blocks, c):
        nb, dim, K, h, P = self.shape(blocks, c); E = self.E
        return E * nb * (self.A_S * P * P + self.A_B * P * self.N) + E * 2 * nb * (self.A_A * (dim * K) ** 2 + self.A_X * (dim * h) ** 2)

    def describe(self):
        n_host = len(self.host_keys)
        nitems = len(self.PRIO) * len(self.chunks) + (len(self.chunks) if self.keep_pm else 0)
        hb = sum(self.items(self.blocks, self.chunks[gi])[n] for n, gi in self.host_keys)
        return (f"{len(self.chunks)} chunks, {nitems - n_host}/{nitems} arrays GPU-resident, "
                f"{_gb(hb):.1f} GB streamed from host{'' if self.keep_pm else ', lean (no stored Pm)'} "
                f"(GPU budget {_gb(self.budget):.1f} GB)")


def _rand_mpf(shape, L, gen, spread=4):
    """random normalised MPF entries of magnitude ~2^[-spread, spread]"""
    n = int(np.prod(shape))
    m = torch.randint(-2 ** 31, 2 ** 31, (L, n), dtype=torch.int32, device=dev, generator=gen)
    e = torch.randint(-spread, spread + 1, (n,), dtype=torch.int64, device=dev, generator=gen) - 32 * L
    return F.normalize(m, e, L).reshape(*shape)


class SDP:
    """SDP in pmp2sdp JSON format (or synthetic(...)), batched by dim in chunks per Layout;
    large arrays go to self.store (GPU or pinned host memory)."""

    def __init__(self, path, L, workers=None, budget=None, chunk=None, stream=None):
        mp.mp.prec = F.NBITS(L) + 64
        t0 = time.time()
        rd = lambda f: json.load(open(os.path.join(path, f)))
        nblocks = rd("control.json")["num_blocks"]
        obj = rd("objectives.json")
        const = mp.mpf(obj.get("constant", "0"))
        b = F.from_strings(obj["b"], L, (len(obj["b"]), 1))
        raw, info = load_raw(path, L, workers, nblocks)
        blocks = [dict(dim=r["dim"], K=r["K"], he=r["bilinear_bases_even"][0], ho=r["bilinear_bases_odd"][0]) for r in raw]

        def data(j):
            r = raw[j]; raw[j] = None                          # free host memory as blocks are assembled
            Kj = r["K"]; npair = r["dim"] * (r["dim"] + 1) // 2
            Vs = []
            for key in ("bilinear_bases_even", "bilinear_bases_odd"):
                hj, arr = r[key]
                Vs.append(F.host_to_mpf(*arr, L, (hj, Kj)) if hj else None)
            return Vs[0], Vs[1], F.host_to_mpf(*r["c"], L, (Kj * npair, 1)), F.host_to_mpf(*r["B"], L, (Kj * npair, len(obj["b"])))

        t1 = time.time()
        self._build(blocks, data, b, const, L, Layout(blocks, len(obj["b"]), L, budget, chunk, stream))
        self.read_info = f"{info}, assemble {time.time() - t1:.1f}s; {self.layout.describe()}"

    @classmethod
    def synthetic(cls, blocks, N, L, seed=0, budget=None, chunk=None, stream=None):
        """random SDP with the given block shapes [dict(dim, K, he, ho)] (benchmarks / memory tests)"""
        self = cls.__new__(cls)
        mp.mp.prec = F.NBITS(L) + 64
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        t1 = time.time()

        def data(j):
            bl = blocks[j]; Kj = bl["K"]; npair = bl["dim"] * (bl["dim"] + 1) // 2
            g = torch.Generator(device=dev); g.manual_seed(seed * 100003 + j)
            Ve = _rand_mpf((bl["he"], Kj), L, g) if bl["he"] else None
            Vo = _rand_mpf((bl["ho"], Kj), L, g) if bl["ho"] else None
            return Ve, Vo, _rand_mpf((Kj * npair, 1), L, g), _rand_mpf((Kj * npair, N), L, g)

        self._build(blocks, data, _rand_mpf((N, 1), L, gen), mp.mpf(0), L, Layout(blocks, N, L, budget, chunk, stream))
        self.read_info = f"synthetic, assemble {time.time() - t1:.1f}s; {self.layout.describe()}"
        return self

    def _build(self, blocks, data, b, const, L, layout):
        from .store import Store
        self.const, self.b, self.N = const, b, b.shape[0]
        N = self.N
        self.layout = layout
        self.store = Store(layout.host_keys)
        self.groups = []
        for gi, c in enumerate(layout.chunks):
            g = Group()
            nb, dim, K, h, P = Layout.shape(blocks, c)
            g.idx = list(c); g.nb = nb; g.dim = dim; g.K = K; g.h = [h]; g.nbp = 2 * nb; g.P = P
            npair = dim * (dim + 1) // 2
            V = F.zeros((2 * nb, h, K), L)
            xm = torch.zeros((2 * nb, dim * h), dtype=torch.bool, device=dev)
            g.c = F.zeros((nb, P, 1), L); B = F.zeros((nb, P, N), L)
            g.xmask = torch.zeros((nb, P), dtype=torch.bool, device=dev)
            for q, j in enumerate(c):
                Kj = blocks[j]["K"]
                Ve, Vo, cj, Bj = data(j)
                for bi, v in enumerate((Ve, Vo)):
                    z = bi * nb + q
                    hj = 0 if v is None else v.shape[0]
                    if hj: V.m[:, z, :hj, :Kj] = v.m; V.e[z, :hj, :Kj] = v.e
                    for cc in range(dim): xm[z, cc * h:cc * h + hj] = True
                for s in range(npair):                       # segments (c, r) of Kj -> K slots
                    g.c.m[:, q, s * K:s * K + Kj] = cj.m[:, s * Kj:(s + 1) * Kj]; g.c.e[q, s * K:s * K + Kj] = cj.e[s * Kj:(s + 1) * Kj]
                    B.m[:, q, s * K:s * K + Kj] = Bj.m[:, s * Kj:(s + 1) * Kj]; B.e[q, s * K:s * K + Kj] = Bj.e[s * Kj:(s + 1) * Kj]
                    g.xmask[q, s * K:s * K + Kj] = True
                del Bj
            Vx = F.zeros((2 * nb, dim * h, dim * K), L)
            for cc in range(dim): put(Vx, cc * h, cc * K, V)
            g.Xmask = [xm]
            self.store.put(("B", gi), B); self.store.put(("V", gi), dict(V=[V], Vx=[Vx])); del B, V, Vx
            g.psd_rows = int(xm.sum())
            self.groups.append(g)
        torch.cuda.synchronize()
        self.total_psd_rows = sum(g.psd_rows for g in self.groups)
        self.nblocks = len(blocks)


# ------------------------------------------------------------------ solver
DEFAULTS = dict(maxIterations=500, findPrimalFeasible=False, findDualFeasible=False,
                detectPrimalFeasibleJump=False, detectDualFeasibleJump=False,
                dualityGapThreshold="1e-30", primalErrorThreshold="1e-30", dualErrorThreshold="1e-30",
                initialMatrixScalePrimal="1e20", initialMatrixScaleDual="1e20",
                feasibleCenteringParameter="0.1", infeasibleCenteringParameter="0.3",
                stepLengthReduction="0.7", maxComplementarity="1e100", minPrimalStep="0", minDualStep="0")


class Solver:
    def __init__(self, sdp, L, **params):
        self.sdp, self.L = sdp, L
        p = dict(DEFAULTS); p.update(params)
        self.p = {k: (mp.mpf(v) if isinstance(v, str) else v) for k, v in p.items()}
        self.S = lambda v: F.scalar(v, L)
        for gi, g in enumerate(sdp.groups):                          # per-chunk state -> store
            Xg, Yg = [], []
            for b in range(len(g.h)):
                n = g.dim * g.h[b]
                I = bexpand(F.eye(n, L, 1), g.nbp)
                Xg.append(set_diag_where(I, self.S(self.p["initialMatrixScalePrimal"]), g.Xmask[b]))
                Yg.append(set_diag_where(I, self.S(self.p["initialMatrixScaleDual"]), g.Xmask[b]))
            sdp.store.put(("xs", gi), dict(x=F.zeros((g.nb, g.P, 1), L), X=Xg, Y=Yg))
            for k in ("dc", "dp", "dir", "cholS", "Pm"): sdp.store.drop((k, gi))
        self.y = F.zeros((sdp.N, 1), L)

    def segments(self, g):
        return [(c, r, (c * (c + 1) // 2 + r) * g.K) for c in range(g.dim) for r in range(c + 1)]

    def cmws(self, g, Vs, a):
        """constraint_matrix_weighted_sum: sum_p a_p A_p, per parity (Vs: the chunk's V item)"""
        out = []
        a2 = dup(a)
        for b in range(len(g.h)):
            h = g.h[b]; R = F.zeros((g.nbp, g.dim * h, g.dim * h), self.L); V = Vs["V"][b]
            for c, r, off in self.segments(g):
                s = gemm(V, col_scale(V, sub(a2, off, off + g.K, 0, 1)), transB=True)
                if r != c: s = s.shift(-1)
                put(R, r * h, c * h, s)
                if r != c: put(R, c * h, r * h, s.T)
            out.append(R)
        return out

    def pairings(self, g, Vs, cholX, Y):
        AXI, AY = [], []
        for b in range(len(g.h)):
            Vx = Vs["Vx"][b]
            T = cholX[b].solve_lower(Vx)
            AXI.append(copy_lower_to_upper(gemm(T, T, transA=True)))
            AY.append(copy_lower_to_upper(gemm(Vx, gemm(Y[b], Vx), transA=True)))
        return AXI, AY

    def schur(self, g, AXI, AY):
        K = g.K
        if g.dim == 1:
            S = fold(mul(AXI[0], AY[0]), g.nb)
        else:
            sb = lambda M, c, r: sub(M, c * K, (c + 1) * K, r * K, (r + 1) * K)       # A_X_inv[c][r]
            sy = lambda M, c, r: sub(M, r * K, (r + 1) * K, c * K, (c + 1) * K)       # A_Y[c][r]
            S = F.zeros((g.nb, g.P, g.P), self.L)
            for c0, r0, ro in self.segments(g):
                for c1, r1, co in self.segments(g):
                    el = None
                    for b in range(len(g.h)):
                        for t in (mul(sb(AXI[b], c0, r1), sy(AY[b], c1, r0)), mul(sb(AXI[b], r0, r1), sy(AY[b], c1, c0)),
                                  mul(sb(AXI[b], c0, c1), sy(AY[b], r1, r0)), mul(sb(AXI[b], r0, c1), sy(AY[b], r1, c0))):
                            el = t if el is None else add(el, t)
                    put(S, ro, co, fold(el, g.nb).shift(-2))
        S = copy_lower_to_upper(S)
        return set_diag_where(S, self.S(1), ~g.xmask)                 # inert padded points

    def seg_diag(self, g, Ms, hs):
        """vector over segments (c, r): sum_b diag of the (r, c) sub-block of Ms[b] (block size hs[b])"""
        v = F.zeros((g.nb, g.P, 1), self.L)
        for c, r, off in self.segments(g):
            d = diag_of(sub(Ms[0], r * hs[0], (r + 1) * hs[0], c * hs[0], (c + 1) * hs[0]))
            put(v, off, 0, fold(d, g.nb))
        return v

    def run(self, max_iter=None, log=None, verbose=True):
        """SDPB's iteration, chunk-major: every pass loops over the chunks and fetches the chunk's
        arrays from the store (GPU-resident or streamed).  Per-chunk store items:
          B (Pm)  constant / L_S^-1 B      V    bilinear bases (V, Vx)      xs   x, X, Y
          ch      Cholesky of X, Y          res  dual residue, P residue, -XY   AXI, AY  pairings
          cholS   Cholesky of S             dir  R, L_S^-1 rhs (one direction)
          dp      predictor dX, dY          dc   corrector dx, dX, dY (applied in the next residue pass)
        Only y, b, Q (cholQ) and scalars are global.  Operation order is that of SDPB / the earlier
        group-major version (bitwise identical results for the same chunks)."""
        p, sdp, L = self.p, self.sdp, self.L
        G = sdp.groups; nG = len(G)
        alphaP = alphaD = mp.mpf(0)
        t0 = time.time()
        it = 0
        if verbose:
            print(f"{'iter':>4} {'time':>7} {'mu':>9} {'P-obj':>11} {'D-obj':>11} {'gap':>9} {'P-err':>9} {'p-err':>9} "
                  f"{'D-err':>9} {'R-err':>9} {'P-step':>7} {'D-step':>7} {'beta':>6}")
        TM = self.timers = Timers()
        ST = sdp.store
        lean = not sdp.layout.keep_pm
        PM = "B" if lean else "Pm"

        def chunks(*names, nxt=()):
            """(gi, g, {name: device object}) over all chunks, prefetching the next chunk's items
            (and, during the last chunk, the items `nxt` that the next pass needs for chunk 0)"""
            items = [tuple((n, gi) for n in names) for gi in range(nG)]
            for gi, objs in enumerate(ST.stream(items, [(n, 0) for n in nxt])):
                yield gi, G[gi], dict(zip(names, objs))

        D1 = ("xs", "ch", "res", "V", "cholS", PM)
        D2 = ("xs", "ch", "res", "V", "dir", "cholS", PM)

        mk2 = lambda g, b: g.Xmask[b][:, :, None] & g.Xmask[b][:, None, :]
        update = None                                                     # (aP, aD) for the dc items
        while True:
            it += 1
            ti = time.time()
            if it == 2: TM("residues"); TM.restart(2)
            TM("residues")
            dobj = sdp.const + dot(sdp.b, self.y)
            psum = musum = mp.mpf(0)
            dual_err = primal_err_P = mp.mpf(0)
            pvec = sdp.b
            for gi, g, o in chunks("xs", "V", "B", *(["dc"] if update else []), nxt=("AXI", "AY", "B")):
                xs = o["xs"]
                if update:
                    aP, aD = update; dc = o["dc"]
                    xs = dict(x=add(xs["x"], mul(dc["dx"], aP)), X=[add(X, mul(d, aP)) for X, d in zip(xs["X"], dc["dX"])],
                              Y=[add(Y, mul(d, aD)) for Y, d in zip(xs["Y"], dc["dY"])])
                    ST.drop(("dc", gi)); del dc
                psum += dot(g.c, xs["x"])
                cX = [Chol(X) for X in xs["X"]]; cY = [Chol(Y) for Y in xs["Y"]]
                Vs = o["V"]
                a, y_ = self.pairings(g, Vs, cX, xs["Y"])
                d = -self.seg_diag(g, y_, [g.K])
                ST.put(("AXI", gi), a); ST.put(("AY", gi), y_); del a, y_
                B = o["B"]; del o
                d = add(d, gemm(B, bexpand(self.y, g.nb)), -1)
                dres = add(d, g.c); del d
                pvec = add(pvec, gemm(flat_rows(B), flat_rows(xs["x"]), transA=True), -1)
                del B
                Ax = self.cmws(g, Vs, xs["x"]); del Vs
                Pres = [masked(add(Ax[b], xs["X"][b], -1), mk2(g, b)) for b in range(len(g.h))]; del Ax
                dual_err = max(dual_err, max_abs(dres))
                primal_err_P = max([primal_err_P] + [max_abs(P) for P in Pres])
                mXY = [masked(-gemm(X, Y), mk2(g, b)) for b, (X, Y) in enumerate(zip(xs["X"], xs["Y"]))]
                for M in mXY: musum += total(diag_of(M))
                ST.put(("xs", gi), xs); ST.put(("ch", gi), dict(X=cX, Y=cY))
                ST.put(("res", gi), dict(dres=dres, Pres=Pres, mXY=mXY))
                del xs, cX, cY, dres, Pres, mXY
            update = None
            pobj = sdp.const + psum
            gap = abs(pobj - dobj) / max(abs(pobj) + abs(dobj), mp.mpf(1))
            primal_err_p = max_abs(pvec)
            primal_err = max(primal_err_P, primal_err_p)
            dual_feas = dual_err < p["dualErrorThreshold"]; primal_feas = primal_err < p["primalErrorThreshold"]
            pd_feas = dual_feas and primal_feas
            reason = None
            if pd_feas and gap < p["dualityGapThreshold"]: reason = "found primal-dual optimal solution"
            elif dual_feas and p["findDualFeasible"]: reason = "found dual feasible solution"
            elif primal_feas and p["findPrimalFeasible"]: reason = "found primal feasible solution"
            elif alphaD == 1 and p["detectDualFeasibleJump"]: reason = "dual feasible jump detected"
            elif alphaP == 1 and p["detectPrimalFeasibleJump"]: reason = "primal feasible jump detected"
            elif it > (max_iter or p["maxIterations"]): reason = "maxIterations exceeded"
            elif it > 1 and alphaP < p["minPrimalStep"]: reason = "primal step too small"
            elif it > 1 and alphaD < p["minDualStep"]: reason = "dual step too small"
            if reason:
                self.result = dict(terminateReason=reason, primalObjective=pobj, dualObjective=dobj, dualityGap=gap,
                                   primalError=primal_err, dualError=dual_err, iterations=it - 1, time=time.time() - t0)
                break
            mu = -musum / sdp.total_psd_rows
            if mu > p["maxComplementarity"]:
                self.result = dict(terminateReason="maxComplementarity exceeded", iterations=it, time=time.time() - t0)
                break
            muS = self.S(mu)
            # ---- Schur complement, its Cholesky factors, Q = sum P^T P ----
            Q = None
            QA = F.GramAcc(sdp.N, L) if lean else None                   # chunked: one CRT for all chunks
            for gi, g, o in chunks("AXI", "AY", "B", nxt=D1):
                ST.drop(("AXI", gi)); ST.drop(("AY", gi))
                TM("schur: assemble S")
                Sg = self.schur(g, o["AXI"], o["AY"])
                TM("schur: chol(S)")
                cs = Chol(Sg); del Sg
                TM("schur: P = L^-1 B")
                if lean: P = solve_panels(cs, o["B"], sdp.layout.panels)    # columns independent: bounded temporaries
                else: P = cs.solve_lower(o["B"])
                del o
                TM("schur: Q += P^T P")
                if lean:
                    QA.add(flat_rows(P))
                else:
                    q = gemm(flat_rows(P), flat_rows(P), transA=True)
                    Q = q if Q is None else add(Q, q)
                    del q
                ST.put(("cholS", gi), cs); del cs
                if lean: del P
                else: ST.put(("Pm", gi), P); del P
            TM("chol(Q)")
            if lean: Q = QA.result(); self.q_flushes = getattr(self, "q_flushes", 0) + QA.flushes; del QA
            cholQ = Chol(copy_lower_to_upper(Q)); del Q

            def direction(beta, corr):
                """predictor (corr=False) or corrector search direction; returns dy and, per chunk, stores
                dp (predictor) or dc (corrector); accumulates the quantities needed afterwards"""
                bm = self.S(beta * mu)
                dy = pvec
                acc = dict(Rerr=mp.mpf(0), fr=mp.mpf(0), lamP=None, lamD=None)
                TM("direction (corrector) 1" if corr else "direction (predictor) 1")
                for gi, g, o in chunks(*D1, *(["dp"] if corr else []), nxt=D2):
                    xs, res, cX = o["xs"], o["res"], o["ch"]["X"]
                    if not corr:
                        acc["Rerr"] = max([acc["Rerr"]] + [max_abs(add_diag(M, muS, g.Xmask[b])) for b, M in enumerate(res["mXY"])])
                    R, Z = [], []
                    for b in range(len(g.h)):
                        Rb = res["mXY"][b] if not corr else add(res["mXY"][b], gemm(o["dp"]["dX"][b], o["dp"]["dY"][b]), -1)
                        Rb = add_diag(Rb, bm, g.Xmask[b])
                        R.append(Rb)
                        Z.append(symmetrize(cX[b].solve(add(gemm(res["Pres"][b], xs["Y"][b]), Rb, -1))))
                    Vx = o["V"]["Vx"]
                    VZV = [gemm(Vx[b], gemm(Z[b], Vx[b]), transA=True) for b in range(len(g.h))]
                    rhs = add(-res["dres"], self.seg_diag(g, VZV, [g.K]), -1)
                    cs, Pm = o["cholS"], o[PM]; del o, xs, res, cX, Z, VZV, Vx
                    w = cs.solve_lower(rhs)
                    ST.put(("dir", gi), dict(R=R, W=w)); del R
                    if lean: w = cs.solve_upper(w)                      # P^T w = B^T L^-T w   (Pm is B)
                    dy = add(dy, gemm(flat_rows(Pm), flat_rows(w), transA=True), -1)
                    del cs, Pm, w, rhs
                dy = cholQ.solve(dy)
                TM("direction (corrector) 2 + step length" if corr else "direction (predictor) 2")
                for gi, g, o in chunks(*D2, nxt=("xs", "V", "B", "dc") if corr else D1 + ("dp",)):
                    xs, res, cX, dr = o["xs"], o["res"], o["ch"]["X"], o["dir"]
                    cs, Pm = o["cholS"], o[PM]
                    Pdy = gemm(Pm, bexpand(dy, g.nb))
                    if lean: Pdy = cs.solve_lower(Pdy)                   # P dy = L^-1 B dy
                    dxg = cs.solve_upper(add(dr["W"], Pdy)); del cs, Pm, Pdy
                    Ad = self.cmws(g, o["V"], dxg)
                    dXg = [add(Ad[b], res["Pres"][b]) for b in range(len(g.h))]; del Ad
                    dYg = [masked(-symmetrize(cX[b].solve(add(gemm(dXg[b], xs["Y"][b]), dr["R"][b], -1))), mk2(g, b))
                           for b in range(len(g.h))]
                    ST.drop(("dir", gi))
                    if not corr:
                        for b in range(len(g.h)):
                            acc["fr"] += dot(masked(add(xs["X"][b], dXg[b]), mk2(g, b)), masked(add(xs["Y"][b], dYg[b]), mk2(g, b)))
                        ST.put(("dp", gi), dict(dX=dXg, dY=dYg))
                    else:
                        acc["lamP"] = min_eig(o["ch"]["X"], dXg, acc["lamP"])
                        acc["lamD"] = min_eig(o["ch"]["Y"], dYg, acc["lamD"])
                        ST.drop(("dp", gi))
                        ST.put(("dc", gi), dict(dx=dxg, dX=dXg, dY=dYg))
                    del o, xs, res, cX, dr, dxg, dXg, dYg
                return dy, acc

            beta_p = mp.mpf(0) if pd_feas else p["infeasibleCenteringParameter"]
            _, acc = direction(beta_p, False)
            R_err = acc["Rerr"]
            r = acc["fr"] / (mu * sdp.total_psd_rows)
            beta = r * r if r < 1 else r
            beta_c = min(max(p["feasibleCenteringParameter"], beta), mp.mpf(1)) if pd_feas else max(p["infeasibleCenteringParameter"], beta)
            dy, acc = direction(beta_c, True)
            for gi in range(nG): ST.drop(("cholS", gi)); ST.drop(("Pm", gi))
            del cholQ
            gm = p["stepLengthReduction"]
            alphaP = step_from_eig(acc["lamP"], gm)
            alphaD = step_from_eig(acc["lamD"], gm)
            if pd_feas:
                alphaP = alphaD = min(alphaP, alphaD)
            TM("update")
            aP, aD = self.S(alphaP), self.S(alphaD)
            update = (aP, aD)                                             # x, X, Y: in the next residue pass
            self.y = add(self.y, mul(dy, aD))
            row = dict(iteration=it, total_time=round(time.time() - t0, 3), iter_time=round(time.time() - ti, 3),
                       mu=mp.nstr(mu, 30), **{"P-obj": mp.nstr(pobj, 30), "D-obj": mp.nstr(dobj, 30), "gap": mp.nstr(gap, 30),
                       "P-err": mp.nstr(primal_err_P, 30), "p-err": mp.nstr(primal_err_p, 30), "D-err": mp.nstr(dual_err, 30),
                       "R-err": mp.nstr(R_err, 30), "P-step": mp.nstr(alphaP, 30), "D-step": mp.nstr(alphaD, 30),
                       "beta": mp.nstr(beta_c, 30)})
            if log is not None: log.append(row)
            if verbose:
                f = lambda v: mp.nstr(v, 2, min_fixed=0, max_fixed=0) if v != 0 else "0"
                print(f"{it:4d} {time.time() - t0:7.1f} {f(mu):>9} {mp.nstr(pobj, 6):>11} {mp.nstr(dobj, 6):>11} {f(gap):>9} "
                      f"{f(primal_err_P):>9} {f(primal_err_p):>9} {f(dual_err):>9} {f(R_err):>9} {mp.nstr(alphaP, 3):>7} "
                      f"{mp.nstr(alphaD, 3):>7} {mp.nstr(beta_c, 3):>6}", flush=True)
        TM.report()
        if getattr(self, "q_flushes", 0):
            print(f"note: Q accumulator flushed {self.q_flushes} times (column scale jumps > 2^{F.GramAcc.HEAD} between chunks)")
        return self.result


def solve_panels(cs, B, panels):
    """L^-1 B in column panels (identical result: alignment and rounding are per column)"""
    N = B.shape[-1]
    if panels <= 1: return cs.solve_lower(B)
    w = -(-N // panels)
    P = F.zeros(B.shape, B.L)
    for j0 in range(0, N, w):
        put(P, 0, j0, cs.solve_lower(sub(B, 0, B.shape[-2], j0, min(N, j0 + w))))
    return P


def min_eig(chols, dMs, lam=None):
    """min over blocks of lambda_min(L^-1 dM L^-T) (eigenvalues from a float64 copy), combined with lam"""
    for ch, D in zip(chols, dMs):
        T = ch.solve_lower(D)
        M = ch.solve_lower(T.T)                                      # L^-1 dM L^-T (dM symmetric)
        tp = F._top(M)
        s = tp.reshape(tp.shape[0], -1).amax(1)                      # per block scale
        ok = s > EZ // 2
        s = torch.where(ok, s, torch.zeros_like(s))
        Mf = M.shift(-s[:, None, None]).to_float()
        Mf = (Mf + np.swapaxes(Mf, 1, 2)) / 2
        ev = np.linalg.eigvalsh(Mf)[:, 0]
        for q in torch.nonzero(ok).reshape(-1).tolist():
            l = mp.mpf(float(ev[q])) * mp.mpf(2) ** int(s[q])
            lam = l if lam is None else min(lam, l)
    return lam


def step_from_eig(lam, gamma):
    """gamma * (-1/lambda_min) capped at 1"""
    if lam is None or lam > -gamma: return mp.mpf(1)
    return -gamma / lam


def step_length(chols, dMs, gamma):
    return step_from_eig(min_eig(chols, dMs), gamma)
