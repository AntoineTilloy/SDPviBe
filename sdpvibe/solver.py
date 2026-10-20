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
import json, os, pickle, sys, time
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
    """worker: read block j of the SDP (JSON or binary) and convert all numbers on the CPU"""
    from .sdp_io import SdpSource, read_block
    path, L = _PARSE
    return read_block(SdpSource(path), j, L)


def _gb(x): return x / 2 ** 30


def load_raw(path, L, workers=None, nblocks=None):
    """parse all blocks of an SDP (pmp2sdp output: directory or archive, JSON or binary blocks) on the
    CPU in parallel, cached as sdpvibe_raw_L<L>.pkl in the SDP directory (next to an archive)
    (python -m sdpvibe.prepare builds the cache ahead of time)"""
    from .sdp_io import SdpSource
    t0 = time.time()
    src = SdpSource(path)
    if nblocks is None:
        nblocks = src.json("control.json")["num_blocks"]
    cdir = src.cache_dir()
    cache = os.path.join(cdir, f"sdpvibe_raw_L{L}.pkl")
    if os.path.exists(cache) and os.path.getmtime(cache) > src.mtime():
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
    fmt = "binary" if "precision" in raw[0] else "json"
    info = f"parse {fmt} {time.time() - t0:.1f}s ({nw} workers)"
    if os.environ.get("SDPVIBE_NO_CACHE") != "1":
        try:
            os.makedirs(cdir, exist_ok=True)
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

    def __init__(self, blocks, N, L, budget=None, chunk=None, stream=None, devices=(0,)):
        E = self.E = 4 * L + 8
        self.N, self.L = N, L
        self.phys = list(devices)                # slot -> physical CUDA device
        self.devices = list(range(len(self.phys)))   # device slots (chunks are assigned to slots)
        devices = self.phys
        self.deterministic = True
        if budget is None and not float(os.environ.get("SDPB_GPU_MEM_GB", 0)):
            # The layout (chunking) changes rounding, so it must not depend on momentary free memory:
            # plan for the card (total minus a fixed reserve for the CUDA context and fragmentation),
            # unless other processes hold so much memory that this plan would not fit.
            for d in devices:                    # multi-GPU: the same per-device budget (smallest card)
                with torch.cuda.device(d):
                    free, total = torch.cuda.mem_get_info()
                    bd = total - max(1.5 * 2 ** 30, 0.08 * total)
                    avail = free + torch.cuda.memory_reserved()
                if avail < bd:
                    print(f"sdpvibe: warning: only {avail / 2**30:.1f} of {total / 2**30:.1f} GB memory free on GPU {d} (other processes); "
                          f"the memory plan, hence the rounding, depends on it (set SDPB_GPU_MEM_GB for reproducible runs)",
                          file=sys.stderr)
                    bd = avail
                    self.deterministic = False
                budget = bd if budget is None else min(budget, bd)
        cap = float(os.environ.get("SDPB_GPU_MEM_GB", 0)) * 2 ** 30
        if cap:                                  # emulate a smaller GPU: plan for it and enforce it
            budget = cap
            for d in set(devices):
                torch.cuda.set_per_process_memory_fraction(min(1.0, cap / torch.cuda.get_device_properties(d).total_memory), d)
        chunk = float(os.environ.get("SDPB_GPU_CHUNK_GB", 0)) * 2 ** 30 or chunk
        stream = os.environ.get("SDPB_GPU_STREAM", stream or "auto")      # auto | all | none
        self.budget = budget
        bydim = {}
        for j, b in enumerate(blocks): bydim.setdefault(b["dim"], []).append(j)
        for d in bydim: bydim[d].sort(key=lambda j: (blocks[j]["K"], max(blocks[j]["he"], blocks[j]["ho"])))
        fixed = self.A_Q * N * N * E + 1.2 * 2 ** 30                     # Q, cholQ; context, GEMM workspace
        # multi-GPU: chunks of at most 1/(2 ndev) of the total work, assigned to devices largest first
        devices = self.devices
        self.cmax = sum(self.cost(blocks, [j]) for j in range(len(blocks))) / (2 * len(devices)) if len(devices) > 1 else float("inf")
        whole = self._split(blocks, bydim, float("inf")) if len(devices) > 1 else [bydim[d] for d in sorted(bydim)]
        tot = lambda ch: sum(self.persistent(blocks, c) for c in ch) + max(self.ws(blocks, c) for c in ch) if ch else 0
        self.keep_pm = True
        self.panels = int(os.environ.get("SDPB_GPU_PANELS", 4))           # column panels for L^-1 B (chunked mode)
        dev_whole = self._assign(blocks, whole)
        fits = all(fixed + tot([whole[gi] for gi in range(len(whole)) if dev_whole[gi] == d]) <= budget for d in devices)
        if stream == "none" or (stream == "auto" and chunk is None and fits):
            self.chunks, self.host_keys, self.dev_of = whole, set(), dev_whole
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
            self.dev_of = self._assign(blocks, self.chunks)
            qc = self.qc_factor(N, E, budget)
            if qc * N * N * E + 1.2 * 2 ** 30 > budget:
                print(f"warning: chol(Q) needs ~{_gb(qc * N * N * E):.1f} GB (N={N}) > GPU budget (estimate)")
            if need > 0.9 * avail:
                print(f"warning: largest chunk needs ~{_gb(need):.1f} GB, available after fixed allocations "
                      f"~{_gb(avail):.1f} GB (estimates; may not fit)")
            self.host_keys = set()
            for d in devices:                                           # per device: what stays resident
                mine = [gi for gi in range(len(self.chunks)) if self.dev_of[gi] == d]
                if not mine: continue
                items = []
                for gi in mine:
                    for name, nbt in self.items(blocks, self.chunks[gi]).items():
                        items.append((-self.PRIO[name], gi, name, nbt))
                if stream == "all":
                    self.host_keys |= {(n, gi) for _, gi, n, _ in items}
                    continue
                slot_d = max(sum(self.items(blocks, self.chunks[gi]).values()) for gi in mine)
                need_d = max(self.ws(blocks, self.chunks[gi]) for gi in mine) + slot_d
                left = min(0.9 * avail - need_d, 0.9 * budget - qc * N * N * E - 1.2 * 2 ** 30)
                for _, gi, name, nbt in sorted(items):
                    if nbt <= left: left -= nbt
                    else: self.host_keys.add((name, gi))
        self.chunks_of = {d: [gi for gi in range(len(self.chunks)) if self.dev_of[gi] == d] for d in devices}
        self.blocks = blocks

    def cost(self, blocks, c):
        """work of a chunk (flop proxy): Cholesky of S, L^-1 B, P^T P"""
        nb, dim, K, h, P = self.shape(blocks, c); N = self.N
        return nb * (P ** 3 / 3 + P * P * N + P * N * N)

    def _assign(self, blocks, chunks):
        """chunk -> device: longest processing time first (deterministic)"""
        load = {d: 0.0 for d in self.devices}
        dev_of = [self.devices[0]] * len(chunks)
        for gi in sorted(range(len(chunks)), key=lambda gi: (-self.cost(blocks, chunks[gi]), gi)):
            d = min(self.devices, key=lambda d: (load[d], self.devices.index(d)))
            dev_of[gi] = d; load[d] += self.cost(blocks, chunks[gi])
        return dev_of

    @classmethod
    def qc_factor(cls, N, E, budget):
        """peak of the Cholesky of Q in units of N^2 MPF entries: monolithic ~9, tiled (blockchol) ~1.6
        (same rule as Solver._q_block)"""
        forced = int(os.environ.get("SDPB_GPU_QBLOCK", 0))
        if (forced and forced < N) or (not forced and cls.A_QC * N * N * E > 0.35 * budget and N > 2048):
            return 1.6
        return cls.A_QC

    def _split(self, blocks, bydim, W):
        chunks = []
        for d in sorted(bydim):
            cur = []
            for j in bydim[d]:
                if cur and (self.ws(blocks, cur + [j]) > W or self.cost(blocks, cur + [j]) > self.cmax):
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
        md = ""
        if len(self.devices) > 1:
            ld = [sum(self.cost(self.blocks, self.chunks[gi]) for gi in self.chunks_of[d]) for d in self.devices]
            md = f", {len(self.devices)} GPUs (work balance {min(ld) / max(ld):.2f})"
        return (f"{len(self.chunks)} chunks, {nitems - n_host}/{nitems} arrays GPU-resident, "
                f"{_gb(hb):.1f} GB streamed from host{'' if self.keep_pm else ', lean (no stored Pm)'} "
                f"(GPU budget {_gb(self.budget):.1f} GB{' each' if len(self.devices) > 1 else ''}){md}")


def _rand_mpf(shape, L, gen, spread=4):
    """random normalised MPF entries of magnitude ~2^[-spread, spread]"""
    n = int(np.prod(shape))
    m = torch.randint(-2 ** 31, 2 ** 31, (L, n), dtype=torch.int32, device=dev, generator=gen)
    e = torch.randint(-spread, spread + 1, (n,), dtype=torch.int64, device=dev, generator=gen) - 32 * L
    return F.normalize(m, e, L).reshape(*shape)


def resolve_devices(spec=None):
    """GPUs to use: a count ("2"), a list ("0,2") or "all"; default SDPVIBE_GPUS, else 1 (GPU 0).
    Testing: SDPVIBE_FAKE_GPUS=k runs the multi-GPU code path with k device slots on GPU 0."""
    fake = int(os.environ.get("SDPVIBE_FAKE_GPUS", 0))
    if fake > 1: return [torch.cuda.current_device()] * fake
    spec = str(spec or os.environ.get("SDPVIBE_GPUS", "1")).strip()
    n = torch.cuda.device_count()
    if spec == "all": return list(range(n))
    if "," in spec: devs = [int(x) for x in spec.split(",") if x.strip()]
    else: devs = list(range(int(spec)))
    if not devs or len(set(devs)) != len(devs) or min(devs) < 0 or max(devs) >= n:
        raise ValueError(f"invalid GPU selection {spec!r} ({n} CUDA devices visible)")
    return devs


class SDP:
    """SDP in pmp2sdp JSON format (or synthetic(...)), batched by dim in chunks per Layout;
    large arrays go to self.store (GPU or pinned host memory).  Multi-GPU (devices): each chunk lives
    on one device (layout.dev_of)."""

    def __init__(self, path, L, workers=None, budget=None, chunk=None, stream=None, devices=None):
        from .sdp_io import SdpSource, read_objectives, read_normalization
        mp.mp.prec = F.NBITS(L) + 64
        devices = resolve_devices(devices)
        torch.cuda.set_device(devices[0])                      # global arrays (b, y, Q) live on the first GPU
        t0 = time.time()
        src = SdpSource(path)
        nblocks = src.json("control.json")["num_blocks"]
        const_s, b_s = read_objectives(src)
        obj = {"b": b_s}
        const = mp.mpf(const_s)
        b = F.from_strings(b_s, L, (len(b_s), 1))
        self.normalization = read_normalization(src)
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
        self._build(blocks, data, b, const, L, Layout(blocks, len(obj["b"]), L, budget, chunk, stream, devices))
        self.read_info = f"{info}, assemble {time.time() - t1:.1f}s; {self.layout.describe()}"

    @classmethod
    def synthetic(cls, blocks, N, L, seed=0, budget=None, chunk=None, stream=None, devices=None):
        """random SDP with the given block shapes [dict(dim, K, he, ho)] (benchmarks / memory tests)"""
        self = cls.__new__(cls)
        mp.mp.prec = F.NBITS(L) + 64
        devices = resolve_devices(devices)
        torch.cuda.set_device(devices[0])
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        t1 = time.time()

        def data(j):
            bl = blocks[j]; Kj = bl["K"]; npair = bl["dim"] * (bl["dim"] + 1) // 2
            g = torch.Generator(device=torch.cuda.current_device()); g.manual_seed(seed * 100003 + j)
            Ve = _rand_mpf((bl["he"], Kj), L, g) if bl["he"] else None
            Vo = _rand_mpf((bl["ho"], Kj), L, g) if bl["ho"] else None
            return Ve, Vo, _rand_mpf((Kj * npair, 1), L, g), _rand_mpf((Kj * npair, N), L, g)

        self._build(blocks, data, _rand_mpf((N, 1), L, gen), mp.mpf(0), L, Layout(blocks, N, L, budget, chunk, stream, devices))
        self.read_info = f"synthetic, assemble {time.time() - t1:.1f}s; {self.layout.describe()}"
        return self

    def _build(self, blocks, data, b, const, L, layout):
        from .store import Store, MultiStore
        self.const, self.b, self.N = const, b, b.shape[0]
        N = self.N
        self.layout = layout
        self.devices, self.phys = layout.devices, layout.phys        # slots, slot -> CUDA device
        self.dev0 = 0
        multi = len(self.devices) > 1
        self.store = MultiStore(layout.host_keys, layout.dev_of, self.phys) if multi else Store(layout.host_keys)
        self.groups = []
        for gi, c in enumerate(layout.chunks):
            with torch.cuda.device(self.phys[layout.dev_of[gi]]):
                g = Group()
                g.dev = layout.dev_of[gi]
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
        for d in set(self.phys): torch.cuda.synchronize(d)
        self.total_psd_rows = sum(g.psd_rows for g in self.groups)
        self.nblocks = len(blocks)


# ------------------------------------------------------------------ solver
DEFAULTS = dict(maxIterations=500, findPrimalFeasible=False, findDualFeasible=False,
                detectPrimalFeasibleJump=False, detectDualFeasibleJump=False,
                dualityGapThreshold="1e-30", primalErrorThreshold="1e-30", dualErrorThreshold="1e-30",
                initialMatrixScalePrimal="1e20", initialMatrixScaleDual="1e20",
                feasibleCenteringParameter="0.1", infeasibleCenteringParameter="0.3",
                stepLengthReduction="0.7", maxComplementarity="1e100", minPrimalStep="0", minDualStep="0")


def fmt_big(v, digits):
    """a number as SDPB prints a BigFloat: like C++ defaultfloat (%g) with `digits` significant digits"""
    v = mp.mpf(v)
    if v == 0: return "0"
    s = mp.nstr(v, digits, min_fixed=-5, max_fixed=digits, strip_zeros=True)
    mant, _, ex = s.partition("e")
    if mant.endswith(".0"): mant = mant[:-2]                      # %g prints 1, not 1.0
    s = mant + ("e" + ex if ex else "")
    if "e" in s:                                                  # C++: e+40 / e-31, at least two digits
        mant, ex = s.split("e"); sign = "-" if ex.startswith("-") else "+"
        ex = ex.lstrip("+-").lstrip("0") or "0"
        s = f"{mant}e{sign}{ex.zfill(2)}"
    return s


def chol_log2_cond_parts(ch):
    """(max, min) over the diagonal of L (log2) for a single-matrix Chol"""
    n = ch.n; L = ch.L
    idx = torch.arange(n, device=dev)
    d = ch.Lt.limbs[:, :, idx, idx]
    top = d[-1].double() * 2.0 ** 32 + (d[-2].double() % 2.0 ** 32) if L > 1 else d[-1].double()
    lg = torch.log2(torch.clamp(top, min=1e-300)) + (ch.Lt.exp + 32 * (L - 2)) + ch.k.double()
    return float(lg.max()), float(lg.min())


def chol_log2_cond(ch, mask=None):
    """per block log2 of SDPB's Cholesky condition number estimate (max/min diagonal of L)^2;
    mask (nb, n): rows that count (padding excluded)"""
    n = ch.n; L = ch.L
    idx = torch.arange(n, device=dev)
    d = ch.Lt.limbs[:, :, idx, idx]                               # (L, nb, n) fixed point, > 0
    top = d[-1].double() * 2.0 ** 32 + (d[-2].double() % 2.0 ** 32) if L > 1 else d[-1].double()
    lg = torch.log2(torch.clamp(top, min=1e-300)) + (ch.Lt.exp + 32 * (L - 2)) + ch.k.double()
    if mask is None: mask = torch.ones_like(lg, dtype=torch.bool)
    hi = torch.where(mask, lg, torch.full_like(lg, -1e300)).amax(1)
    lo = torch.where(mask, lg, torch.full_like(lg, 1e300)).amin(1)
    return torch.where(mask.any(1), 2 * (hi - lo), torch.zeros_like(hi))


class Solver:
    def __init__(self, sdp, L, precision=None, **params):
        self.sdp, self.L = sdp, L
        self.precision = precision or (32 * (L - 1) - 2)
        self.digits = int(mp.ceil(self.precision * mp.log10(2))) + 1  # SDPB's set_stream_precision
        self._pending_update = None
        self.c_minus_By = [None] * len(sdp.groups)
        p = dict(DEFAULTS); p.update(params)
        self.p = {k: (mp.mpf(v) if isinstance(v, str) else v) for k, v in p.items()}
        self.S = lambda v: F.scalar(v, L)
        for gi, g in enumerate(sdp.groups):                          # per-chunk state -> store
            with torch.cuda.device(sdp.phys[g.dev]):
                Xg, Yg = [], []
                for b in range(len(g.h)):
                    n = g.dim * g.h[b]
                    I = bexpand(F.eye(n, L, 1), g.nbp)
                    Xg.append(set_diag_where(I, self.S(self.p["initialMatrixScalePrimal"]), g.Xmask[b]))
                    Yg.append(set_diag_where(I, self.S(self.p["initialMatrixScaleDual"]), g.Xmask[b]))
                sdp.store.put(("xs", gi), dict(x=F.zeros((g.nb, g.P, 1), L), X=Xg, Y=Yg))
                for k in ("dc", "dp", "dir", "cholS", "Pm"): sdp.store.drop((k, gi))
        self.y = F.zeros((sdp.N, 1), L)

    def _q_block(self, Q):
        """use the tiled Cholesky of Q? (SDPB_GPU_QBLOCK=<tile> forces it; else when the monolithic
        factorisation, ~9 Q, would not comfortably fit the GPU budget)"""
        forced = int(os.environ.get("SDPB_GPU_QBLOCK", 0))
        N = Q.shape[0]
        if forced:
            self._q_block_size = forced
            return forced < N
        E = 4 * self.L + 8
        if 9.0 * N * N * E > 0.35 * self.sdp.layout.budget and N > 2048:
            self._q_block_size = 2048
            return True
        return False

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

    def _to(self, obj, d):
        """obj (tree of CUDA tensors) on device d (no copy if already there)"""
        from .store import tmap
        dv = torch.device("cuda", self.sdp.phys[d])
        return tmap(lambda t: t.to(dv), obj)

    def _bcast(self, obj):
        """{device: obj on that device}"""
        return {d: self._to(obj, d) for d in self.sdp.devices}

    def _par(self, fn, names, nxt=()):
        """fn(gi, g, o) over all chunks (o: the chunk's store items `names`) -> results by chunk index.
        One thread per GPU, each over its own chunks in order, prefetching (and, during its last chunk,
        the items `nxt` of its first chunk for the next pass).  Single GPU: inline, in chunk order."""
        sdp = self.sdp; G = sdp.groups; out = [None] * len(G)
        multi = len(sdp.devices) > 1

        def work(d):
            mine = sdp.layout.chunks_of[d] if multi else list(range(len(G)))
            if not mine: return
            with torch.cuda.device(sdp.phys[d]):
                st = sdp.store.stores[d] if multi else sdp.store
                items = [tuple((n, gi) for n in names) for gi in mine]
                for gi, objs in zip(mine, st.stream(items, [(n, mine[0]) for n in nxt])):
                    out[gi] = fn(gi, G[gi], dict(zip(names, objs)))
        if not multi:
            work(sdp.devices[0])
        else:
            for f in [self._pool.submit(work, d) for d in sdp.devices]: f.result()
        return out

    def _sum_to0(self, parts):
        """sum of N x N MPF matrices living on different devices, on device 0, in device order;
        added in column panels (peak: one panel beyond the result)"""
        Q = parts[0]
        d0 = self.sdp.dev0
        for P in parts[1:]:
            if P is None: continue
            if Q is None: Q = self._to(P, d0); continue
            N = Q.shape[-1]; w = max(1, -(-N // 8))
            for c0 in range(0, N, w):
                c1 = min(N, c0 + w)
                blk = MPF(P.m[:, :, c0:c1].to(Q.m.device), P.e[:, c0:c1].to(Q.e.device))
                put(Q, 0, c0, add(sub(Q, 0, N, c0, c1), blk))
                del blk
        return Q

    def run(self, max_iter=None, log=None, verbose=True, on_iteration=None, max_runtime=None,
            checkpoint=None, checkpoint_interval=None, stop=None):
        """max_runtime (s): SDPB's maxRuntime; checkpoint(solver) is called at the start of an iteration
        when checkpoint_interval seconds have passed since the last one; stop() -> True ends the run
        with SDPB's 'SIGTERM signal received'; on_iteration(row) gets SDPB's iterations.json record."""
        """SDPB's iteration, chunk-major: every pass loops over the chunks and fetches the chunk's
        arrays from the store (GPU-resident or streamed).  Per-chunk store items:
          B (Pm)  constant / L_S^-1 B      V    bilinear bases (V, Vx)      xs   x, X, Y
          ch      Cholesky of X, Y          res  dual residue, P residue, -XY   AXI, AY  pairings
          cholS   Cholesky of S             dir  R, L_S^-1 rhs (one direction)
          dp      predictor dX, dY          dc   corrector dx, dX, dY (applied in the next residue pass)
        Only y, b, Q (cholQ) and scalars are global.  Operation order is that of SDPB / the earlier
        group-major version (bitwise identical results for the same chunks).
        Multi-GPU: each pass runs on all devices at once (one thread per device, over its chunks);
        per-chunk contributions are reduced on device 0 in chunk order (as on one GPU), Q is
        accumulated per device and summed, factorised on device 0, and dy is broadcast."""
        p, sdp, L = self.p, self.sdp, self.L
        G = sdp.groups; nG = len(G)
        devs = sdp.devices; d0 = sdp.dev0; multi = len(devs) > 1
        if multi:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(len(devs))
        to0 = lambda o: self._to(o, d0)
        alphaP = alphaD = mp.mpf(0)
        t0 = time.time()
        it = 0
        if verbose:
            print(f"{'iter':>4} {'time':>7} {'mu':>9} {'P-obj':>11} {'D-obj':>11} {'gap':>9} {'P-err':>9} {'p-err':>9} "
                  f"{'D-err':>9} {'R-err':>9} {'P-step':>7} {'D-step':>7} {'beta':>6}")
        TM = self.timers = Timers()
        tm = TM if not multi else (lambda name: None)                     # per-phase timing: single GPU only
        ST = sdp.store
        lean = not sdp.layout.keep_pm
        PM = "B" if lean else "Pm"

        D1 = ("xs", "ch", "res", "V", "cholS", PM)
        D2 = ("xs", "ch", "res", "V", "dir", "cholS", PM)

        mk2 = lambda g, b: g.Xmask[b][:, :, None] & g.Xmask[b][:, None, :]
        update = self._pending_update                                     # (aP, aD) for the dc items
        last_ck = time.time()
        pobj = dobj = gap = primal_err = dual_err = mp.mpf(0)
        fmt = lambda v: fmt_big(v, self.digits)

        def result(reason, iterations):
            return dict(terminateReason=reason, primalObjective=pobj, dualObjective=dobj, dualityGap=gap,
                        primalError=primal_err, dualError=dual_err, iterations=iterations, time=time.time() - t0)

        try:
            while True:
                it += 1
                ti = time.time()
                if stop is not None and stop():
                    self.result = result("SIGTERM signal received", it - 1)
                    break
                if checkpoint is not None and checkpoint_interval is not None and time.time() - last_ck >= checkpoint_interval:
                    self._pending_update = update
                    checkpoint(self)
                    last_ck = time.time()
                cond = [mp.mpf(0), ""]                                        # max block condition number, name

                def upd_cond(lg2, name):
                    if lg2 > float(mp.log(cond[0], 2) if cond[0] > 0 else -1):
                        cond[0] = mp.mpf(2) ** mp.mpf(lg2); cond[1] = name
                if it == 2: TM("residues"); TM.restart(2)
                TM("residues")
                dobj = sdp.const + dot(sdp.b, self.y)
                y_d = self._bcast(self.y)
                upd_d = self._bcast(update) if update else None

                def residues(gi, g, o):
                    xs = o["xs"]; r = {}
                    if upd_d:
                        aP, aD = upd_d[g.dev]; dc = o["dc"]
                        xs = dict(x=add(xs["x"], mul(dc["dx"], aP)), X=[add(X, mul(d, aP)) for X, d in zip(xs["X"], dc["dX"])],
                                  Y=[add(Y, mul(d, aD)) for Y, d in zip(xs["Y"], dc["dY"])])
                        ST.drop(("dc", gi)); del dc
                    r["psum"] = dot(g.c, xs["x"])
                    cX = [Chol(X) for X in xs["X"]]; cY = [Chol(Y) for Y in xs["Y"]]
                    r["cond"] = []
                    for tag, chs in (("X", cX), ("Y", cY)):
                        lg = chol_log2_cond(chs[0], g.Xmask[0]).cpu().tolist()
                        z = max(range(len(lg)), key=lambda i: lg[i])
                        r["cond"].append((lg[z], f"{tag}_cholesky.block_{g.idx[z % g.nb]}_{z // g.nb}"))
                    Vs = o["V"]
                    a, y_ = self.pairings(g, Vs, cX, xs["Y"])
                    d = -self.seg_diag(g, y_, [g.K])
                    ST.put(("AXI", gi), a); ST.put(("AY", gi), y_); del a, y_
                    B = o["B"]; del o
                    By = gemm(B, bexpand(y_d[g.dev], g.nb))
                    self.c_minus_By[gi] = add(g.c, By, -1)
                    d = add(d, By, -1); del By
                    dres = add(d, g.c); del d
                    r["pvec"] = gemm(flat_rows(B), flat_rows(xs["x"]), transA=True)
                    del B
                    Ax = self.cmws(g, Vs, xs["x"]); del Vs
                    Pres = [masked(add(Ax[b], xs["X"][b], -1), mk2(g, b)) for b in range(len(g.h))]; del Ax
                    r["dual_err"] = max_abs(dres)
                    r["perrP"] = [max_abs(P) for P in Pres]
                    mXY = [masked(-gemm(X, Y), mk2(g, b)) for b, (X, Y) in enumerate(zip(xs["X"], xs["Y"]))]
                    r["mu"] = [total(diag_of(M)) for M in mXY]
                    ST.put(("xs", gi), xs); ST.put(("ch", gi), dict(X=cX, Y=cY))
                    ST.put(("res", gi), dict(dres=dres, Pres=Pres, mXY=mXY))
                    return r
                psum = musum = mp.mpf(0)
                dual_err = primal_err_P = mp.mpf(0)
                pvec = sdp.b
                for r in self._par(residues, ("xs", "V", "B", *(["dc"] if update else [])), nxt=("AXI", "AY", "B")):
                    psum += r["psum"]
                    for c in r["cond"]: upd_cond(*c)
                    pvec = add(pvec, to0(r["pvec"]), -1)
                    dual_err = max(dual_err, r["dual_err"])
                    primal_err_P = max([primal_err_P] + r["perrP"])
                    for v in r["mu"]: musum += v
                del r
                update = upd_d = None; self._pending_update = None
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
                elif max_runtime is not None and time.time() - t0 >= max_runtime: reason = "maxRuntime exceeded"
                elif it > 1 and alphaP < p["minPrimalStep"]: reason = "primal step too small"
                elif it > 1 and alphaD < p["minDualStep"]: reason = "dual step too small"
                if reason:
                    self.result = result(reason, it - 1)
                    break
                mu = -musum / sdp.total_psd_rows
                if mu > p["maxComplementarity"]:
                    self.result = result("maxComplementarity exceeded", it - 1)
                    break
                # ---- Schur complement, its Cholesky factors, Q = sum P^T P ----
                QA, Qd = {}, {}
                for d in devs:
                    with torch.cuda.device(sdp.phys[d]):
                        ref = getattr(self, "_q_ref", None)
                        QA[d] = F.GramAcc(sdp.N, L, None if ref is None else self._to(ref, d)) if lean else None   # chunked: one CRT
                        Qd[d] = None

                def schur_q(gi, g, o):
                    ST.drop(("AXI", gi)); ST.drop(("AY", gi))
                    tm("schur: assemble S")
                    Sg = self.schur(g, o["AXI"], o["AY"])
                    tm("schur: chol(S)")
                    cs = Chol(Sg); del Sg
                    lg = chol_log2_cond(cs, g.xmask).cpu().tolist()
                    q = max(range(len(lg)), key=lambda i: lg[i])
                    c = (lg[q], f"schur_complement_cholesky.block_{g.idx[q]}")
                    tm("schur: P = L^-1 B")
                    if lean: P = solve_panels(cs, o["B"], sdp.layout.panels)    # columns independent: bounded temporaries
                    else: P = cs.solve_lower(o["B"])
                    del o
                    tm("schur: Q += P^T P")
                    if lean:
                        QA[g.dev].add(flat_rows(P))
                    else:
                        q = gemm(flat_rows(P), flat_rows(P), transA=True)
                        Qd[g.dev] = q if Qd[g.dev] is None else add(Qd[g.dev], q)
                        del q
                    ST.put(("cholS", gi), cs); del cs
                    if not lean: ST.put(("Pm", gi), P)
                    return c
                for c in self._par(schur_q, ("AXI", "AY", "B"), nxt=D1): upd_cond(*c)
                TM("chol(Q)")
                if lean:
                    parts = [None] * len(devs)

                    def finish(k):
                        with torch.cuda.device(sdp.phys[devs[k]]): parts[k] = QA[devs[k]].result()
                    if multi:
                        for f in [self._pool.submit(finish, k) for k in range(len(devs))]: f.result()
                    else: finish(0)
                    Q = self._sum_to0(parts) if multi else parts[0]; del parts
                    self.q_flushes = getattr(self, "q_flushes", 0) + sum(QA[d].flushes for d in devs)
                    loss = max(QA[d].precision_loss() for d in devs)
                    if loss > 28: self.q_lossy = getattr(self, "q_lossy", 0) + 1     # reference > 28 bits above the column max
                    if multi:                                                # combined column maxima -> next reference
                        for d in devs[1:]: QA[d0].Tmax = torch.maximum(QA[d0].Tmax, to0(QA[d].Tmax))
                    self._q_ref = QA[d0].next_reference()
                else:
                    Q = self._sum_to0([Qd[d] for d in devs]) if multi else Qd[d0]
                del QA, Qd
                if self._q_block(Q):                                         # tiled: ~Q/2 instead of ~9 Q
                    from .blockchol import BlockChol
                    holder = [Q]; del Q
                    cholQ = BlockChol(holder, self._q_block_size)
                    Q_cond = mp.mpf(2) ** mp.mpf(cholQ.log2_cond())
                else:
                    cholQ = Chol(copy_lower_to_upper(Q)); del Q
                    Q_cond = mp.mpf(2) ** mp.mpf(float(chol_log2_cond(cholQ)[0]))

                def direction(beta, corr):
                    """predictor (corr=False) or corrector search direction; returns dy and, per chunk, stores
                    dp (predictor) or dc (corrector); accumulates the quantities needed afterwards"""
                    bm_d, muS_d = {}, {}
                    for d in devs:
                        with torch.cuda.device(sdp.phys[d]): bm_d[d] = self.S(beta * mu); muS_d[d] = self.S(mu)
                    acc = dict(Rerr=mp.mpf(0), fr=mp.mpf(0), lamP=None, lamD=None)
                    TM("direction (corrector) 1" if corr else "direction (predictor) 1")

                    def dir1(gi, g, o):
                        xs, res, cX = o["xs"], o["res"], o["ch"]["X"]
                        bm = bm_d[g.dev]; r = {}
                        if not corr:
                            r["Rerr"] = [max_abs(add_diag(M, muS_d[g.dev], g.Xmask[b])) for b, M in enumerate(res["mXY"])]
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
                        r["dy"] = gemm(flat_rows(Pm), flat_rows(w), transA=True)
                        return r
                    dy = pvec
                    for r in self._par(dir1, (*D1, *(["dp"] if corr else [])), nxt=D2):
                        if not corr: acc["Rerr"] = max([acc["Rerr"]] + r["Rerr"])
                        dy = add(dy, to0(r["dy"]), -1)
                    dy = cholQ.solve(dy)
                    dy_d = self._bcast(dy)
                    TM("direction (corrector) 2 + step length" if corr else "direction (predictor) 2")

                    def dir2(gi, g, o):
                        xs, res, cX, dr = o["xs"], o["res"], o["ch"]["X"], o["dir"]
                        cs, Pm = o["cholS"], o[PM]; r = {}
                        Pdy = gemm(Pm, bexpand(dy_d[g.dev], g.nb))
                        if lean: Pdy = cs.solve_lower(Pdy)                   # P dy = L^-1 B dy
                        dxg = cs.solve_upper(add(dr["W"], Pdy)); del cs, Pm, Pdy
                        Ad = self.cmws(g, o["V"], dxg)
                        dXg = [add(Ad[b], res["Pres"][b]) for b in range(len(g.h))]; del Ad
                        dYg = [masked(-symmetrize(cX[b].solve(add(gemm(dXg[b], xs["Y"][b]), dr["R"][b], -1))), mk2(g, b))
                               for b in range(len(g.h))]
                        ST.drop(("dir", gi))
                        if not corr:
                            r["fr"] = [dot(masked(add(xs["X"][b], dXg[b]), mk2(g, b)), masked(add(xs["Y"][b], dYg[b]), mk2(g, b)))
                                       for b in range(len(g.h))]
                            ST.put(("dp", gi), dict(dX=dXg, dY=dYg))
                        else:
                            r["lamP"] = min_eig(o["ch"]["X"], dXg)
                            r["lamD"] = min_eig(o["ch"]["Y"], dYg)
                            ST.drop(("dp", gi))
                            ST.put(("dc", gi), dict(dx=dxg, dX=dXg, dY=dYg))
                        return r
                    for r in self._par(dir2, D2, nxt=("xs", "V", "B", "dc") if corr else D1 + ("dp",)):
                        if not corr:
                            for v in r["fr"]: acc["fr"] += v
                        else:
                            for k in ("lamP", "lamD"):
                                if r[k] is not None: acc[k] = r[k] if acc[k] is None else min(acc[k], r[k])
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
                self._pending_update = update
                self.y = add(self.y, mul(dy, aD))
                row = dict(iteration=it, total_time=round(time.time() - t0, 3), iter_time=round(time.time() - ti, 3),
                           mu=fmt(mu), **{"P-obj": fmt(pobj), "D-obj": fmt(dobj), "gap": fmt(gap),
                           "P-err": fmt(primal_err_P), "p-err": fmt(primal_err_p), "D-err": fmt(dual_err),
                           "R-err": fmt(R_err), "P-step": fmt(alphaP), "D-step": fmt(alphaD), "beta": fmt(beta_c),
                           "Q_cond_number": fmt(Q_cond), "max_block_cond_number": fmt(cond[0]), "block_name": cond[1]})
                self.last_mu, self.last_steps = mu, (alphaP, alphaD, beta_c)
                if log is not None: log.append(row)
                if on_iteration is not None: on_iteration(row)
                if verbose:
                    f = lambda v: mp.nstr(v, 2, min_fixed=0, max_fixed=0) if v != 0 else "0"
                    print(f"{it:4d} {time.time() - t0:7.1f} {f(mu):>9} {mp.nstr(pobj, 6):>11} {mp.nstr(dobj, 6):>11} {f(gap):>9} "
                          f"{f(primal_err_P):>9} {f(primal_err_p):>9} {f(dual_err):>9} {f(R_err):>9} {mp.nstr(alphaP, 3):>7} "
                          f"{mp.nstr(alphaD, 3):>7} {mp.nstr(beta_c, 3):>6}", flush=True)
        finally:
            if multi: self._pool.shutdown()
        TM.report()
        if getattr(self, "q_flushes", 0) and os.environ.get("SDPB_GPU_TIMING") == "1":
            print(f"note: Q accumulator flushed {self.q_flushes} times (column scale jumps > 2^{F.GramAcc.HEAD} between chunks)")
        if getattr(self, "q_lossy", 0):
            print(f"sdpvibe: warning: in {self.q_lossy} iterations a Q column lost more than 28 bits (scale drop between iterations)",
                  file=sys.stderr)
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
