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
import json, os, time
import mpmath as mp
import numpy as np
import torch
from sdpvibe import mpf as F
from sdpvibe.mpf import MPF, mul, add, gemm, symmetrize, total, dot, max_abs, Chol, EZ
from sdpvibe.rns import dev


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
        self.acc = {}; self.t = None

    def __call__(self, name):
        if not self.on: return
        torch.cuda.synchronize(); now = time.time()
        if self.t is not None:
            self.acc[self.cur] = self.acc.get(self.cur, 0.0) + now - self.t
        self.t, self.cur = now, name

    def report(self):
        if not self.on: return
        self("end"); tot = sum(self.acc.values())
        print("phase timings: " + ", ".join(f"{k} {v:.2f}s ({100 * v / tot:.0f}%)" for k, v in
                                            sorted(self.acc.items(), key=lambda kv: -kv[1])))


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


class SDP:
    def __init__(self, path, L, workers=None):
        mp.mp.prec = F.NBITS(L) + 64
        t0 = time.time()
        rd = lambda f: json.load(open(os.path.join(path, f)))
        nblocks = rd("control.json")["num_blocks"]
        obj = rd("objectives.json")
        self.const = mp.mpf(obj.get("constant", "0"))
        self.b = F.from_strings(obj["b"], L, (len(obj["b"]), 1))
        self.N = N = len(obj["b"])
        cache = os.path.join(path, f"sdpb_gpu_cache_L{L}.pt")
        if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(os.path.join(path, "control.json")):
            st = torch.load(cache, map_location=dev, weights_only=False)
            self.groups = []
            for gd in st["groups"]:
                g = Group(); g.__dict__.update(gd); self.groups.append(g)
            self._finish(nblocks)
            self.read_info = f"cache {time.time() - t0:.1f}s"
            return
        global _PARSE
        _PARSE = (path, L)
        import multiprocessing
        nw = workers or min(nblocks, os.cpu_count() or 1)
        if nw > 1:
            with multiprocessing.get_context("fork").Pool(nw) as pool:
                raw = pool.map(_parse_block, range(nblocks))
        else:
            raw = [_parse_block(j) for j in range(nblocks)]
        t1 = time.time()
        self.groups = []
        for dim in sorted({r["dim"] for r in raw}):
            mem = [(j, r) for j, r in enumerate(raw) if r["dim"] == dim]
            g = Group()
            g.idx = [j for j, _ in mem]; g.nb = nb = len(mem); g.dim = dim
            g.K = K = max(r["K"] for _, r in mem)
            h = max(r[k][0] for _, r in mem for k in ("bilinear_bases_even", "bilinear_bases_odd"))
            g.h = [h]; g.nbp = 2 * nb
            g.P = P = K * dim * (dim + 1) // 2
            npair = dim * (dim + 1) // 2
            V = F.zeros((2 * nb, h, K), L)
            xm = torch.zeros((2 * nb, dim * h), dtype=torch.bool, device=dev)
            for bi, key in enumerate(("bilinear_bases_even", "bilinear_bases_odd")):
                for q, (_, r) in enumerate(mem):
                    hj, arr = r[key]; Kj = r["K"]; z = bi * nb + q
                    if hj:
                        v = F.host_to_mpf(*arr, L, (hj, Kj))
                        V.m[:, z, :hj, :Kj] = v.m; V.e[z, :hj, :Kj] = v.e
                    for c in range(dim): xm[z, c * h:c * h + hj] = True
            Vx = F.zeros((2 * nb, dim * h, dim * K), L)
            for c in range(dim): put(Vx, c * h, c * K, V)
            g.V, g.Vx, g.Xmask = [V], [Vx], [xm]
            g.c = F.zeros((nb, P, 1), L); g.B = F.zeros((nb, P, N), L)
            g.xmask = torch.zeros((nb, P), dtype=torch.bool, device=dev)
            for q, (_, r) in enumerate(mem):
                Kj = r["K"]
                cj = F.host_to_mpf(*r["c"], L, (Kj * npair, 1))
                Bj = F.host_to_mpf(*r["B"], L, (Kj * npair, N))
                for s in range(npair):                       # segments (c, r) of Kj -> K slots
                    g.c.m[:, q, s * K:s * K + Kj] = cj.m[:, s * Kj:(s + 1) * Kj]; g.c.e[q, s * K:s * K + Kj] = cj.e[s * Kj:(s + 1) * Kj]
                    g.B.m[:, q, s * K:s * K + Kj] = Bj.m[:, s * Kj:(s + 1) * Kj]; g.B.e[q, s * K:s * K + Kj] = Bj.e[s * Kj:(s + 1) * Kj]
                    g.xmask[q, s * K:s * K + Kj] = True
            self.groups.append(g)
        del raw
        self._finish(nblocks)
        t2 = time.time()
        try:
            torch.save({"groups": [g.__dict__ for g in self.groups]}, cache)
        except OSError:
            pass
        self.read_info = f"parse {t1 - t0:.1f}s ({nw} workers), assemble {t2 - t1:.1f}s, cache {time.time() - t2:.1f}s"

    def _finish(self, nblocks):
        for g in self.groups:
            g.psd_rows = int(sum(m.sum() for m in g.Xmask))
        self.total_psd_rows = sum(g.psd_rows for g in self.groups)
        self.nblocks = nblocks


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
        one = self.S(1)
        self.X, self.Y, self.x = [], [], []
        for g in sdp.groups:
            Xg, Yg = [], []
            for b in range(len(g.h)):
                n = g.dim * g.h[b]
                I = bexpand(F.eye(n, L, 1), g.nbp)
                Xg.append(set_diag_where(I, self.S(self.p["initialMatrixScalePrimal"]), g.Xmask[b]))
                Yg.append(set_diag_where(I, self.S(self.p["initialMatrixScaleDual"]), g.Xmask[b]))
            self.X.append(Xg); self.Y.append(Yg)
            self.x.append(F.zeros((g.nb, g.P, 1), L))
        self.y = F.zeros((sdp.N, 1), L)

    def segments(self, g):
        return [(c, r, (c * (c + 1) // 2 + r) * g.K) for c in range(g.dim) for r in range(c + 1)]

    def cmws(self, g, a):
        """constraint_matrix_weighted_sum: sum_p a_p A_p, per parity"""
        out = []
        a2 = dup(a)
        for b in range(len(g.h)):
            h = g.h[b]; R = F.zeros((g.nbp, g.dim * h, g.dim * h), self.L)
            for c, r, off in self.segments(g):
                s = gemm(g.V[b], col_scale(g.V[b], sub(a2, off, off + g.K, 0, 1)), transB=True)
                if r != c: s = s.shift(-1)
                put(R, r * h, c * h, s)
                if r != c: put(R, c * h, r * h, s.T)
            out.append(R)
        return out

    def pairings(self, g, cholX, Y):
        AXI, AY = [], []
        for b in range(len(g.h)):
            T = cholX[b].solve_lower(g.Vx[b])
            AXI.append(copy_lower_to_upper(gemm(T, T, transA=True)))
            AY.append(copy_lower_to_upper(gemm(g.Vx[b], gemm(Y[b], g.Vx[b]), transA=True)))
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
        p, sdp, L = self.p, self.sdp, self.L
        G = sdp.groups
        alphaP = alphaD = mp.mpf(0)
        t0 = time.time()
        it = 0
        if verbose:
            print(f"{'iter':>4} {'time':>7} {'mu':>9} {'P-obj':>11} {'D-obj':>11} {'gap':>9} {'P-err':>9} {'p-err':>9} "
                  f"{'D-err':>9} {'R-err':>9} {'P-step':>7} {'D-step':>7} {'beta':>6}")
        TM = self.timers = Timers()
        while True:
            it += 1
            ti = time.time()
            TM("objectives+chol(X,Y)")
            pobj = sdp.const + sum((dot(g.c, x) for g, x in zip(G, self.x)), mp.mpf(0))
            dobj = sdp.const + dot(sdp.b, self.y)
            gap = abs(pobj - dobj) / max(abs(pobj) + abs(dobj), mp.mpf(1))
            cholX = [[Chol(X) for X in Xg] for Xg in self.X]
            cholY = [[Chol(Y) for Y in Yg] for Yg in self.Y]
            AXI, AY, dres, Pres = [], [], [], []
            TM("pairings+residues")
            for gi, g in enumerate(G):
                a, y_ = self.pairings(g, cholX[gi], self.Y[gi])
                AXI.append(a); AY.append(y_)
                d = -self.seg_diag(g, y_, [g.K])
                d = add(d, gemm(g.B, bexpand(self.y, g.nb)), -1)
                dres.append(add(d, g.c))
                Ax = self.cmws(g, self.x[gi])
                Pres.append([masked(add(Ax[b], self.X[gi][b], -1), g.Xmask[b][:, :, None] & g.Xmask[b][:, None, :])
                             for b in range(len(g.h))])
            dual_err = max(max_abs(d) for d in dres)
            primal_err_P = max(max_abs(P) for Pg in Pres for P in Pg)
            pvec = sdp.b
            for gi, g in enumerate(G):
                pvec = add(pvec, gemm(flat_rows(g.B), flat_rows(self.x[gi]), transA=True), -1)
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
            # ---- step ----
            cholS, Pm, Q = [], [], None
            TM("schur+chol(S)+P+Q")
            for gi, g in enumerate(G):
                cs = Chol(self.schur(g, AXI[gi], AY[gi])); cholS.append(cs)
                P = cs.solve_lower(g.B); Pm.append(P)
                q = gemm(flat_rows(P), flat_rows(P), transA=True)
                Q = q if Q is None else add(Q, q)
            TM("chol(Q)")
            cholQ = Chol(copy_lower_to_upper(Q))
            TM("XY,mu")
            mXY = [[masked(-gemm(X, Y), g.Xmask[b][:, :, None] & g.Xmask[b][:, None, :])
                    for b, (X, Y) in enumerate(zip(Xg, Yg))] for g, Xg, Yg in zip(G, self.X, self.Y)]
            mu = -sum((total(diag_of(M)) for Mg in mXY for M in Mg), mp.mpf(0)) / sdp.total_psd_rows
            if mu > p["maxComplementarity"]:
                self.result = dict(terminateReason="maxComplementarity exceeded", iterations=it, time=time.time() - t0)
                break
            muS = self.S(mu)
            R_err = max(max_abs(add_diag(M, muS, g.Xmask[b])) for g, Mg in zip(G, mXY) for b, M in enumerate(Mg))

            def direction(beta, dXdY=None):
                bm = self.S(beta * mu)
                R, Z, W = [], [], []
                dy = pvec
                for gi, g in enumerate(G):
                    Rg, Zg = [], []
                    for b in range(len(g.h)):
                        Rb = mXY[gi][b] if dXdY is None else add(mXY[gi][b], gemm(dXdY[0][gi][b], dXdY[1][gi][b]), -1)
                        Rb = add_diag(Rb, bm, g.Xmask[b])
                        Rg.append(Rb)
                        Zg.append(symmetrize(cholX[gi][b].solve(add(gemm(Pres[gi][b], self.Y[gi][b]), Rb, -1))))
                    R.append(Rg)
                    VZV = [gemm(g.Vx[b], gemm(Zg[b], g.Vx[b]), transA=True) for b in range(len(g.h))]
                    rhs = add(-dres[gi], self.seg_diag(g, VZV, [g.K]), -1)
                    w = cholS[gi].solve_lower(rhs); W.append(w)
                    dy = add(dy, gemm(flat_rows(Pm[gi]), flat_rows(w), transA=True), -1)
                dy = cholQ.solve(dy)
                dx, dX, dY = [], [], []
                for gi, g in enumerate(G):
                    dxg = cholS[gi].solve_upper(add(W[gi], gemm(Pm[gi], bexpand(dy, g.nb))))
                    dx.append(dxg)
                    Ad = self.cmws(g, dxg)
                    dXg = [add(Ad[b], Pres[gi][b]) for b in range(len(g.h))]
                    dX.append(dXg)
                    dY.append([masked(-symmetrize(cholX[gi][b].solve(add(gemm(dXg[b], self.Y[gi][b]), R[gi][b], -1))),
                                      g.Xmask[b][:, :, None] & g.Xmask[b][:, None, :]) for b in range(len(g.h))])
                return dx, dX, dy, dY

            beta_p = mp.mpf(0) if pd_feas else p["infeasibleCenteringParameter"]
            TM("search direction x2")
            dx, dX, dy, dY = direction(beta_p)
            fr = mp.mpf(0)
            for gi, g in enumerate(G):
                for b in range(len(g.h)):
                    mk = g.Xmask[b][:, :, None] & g.Xmask[b][:, None, :]
                    fr += dot(masked(add(self.X[gi][b], dX[gi][b]), mk), masked(add(self.Y[gi][b], dY[gi][b]), mk))
            r = fr / (mu * sdp.total_psd_rows)
            beta = r * r if r < 1 else r
            beta_c = min(max(p["feasibleCenteringParameter"], beta), mp.mpf(1)) if pd_feas else max(p["infeasibleCenteringParameter"], beta)
            dx, dX, dy, dY = direction(beta_c, (dX, dY))
            TM("step length")
            gm = p["stepLengthReduction"]
            alphaP = step_length([c for cg in cholX for c in cg], [d for dg in dX for d in dg], gm)
            alphaD = step_length([c for cg in cholY for c in cg], [d for dg in dY for d in dg], gm)
            if pd_feas:
                alphaP = alphaD = min(alphaP, alphaD)
            TM("update")
            aP, aD = self.S(alphaP), self.S(alphaD)
            self.x = [add(x, mul(d, aP)) for x, d in zip(self.x, dx)]
            self.X = [[add(X, mul(d, aP)) for X, d in zip(Xg, dg)] for Xg, dg in zip(self.X, dX)]
            self.y = add(self.y, mul(dy, aD))
            self.Y = [[add(Y, mul(d, aD)) for Y, d in zip(Yg, dg)] for Yg, dg in zip(self.Y, dY)]
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
        return self.result


def step_length(chols, dMs, gamma):
    """gamma * (-1/lambda_min(L^-1 dM L^-T)) capped at 1 (eigenvalues from a float64 copy)"""
    lam = None
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
    if lam is None or lam > -gamma: return mp.mpf(1)
    return -gamma / lam
