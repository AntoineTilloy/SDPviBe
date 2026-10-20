"""Solver state in SDPB's block order, independent of sdpvibe's chunk layout.

  state = {"y": MPF (N, 1),
           "x": [MPF (P_j, 1) for each block j],                 P_j = K_j dim_j (dim_j + 1) / 2
           "X": [[MPF (n_j0, n_j0), MPF (n_j1, n_j1)] ...],       n_jp = dim_j * (bilinear basis rows of parity p)
           "Y": same}
All MPF on the CPU.  Index conventions are SDPB's: x_j follows the rows of B_j (segments (c, r), r <= c,
of K_j points), X_j,p has dim_j x dim_j sub-blocks of size (basis rows), as in SDPB's x_<j>.txt and
X_matrix_<2j+p>.txt.  get_state applies a pending step (sdpvibe applies x += aP dx etc. lazily, at the
start of the next iteration) so the state is that of SDPB at the same point.
"""
import torch
from . import mpf as F
from .mpf import MPF, add, mul


def _cpu(A):
    return MPF(A.m.cpu(), A.e.cpu())


def _block_shapes(sdp, j):
    b = sdp.layout.blocks[j]
    return b["dim"], b["K"], (b["he"], b["ho"])


def _x_index(g, K_j):
    """positions of block x_j (SDPB order) inside the chunk's padded (P) layout"""
    idx = []
    npair = g.dim * (g.dim + 1) // 2
    for s in range(npair):
        idx.extend(range(s * g.K, s * g.K + K_j))
    return torch.tensor(idx, dtype=torch.long)


def _X_index(g, h_j):
    """positions of the real rows of a parity block inside the chunk's padded (dim * h) layout"""
    h = g.h[0]
    return torch.tensor([c * h + r for c in range(g.dim) for r in range(h_j)], dtype=torch.long)


def get_state(solver):
    sdp = solver.sdp
    ST = sdp.store
    st = {"y": _cpu(solver.y), "x": [None] * sdp.nblocks, "X": [None] * sdp.nblocks, "Y": [None] * sdp.nblocks}
    upd = solver._pending_update
    for gi, g in enumerate(sdp.groups):
        with torch.cuda.device(sdp.phys[g.dev] if hasattr(sdp, "phys") else 0):
            xs = ST.get(("xs", gi))
            if upd is not None:
                aP, aD = (solver._to(u, g.dev) for u in upd); dc = ST.get(("dc", gi))
                xs = dict(x=add(xs["x"], mul(dc["dx"], aP)), X=[add(X, mul(d, aP)) for X, d in zip(xs["X"], dc["dX"])],
                          Y=[add(Y, mul(d, aD)) for Y, d in zip(xs["Y"], dc["dY"])])
            x = _cpu(xs["x"]); X = _cpu(xs["X"][0]); Y = _cpu(xs["Y"][0])
            for q, j in enumerate(g.idx):
                dim, K_j, hs = _block_shapes(sdp, j)
                ix = _x_index(g, K_j)
                st["x"][j] = MPF(x.m[:, q, ix].contiguous(), x.e[q, ix].contiguous())
                Xs, Ys = [], []
                for p in (0, 1):
                    z = p * g.nb + q
                    ii = _X_index(g, hs[p])
                    Xs.append(MPF(X.m[:, z][:, ii][:, :, ii].contiguous(), X.e[z][ii][:, ii].contiguous()))
                    Ys.append(MPF(Y.m[:, z][:, ii][:, :, ii].contiguous(), Y.e[z][ii][:, ii].contiguous()))
                st["X"][j] = Xs; st["Y"][j] = Ys
    return st


def _to_dev_L(A, L):
    """MPF (any limb count, CPU or GPU) -> normalised L-limb MPF on the GPU"""
    from .rns import dev
    m = A.m.to(dev); e = A.e.to(dev)
    shape = tuple(e.shape)
    if m.shape[0] == L:
        return MPF(m, e)
    return F.normalize(m.reshape(m.shape[0], -1).contiguous(), e.reshape(-1).contiguous(), L).reshape(*shape)


def set_state(solver, st):
    """initialise the solver from a state (e.g. a checkpoint); padding stays inert"""
    sdp = solver.sdp
    L = solver.L
    ST = sdp.store
    if len(st["x"]) != sdp.nblocks:
        raise ValueError(f"checkpoint has {len(st['x'])} blocks, the SDP has {sdp.nblocks}")
    solver.y = _to_dev_L(st["y"], L)
    if solver.y.shape != (sdp.N, 1):
        raise ValueError(f"checkpoint y has shape {solver.y.shape}, the SDP has N={sdp.N}")
    for gi, g in enumerate(sdp.groups):
        with torch.cuda.device(sdp.phys[g.dev] if hasattr(sdp, "phys") else 0):
            xs = ST.get(("xs", gi))
            x = xs["x"].clone(); X = xs["X"][0].clone(); Y = xs["Y"][0].clone()
            for q, j in enumerate(g.idx):
                dim, K_j, hs = _block_shapes(sdp, j)
                ix = _x_index(g, K_j).to(x.e.device)
                v = _to_dev_L(st["x"][j], L).reshape(-1)
                if v.shape[0] != len(ix):
                    raise ValueError(f"checkpoint x_{j} has {v.shape[0]} entries, expected {len(ix)}")
                x.m[:, q, ix, 0] = v.m; x.e[q, ix, 0] = v.e
                for p in (0, 1):
                    z = p * g.nb + q
                    ii = _X_index(g, hs[p]).to(x.e.device)
                    for M, src in ((X, st["X"][j][p]), (Y, st["Y"][j][p])):
                        s = _to_dev_L(src, L)
                        if s.shape != (len(ii), len(ii)):
                            raise ValueError(f"checkpoint matrix block {2 * j + p} has shape {s.shape}, expected {len(ii)}")
                        if len(ii):
                            sub_m = M.m[:, z]; sub_e = M.e[z]
                            sub_m[:, ii[:, None], ii[None, :]] = s.m
                            sub_e[ii[:, None], ii[None, :]] = s.e
            ST.put(("xs", gi), dict(x=x, X=[X], Y=[Y]))


def c_minus_By_blocks(solver):
    """c - B y per block (SDPB order), from the last residue pass"""
    sdp = solver.sdp
    out = [None] * sdp.nblocks
    for gi, g in enumerate(sdp.groups):
        v = solver.c_minus_By[gi]
        if v is None:
            continue
        v = _cpu(v)
        for q, j in enumerate(g.idx):
            ix = _x_index(g, sdp.layout.blocks[j]["K"])
            out[j] = MPF(v.m[:, q, ix, 0].contiguous(), v.e[q, ix, 0].contiguous())
    return [b for b in out if b is not None]
