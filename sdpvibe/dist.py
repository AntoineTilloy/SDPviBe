"""Multi-GPU: one process per GPU (like SDPB's MPI ranks).

`sdpb --gpus k` (cli.main) re-executes itself k times (launch()); worker r drives GPU phys[r] and owns the
chunks the layout assigns to it.  After every pass over the chunks the per-chunk results (scalars, N-vectors)
are all-gathered, so every rank performs the same reductions in chunk order and holds the same global
state (y, dy, Q) without further broadcasts; Q's per-rank partial sums are exchanged with NCCL (or gloo:
SDPVIBE_DIST_BACKEND=gloo, e.g. several ranks on one GPU for testing, SDPVIBE_FAKE_GPUS=k).

Environment of a worker: SDPVIBE_RANK, SDPVIBE_WORLD, SDPVIBE_DEVICE, SDPVIBE_PORT.
"""
import os, signal, socket, subprocess, sys
import torch

_cpu_group = None


def rank(): return int(os.environ.get("SDPVIBE_RANK", 0))
def world(): return int(os.environ.get("SDPVIBE_WORLD", 1))
def is_worker(): return "SDPVIBE_RANK" in os.environ
def multi(): return world() > 1


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def launch(argv, phys):
    """parent: run `python -m sdpvibe argv` once per GPU; rank 0's output is ours; SIGTERM/SIGINT are
    forwarded; returns rank 0's exit code (or a failing worker's)"""
    port = _free_port()
    procs = []
    for r, d in enumerate(phys):
        env = dict(os.environ, SDPVIBE_RANK=str(r), SDPVIBE_WORLD=str(len(phys)), SDPVIBE_DEVICE=str(d),
                   SDPVIBE_PORT=str(port))
        if len(set(phys)) < len(phys): env.setdefault("SDPVIBE_DIST_BACKEND", "gloo")   # NCCL: one rank per GPU
        out = None if r == 0 else subprocess.DEVNULL
        procs.append(subprocess.Popen([sys.executable, "-m", "sdpvibe"] + list(argv), env=env, stdout=out))

    def fwd(sig, _):
        for p in procs:
            if p.poll() is None: p.send_signal(sig)
    old = {s: signal.signal(s, fwd) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        codes = [None] * len(procs)
        while any(c is None for c in codes):
            for i, p in enumerate(procs):
                if codes[i] is None:
                    try: codes[i] = p.wait(timeout=0.5)
                    except subprocess.TimeoutExpired: pass
                    if codes[i] not in (None, 0, 15):            # a worker failed: the others would hang
                        for q in procs:
                            if q.poll() is None: q.kill()
    finally:
        for s, h in old.items(): signal.signal(s, h)
    bad = [c for c in codes[1:] if c not in (0, 15)]
    return codes[0] if codes[0] not in (0, 15) or not bad else bad[0]


def init():
    """worker: bind the GPU, join the process group"""
    global _cpu_group
    if not multi(): return
    import torch.distributed as D
    torch.cuda.set_device(int(os.environ["SDPVIBE_DEVICE"]))
    backend = os.environ.get("SDPVIBE_DIST_BACKEND", "nccl")
    D.init_process_group(backend, init_method=f"tcp://127.0.0.1:{os.environ['SDPVIBE_PORT']}",
                         rank=rank(), world_size=world(), device_id=torch.device("cuda", torch.cuda.current_device())
                         if backend == "nccl" else None)
    _cpu_group = D.new_group(backend="gloo") if backend == "nccl" else D.group.WORLD


def allgather(obj):
    """list over ranks of a picklable object (tensors: move to CPU first)"""
    if not multi(): return [obj]
    import torch.distributed as D
    out = [None] * world()
    D.all_gather_object(out, obj, group=_cpu_group)
    return out


def bcast(obj, src=0):
    if not multi(): return obj
    import torch.distributed as D
    lst = [obj]
    D.broadcast_object_list(lst, src=src, group=_cpu_group)
    return lst[0]


def barrier():
    if multi():
        import torch.distributed as D
        D.barrier(group=_cpu_group)


def _tensor_bcast(t, src):
    """broadcast a CUDA tensor in place (gloo: staged through host memory)"""
    import torch.distributed as D
    if os.environ.get("SDPVIBE_DIST_BACKEND", "nccl") == "nccl":
        D.broadcast(t, src=src)
    else:
        h = t.cpu(); D.broadcast(h, src=src); t.copy_(h)


def allreduce_max_(t):
    """elementwise max over ranks, in place (CUDA int64 tensor)"""
    if not multi(): return t
    import torch.distributed as D
    if os.environ.get("SDPVIBE_DIST_BACKEND", "nccl") == "nccl":
        D.all_reduce(t, op=D.ReduceOp.MAX)
    else:
        h = t.cpu(); D.all_reduce(h, op=D.ReduceOp.MAX); t.copy_(h)
    return t


def sum_mpf(Q, add, panels=8):
    """sum over ranks of N x N MPF matrices (Q: this rank's part, or None), in rank order, identical on
    every rank; column panels bound the extra memory"""
    if not multi(): return Q
    from .mpf import MPF
    have = allgather(None if Q is None else (tuple(Q.m.shape), tuple(Q.e.shape)))
    shp = next((h for h in have if h is not None), None)
    if shp is None: return None
    (L, N, N2), _ = shp
    w = max(1, -(-N2 // panels))
    out_m = torch.empty((L, N, N2), dtype=torch.int32, device="cuda"); out_e = torch.empty((N, N2), dtype=torch.int64, device="cuda")
    for c0 in range(0, N2, w):
        c1 = min(N2, c0 + w)
        acc = None
        for r in range(world()):
            if have[r] is None: continue
            if r == rank(): m = Q.m[:, :, c0:c1].contiguous(); e = Q.e[:, c0:c1].contiguous()
            else:
                m = torch.empty((L, N, c1 - c0), dtype=torch.int32, device="cuda"); e = torch.empty((N, c1 - c0), dtype=torch.int64, device="cuda")
            _tensor_bcast(m, r); _tensor_bcast(e, r)
            acc = MPF(m, e) if acc is None else add(acc, MPF(m, e))
            del m, e
        out_m[:, :, c0:c1] = acc.m; out_e[:, c0:c1] = acc.e
        del acc
    return MPF(out_m, out_e)
