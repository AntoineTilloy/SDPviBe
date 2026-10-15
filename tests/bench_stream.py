"""Synthetic SDP with the block structure of the stress-tensor system (Chang et al. 2024), scaled
from its nmax=6 instance (N=742; 113 blocks: dims {1:45, 2:53, 3:1, 4:1, 5:13}; K <= 57):
N ~ nmax^2, number of blocks ~ nmax, points per block ~ nmax.  Runs a few iterations in one
memory layout and reports per-iteration time, peak memory and host<->GPU traffic.
usage: python tests/bench_stream.py --nmax 10 [--prec 1024] [--its 4] [--stream none|auto|all]
                                    [--budget GB] [--chunk GB] [--scale-blocks 1.0]"""
import argparse, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # streaming: less fragmentation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from sdpvibe.solver import SDP, Solver, Layout


def st3d_like(nmax, scale_blocks=1.0, seed=0):
    s = nmax / 6
    N = int(round(742 * s * s))
    Kmax = int(round(57 * s))
    rng = np.random.default_rng(seed)
    blocks = []
    for dim, cnt in {1: 45, 2: 53, 3: 1, 4: 1, 5: 13}.items():
        n = max(1, int(round(cnt * s * scale_blocks)))
        for i in range(n):
            K = 1 if (dim == 1 and i < 3) else int(rng.integers(int(0.7 * Kmax), Kmax + 1))
            blocks.append(dict(dim=dim, K=K, he=(K - 1) // 2 + 1, ho=(K - 2) // 2 + 1 if K > 1 else 0))
    return blocks, N


ap = argparse.ArgumentParser()
ap.add_argument("--nmax", type=float, default=6)
ap.add_argument("--prec", type=int, default=1024)
ap.add_argument("--its", type=int, default=4)
ap.add_argument("--stream", default="auto")
ap.add_argument("--budget", type=float)
ap.add_argument("--chunk", type=float)
ap.add_argument("--scale-blocks", type=float, default=1.0)
ap.add_argument("--plan-only", action="store_true")
ap.add_argument("--bw", action="store_true", help="measure pinned host<->GPU bandwidth and exit")
a = ap.parse_args()
if a.bw:
    h = torch.empty(2 ** 30, dtype=torch.uint8).pin_memory(); d = torch.empty(2 ** 30, dtype=torch.uint8, device="cuda")
    for name, f in (("H2D", lambda: d.copy_(h, non_blocking=True)), ("D2H", lambda: h.copy_(d, non_blocking=True))):
        f(); torch.cuda.synchronize(); t = time.time()
        for _ in range(3): f()
        torch.cuda.synchronize(); print(f"{name} {3 / (time.time() - t):.1f} GB/s")
    sys.exit()
if a.budget:                                     # emulate a smaller GPU (hard cap)
    os.environ["SDPB_GPU_MEM_GB"] = str(a.budget)
L = (a.prec + 2 + 31) // 32 + 1
blocks, N = st3d_like(a.nmax, a.scale_blocks)
G = 2 ** 30
lay = Layout(blocks, N, L, a.budget and a.budget * G, a.chunk and a.chunk * G, a.stream)
tot = {}
for c in lay.chunks:
    for k, v in lay.items(blocks, c).items(): tot[k] = tot.get(k, 0) + v
print(f"nmax~{a.nmax}: N={N}, {len(blocks)} blocks, max P={max(Layout.shape(blocks, c)[4] for c in lay.chunks)}, L={L}; "
      f"arrays: " + ", ".join(f"{k} {v / G:.1f} GB" for k, v in tot.items()) + f"; Q {N * N * (4 * L + 8) / G:.1f} GB", flush=True)
print("layout:", lay.describe(), flush=True)
if a.plan_only: sys.exit()
t0 = time.time()
sdp = SDP.synthetic(blocks, N, L, budget=a.budget and a.budget * G, chunk=a.chunk and a.chunk * G, stream=a.stream)
print(f"built in {time.time() - t0:.1f}s: {sdp.read_info}; host pinned {sdp.store.host_bytes / G:.1f} GB", flush=True)
S = Solver(sdp, L)
log = []
torch.cuda.reset_peak_memory_stats()
S.run(max_iter=a.its, log=log, verbose=True)
st = sdp.store
ti = [r["iter_time"] for r in log]
print(f"RESULT nmax={a.nmax} stream={a.stream} budget={a.budget} chunks={len(sdp.groups)}: iteration times {ti}; "
      f"median after first {np.median(ti[1:]):.2f} s; peak GPU {max(S.timers.peak.values(), default=torch.cuda.max_memory_allocated()) / G:.2f} GB; "
      f"H2D {st.bytes_h2d / G / len(ti):.1f} GB/it, D2H {st.bytes_d2h / G / len(ti):.1f} GB/it", flush=True)
