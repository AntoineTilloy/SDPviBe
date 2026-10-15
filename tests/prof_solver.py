"""GPU busy time vs wall time for a few solver iterations (torch profiler / CUPTI sees all kernels).
usage: python tests/prof_solver.py <sdp dir> [precision] [iterations]"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.profiler import profile, ProfilerActivity
from sdpvibe.solver import SDP, Solver

path = sys.argv[1]; prec = int(sys.argv[2]) if len(sys.argv) > 2 else 768
its = int(sys.argv[3]) if len(sys.argv) > 3 else 3
L = (prec + 2 + 31) // 32 + 1
sdp = SDP(path, L)
s = Solver(sdp, L)
s.run(max_iter=2, verbose=False)                         # warm-up: compile all kernels
s2 = Solver(sdp, L)
s2.run(max_iter=1, verbose=False)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    t0 = time.time()
    s3 = Solver(sdp, L)
    s3.run(max_iter=its, verbose=False)
    torch.cuda.synchronize()
    wall = time.time() - t0
ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
busy = sum(e.device_time for e in ev) / 1e6
print(f"{its} iterations: wall {wall:.2f} s, GPU kernel time {busy:.2f} s ({100 * busy / wall:.0f}% busy), {len(ev)} kernels "
      f"({len(ev) / its:.0f}/iteration)")
agg = {}
for e in ev:
    k = e.name[:60]; a = agg.setdefault(k, [0, 0.0]); a[0] += 1; a[1] += e.device_time / 1e6
for k, (n, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]:
    print(f"  {t:7.3f} s  {n:6d}x  {k}")
