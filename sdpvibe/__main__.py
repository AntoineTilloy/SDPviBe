"""SDPviBe command line. Usage mirrors sdpb:
  python -m sdpvibe -s runs/x/sdp --precision 768 [--findPrimalFeasible] [--findDualFeasible] [-o out]
(input: pmp2sdp --outputFormat json)"""
import argparse, json, os, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # streaming: less fragmentation
import mpmath as mp
from sdpvibe.solver import SDP, Solver, DEFAULTS

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sdpDir", required=True)
ap.add_argument("-o", "--outDir")
ap.add_argument("--precision", type=int, default=768)
ap.add_argument("--guardLimbs", type=int, default=1)
for k, v in DEFAULTS.items():
    if isinstance(v, bool): ap.add_argument(f"--{k}", action="store_true")
    else: ap.add_argument(f"--{k}", type=type(v), default=v)
a = ap.parse_args()
L = (a.precision + 2 + 31) // 32 + a.guardLimbs
t0 = time.time()
sdp = SDP(a.sdpDir, L)
print(f"read SDP: {sdp.nblocks} blocks in {len(sdp.groups)} groups, N={sdp.N}, psd rows={sdp.total_psd_rows}, L={L} limbs "
      f"({32 * L - 2} bits) in {time.time() - t0:.1f}s [{sdp.read_info}]", flush=True)
params = {k: getattr(a, k) for k in DEFAULTS}
solver = Solver(sdp, L, **params)
out = a.outDir or a.sdpDir.rstrip("/") + "_gpu_out"
os.makedirs(out, exist_ok=True)


class IterLog(list):
    """iterations.json rewritten after every iteration (survives an interrupted run)"""
    def append(self, row):
        super().append(row)
        tmp = os.path.join(out, "iterations.json.tmp")
        json.dump(self, open(tmp, "w"), indent=0); os.replace(tmp, os.path.join(out, "iterations.json"))


log = IterLog()
res = solver.run(log=log)
with open(os.path.join(out, "out.txt"), "w") as f:
    for k, v in res.items():
        f.write(f"{k} = {mp.nstr(v, 40) if isinstance(v, mp.mpf) else v};\n")
print("-----", res["terminateReason"], "-----")
for k, v in res.items():
    if k != "terminateReason": print(f"{k:16s} = {mp.nstr(v, 20) if isinstance(v, mp.mpf) else v}")
