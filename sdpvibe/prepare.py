"""Parse a pmp2sdp JSON SDP directory once (CPU, all cores) into sdpvibe's cache, and print its shape.
  python -m sdpvibe.prepare -s sdp_dir --precision 832 [--workers 128] [--json shape.json]
The cache (sdp_dir/sdpvibe_raw_L<L>.pkl) is what the solver reads; with control.json, objectives.json
and block_info_*.json it is all that needs to be copied to a GPU machine."""
import argparse, collections, json, os, time
from .solver import load_raw

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sdpDir", required=True)
ap.add_argument("--precision", type=int, default=768)
ap.add_argument("--guardLimbs", type=int, default=1)
ap.add_argument("--workers", type=int)
ap.add_argument("--json")
a = ap.parse_args()
L = (a.precision + 2 + 31) // 32 + a.guardLimbs
t0 = time.time()
raw, info = load_raw(a.sdpDir, L, a.workers)
N = len(json.load(open(os.path.join(a.sdpDir, "objectives.json")))["b"])
E = 4 * L + 8
dims = collections.Counter(r["dim"] for r in raw)
Ks = collections.defaultdict(list)
for r in raw: Ks[r["dim"]].append(r["K"])
P = [r["K"] * r["dim"] * (r["dim"] + 1) // 2 for r in raw]
shape = dict(N=N, blocks=len(raw), L=L, dims=dict(sorted(dims.items())),
             K={d: [min(k), max(k), round(sum(k) / len(k), 1)] for d, k in sorted(Ks.items())},
             sum_P=sum(P), max_P=max(P), B_GB=round(sum(P) * N * E / 2 ** 30, 2),
             S_GB=round(sum(p * p for p in P) * E / 2 ** 30, 2), Q_GB=round(N * N * E / 2 ** 30, 2),
             gram_MACs=float(sum(P)) * N * N, trsm_MACs=float(sum(p * p for p in P)) * N, read=info,
             block_shapes=[[r["dim"], r["K"], r["bilinear_bases_even"][0], r["bilinear_bases_odd"][0]] for r in raw],
             seconds=round(time.time() - t0, 1))
print(json.dumps({k: v for k, v in shape.items() if k != "block_shapes"}))
if a.json:
    json.dump(shape, open(a.json, "w"), indent=1)
