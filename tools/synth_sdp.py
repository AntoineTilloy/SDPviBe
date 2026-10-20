"""Write a synthetic SDP (random data, block structure of the stress-tensor system scaled from its real
nmax=6 instance, see tests/bench_stream.py) in pmp2sdp's binary format, readable by sdpb and sdpvibe.
For benchmarking per-iteration times of both solvers on identical inputs.
  python tools/synth_sdp.py --nmax 8 --precision 832 --out runs/synth_n8
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mpmath as mp
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--nmax", type=float, required=True)
ap.add_argument("--precision", type=int, default=832)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from sdpvibe import sdp_io
from sdpvibe.solver import _rand_mpf
from sdpvibe.synthetic import st3d_like

L = (a.precision + 2 + 31) // 32 + 1
blocks, N = st3d_like(a.nmax)
t0 = time.time()


def host(M):
    """MPF (..) on GPU -> (limbs (n, L) int32, exps (n,) int64) host"""
    m = M.m.reshape(M.L, -1).t().contiguous().cpu().numpy(); e = M.e.reshape(-1).cpu().numpy()
    return m, e


mp.mp.prec = 32 * L
g = torch.Generator(device="cuda"); g.manual_seed(a.seed)
b = _rand_mpf((N,), L, g).to_mp()
b_strings = [mp.nstr(v, int(a.precision * 0.30103) + 2) for v in b]
os.makedirs(a.out, exist_ok=True)
total = 0
for j, bl in enumerate(blocks):
    K = bl["K"]; npair = bl["dim"] * (bl["dim"] + 1) // 2
    gj = torch.Generator(device="cuda"); gj.manual_seed(a.seed * 100003 + j)
    blk = {"dim": bl["dim"], "K": K}
    for key, h in (("bilinear_bases_even", bl["he"]), ("bilinear_bases_odd", bl["ho"])):
        blk[key] = (h, host(_rand_mpf((h, K), L, gj)) if h else None)
    blk["c"] = host(_rand_mpf((K * npair,), L, gj))
    blk["B"] = host(_rand_mpf((K * npair, N), L, gj))
    if j == 0:
        sdp_io.write_sdp(a.out, [], b_strings, a.precision, L)      # control/objectives first (rewritten below)
    with open(os.path.join(a.out, f"block_info_{j}.json"), "w") as f:
        import json; json.dump({"dim": bl["dim"], "num_points": K}, f)
    sdp_io.write_block_bin(os.path.join(a.out, f"block_data_{j}.bin"), blk, a.precision, L, N)
    total += os.path.getsize(os.path.join(a.out, f"block_data_{j}.bin"))
import json
json.dump({"num_blocks": len(blocks), "command": f"tools/synth_sdp.py --nmax {a.nmax}"}, open(os.path.join(a.out, "control.json"), "w"))
print(f"synthetic nmax~{a.nmax}: N={N}, {len(blocks)} blocks, {total / 2**30:.1f} GB, {time.time() - t0:.0f} s -> {a.out}")
