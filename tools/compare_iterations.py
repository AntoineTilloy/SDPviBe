"""Compare SDPB's iterations.json with SDPviBe's: per-iteration relative differences.
usage: python tools/compare_iterations.py runs/x/out/iterations.json runs/x/sdp_gpu_out/iterations.json"""
import json, sys
import mpmath as mp
mp.mp.dps = 40
def load(f):
    t = open(f).read().strip()
    try: return json.loads(t)
    except json.JSONDecodeError:                          # run still in progress: close the list
        return json.loads(t[:t.rfind("}") + 1] + "]")
a = load(sys.argv[1]); b = load(sys.argv[2])
keys = ["mu", "P-obj", "D-obj", "P-err", "D-err", "P-step", "D-step", "beta"]
print(f"{'it':>3} " + " ".join(f"{k:>9}" for k in keys) + "   (log10 relative difference)")
worst = {k: -99 for k in keys}
for ra, rb in zip(a, b):
    out = []
    for k in keys:
        x, y = mp.mpf(ra[k]), mp.mpf(rb[k])
        d = abs(x - y) / max(abs(x), abs(y), mp.mpf(10) ** -300)
        v = float(mp.log10(d)) if d > 0 else -99
        worst[k] = max(worst[k], v)
        out.append(f"{v:9.1f}")
    print(f"{ra['iteration']:3d} " + " ".join(out))
print("worst " + " ".join(f"{k}={worst[k]:.1f}" for k in keys))
print(f"SDPB iterations: {len(a)}, GPU iterations: {len(b)}")
