"""Summarise a tests/dropin_suite.py work directory as a markdown table.
  python tools/dropin_report.py runs/dropin
Per-iteration agreement uses mu, P-obj, D-obj, P-step, D-step, beta (the error columns reach the round-off
level near convergence, where relative agreement is meaningless); 'median' is over iterations."""
import json, os, sys
import mpmath as mp

mp.mp.prec = 1200
KEYS = ["mu", "P-obj", "D-obj", "P-step", "D-step", "beta"]


def dig(a, b):
    a, b = mp.mpf(a), mp.mpf(b)
    if a == b: return 99.0
    s = max(abs(a), abs(b))
    return float(-mp.log10(abs(a - b) / s))


def main(work):
    rep = json.load(open(os.path.join(work, "report.json")))
    print("| case | N | verdict (SDPB = SDPviBe?) | iterations SDPB / SDPviBe | agreement per iteration: worst / median digits | "
          "y digits | spectrum | s/it SDPB (CPU) | s/it SDPviBe (GPU) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, r in rep["cases"].items():
        W = os.path.join(work, name)
        try:
            a = json.load(open(os.path.join(W, "out_sdpb", "iterations.json")))
            b = json.load(open(os.path.join(W, "out_sdpvibe", "iterations.json")))
        except OSError:
            print(f"| {name} | | {r.get('error', 'failed')} |"); continue
        per = [min(dig(x[k], y[k]) for k in KEYS) for x, y in zip(a, b)]
        per_s = sorted(per)
        N = None
        try: N = int(open(os.path.join(W, "out_sdpb", "y.txt")).readline().split()[0])
        except (OSError, ValueError, IndexError): pass
        sp = r.get("spectrum", {})
        sps = f"{sp.get('zeros')} zeros, {sp.get('zeros_digits')} digits" if "zeros" in sp else "-"
        print(f"| {name} | {N} | {r['reason_sdpb']} ({'yes' if r['same_reason'] else 'NO'}) | {r['iterations_sdpb']} / "
              f"{r['iterations_sdpvibe']} | {per_s[0]:.1f} / {per_s[len(per_s) // 2]:.1f} | {r.get('y_digits') and round(r['y_digits'], 1)} | "
              f"{sps} | {r['iter_time_sdpb']} | {r['iter_time_sdpvibe']} |")
    if rep.get("features"):
        print("\nFeatures:")
        for k, v in rep["features"].items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1])
