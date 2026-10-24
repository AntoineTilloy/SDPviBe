"""Bar chart of SDPviBe's speedup over SDPB (per iteration) from a benchmark JSON:
  {"sdpb": "SDPB 3.1, 64 cores (AMD EPYC)", "gpu": "RTX 3090",
   "problems": [{"name": "nmax~6", "N": 742, "sdpb_s": 108.1, "gpu_s": {"1": 6.97, "2": 4.1, ...}}, ...]}
  python tools/plot_speedup.py docs/bench_3090.json docs/speedup_3090.svg
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
P = d["problems"]
series = d.get("series") or [{"key": str(g), "label": f"{g} x {d['gpu']}"} for g in sorted({int(g) for p in P for g in p["gpu_s"]})]
fig, ax = plt.subplots(figsize=(9, 4.8 if any(p.get("cpu_label") for p in P) else 4.4), dpi=100)
w = 0.8 / len(series)
colors = ["#a6d96a", "#66bd63", "#1a9850", "#3b6fb6", "#5e3c99"]
top = 0
for k, sr in enumerate(series):
    xs, ys = [], []
    for i, p in enumerate(P):
        if sr["key"] in p["gpu_s"]:
            xs.append(i + (k - (len(series) - 1) / 2) * w); ys.append(p["sdpb_s"] / p["gpu_s"][sr["key"]])
    bars = ax.bar(xs, ys, w * 0.92, color=colors[k % len(colors)], label=sr["label"])
    for x, y in zip(xs, ys):
        ax.text(x, y, f"{y:.0f}x", ha="center", va="bottom", fontsize=8)
        top = max(top, y)
ax.axhline(1, color="#888", lw=0.8)
ax.set_xticks(range(len(P)))
ax.set_xticklabels([f"{p['name']}\nN={p['N']}\nSDPB {p['sdpb_s']:.1f} s/it"
                    + (f"\n{p['cpu_label']}" if p.get("cpu_label") else "") for p in P], fontsize=8)
ax.set_ylabel(f"speedup per iteration vs {d['sdpb']}", fontsize=9)
ax.set_ylim(0, top * 1.15)
ax.set_title(d.get("title", "SDPviBe vs SDPB"), fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(sys.argv[2])
print("wrote", sys.argv[2])
