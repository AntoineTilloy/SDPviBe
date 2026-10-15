"""Rank vast.ai GPU offers by price-performance for our INT8-tensor-core multiprecision kernels.

Usage: python3 vast/gpu_value.py [--refresh]
(--refresh re-downloads offers with the vastai CLI; searching needs no API key)

Throughput model: ~70% of the time scales with dense INT8 tensor TOPS (RNS GEMMs), ~30% with
memory bandwidth (residues, CRT, limb kernels) -- as measured on the RTX A3000 laptop GPU.
'eff' = relative speed vs an RTX 3090 under that model.  Volta (V100/Titan V) has no INT8
tensor cores: its figure is the FP16-tensor-core route (int8-valued fp16, fp32 accumulate),
which would need a port of the GEMM kernel.  FP64 (Ozaki-style DGEMM) is worse on every card
listed: ~110 DGEMMs vs 640 INT8 GEMMs at 1024 bits.
"""
import glob, json, os, statistics as st, subprocess, sys

# dense INT8 tensor TOPS (V100/Titan V: FP16 tensor TFLOPS), FP64 TFLOPS (incl. FP64 tensor cores)
SPEC = {
    "Tesla_V100": (125, 7.8), "Titan_V": (110, 6.9), "Tesla_P100": (0, 4.7),
    "RTX_2080_Ti": (215, 0.4), "RTX_3090": (284, 0.56), "RTX_3090_Ti": (320, 0.63),
    "RTX_A5000": (222, 0.43), "RTX_A6000": (310, 0.6), "A10": (250, 0.98), "A40": (299, 0.58),
    "RTX_4080": (390, 0.76), "RTX_4090": (660, 1.3), "L40": (362, 1.4), "L40S": (733, 1.4),
    "RTX_6000Ada": (728, 1.4), "RTX_5080": (450, 0.88), "RTX_5090": (838, 1.6),
    "RTX_PRO_6000_WS": (1008, 2.0), "A100_PCIE": (624, 19.5), "A100_SXM4": (624, 19.5),
    "H100_PCIE": (1513, 51), "H100_NVL": (1671, 60), "H100_SXM": (1979, 67), "H200": (1979, 67),
    "B200": (4500, 40),
}
REF = "RTX_3090"
HERE = os.path.dirname(os.path.abspath(__file__))
VAST = os.path.expanduser("~/.local/bin/vastai")

if "--refresh" in sys.argv:
    for g in SPEC:
        out = subprocess.run([VAST, "search", "offers", f"rentable=true gpu_name={g} num_gpus>=1",
                              "--raw", "--limit", "400"], capture_output=True, text=True).stdout
        open(os.path.join(HERE, f"o_{g}.json"), "w").write(out or "[]")

rows = {}
for f in glob.glob(os.path.join(HERE, "o_*.json")):
    g = os.path.basename(f)[2:-5]
    d = json.load(open(f))
    if not d or g not in SPEC: continue
    p = sorted(o["dph_total"] / o["num_gpus"] for o in d)
    rows[g] = dict(n=len(d), pmin=p[0], pmed=st.median(p), bw=st.median(o["gpu_mem_bw"] for o in d),
                   vram=sorted({round(o["gpu_ram"] / 1024) for o in d}),
                   ram=st.median(o["cpu_ram"] / 1024 / o["num_gpus"] for o in d))

ref = rows[REF]
def eff(g):
    tops, _ = SPEC[g]
    if tops == 0: return 0.0
    return 1 / (0.7 * SPEC[REF][0] / tops + 0.3 * ref["bw"] / rows[g]["bw"])

print(f"{'GPU':16s} {'offers':>6s} {'$/h min':>8s} {'$/h med':>8s} {'VRAM GB':>10s} {'host GB/GPU':>11s}"
      f" {'INT8':>6s} {'FP64':>5s} {'speed':>6s} {'perf/$ (med)':>12s}")
ref_val = eff(REF) / ref["pmed"]
for g, r in sorted(rows.items(), key=lambda kv: -eff(kv[0]) / kv[1]["pmed"]):
    e = eff(g)
    print(f"{g:16s} {r['n']:6d} {r['pmin']:8.3f} {r['pmed']:8.3f} {str(r['vram']):>10s} {r['ram']:11.0f}"
          f" {SPEC[g][0]:6d} {SPEC[g][1]:5.1f} {e:6.2f} {e / r['pmed'] / ref_val:12.2f}")
