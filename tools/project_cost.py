"""Per-iteration time model of the sdpvibe solver, from per-phase operation counts.

  phase                 dominant count                       (P = K dim(dim+1)/2 per block, N = #functionals)
  Q += P^T P            sum P * N^2
  P = L^-1 B            sum P^2 * N
  chol(S)               sum P^3
  assemble S            sum P^2
  residues, directions  sum P * N   (+ a fixed cost per chunk: kernel latency, small-batch Cholesky/solves)
  chol(Q)               N^3
Rates are calibrated on one timed run (SDPB_GPU_TIMING=1 phase breakdown) of a known shape, then applied to
another shape.  Used by docs/SOTA_PLAN.md.
"""
import json, re, sys


def counts(blocks, N):
    P = [b["K"] * b["dim"] * (b["dim"] + 1) // 2 for b in blocks]
    return dict(gram=sum(P) * N * N, trsm=sum(p * p for p in P) * N, chol=sum(p ** 3 for p in P),
                asm=sum(p * p for p in P), pn=sum(P) * N, cholQ=N ** 3)


PHASES = {   # phase-name regex -> count it scales with
    r"Q \+= P\^T P": "gram", r"P = L\^-1 B": "trsm", r"chol\(S\)": "chol", r"assemble S": "asm",
    r"^residues": "pn", r"direction": "pn", r"chol\(Q\)": "cholQ"}


def parse_phases(line):
    """'phase timings (iterations 2+): name 1.2s (3%), ...' -> {name: seconds}"""
    out = {}
    for m in re.finditer(r"([^,:]+?) ([0-9.]+)s \(\d+%\)", line.split(":", 1)[1]):
        out[m.group(1).strip()] = float(m.group(2))
    return out


def calibrate(phases, its, c, nchunks, chunk_frac=0.0):
    """rates[count] = seconds per unit count (per iteration); chunk_frac of the pn phases is per-chunk cost"""
    agg = {}
    for name, t in phases.items():
        for pat, key in PHASES.items():
            if re.search(pat, name):
                agg[key] = agg.get(key, 0) + t / its
    rates = {k: v / c[k] for k, v in agg.items()}
    if chunk_frac:
        rates["chunk"] = chunk_frac * agg["pn"] / nchunks
        rates["pn"] *= (1 - chunk_frac)
    return rates


def predict(rates, c, nchunks=0, scale=1.0):
    t = {k: rates[k] * c[k] * scale for k in rates if k in c}
    if "chunk" in rates: t["chunk"] = rates["chunk"] * nchunks
    return t
