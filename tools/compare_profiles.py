"""Check numerical agreement and summarize a pair of unprofiled large-SDP runs.

Require 100 relative digits, allowing absolute differences below 1e-100 at zero.
Report exact identity separately: correcting integer bit lengths can change the
last limbs without changing the solver trajectory at meaningful precision.
"""
import argparse
import json
from pathlib import Path
import mpmath as mp


def compare(before, after):
    for run in (before, after):
        assert not run["arguments"]["profile"] and not run["arguments"]["phases"], "Use uninstrumented timings"
    for key in ("nmax", "precision", "iterations", "sdp", "seed", "stream", "chunk"):
        assert before["arguments"][key] == after["arguments"][key], f"Different input option: {key}"
    for key in ("gpu", "torch", "cuda", "N", "L", "chunks", "host_pinned_bytes"):
        assert before[key] == after[key], f"Different environment/layout: {key}"
    assert before.get("environment") == after.get("environment"), "Different runtime options"
    a, b = before["iterations"], after["iterations"]
    assert len(a) == len(b) >= 3, "Different iteration counts or insufficient warmup"
    identical, worst, noise = True, mp.inf, mp.mpf(0)
    with mp.workprec(4 * before["L"] * 32):
        for i, (x, y) in enumerate(zip(a, b), 1):
            assert x.keys() == y.keys(), f"Different fields at iteration {i}"
            for key in x.keys() - {"total_time", "iter_time"}:
                if x[key] == y[key]: continue
                identical = False
                assert key not in {"iteration", "block_name"}, f"Different {key} at iteration {i}"
                u, v = mp.mpf(x[key]), mp.mpf(y[key])
                delta = abs(u - v)
                if not delta: continue
                if delta < mp.mpf("1e-100"):
                    noise = max(noise, delta)
                else:
                    digits = -mp.log10(delta / max(abs(u), abs(v)))
                    worst = min(worst, digits)
                    assert digits >= 100, f"Only {float(digits):.1f} digits at iteration {i}: {key}"
    old, new = before["steady_median_seconds"], after["steady_median_seconds"]
    return dict(N=before["N"], iterations=len(a), identical=identical,
                worst_significant_digits=float(worst) if mp.isfinite(worst) else None,
                max_noise_abs=mp.nstr(noise, 12),
                before_seconds=old, after_seconds=new, speedup=old / new,
                before_peak_gpu_bytes=before["peak_gpu_bytes"], after_peak_gpu_bytes=after["peak_gpu_bytes"],
                before_source_sha256=before.get("source_sha256"), after_source_sha256=after.get("source_sha256"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    a = p.parse_args()
    print(json.dumps(compare(json.loads(a.before.read_text()), json.loads(a.after.read_text())), indent=2))
