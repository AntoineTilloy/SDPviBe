"""Reproducible large-SDP timings and optional one-iteration CUDA profile.

Run on a GPU host, e.g. python tools/profile_large.py --nmax 10 --output results/run.json.
The first two iterations are excluded from steady-state timings. --profile records
iteration three; use a separate run without profiling for performance comparisons.
"""
import argparse
import contextlib
import functools
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from sdpvibe.solver import SDP, Solver, Timers
from sdpvibe.synthetic import st3d_like


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nmax", type=float, default=10)
    ap.add_argument("--precision", type=int, default=832)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--sdp", help="Use an actual SDP instead of synthetic stress-tensor shapes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream", choices=["auto", "all", "none"], default="auto")
    ap.add_argument("--chunk", type=float, help="Chunk budget in GiB")
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--profile", action="store_true", help="Record iteration three; the trace can exceed 1 GiB")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.iterations < 3:
        ap.error("At least three iterations are needed to exclude warmup")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.phases:
        os.environ["SDPB_GPU_TIMING"] = "1"
    a.phases = os.environ.get("SDPB_GPU_TIMING") == "1"
    L = (a.precision + 33) // 32 + 1
    kw = dict(stream=a.stream, chunk=a.chunk and a.chunk * 2**30)
    t0 = time.perf_counter()
    if a.sdp:
        sdp = SDP(a.sdp, L, workers=4, **kw)
    else:
        blocks, N = st3d_like(a.nmax)
        sdp = SDP.synthetic(blocks, N, L, seed=a.seed, **kw)
    torch.cuda.synchronize()
    build_seconds = time.perf_counter() - t0
    print(f"Built N={sdp.N}, L={L}: {sdp.layout.describe()}", flush=True)
    solver = Solver(sdp, L, precision=a.precision)
    log, kernels = [], []
    active_range = [None]
    original_timer = Timers.__call__

    def annotated_timer(timer, name):
        if active_range[0] is not None:
            active_range[0].__exit__(None, None, None)
        active_range[0] = torch.profiler.record_function("phase: " + name)
        active_range[0].__enter__()
        return original_timer(timer, name)

    def save_profile(prof):
        prof.export_chrome_trace(str(a.output.with_suffix(".trace.json")))
        for e in prof.key_averages():
            kernels.append(dict(name=e.key, count=e.count,
                                self_cpu_us=e.self_cpu_time_total,
                                self_cuda_us=e.self_device_time_total,
                                total_cuda_us=e.device_time_total))

    prof = None
    if a.profile:
        Timers.__call__ = annotated_timer
        from sdpvibe import batched, rns
        for owner, name in ((rns.RNS, "residues"), (rns.RNS, "crt_block"),
                            (batched, "bgemm_nt"), (batched, "bchol_tile")):
            fn = getattr(owner, name)
            def annotate(fn, name):
                @functools.wraps(fn)
                def wrapped(*args, **kwargs):
                    operand = args[1] if isinstance(args[0], rns.RNS) else args[0]
                    shape = getattr(operand, "shape", "")
                    with torch.profiler.record_function(f"op: {name} {tuple(shape)}"):
                        return fn(*args, **kwargs)
                return wrapped
            setattr(owner, name, annotate(fn, name))
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=save_profile)

    torch.cuda.reset_peak_memory_stats()
    with prof if prof is not None else contextlib.nullcontext():
        def iteration(row):
            # Persist partial progress if a bounded job is interrupted.
            a.output.with_suffix(".iterations.json").write_text(json.dumps(log, indent=2) + "\n")
            if prof is not None:
                if active_range[0] is not None:
                    active_range[0].__exit__(None, None, None)
                    active_range[0] = None
                prof.step()
        solver.run(max_iter=a.iterations, log=log, on_iteration=iteration)
        if active_range[0] is not None:
            active_range[0].__exit__(None, None, None)
    torch.cuda.synchronize()
    Timers.__call__ = original_timer
    stable = [r["iter_time"] for r in log[2:]]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True,
                              cwd=Path(__file__).resolve().parents[1])
    result = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        git_commit=revision.stdout.strip() if revision.returncode == 0 else None,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((Path(__file__).resolve().parents[1] / "sdpvibe").glob("*.py"))},
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        environment={k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
            "SDPB_GPU_TIMING", "SDPB_GPU_MEM_GB", "SDPB_GPU_PANELS", "SDPB_GPU_STREAM",
            "SDPB_GPU_CHUNK_GB", "SDPB_GPU_SOLVE_REFINE", "SDPB_GPU_QBLOCK")},
        N=sdp.N, L=L, chunks=len(sdp.groups), layout=sdp.layout.describe(),
        build_seconds=build_seconds, iterations=log,
        steady_median_seconds=statistics.median(stable) if stable else None,
        peak_gpu_bytes=max(solver.timers.peak.values(), default=torch.cuda.max_memory_allocated()),
        host_pinned_bytes=sdp.store.host_bytes, h2d_bytes=sdp.store.bytes_h2d,
        d2h_bytes=sdp.store.bytes_d2h, phases_seconds=solver.timers.acc,
        profile=sorted(kernels, key=lambda k: -k["self_cuda_us"]))
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"RESULT median={result['steady_median_seconds']} s; {a.output}", flush=True)


if __name__ == "__main__":
    main()
