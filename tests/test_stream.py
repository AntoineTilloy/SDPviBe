"""Streaming / chunking consistency: run a few iterations of one SDP in several memory layouts.
  whole      one chunk per dim, everything resident (must equal the pre-streaming solver bit for bit)
  chunked    chunks of <= CHUNK GB working set, resident
  streamed   same chunks, every large array in pinned host memory (must equal 'chunked' bit for bit)
usage: python tests/test_stream.py <sdp dir> [precision] [iterations] [chunk GB] [--old]"""
import sys, os, time, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import mpmath as mp
from sdpvibe.solver import SDP, Solver

path = sys.argv[1]
prec = int(sys.argv[2]) if len(sys.argv) > 2 else 768
its = int(sys.argv[3]) if len(sys.argv) > 3 else 5
chunk = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
L = (prec + 2 + 31) // 32 + 1
KEYS = ["mu", "P-obj", "D-obj", "P-err", "D-err", "P-step", "D-step", "beta"]      # p-err, R-err: round-off level


def run(mode):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if mode == "old":
        old = importlib.import_module("tests._solver_v1_resident")
        sdp = old.SDP(path, L); S = old.Solver(sdp, L)
    else:
        kw = dict(whole=dict(stream="none"), chunked=dict(chunk=chunk * 2 ** 30, budget=1e15),
                  streamed=dict(chunk=chunk * 2 ** 30, stream="all"))[mode]
        sdp = SDP(path, L, **kw); S = Solver(sdp, L)
    log = []
    t0 = time.time()
    S.run(max_iter=its, log=log, verbose=False)
    torch.cuda.synchronize()
    st = getattr(sdp, "store", None)
    extra = f", {len(sdp.groups)} groups" + (f", H2D {st.bytes_h2d / 2**30:.2f} GB D2H {st.bytes_d2h / 2**30:.2f} GB" if st else "")
    print(f"{mode:9s} {time.time() - t0:7.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB{extra}", flush=True)
    return log


def agree(a, b):
    worst = 99.0
    for ra, rb in zip(a, b):
        for k in KEYS:
            x, y = mp.mpf(ra[k]), mp.mpf(rb[k])
            if x == y: continue
            d = abs(x - y) / max(abs(x), abs(y))
            worst = min(worst, -float(mp.log10(d)))
    return worst


modes = (["old"] if "--old" in sys.argv else []) + ["whole", "chunked", "streamed"]
logs = {m: run(m) for m in modes}
ref = logs["whole"]
for m in modes:
    if m != "whole":
        same = all(ra == rb for ra, rb in zip(
            [{k: r[k] for k in KEYS} for r in logs[m]], [{k: r[k] for k in KEYS} for r in ref]))
        print(f"{m:9s} vs whole: {'bitwise identical' if same else f'agree to {agree(logs[m], ref):.1f} digits (30 printed)'}")
same = [{k: r[k] for k in KEYS} for r in logs["chunked"]] == [{k: r[k] for k in KEYS} for r in logs["streamed"]]
print("streamed vs chunked:", "bitwise identical" if same else f"DIFFER ({agree(logs['chunked'], logs['streamed']):.1f} digits)")
