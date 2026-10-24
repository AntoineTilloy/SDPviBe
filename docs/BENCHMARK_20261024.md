# README benchmark rerun — 2026-10-24

The README's GPU columns were rerun on one, two and four RTX 3090s and one
H100 SXM. The eight-GPU column was removed. Production code is commit
`a7e26146f30f88640566c24c6e0ee5cee54c0937`; the same-host baseline is
`63683ab67d95d2416fb3115346375fc5bc9bc73b`.

This extends the [single-3090 optimization study](PERFORMANCE_3090.md), which
measured a 1.20–1.26× improvement with slightly lower peak allocated GPU memory.
The twofold whole-solver target was not reached. The new measurements use the
public command-line solver and serialized inputs, including normal streaming,
host work and multi-GPU communication.

## Results

| problem | N | SDP size | SDPB CPU reference | 1× 3090 | 2× 3090 | 4× 3090 | 1× H100 SXM |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed Ising nmax=10 (real, 768 bits) | 274 | 0.1 GiB | 10.15 (16 ranks) | 0.82 (12.4×) | 0.58 (17.4×) | 0.60 (16.8×) | 0.55 (18.6×) |
| stress-tensor nmax~6 | 742 | 1.8 GiB | 108 (64 threads, historical) | 5.13 (21.1×) | 3.40 (31.8×) | 2.81 (38.4×) | 2.56 (42.2×) |
| stress-tensor nmax~8 | 1319 | 5.9 GiB | 344 (64 threads, historical) | 12.80 (26.9×) | 6.57 (52.4×) | 4.13 (83.2×) | 4.95 (69.5×) |
| stress-tensor nmax~10 | 2061 | 14.7 GiB | 964 (64 threads, historical) | 24.40 (39.5×) | 13.52 (71.3×) | 8.17 (117.9×) | 11.30 (85.3×) |
| stress-tensor nmax~12 | 2968 | 30.1 GiB | not run | 49.80 | 27.10 | 15.09 | 18.62 |
| stress-tensor nmax~14 | 4040 | 54.3 GiB | not run | — | — | — | 29.75 |

Times are seconds per iteration, averaged over iterations 2 onward. Synthetic
cases run six iterations; they measure throughput, not convergence. The real
mixed-Ising case runs to a feasibility verdict. Startup, parsing, input
generation and first-iteration compilation are excluded from these means;
wall times and every iteration time are retained in the evidence.

The real CPU reference is a new official SDPB 3.1.0 run with 16 MPI ranks on the
H100 host's Xeon Platinum 8480+, using exactly the same input files. The three
synthetic CPU timings are historical SDPB measurements on 64 EPYC threads.
Their original files and hashes are unavailable: they represent the same
problem shapes and generator parameters, but cannot establish a fresh
identical-input CPU/GPU comparison. Those synthetic speedup ratios are
indicative. All current GPU columns do use identical serialized files.

The regenerated real case has 64 positivity blocks and N=274. It is not claimed
to be the older README's exact 131-iteration fixture. Its generation parameters
are nmax=10, lmax=40, order=80, 20 poles, 665-bit conformal blocks,
sigma dimension 0.5181489, epsilon dimension 1.412625 and gaps 3; the solver
uses 768 bits. Synthetic cases use `tools/synth_sdp.py`, seed 0 and 832 bits.

## Same-host optimization and memory checks

| GPUs | Baseline s/it | Candidate s/it | Speedup |
|---|---:|---:|---:|
| 1 × RTX3090 | 29.99 | 24.40 | 1.23× |
| 2 × RTX3090 | 16.43 | 13.52 | 1.21× |
| 4 × RTX3090 | 9.71 | 8.17 | 1.19× |
| 1 × H100 | 13.26 | 11.30 | 1.17× |

The baseline/candidate pairs above use the same nmax=10 files, environment,
GPU count and default memory planner, running sequentially on each host.
They are single six-iteration runs, so small differences should not be
interpreted as precise speedup estimates. The earlier single-GPU study used
three problem sizes and alternating baseline/candidate order.

| GPUs | Baseline sampled peak, MiB per active GPU | Candidate sampled peak, MiB per active GPU | Change in largest peak |
|---|---|---|---:|
| 1 × RTX3090 | 22478 | 22526 | +0.21% |
| 2 × RTX3090 | 20008, 20482 | 19998, 20592 | +0.54% |
| 4 × RTX3090 | 18390, 14704, 15192, 13010 | 18300, 14634, 15182, 12980 | -0.49% |
| 1 × H100 | 44743 | 44799 | +0.13% |

The largest per-device increase in these pairs is **0.54%**.
These memory values are device memory reported by `nvidia-smi` at one-second
intervals, including the CUDA context and allocator reservations. They can miss
brief peaks and are not equivalent to PyTorch's peak allocated memory. The
earlier study directly measured peak allocations at all three large sizes and
found small decreases. No larger memory budget, extra solve panels, reduced
precision or relaxed stopping rules were introduced.

## Numerical validation

The real case converged to the same primal-feasible verdict in **108 iterations**
with official SDPB and all four GPU configurations. The checked trajectory
fields and final y vector are identical across one, two and four RTX 3090s.
Against the single-3090 run, H100 agrees to at least **13.6 significant digits**
throughout the trajectory and 14.4 digits in y; official SDPB agrees to at least
**13.9 significant digits** and 14.9 digits in y.

All 12 synthetic comparisons across GPU counts and architectures agree to at
least 13.6 significant digits, with matching iteration counts and termination
reasons. The four same-host baseline/candidate pairs differ by at most `1e-232`
in absolute value across the checked trajectory fields. All 25 measured runs,
including the H100-only nmax=14 case, have finite trajectory values, consecutive
iteration indices and positive finite iteration times.

The comparison covers mu, primal/dual objectives, gap, primal/dual residuals,
step lengths and beta, plus the final y vector for the real case. It requires
the same iteration counts and termination reasons. Differences below `1e-100`
in absolute value are reported separately from significant relative agreement;
the raw relative metric remains in the evidence. This matters for residuals
near multiprecision roundoff, far below the solver's `1e-30` tolerance.

All 85 exact arithmetic tests passed on H100 as well as in the earlier RTX
3090 study. The earlier study also passed the eight official-SDPB comparison
cases, checkpoint/CLI/streaming checks and BlocksviBe's four test scripts.
No production arithmetic changed during this README rerun.

## Hardware and protocol

| | RTX 3090 host | H100 host |
|---|---|---|
| GPUs | Four 24 GiB RTX 3090s, 350 W each | One H100 80 GB SXM, 700 W |
| Host CPU | AMD EPYC 7282, 16 cores / 32 threads | Intel Xeon Platinum 8480+ |
| GPU connection | PCIe 3 ×16, no NVLink; NODE/PHB paths | SXM; PCIe 5 ×16 host interface |
| RAM | 256 GB advertised; about 251 GiB reported | 256 GB advertised; larger shared host |
| NVIDIA driver | 570.133.20 | 615.71.09 |

Both use Python 3.11.13, PyTorch 2.7.1+cu128, CUDA 12.8, Triton 3.3.1 and CuPy
14.2.0. `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` are set for every GPU run.
Automatic chunking and streaming remain enabled. Timings run one job at a time
on each host, without the profiler or synchronized phase instrumentation.
The fresh CPU reference starts only after the H100 GPU measurements finish.

These Vast.ai rentals cost approximately $1.23/hour for the four-3090 host
and $2.43/hour for the H100 host, including their 160 GB disks. Those are the
observed rental prices for this run, not a general pricing claim. All numerical
work, validation and chart generation ran remotely; the local workstation was
used only for source editing, file transfer and orchestration.

## Input identity and reproduction

A CUDA random seed alone does not reproduce these synthetic inputs across GPU
architectures. PyTorch's normal generator chooses its Philox launch geometry
using device properties. An unmodified RTX 3090 generation differed from the
H100 generation despite the same seed. This was detected before the RTX
timing runs.

The H100 files were used as canonical inputs. Because transferring the large
dataset between rentals was too slow, an isolated input-generation process on
the RTX host reproduced the H100 generator's launch geometry (132 SMs and 2048
threads per SM). Every serialized input file then matched the H100 SHA-256
manifest. That process exited before any solver timing; solver processes used
the normal device properties. This workaround is not part of the solver.
For ordinary reproduction, generate once and copy the resulting files.

On a remote CUDA host with official `scalar_blocks` and SDPB tools installed,
generate and convert the real input once:

```sh
python tools/ising_mixed_pmp.py --nmax 10 --lmax 40 --order 80 --poles 20 \
  --prec 665 --ds 0.5181489 --de 1.412625 --out mixed_n10.json
mpirun -n 16 pmp2sdp --precision 768 -i mixed_n10.json -o mixed_n10
```

Prepare a dedicated canonical synthetic input directory:

```sh
for n in 6 8 10 12; do
  python tools/synth_sdp.py --nmax "$n" --precision 832 --seed 0 \
    --out "canonical/synth_n$n"
done
python tools/benchmark_readme.py --repo . --output results/readme \
  --input-root canonical --gpus 1 2 4 --sizes 6 8 10 12 \
  --real-sdp path/to/mixed_n10 --baseline path/to/baseline
```

Copy the canonical inputs to the H100 host and run the same command with
`--gpus 1`. Generate the nmax=14 case there and add it to `--sizes` to reproduce
the largest H100 row. The harness records input/source hashes, commands,
iteration timings and sampled device memory; it rejects a resumed run with
different inputs or source files. Use a separate output directory for a new
experiment. Each subprocess has a 40-minute limit; rented hosts also had
independent destruction deadlines.

[Plot data](bench_3090.json) and [measurement evidence](benchmark_20261024.json)
include all measured iteration times, source/input hashes, baseline pairs,
sampled memory and numerical comparison metrics. The chart can be rebuilt on
a remote host with:

```sh
python tools/plot_speedup.py docs/bench_3090.json docs/speedup_3090.svg
```
