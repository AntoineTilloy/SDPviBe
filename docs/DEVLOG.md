# sdpb_gpu — multiprecision linear algebra for SDPB on GPUs (prototype)

Goal: test whether the expensive multiprecision steps of an SDPB interior-point iteration
(Q = PᵀP, Cholesky of the Schur blocks S_j and of Q, P = L⁻¹B, ...) can run orders of magnitude
faster on (consumer) GPUs, at SDPB precisions (768–1024+ bits).

Everything here was measured on a laptop: RTX A3000 Laptop GPU (Ampere, 32 SMs, ~3.3 GB usable,
weak FP64) vs. the same laptop's i7-11850H (8 cores / 16 threads).

## Approach

* **Exact multiprecision GEMM on INT8 tensor cores** (`mpgpu/rns.py`): residue number system with
  ~160 primes of 15 bits; residues split into int8 halves; per prime one fused Triton kernel does
  the 4 int8 products + modular accumulation; CRT reconstruction is itself two int8 GEMMs plus a
  carry kernel; correct rounding (half-up) to the target exponent. Bit-exact against Python ints.
* **Number format** (`mpgpu/mpmat.py`): 32-bit limbs in *limb-plane* layout `(L, rows, cols)`
  (coalesced kernels), one exponent per matrix or per column ("block floating point").
  Cholesky works on power-of-two-equilibrated matrices, so |L_ij| ≤ 1 and fixed point is as
  accurate as floating point.
* **Base cases on CUDA cores** (`mpgpu/basecase.py`, CUDA via CuPy/NVRTC): batched 32×32
  multiprecision Cholesky (warp-cooperative Newton rsqrt for the pivots) and triangular inverse.
* **Recursive algorithms** (`mpgpu/linalg.py`): Cholesky with recursive TRSM; leaves applied by
  tensor-core GEMM with explicit 32×32 tile inverses (+1 refinement step for the Cholesky panels,
  which restores floating-point accuracy). P = L⁻¹B by recursive forward substitution.
* **Batched versions** (`mpgpu/batched.py`): all equal-sized blocks (SDPB's S_j) in one launch.

## Results (1024-bit unless noted; GPU warm timings; CPU = GMP mpf naive loops, 16 threads,
mimicking SDPB's El::BigFloat path — `cpu_baseline/mpf_baseline.c`)

| operation | size | GPU | CPU (16 thr) | speedup |
|---|---|---|---|---|
| Q = PᵀP exact (bench `rns_syrk5.py`) | N=2048, K=65536 | 4.1 s | 87 s (SDPB-3 BLAS+CRT path, BLAS only) | 21× |
| GEMM | 1024³ | 0.07 s | 38 s (naive mpf) | ~540× |
| Cholesky | n=1024 | 0.40 s | 7.3 s | 18× |
| TRSM P = L⁻¹B | n=1024, 1024 rhs | 0.22 s | 16.1 s | 73× |
| Cholesky | n=2048 | 1.18 s | 54 s | 46× |
| TRSM | n=2048, 1024 rhs | 0.53 s | ~77 s | ~145× |
| batched Cholesky + TRSM | 32 blocks × 256, 256 rhs | 0.60 s | ~9.7 s (est., best CPU throughput) | ~16× |
| 832-bit: Cholesky / TRSM | n=1024 | 0.35 / 0.17 s | 5.3 / 11.8 s | 15× / 69× |

Accuracy (n=256, eigenvalue spread 2^400, vs mpmath at 2296 bits): forward error of L 2^-847
(same as 1048-bit floating-point Cholesky: 2^-846), backward error ‖LLᵀ−A‖ 2^-1047.

## Cloud results (vast.ai, 2026-10-09; raw logs in `results/`)

All exactness/accuracy tests pass on both cards (bit-exact GEMM/Q, Cholesky forward error 2^-847).
1024-bit unless noted. Price = what we paid per hour.

| operation | size | laptop A3000 | RTX 3090 ($0.19/h) | RTX 5090 ($0.56/h) |
|---|---|---|---|---|
| Q = PᵀP exact | N=2048, K=65536 | 4.09 s | 1.15 s | 0.44 s |
| Q = PᵀP exact | N=4096, K=65536 | – | 3.96 s | 1.44 s |
| Q = PᵀP exact, 832 bits | N=4096, K=65536 | – | 3.42 s | 1.21 s |
| int8 rate in fused syrk | | 63 TOPS | 220 TOPS (78% of peak) | 697 TOPS (83%) |
| GEMM | 4096² × 2048 | – | 0.58 s | 0.25 s |
| Cholesky | n=2048 | 1.18 s | 0.76 s | 0.56 s |
| Cholesky | n=4096 | – | 2.06 s | 1.35 s |
| TRSM | n=4096, 2048 rhs | – | 0.76 s | 0.32 s |
| batched Cholesky + TRSM | 32 × 256 | 0.60 s | 0.20 s | 0.10 s |
| batched Cholesky + TRSM | 128 × 256 | – | 0.75 s | 0.34 s |
| batched Cholesky + TRSM | 32 × 512 | – | 0.82 s | 0.40 s |

GEMM-bound steps scale with INT8 throughput (5090 ≈ 2.5–2.75× a 3090 at ~3× the price);
the single large Cholesky is latency-bound (5090 only 1.5× a 3090) → next optimisation target.
n=8192 currently OOMs in the test-matrix construction (bench script), not in the solver.


## End-to-end GPU solver (2026-10-09)

`sdpb_gpu.py` is a line-by-line port of SDPB 3.1's interior point method (`mpgpu/solver.py`) on a
per-entry multiprecision floating type (`mpgpu/mpf.py`): elementwise CUDA kernels, exact RNS/INT8
GEMMs rounded once per entry, equilibrated batched Cholesky/TRSM; blocks batched in groups with
inert padding.  Input: `pmp2sdp --outputFormat json`.  Official SDPB 3.1 and scalar_blocks run locally
without docker (`tools/pull_image.py`, `tools/run_in_image.sh`, `source tools/sdpb_env.sh`) and
natively with MPI on vast instances (`vast/remote_sdpb_setup.sh`).

Validation (iterations.json compared with `tools/compare_iterations.py`): every iteration agrees with
SDPB to ~1e-13..1e-15 relative (limited by float64 step-length eigenvalues) on
* SDPB's 1-block example (160 iterations), single-correlator Ising nmax=6 (28 / 100 iterations),
* mixed sigma/eps Ising (KPSV 1406.4858) nmax=6 (61 iterations) and nmax=10 (131 iterations).
Physics: single-correlator nmax=10 bound at Ds=0.5181 is De <= 1.4122(3) (paper: ~1.412); the
nmax=6 mixed island of KPSV fig. 4 is reproduced (inside point allowed, both sides excluded) after
fixing an odd-spin (-1)^l convention between scalar_blocks and KPSV (see tools/ising_mixed_pmp.py).

| problem | official SDPB | GPU solver | per-iteration speedup |
|---|---|---|---|
| mixed Ising nmax=6 (N=104, 49 blocks), 61 it. | 48 min, 1 core (laptop) | 108 s, laptop RTX A3000 | 27x vs 1 core |
| mixed Ising nmax=10 (N=274, 64 blocks), 131 it. | 21.2 min, 56 cores EPYC 7663 | 4.2 min, RTX 5090 | 6.2x vs 56 cores |
| mixed Ising nmax=14 (N=524, 79 blocks, 958 bits) | 17.96 s/it, 64 cores EPYC 7B13 | 2.04 s/it, RTX 5090 | 8.8x vs 64 cores |
| mixed Ising nmax=18 (N=854, 94 blocks, 1086 bits) | 36.68 s/it, 64 cores EPYC 7B13 | 2.03 s/it, RTX 5090 | 18x vs 64 cores |

nmax=18 at the best-known Ising point (0.5181489, 1.412625): GPU finds it allowed (primal feasible) in
167 iterations, 6 min 41 s end to end (SDP read 12.7 s with the parallel reader); SDPB's first 20
iterations agree to 1e-14.3; a full SDPB run would take ~100 min on 64 cores.  Profile: 76% GPU busy,
~47k kernels/iteration.  Blocks for this single point took 6 h on 64 cores (one-time per Ds-De value).

nmax=14: the first 20 iterations agree with SDPB to ~1e-14; the GPU run reaches "dual feasible"
(excluded) at (0.5182, 1.4127) after 149 iterations (22 min, of which 16 min was reading the JSON SDP
-- the Python reader is now the bottleneck).  nmax=6 island scan (results/scan_n6.csv, rows of
constant Ds-De, step 0.001 in Ds): allowed points (0.518, 1.413) and (0.520, 1.425), all other 30
excluded -- consistent with the thin island of KPSV fig. 4.

At these sizes the GPU is only ~50% busy (~27k kernel launches / iteration): launch overhead and
the fixed per-entry CRT cost dominate; larger Lambda shifts the balance towards the GPU.


## Stress-tensor system (Chang et al. 2024, gitlab.com/davidsd/stress-tensors-3d)

Their open-source pipeline (hyperion/Haskell + blocks_3d) builds and runs on a single vast box:
`vast/st3d_setup.sh` (stack build; musl loader must be absent during the build), `vast/fake_slurm.sh`
(hyperion needs Slurm; vast containers have a read-only cgroupfs, so sbatch/scontrol stand-ins run
jobs locally, ssh-to-self via authorized_keys2, hostname must resolve to one address), wrappers in
`hyperion-config/vast` (unset SLURM_* before mpirun).  `vast/st3d_compare.sh` converts the PMP it
builds and runs official SDPB and the GPU solver with identical parameters.

TSigEps_test_nmax6 at (0.5181489, 1.412625): SDP with N=742, 113 blocks, positivity matrices up to
5x5 (13 blocks), 1024 bits.  Block/PMP build: ~90 min on 64 cores.
| | official SDPB, 64 cores EPYC 7B13 | GPU solver, RTX 5090 |
|---|---|---|
| verdict / iterations | primal feasible (allowed), 115 | primal feasible, 115 |
| wall time | 129.5 min | 8.4 min (15.5x) |
| per iteration | 65.5 s | 3.8 s (17x) |
Iterates agree to 1e-13 throughout.  GPU memory: 20 GB already at nmax=6 (5x5 blocks -> Schur
blocks ~855x855); higher Lambda needs host-RAM streaming or bigger GPUs.

## Host-RAM streaming (2026-10-15)

Large Lambda does not fit on one GPU: at the stress-tensor system's shapes, B alone is ~60 GB at nmax~14
and ~370 GB at nmax~22.  `mpgpu/store.py` + `Layout` and the chunk-major `Solver.run` in `mpgpu/solver.py`:

* blocks of equal dim are batched in **chunks**; every per-chunk array (B, Cholesky of S, bilinear bases,
  x/X/Y, Cholesky of X/Y, residues, direction temporaries, pairings) is a store item that is either
  GPU-resident or kept in pinned host memory and **prefetched on side CUDA streams** (next chunk, and the
  first chunk of the next pass).  Only y, b, Q/cholQ and scalars are global.
* chunked mode is **lean**: Pm = L_S^-1 B is not stored; the search direction uses B and the Cholesky
  factor (P^T w = B^T L^-T w, P dy = L^-1 B dy) -> half the memory and traffic, same cost when resident.
* **Q in RNS form** (`mpf.GramAcc`): residues of every chunk's P^T P are accumulated exactly and the CRT
  runs once per iteration (a GEMM's CRT costs ~N^2 whatever the inner size; one CRT per chunk dominated
  with many chunks).  Columns aligned to running reference exponents, 28 bits of headroom in an extra
  limb, flush on larger scale jumps (none seen on real SDPs).  Never less precise than one global GEMM.
* L^-1 B in column panels (bitwise identical, bounded temporaries); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`.
* planner: streaming is cheap but small batches are not (Cholesky/trsm tile latency, per-chunk work), so
  it picks the **largest chunks that fit** and fills the remaining memory with resident arrays.
  `SDPB_GPU_MEM_GB` caps the GPU (emulates a smaller card, enforced), `SDPB_GPU_STREAM=all|none`.

Correctness: unchunked = previous solver bit for bit; streamed = chunked-resident bit for bit
(`tests/test_stream.py`); real SDP (mixed Ising nmax=10), resident vs every array streamed: same 131
iterations and verdict, iterates agree to 1e-13.8 (the GPU-vs-SDPB level).  `tests/test_gramacc.py`.

Speed (`tests/bench_stream.py`: synthetic SDP with the stress-tensor block structure scaled from its
nmax=6 instance; RTX PRO 6000 96 GB, PCIe 5 at 53 GB/s; "32 GB" = hard cap emulating an RTX 5090):

| nmax~ | N | host arrays | 96 GB card | 32 GB cap | first streaming version, 96 / 32 GB |
|---|---|---|---|---|---|
| 8  | 1319 | 18 GB  | 7.56 s/it (all resident, original code) | 7.93 (+5%) | 7.63 / 8.61 |
| 10 | 2061 | 36 GB  | 17.57 | 15.63 | 16.45 / 30.29 |
| 12 | 2968 | 62 GB  | 30.39 | 30.54 | 33.54 / 39.77 |
| 14 | 4040 | 102 GB | 52.04 | 61.86 (+19%) | 60.39 / 90.36 |
| 16 | 5276 | 172 GB | 86.94 | 120.67 (+39%) | - |

Before streaming, nmax~10 and up did not fit even the 96 GB card.  Host<->GPU traffic is 70 GB/it
(nmax~8) to ~1 TB/it (nmax~16), mostly hidden behind compute: with the same batches, streaming
everything cost +1.6% (nmax~8).  What remains on 32 GB at nmax>=14 is the cost of small batches
(single 5x5 blocks per chunk): Q accumulation GEMMs with few rows, latency-bound Cholesky/trsm of S
and of X/Y.  Limits: chol(Q) needs ~9x Q (58 GB at nmax~18) -> 96 GB card above nmax~16; host RAM
~1.2x the host arrays above; PCIe 4 hosts (most 5090s) have half the bandwidth.

## Where the time goes / next steps

* Large GEMMs run at ~15 G multiprecision-MAC/s (~70 TOPS int8). Small-inner-dimension GEMMs are
  dominated by the fixed per-output CRT cost (~45 ns/entry): use larger explicit diagonal-block
  inverses + one refinement step (refinement squares the error) to get fewer, bigger GEMMs.
* Cholesky tiles (32×32) are latency-bound on CUDA cores (~7 ms/tile, sequential along the
  diagonal); batching over blocks hides this for the S_j, not for the single Q.
* Not yet done: step length (min eigenvalue; plan: Cholesky-based test or multiprecision Lanczos),
  Schur-complement assembly (elementwise), integration into SDPB (BigFloat ↔ limbs).
* Memory: the laptop GPU limits sizes to n≈2048; an RTX 3090 (24 GB, ~3× INT8 throughput and
  bandwidth) should handle N≈5000 and run ~3× faster; independent moduli split across GPUs.

## Layout

```
mpgpu/        rns.py (residues, fused INT8 GEMM, CRT)   limbops.py (Triton limb kernels)
              mpmat.py (MP matrices, gemm_nt, add, rescale)   basecase.py (CUDA tile kernels)
              linalg.py (chol, trsm_left/right)   batched.py (batched versions)
tests/        test_gemm.py test_basecase.py test_linalg.py (exactness / accuracy)
              bench_linalg.py bench_batched.py prof_*.py (benchmarks / profiles)
cpu_baseline/ mpf_baseline.c (+ results *.txt)
rns_syrk*.py  earlier stand-alone Q = PᵀP benchmarks (runs_K65536.txt)
vast/         gpu_value.py (rank vast.ai offers by perf/$), remote_setup.sh, bench_suite.sh, sync.sh
```

## Cloud runs (vast.ai)

CLI: `~/.local/bin/vastai` (pipx). Workflow: `vastai create instance <offer> --image vastai/base-image
--disk 30 --ssh` → `bash vast/sync.sh push <id>` → on the instance `bash sdpb_gpu/vast/remote_setup.sh`
then `cd sdpb_gpu && bash vast/bench_suite.sh quick|full` → `bash vast/sync.sh pull <id>` →
`vastai destroy instance <id>`.

Run: `.venv/bin/python tests/test_linalg.py --n 256 --ref 1` (accuracy),
`.venv/bin/python tests/bench_linalg.py --n 1024 --nrhs 1024` (speed).
