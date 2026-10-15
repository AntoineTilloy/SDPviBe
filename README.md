# SDPviBe

A GPU port of the interior-point solver of [SDPB](https://github.com/davidsd/sdpb) (the semidefinite
program solver of the conformal bootstrap), in multiprecision arithmetic on consumer and datacenter GPUs.

It reads the SDPs produced by SDPB's `pmp2sdp` and follows SDPB 3.1's algorithm step by step (same
residues, Schur complement, Mehrotra predictor-corrector, step lengths and termination criteria), so its
iterations can be compared one to one with SDPB's `iterations.json`.

**Status: research prototype.** Validated against official SDPB on 3d Ising problems (below); used so far
for single-correlator and mixed (sigma, epsilon) Ising bounds and the stress-tensor system of Chang et al.

## How it works

* **Numbers**: per-entry floating point, mantissas of L 32-bit limbs (two's complement) plus a 64-bit
  exponent, like SDPB's BigFloat (`sdpvibe/mpf.py`).
* **Matrix products are exact**: operands are split into residues modulo ~150 primes below 2^15, multiplied
  on **INT8 tensor cores**, and reconstructed by the Chinese remainder theorem (also on tensor cores), so
  every output entry is rounded once (`rns.py`, `limbops.py`).
* **Cholesky and triangular solves**: recursive, equilibrated fixed point, 32x32 base tiles on CUDA cores,
  batched over the SDP's blocks (`linalg.py`, `batched.py`, `basecase.py`).
* **Solver** (`solver.py`): blocks batched by matrix dimension with inert padding; the step-length
  eigenvalue uses a float64 copy.
* **Large problems** (`store.py`, `Layout` in `solver.py`): blocks are processed in chunks, and every
  per-chunk array is either GPU-resident or kept in pinned host memory and prefetched on side CUDA
  streams. The Gram matrix Q is accumulated in residue form across chunks with a single reconstruction per
  iteration (`mpf.GramAcc`). A planner picks the largest chunks that fit the GPU.

## Validation against official SDPB 3.1

Iterates (mu, objectives, errors, step lengths) agree with SDPB to 1e-13 to 1e-15 relative, with the same
iteration counts and verdicts:
SDPB's example; single-correlator Ising nmax=6; mixed sigma-epsilon Ising nmax=6, 10, 14, 18 (first 20
iterations for the larger ones); stress-tensor system (T, sigma, epsilon) nmax=6, all 115 iterations.
The KPSV nmax=6 island is reproduced. Streaming does not change results: unchunked runs are bitwise
identical to the original solver, streamed runs bitwise identical to resident ones with the same chunks,
and a fully streamed real SDP (mixed Ising nmax=10) agrees with the resident run to 1e-14 over all 131
iterations.

## Performance

Per iteration, against official SDPB on 56-64 cores (AMD EPYC):

| problem | SDPB (CPU) | SDPviBe (one RTX 5090) |
|---|---|---|
| mixed Ising nmax=10 | 9.45 s (56 cores) | 1.53 s |
| mixed Ising nmax=14 | 17.96 s (64 cores) | 2.04 s |
| mixed Ising nmax=18 (N=854, 1086 bits) | 36.68 s (64 cores) | 2.03 s (18x) |
| stress-tensor system nmax=6 (N=742, 5x5 blocks) | 65.5 s (64 cores) | 3.8 s (17x; 129.5 min vs 8.4 min end to end) |

Streaming (synthetic SDPs with the stress-tensor system's block structure, scaled up; RTX PRO 6000 with
96 GB, PCIe 5; "32 GB" is a hard cap emulating an RTX 5090):

| nmax~ | N | data in host RAM | 96 GB card | 32 GB cap |
|---|---|---|---|---|
| 8  | 1319 | 18 GB  | 7.56 s/it (all resident) | 7.93 |
| 10 | 2061 | 36 GB  | 17.6 | 15.6 |
| 12 | 2968 | 62 GB  | 30.4 | 30.5 |
| 14 | 4040 | 102 GB | 52.0 | 61.9 |
| 16 | 5276 | 172 GB | 86.9 | 120.7 |

Details, history and design notes: [docs/DEVLOG.md](docs/DEVLOG.md).

## Install

Needs an NVIDIA GPU (developed on Ampere and Blackwell), CUDA, PyTorch with Triton, and CuPy:

```bash
pip install torch numpy mpmath
pip install cupy-cuda12x        # or cupy-cuda13x, matching your CUDA
pip install -e .
```

## Usage

```bash
pmp2sdp --precision 768 --input pmp.json --output sdp --outputFormat json     # official SDPB tool
python -m sdpvibe -s sdp --precision 768 --findPrimalFeasible --findDualFeasible -o out
```

Options follow SDPB (`--precision`, `--maxIterations`, `--dualityGapThreshold`, `--primalErrorThreshold`,
`--dualErrorThreshold`, `--initialMatrixScalePrimal/Dual`, `--feasibleCenteringParameter`,
`--infeasibleCenteringParameter`, `--stepLengthReduction`, `--maxComplementarity`, `--findPrimalFeasible`,
`--findDualFeasible`, `--detectPrimalFeasibleJump`, `--detectDualFeasibleJump`). Output: `out/out.txt`
and `out/iterations.json` (same fields as SDPB's).

Environment variables: `SDPB_GPU_TIMING=1` (per-phase times and memory peaks), `SDPB_GPU_MEM_GB=<GB>`
(plan for and enforce a smaller GPU), `SDPB_GPU_STREAM=all|none` (force streaming / residency),
`SDPB_GPU_CHUNK_GB`, `SDPB_GPU_PANELS`.

## Examples and tools

See [examples/README.md](examples/README.md): Ising single-correlator and mixed-correlator bounds and
scans with `scalar_blocks`, comparison with official SDPB, the stress-tensor system with the
`stress-tensors-3d` pipeline, and running everything on rented GPUs (vast.ai).

```
sdpvibe/   the solver package (python -m sdpvibe)
tests/     exactness and consistency tests, benchmarks (bench_stream.py: synthetic large SDPs)
tools/     PMP generators for Ising, SDPB/scalar_blocks images without docker, iteration comparison
vast/      scripts for rented GPU machines (setup, sync, self-destroying batch runs, stress-tensor pipeline)
cpu_baseline/  GMP reference for the multiprecision kernels
docs/      development log
```

## Limitations

* Input is `pmp2sdp`'s JSON format; very large SDPs need a binary reader (not written yet).
* No checkpoints / hot start yet; single GPU per SDP.
* Factoring Q currently needs ~9x the memory of Q (limits 32 GB cards to nmax~16 for the stress-tensor
  system).

## Credits

SDPB (D. Simmons-Duffin, W. Landry, V. Dommes et al.), whose algorithm and file formats this follows;
`scalar_blocks` and `blocks_3d` (bootstrap collaboration) for conformal blocks; the `stress-tensors-3d`
pipeline (Chang, Dommes, Erramilli, Homrich, Kravchuk, Liu, Mitchell, Poland, Simmons-Duffin) for the
stress-tensor system. Mixed Ising system following Kos, Poland, Simmons-Duffin, Vichi (arXiv:1406.4858).
