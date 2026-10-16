# Plan: a state-of-the-art 3d Ising island with SDPviBe

Draft, 2026-10-16. "Measured" numbers come from runs described here; the rest are projections from the
per-phase model `tools/project_cost.py` (reproduces measured 5090 iteration times to ~10-15% for
nmax~8-16). Problem shapes beyond nmax~16 are extrapolated (see section 3): read everything below as
+-50%, and the cost ranges as order-of-magnitude guidance until one real nmax=22 SDP has been measured.

## 1. The target

The best 3d Ising bounds come from the stress-tensor system (T, sigma, epsilon) of Chang, Dommes, Erramilli,
Homrich, Kravchuk, Liu, Mitchell, Poland, Simmons-Duffin, computed with their public pipeline
<https://gitlab.com/davidsd/stress-tensors-3d> (hyperion + blocks_3d + SDPB). Their repository shows:

| nmax (Lambda = 2 nmax - 1) | island data in the repo | SDPB job size per point |
|---|---|---|
| 14 (27) | done | 5 nodes x 128 cores, 12-48 h |
| 18 (35) | ~100 points | |
| 22 (43) | 41 allowed + 38 excluded points (default gaps), ~80 more with aggressive gaps | 16 nodes x 128 cores, <= 20 h |
| 26 (51) | 1 allowed point, 20 test intervals (in progress) | 16 nodes x 128 cores, <= 20-30 h |

So reproducing the nmax=22 island is ~80 points x (one or more) 2048-core x 20 h jobs, i.e. up to a few
million CPU core-hours for the SDP solves alone. Each point is an OPE scan: several hot-started SDPB solves.

## 2. What was measured

**Solver correctness at scale**: streamed/chunked runs agree with resident runs and with official SDPB to
~1e-13-1e-14 (README). Real stress-tensor nmax=6: 17x per iteration vs 64 CPU cores.

**GPU iteration times**, synthetic SDPs with the stress-tensor block structure (shapes scaled from the real
nmax=6 SDP), 832 bits, whole card, host-RAM streaming as needed (s/iteration):

| nmax~ | N | RTX 3090 24 GB (PCIe 4) | RTX 5090 32 GB (PCIe 4) | RTX PRO 6000 96 GB (PCIe 5, *1024 bits*) |
|---|---|---|---|---|
| 8  | 1319 | 18.4 | 10.0 | 7.6 |
| 10 | 2061 | 45.1 | 20.2 | 17.6 |
| 12 | 2968 | 106 | 39.4 | 30.4 |
| 14 | 4040 | 201 | 68.4 | 52.0 |
| 16 | 5276 | - | 161 | 86.9 |

The 3090 is ~3x slower than a 5090 at nmax>=12 (half the INT8 throughput, and 24 GB forces smaller
batches). Per iteration at nmax~14 a 5090 costs ~$0.011 (at $0.6/h), a 3090 ~$0.017, a PRO 6000 ~$0.015;
above nmax~16 the 96 GB card is cheaper per iteration (fewer, larger batches) and is the only single card
that holds the Q-related arrays without more engineering.

**Blocks** (their task statistics, plus a real nmax=10 point started here):
* 86% of blocks_3d time is spent on <TTTT> (zero dimension differences, Delta_1 + Delta_2 = 6): identical
  for every point, computed once per island and cached by hyperion. 11% (<TTT x>-type) depends only on
  Delta_epsilon; ~3% depends on both external dimensions.
* Per point (TTTT cached), their statistics give ~1.6k core-hours of block + PMP tasks at nmax=22 and
  ~4.5k at nmax=26; the one-time TTTT is ~18k core-hours at nmax=26.
* A real nmax=10 point (production nmax=10 parameters, all blocks computed from scratch, 64 cores / 128
  threads EPYC 7B13): the 550 blocks_3d tasks took ~7 h, and the remaining hyperion stages (mixed and
  three-point blocks, PMP assembly) did not finish within the 22 h limit of the run. **The end-to-end real
  run at nmax=10 therefore did not complete**; the real-shape solve is still to be done.

## 3. Projected problem sizes

N follows exactly from the derivative sets of blocks-core (`xtDerivs`: m + 2n <= 2 nmax - 1, by parity of
m): N ~ n(n+1)/2 per crossing-equation component, i.e. N ~ 10^4 at nmax=22 and ~1.4x10^4 at nmax=26.
Points per block K follow the polynomial degree, which grows with the number of kept poles (104 poles at
order 60; their nmax=22/26 runs use order 130/152 with pole shifting off); taken here as the synthetic
scaling (K ~ 250 at nmax=26) and 1.3x that.

| | nmax=22 (832 bits) | nmax=26 (896 bits) |
|---|---|---|
| N | ~10,000 | ~14,000 |
| blocks | ~415 | ~490 |
| B + per-block state (host RAM / NVMe) | 450-640 GB | 0.85-1.2 TB |
| Q (N x N) | 11 GB | 23 GB |

## 4. Cost on rented GPUs

Per-iteration projections (with the engineering of section 6: out-of-core Q; current kernels):

| | nmax=22 | nmax=26 |
|---|---|---|
| RTX PRO 6000 96 GB | 330-490 s | 810-1200 s |
| RTX 5090 32 GB | 660-900 s | 1400-2000 s |
| RTX 3090 24 GB | 1900-2600 s | 4100-5700 s |
| official SDPB, 2048 cores (scaled from measured 64-core nmax=6, ideal scaling) | ~40-55 min | ~2-2.7 h |

One of their 2048-core x 20 h jobs therefore holds only ~20-30 SDPB iterations at nmax=22 (~7-10 at
nmax=26); the same iterations take ~2-4 h on one RTX PRO 6000 (~$3-6 at $1.4/h), i.e. **one GPU ~ 5-10x
their 16-node job** (more if SDPB scales less than ideally).

**nmax=22 island (~80 points)**, assuming 1-3 such job-equivalents per point:
* GPU: 160-960 GPU-hours on RTX PRO 6000 -> **$250-1,350**
* blocks + PMP on rented CPUs (~$0.005-0.01 per core-hour on vast): one-time TTTT ~$50-150; per point
  1.6k core-hours -> $1-1.5k for 80 points as done now, **~$200-400 with Delta_epsilon interpolation**
* total **~$0.5-2k**, a few days of wall time with ~5-10 GPUs in parallel (points are independent).

**nmax=26** (new territory): ~2.5x the GPU time per iteration but ~3x fewer iterations per job-equivalent;
the one-time TTTT (~18k core-hours, ~$100-200) and the per-point blocks dominate unless interpolated; a
first island at nmax=26 is plausibly in the **$1-4k** range.

Best option: **RTX PRO 6000 (96 GB) boxes with >= 1 TB RAM or fast NVMe** for the solves, cheap many-core
CPU boxes for the blocks. H100/H200 are untested (likely similar performance per dollar).

## 5. Home machine: Threadripper 24 cores, 3x RTX 3090, 4 TB NVMe

* SDP data (450-640 GB at nmax=22, ~1 TB at nmax=26) lives on the NVMe and is streamed: ~2.5 TB read per
  iteration at nmax=22, i.e. ~6 min/iteration at 7 GB/s, overlapped with compute.
* With the three 3090s sharing the chunks: ~11-15 min/iteration at nmax=22 (~25-35 min at nmax=26).
  One of their 20 h 2048-core jobs ~ 5-7 h on this box.
* The 24 CPU cores are the bottleneck: per-point blocks at nmax=22 are ~1.6k core-hours, ~3 days per point
  on 24 cores (one-time TTTT another ~2-3 weeks at nmax=22, ~1 month at nmax=26). With Delta_epsilon
  interpolation and TTTT computed once (or rented for ~$100), per point drops to ~0.1-0.3k core-hours,
  4-12 h, overlapping with the GPU solves.
* **nmax=22 island (~80 points): ~1-2 months on this box** (GPU and CPU running in parallel), versus a few
  million core-hours on a cluster. nmax=26: ~2-3 months, with the one-time TTTT rented.

## 6. Engineering still needed (in order)

1. **Drop-in `sdpb` for hyperion**: SDPB's binary SDP input (JSON is ~2.5x larger and slow to parse at these
   sizes), SDPB-compatible outputs and checkpoints (`--initialCheckpointDir`, hot start: OPE scans and
   Delaunay searches rely on it), and the options their pipeline passes. Then their search machinery
   (Delaunay islands, OPE scans, skydiving) drives SDPviBe unchanged.
2. **Q (N x N) out of core**: the RNS accumulator is ~2.3x Q (lower triangle only: ~1.2x) and the Cholesky of
   Q currently ~9x Q; a tiled, host-streamed factorisation needs ~1x Q plus a panel. Needed above nmax~16
   on 24-32 GB cards, above nmax~20 on 96 GB cards.
3. **Several GPUs per problem**: chunks split across GPUs, Q partial sums reduced on the host.
4. **B on NVMe** (constant data, read 4x per iteration) for boxes with less RAM than the SDP.
5. **Faster kernels**: the fused INT8 Q kernel prototyped earlier (~5x the in-solver rate); larger diagonal
   tiles for the latency-bound Cholesky/triangular solves of S; these dominate at nmax=22-26.
6. **Blocks**: Delta_epsilon interpolation of the <TTT x> family; longer term a GPU blocks_3d.
7. **Measure one real nmax=22 SDP** (shape, iteration count of a hot-started OPE step) before committing
   money: it fixes the +-50% above. Cheapest route: their pipeline on a many-core CPU box with a budget-
   derived deadline and periodic log pulls (the nmax=10 attempt here lost its logs).

## 7. Risks

* Problem shapes at nmax=22/26 are extrapolated (K in particular); a 2x larger K roughly doubles memory and
  time.
* Iterations per point in their OPE-scan workflow are inferred from job sizes, not measured.
* Numerical behaviour of SDPviBe at 832-896 bits on these much larger SDPs is untested beyond nmax=6 (real)
  and the agreement tests; SDPB cross-checks on a few points remain necessary.
* Block generation costs are taken from their statistics; our only real measurement (nmax=10) suggests
  the full pipeline on rented CPUs can be slower than those statistics imply.
