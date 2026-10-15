# Examples

All commands run from the repository root.

## 0. Official SDPB and scalar_blocks without docker

The comparison runs and the PMP generators use the official images, unpacked locally (no root needed):

```bash
python3 tools/pull_image.py bootstrapcollaboration/sdpb 3.1.0-amd64 tools/sdpb_root
python3 tools/pull_image.py bootstrapcollaboration/scalar_blocks master tools/sb_root
source tools/sdpb_env.sh        # defines pmp2sdp and sdpb (single MPI rank) via the musl loader
```

On a rented machine, `vast/remote_sdpb_setup.sh` does the same and defines multi-rank `mpipmp2sdp` /
`mpisdpb` helpers.

## 1. SDPB's example

```bash
pmp2sdp --precision 768 --input tools/example_pmp.json --output runs/example/sdp --outputFormat json
python -m sdpvibe -s runs/example/sdp --precision 768 -o runs/example/gpu_out
```

## 2. Single-correlator Ising bound

`tools/ising_pmp.py` builds the <sigma sigma sigma sigma> PMP from `scalar_blocks` (cached in `data/blocks`):

```bash
python3 tools/ising_pmp.py --nmax 6 --ds 0.518 --de 1.41 --out runs/ising/pmp.json
pmp2sdp --precision 768 --input runs/ising/pmp.json --output runs/ising/sdp --outputFormat json
python -m sdpvibe -s runs/ising/sdp --precision 768 --findPrimalFeasible --findDualFeasible -o runs/ising/gpu_out
```

`tools/bisect_single.sh` bisects the upper bound on Delta_epsilon at fixed Delta_sigma.

## 3. Mixed (sigma, epsilon) Ising island

`tools/ising_mixed_pmp.py` builds the KPSV system (arXiv:1406.4858; note the (-1)^l sign of the V_- vectors
in the scalar_blocks conventions), with gaps Delta_sigma' , Delta_epsilon' >= 3:

```bash
python3 tools/ising_mixed_pmp.py --nmax 6 --ds 0.5181 --de 1.4126 --out runs/mixed/pmp.json
python3 tools/scan_mixed.py --nmax 6 --deltas -0.885,-0.895 --ds 0.516:0.523:0.001 --out results/scan.csv
```

## 4. Comparing with official SDPB

Run both with the same parameters, then

```bash
sdpb -s runs/x/sdp_bin -o runs/x/sdpb_out --precision 768 --findPrimalFeasible --findDualFeasible
python3 tools/compare_iterations.py runs/x/sdpb_out/iterations.json runs/x/gpu_out/iterations.json
```

prints log10 of the relative difference per iteration and quantity.

## 5. Stress-tensor system (T, sigma, epsilon)

The [stress-tensors-3d](https://gitlab.com/davidsd/stress-tensors-3d) pipeline (Haskell, hyperion,
blocks_3d) needs Slurm; on a single rented machine:

```bash
bash vast/remote_setup.sh && bash vast/remote_sdpb_setup.sh
git clone https://gitlab.com/davidsd/stress-tensors-3d ext/stress-tensors-3d
bash vast/st3d_setup.sh          # stack build, blocks_3d image, 'vast' hyperion config
bash vast/fake_slurm.sh          # sbatch/scontrol stand-ins, ssh-to-self
# run one of their programs (e.g. TSigEps_test_nmax6), then convert and compare with official SDPB:
bash vast/st3d_compare.sh <hyperion object dir>
```

## 6. Large SDPs and streaming

```bash
python tests/bench_stream.py --nmax 12 --budget 32      # synthetic stress-tensor-like SDP, 32 GB GPU cap
python tests/test_stream.py runs/x/sdp 768 6 0.03       # resident vs chunked vs streamed consistency
```

## 7. Rented GPUs (vast.ai)

`vast/gpu_value.py` ranks offers by throughput per dollar; `vast/sync.sh push|pull <id>` copies code and
results; the batch scripts (`run_all.sh`, `run_n18.sh`, `bench_stream.sh`) run detached, log every step,
and destroy the instance after the results are pulled or after a deadline (`watch_bench.sh` follows a run
from the local machine and pulls the results).
