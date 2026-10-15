#!/usr/bin/env bash
# Run ON the instance from ~/SDPviBe after remote_setup.sh + remote_sdpb_setup.sh:
#   bash vast/scale_test.sh <nmax> <ds> <de> <lmax> <cores> [precision] [sdpb_max_iter]
# Generates the mixed Ising PMP, converts it, runs official SDPB on <cores> MPI ranks and the GPU solver.
set -uo pipefail
NMAX=$1; DS=$2; DE=$3; LMAX=$4; CORES=$5; PREC=${6:-768}; SDPB_ITERS=${7:-500}
. tools/sdpb_env_native.sh
R=runs/mixed_n${NMAX}_${DS}_${DE}
mkdir -p $R results
OUT=results/scale_n${NMAX}_$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_')_${CORES}c.txt
log() { echo "$*" | tee -a $OUT; }
log "### nmax=$NMAX ds=$DS de=$DE lmax=$LMAX cores=$CORES prec=$PREC  $(date)"
if [ ! -f $R/pmp.json ]; then
  ( time python tools/ising_mixed_pmp.py --nmax $NMAX --lmax $LMAX --order ${ORDER:-80} --poles ${POLES:-20} --prec ${BPREC:-665} --ds $DS --de $DE --out $R/pmp.json ) 2>&1 | tail -4 | tee -a $OUT
fi
if [ ! -f $R/sdp/control.json ]; then
  ( time mpipmp2sdp $CORES --precision $PREC -i $R/pmp.json -o $R/sdp --outputFormat json ) 2>&1 | tail -4 | tee -a $OUT
fi
log "--- GPU solver"
( time python -m sdpvibe -s $R/sdp --precision $PREC --findPrimalFeasible --findDualFeasible -o $R/gpu_out ) > $R/gpu.log 2>&1
grep -E "read SDP|-----|iterations|real" $R/gpu.log | tee -a $OUT
log "--- SDPB $CORES ranks"
( time mpisdpb $CORES -s $R/sdp -o $R/sdpb_out --precision $PREC --noFinalCheckpoint --findPrimalFeasible --findDualFeasible --maxIterations $SDPB_ITERS --verbosity 1 ) > $R/sdpb.log 2>&1
grep -E "terminateReason|real" $R/sdpb.log $R/sdpb_out/out.txt | tee -a $OUT
python tools/compare_iterations.py $R/sdpb_out/iterations.json $R/gpu_out/iterations.json | tail -2 | tee -a $OUT
python - <<EOF | tee -a $OUT
import json
a = json.load(open("$R/sdpb_out/iterations.json")); b = json.load(open("$R/gpu_out/iterations.json"))
ta = [r["iter_time"] for r in a[2:]]; tb = [r["iter_time"] for r in b[2:]]
print(f"per-iteration (median, after warm-up): SDPB {sorted(ta)[len(ta)//2]:.2f} s on $CORES cores, GPU {sorted(tb)[len(tb)//2]:.2f} s")
EOF
