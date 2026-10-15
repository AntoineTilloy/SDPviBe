#!/usr/bin/env bash
# Run ON the instance (detached) from ~/SDPviBe.
#   1. nmax=18 mixed-Ising blocks + PMP + pmp2sdp on the CPUs
#   2. wait for results/GO (client pushes updated solver code, then touches it)
#   3. GPU solver (full run, phase timings) + GPU profile; SDPB on all cores (20 iterations)
#   4. touch results/DONE; wait (<= 3 h) for results/PULLED; destroy the instance (fallback: stop)
#   Hard deadline DEADLINE_H hours overall.
cd ~/SDPviBe
CORES=${CORES:-$(nproc)}; DEADLINE_H=${DEADLINE_H:-16}
NMAX=18; DS=0.5181489; DE=1.412625; PREC=1024
mkdir -p results
log() { echo "$(date) $*" >> results/run_n18.log; }
selfkill() {
    log "shutting down ($1)"
    vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> results/run_n18.log 2>&1 \
      || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> results/run_n18.log 2>&1
}
( sleep $((DEADLINE_H * 3600)); selfkill "deadline ${DEADLINE_H}h" ) &
WATCHDOG=$!
. tools/sdpb_env_native.sh
R=runs/mixed_n${NMAX}_${DS}_${DE}
mkdir -p $R
log "start, $CORES cores"
( time python tools/ising_mixed_pmp.py --nmax $NMAX --lmax 60 --order 100 --poles 28 --prec 896 --ds $DS --de $DE \
    --out $R/pmp.json ) > $R/gen.log 2>&1
log "PMP done: $(tail -4 $R/gen.log | tr '\n' ' ')"
( time mpipmp2sdp $CORES --precision $PREC -i $R/pmp.json -o $R/sdp --outputFormat json ) > $R/pmp2sdp.log 2>&1
log "pmp2sdp done: $(tail -3 $R/pmp2sdp.log | tr '\n' ' ')"
touch results/READY
while [ ! -f results/GO ]; do sleep 30; done
log "GO received"
( time SDPB_GPU_TIMING=1 python -m sdpvibe -s $R/sdp --precision $PREC --findPrimalFeasible --findDualFeasible \
    -o $R/gpu_out ) > $R/gpu.log 2>&1
log "GPU done: $(grep -E 'read SDP|-----|iterations|phase|real' $R/gpu.log | tr '\n' ' ')"
python tests/prof_solver.py $R/sdp $PREC 3 > results/prof_n18.txt 2>&1
log "profile done"
( time mpisdpb $CORES -s $R/sdp -o $R/sdpb_out --precision $PREC --noFinalCheckpoint --findPrimalFeasible \
    --findDualFeasible --maxIterations 20 --verbosity 1 ) > $R/sdpb.log 2>&1
log "SDPB done: $(grep real $R/sdpb.log)"
python tools/compare_iterations.py $R/sdpb_out/iterations.json $R/gpu_out/iterations.json > results/compare_n18.txt 2>&1
python - >> results/run_n18.log <<EOF
import json
a = json.load(open("$R/sdpb_out/iterations.json")); b = json.load(open("$R/gpu_out/iterations.json"))
ta = sorted(r["iter_time"] for r in a[2:]); tb = sorted(r["iter_time"] for r in b[2:])
print(f"per-iteration (median after warm-up): SDPB {ta[len(ta)//2]:.2f} s on $CORES cores, GPU {tb[len(tb)//2]:.2f} s")
EOF
cp $R/gpu.log $R/sdpb.log $R/gen.log $R/pmp2sdp.log results/ 2>/dev/null
mkdir -p results/n18_iter && cp $R/gpu_out/iterations.json results/n18_iter/gpu.json && cp $R/sdpb_out/iterations.json results/n18_iter/sdpb.json
touch results/DONE
for i in $(seq 1 180); do [ -f results/PULLED ] && break; sleep 60; done
kill $WATCHDOG 2>/dev/null
selfkill "finished"
