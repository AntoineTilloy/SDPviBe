#!/usr/bin/env bash
# Run ON a multi-GPU instance (detached) from ~/SDPviBe, with SDPs uploaded to testdata/sdp_<name>/:
#  1. correctness: a real SDP solved with --gpus 1, 2, 4, ... (same verdict / iterations, iterate agreement),
#     plus a streamed (lean) run on all GPUs vs 1 GPU
#  2. performance: synthetic stress-tensor-like SDPs (tools/synth_sdp.py, same files as vast/bench_vs_sdpb.sh)
#     at nmax~6..12, a few iterations with 1, 2, 4, ... GPUs
# Log: results/mgpu.log (last line DONE).  Waits <= HOLD_H h for results/PULLED, then destroys the instance;
# hard deadline DEADLINE_H.
cd ~/SDPviBe
DEADLINE_H=${DEADLINE_H:-3}; HOLD_H=${HOLD_H:-0.5}; PREC=${PREC:-832}
NG=$(nvidia-smi -L | wc -l)
GPUS=${GPUS:-$(for g in 1 2 4 8; do [ $g -le $NG ] && echo -n "$g "; done)}
[ -n "${CONTAINER_API_KEY:-}" ] || eval "$(tr '\0' '\n' < /proc/1/environ | grep '^CONTAINER_\(ID\|API_KEY\)=' | sed 's/^/export /')"
W=${W:-runs/mgpu}; mkdir -p results $W; LOG=results/${LOGNAME_:-mgpu}.log
log() { echo "$(date +%T) $*" >> $LOG; }
selfkill() { log "shutting down ($1)"; vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> $LOG 2>&1 \
    || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1; }
( sleep $(python3 -c "print(int($DEADLINE_H * 3600))"); selfkill "deadline ${DEADLINE_H}h" ) &
log "start: $NG x $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1); $(nproc) cores; $(free -g | awk '/Mem:/{print $2}') GB RAM; GPU counts: $GPUS"
nvidia-smi topo -m > results/mgpu_topo.txt 2>&1
bash vast/remote_setup.sh > results/setup.log 2>&1 || log "FAILED setup: $(tail -3 results/setup.log | tr '\n' ' ')"
. ~/venv/bin/activate
mt() { python3 -c "
import json,sys; a=json.load(open(sys.argv[1])); t=[r['iter_time'] for r in a]; print('iteration times', t, 'median(2+):', sorted(t[1:])[len(t[1:])//2] if len(t)>1 else t)" $1 2>&1; }
sv() {  # tag gpus sdp precision extra-args...
    local tag=$1 g=$2 sdp=$3 prec=$4; shift 4
    local t0=$(date +%s)
    python3 -m sdpvibe -s $sdp -o $W/$tag -c $W/$tag.ck --precision $prec --gpus $g --verbosity 2 "$@" > $W/$tag.log 2>&1
    local rc=$?
    log "$tag (gpus=$g): rc=$rc $(( $(date +%s) - t0 )) s; $(grep -m1 'chunks' $W/$tag.log | sed 's/^\s*//' | cut -c1-240); $(grep -m1 'terminateReason' $W/$tag/out.txt 2>/dev/null); $(mt $W/$tag/iterations.json)"
    [ $rc = 0 ] || log "  error: $(grep -v '^\s*$' $W/$tag.log | tail -3 | tr '\n' ' ' | cut -c1-400)"
}
cmp_it() {  # a b
    python3 - "${BASEW:-$W}/$1/iterations.json" "$W/$2/iterations.json" >> $LOG 2>&1 <<'EOF'
import json, sys, mpmath as mp
mp.mp.prec = 1200
a, b = (json.load(open(f)) for f in sys.argv[1:3])
ks = ["mu", "P-obj", "D-obj", "P-step", "D-step", "beta"]
def dig(x, y):
    x, y = mp.mpf(x), mp.mpf(y)
    return 99.0 if x == y else float(-mp.log10(abs(x - y) / max(abs(x), abs(y))))
per = [min(dig(x[k], y[k]) for k in ks) for x, y in zip(a, b)]
same = sum(all(x[k] == y[k] for k in ks) for x, y in zip(a, b))
print(f"  compare {sys.argv[1].split('/')[-2]} vs {sys.argv[2].split('/')[-2]}: iterations {len(a)} / {len(b)}, "
      f"bitwise-equal rows {same}, worst digits {min(per):.1f}, median {sorted(per)[len(per) // 2]:.1f}")
EOF
}
log "setup done"
# 1. correctness on a real SDP
F="--findPrimalFeasible --findDualFeasible --detectPrimalFeasibleJump --detectDualFeasibleJump --noFinalCheckpoint"
S=testdata/sdp_mixed_n10_ising
for g in ${REAL_GPUS-$GPUS}; do sv real_g$g $g $S 768 $F; [ $g = 1 ] || cmp_it real_g1 real_g$g; done
GMAX=$(echo ${REAL_GPUS-$GPUS} | awk '{print $NF}')
[ -n "$GMAX" ] && [ "${LEAN_TEST:-1}" = 1 ] && { SDPB_GPU_CHUNK_GB=0.3 sv real_lean_g$GMAX $GMAX $S 768 $F; cmp_it real_g1 real_lean_g$GMAX; }
# 2. performance on synthetic SDPs
for n in ${SIZES:-6 8 10 12}; do
    D=$W/synth_n$n
    python3 tools/synth_sdp.py --nmax $n --precision $PREC --out $D >> $LOG 2>&1 || { log "FAILED synth n$n"; continue; }
    for g in $GPUS; do sv synth_n${n}_g$g $g $D $PREC --maxIterations 4 --noFinalCheckpoint; [ $g = 1 ] || cmp_it synth_n${n}_g1 synth_n${n}_g$g; done
    rm -rf $D
done
log "DONE"
t=0; while [ ! -f results/PULLED ] && [ $t -lt $(python3 -c "print(int($HOLD_H * 3600))") ]; do sleep 60; t=$((t + 60)); done
selfkill "finished"
