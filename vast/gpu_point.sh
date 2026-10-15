#!/usr/bin/env bash
# Run ON a GPU instance (detached) from ~/SDPviBe, next to a CPU instance running vast/st3d_point.sh:
#   SRC="-p <port> root@<host>"  (ssh to the CPU instance; this machine's key must be authorized there)
#   for each program: wait for results/<program>_READY on SRC, copy the prepared SDP (sdpvibe cache +
#   small json files), run a short timed solve (phase breakdown, memory) and a full solve (capped at
#   MAXH hours), compare its first iterations with official SDPB's.
#   Then log DONE, wait for results/PULLED (<= HOLD_H h) and destroy the instance. Hard deadline DEADLINE_H.
set -u
cd ~/SDPviBe
DEADLINE_H=${DEADLINE_H:-20}; HOLD_H=${HOLD_H:-6}; MAXH=${MAXH:-6}
PROGRAMS=${PROGRAMS:-"SDPviBe_nmax10_point SDPviBe_nmax14_point"}
[ -n "${CONTAINER_API_KEY:-}" ] || eval "$(tr '\0' '\n' < /proc/1/environ | grep '^CONTAINER_\(ID\|API_KEY\)=' | sed 's/^/export /')"
mkdir -p results runs
LOG=results/gpu_point.log
log() { echo "$(date +%T) $*" >> $LOG; }
selfkill() {
    log "shutting down ($1)"
    vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> $LOG 2>&1 \
      || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1
}
( sleep $((DEADLINE_H * 3600)); selfkill "deadline ${DEADLINE_H}h" ) &
# SRC=local: the CPU pipeline runs on this machine (no copy)
if [ "$SRC" = local ]; then SSH="bash -c"; else SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 $SRC"; fi
log "start: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) x $(nvidia-smi -L | wc -l); $(nproc) cores; $(free -g | awk '/Mem:/{print $2}') GB RAM"
for PROG in $PROGRAMS; do
    log "== $PROG: waiting for the SDP"
    while true; do
        R=$($SSH "cd ~; cat SDPviBe/results/${PROG}_READY 2>/dev/null; grep -c 'FAILED $PROG\|^.* DONE$' SDPviBe/results/st3d_point.log" 2>/dev/null)
        set -- $R
        [ $# -ge 3 ] && break                                   # path, precision, count
        [ "${1:-0}" != 0 ] && [ $# -eq 1 ] && { log "$PROG: not available on the CPU instance"; continue 2; }
        sleep 120
    done
    DIR=$1; PREC=$2; W=runs/$PROG; mkdir -p $W
    t0=$(date +%s)
    if [ "$SRC" = local ]; then
        ln -sfn $DIR $W/sdp; cp results/${PROG}_sdpb_iterations.json $W/sdpb_iterations.json 2>/dev/null
    else
        mkdir -p $W/sdp
        $SSH "tar cf - -C $DIR --exclude 'block_data_*' ." | tar xf - -C $W/sdp || { log "FAILED $PROG copy"; continue; }
        $SSH "cat SDPviBe/results/${PROG}_sdpb_iterations.json" > $W/sdpb_iterations.json 2>/dev/null
        log "$PROG: copied $(du -sh $W/sdp/ | cut -f1) in $(( $(date +%s) - t0 )) s"
    fi
    P="--precision $PREC --findPrimalFeasible --findDualFeasible --detectPrimalFeasibleJump --detectDualFeasibleJump"
    ( time SDPB_GPU_TIMING=1 python -m sdpvibe -s $W/sdp $P --maxIterations 3 -o $W/gpu_timed ) > results/${PROG}_gpu_timed.log 2>&1 \
        || log "FAILED $PROG timed run: $(grep -v '^\[W' results/${PROG}_gpu_timed.log | tail -3 | tr '\n' ' ')"
    log "$PROG: timed: $(grep -h 'read SDP\|phase\|^real' results/${PROG}_gpu_timed.log | tr '\n' ' ')"
    ( time timeout $((MAXH * 3600)) python -m sdpvibe -s $W/sdp $P -o $W/gpu_out ) > results/${PROG}_gpu.log 2>&1
    log "$PROG: full: $(grep -h -- '-----\|^iterations\|^real' results/${PROG}_gpu.log | tr '\n' ' ') last: $(grep -v '^\[W' results/${PROG}_gpu.log | grep '^ *[0-9]' | tail -1)"
    cp $W/gpu_out/iterations.json results/${PROG}_gpu_iterations.json 2>/dev/null
    [ -s $W/sdpb_iterations.json ] && python tools/compare_iterations.py $W/sdpb_iterations.json $W/gpu_out/iterations.json \
        > results/${PROG}_compare.txt 2>&1 && log "$PROG: vs SDPB: $(tail -2 results/${PROG}_compare.txt | tr '\n' ' ')"
done
log DONE
for i in $(seq $((HOLD_H * 120))); do [ -f results/PULLED ] && break; sleep 30; done
selfkill "finished"
