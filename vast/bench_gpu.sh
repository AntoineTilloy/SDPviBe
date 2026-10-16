#!/usr/bin/env bash
# Run ON a GPU instance (detached) from ~/SDPviBe: synthetic stress-tensor-like SDPs (tests/bench_stream.py)
# at nmax in $SIZES and precision $PREC, on the whole card; one timed run (phase breakdown) at $TIMED.
# results/bench_gpu.log: one line per step; last line DONE. Then waits <= 2 h for results/PULLED and
# destroys the instance. Hard deadline DEADLINE_H.
cd ~/SDPviBe
DEADLINE_H=${DEADLINE_H:-3}
[ -n "${CONTAINER_API_KEY:-}" ] || eval "$(tr '\0' '\n' < /proc/1/environ | grep '^CONTAINER_\(ID\|API_KEY\)=' | sed 's/^/export /')"
mkdir -p results
LOG=results/bench_gpu.log
log() { echo "$(date +%T) $*" >> $LOG; }
selfkill() {
    log "shutting down ($1)"
    vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> $LOG 2>&1 \
      || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1
}
( sleep $((DEADLINE_H * 3600)); selfkill "deadline ${DEADLINE_H}h" ) &
step() {   # step <name> <timeout s> <command...>
    local name=$1 to=$2; shift 2
    log "step $name: $*"
    ( time timeout $to "$@" ) > results/$name.out 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then log "$(grep -h 'RESULT\|GB/s\|phase\|^real' results/$name.out | tr '\n' ' ')"
    else log "STEP FAILED $name (rc $rc): $(grep -v '^\[W' results/$name.out | tail -3 | tr '\n' ' ')"; fi
}
bash vast/remote_setup.sh > results/setup.log 2>&1 || log "FAILED setup: $(tail -3 results/setup.log | tr '\n' ' ')"
log "start: $(nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader); $(nproc) cores; $(free -g | awk '/Mem:/{print $2}') GB RAM"
B="python tests/bench_stream.py --prec ${PREC:-832}"
step bw 300 python tests/bench_stream.py --bw
for n in ${SIZES:-8 10 12 14}; do
    its=3; [ $n -ge 14 ] && its=2
    step n$n 7200 $B --nmax $n --its $its
done
SDPB_GPU_TIMING=1 step t_n${TIMED:-12} 7200 $B --nmax ${TIMED:-12} --its 2
log DONE
for i in $(seq 240); do [ -f results/PULLED ] && break; sleep 30; done
selfkill "finished"
