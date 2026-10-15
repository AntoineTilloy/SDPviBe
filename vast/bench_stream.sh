#!/usr/bin/env bash
# Run ON the instance (detached) from ~/SDPviBe: streaming benchmarks + validation.
#   results/bench_stream.log gets one line per step (RESULT ... or STEP FAILED ...); last line DONE.
#   Afterwards waits <= 2 h for results/PULLED, then destroys the instance. Hard deadline DEADLINE_H.
cd ~/SDPviBe
DEADLINE_H=${DEADLINE_H:-3}
# vast puts the instance's id / API key in PID 1's environment only
[ -n "$CONTAINER_API_KEY" ] || eval "$(tr '\0' '\n' < /proc/1/environ | grep '^CONTAINER_\(ID\|API_KEY\)=' | sed 's/^/export /')"
mkdir -p results
LOG=results/bench_stream3.log
log() { echo "$(date +%T) $*" >> $LOG; }
selfkill() {
    log "shutting down ($1)"
    vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> $LOG 2>&1 \
      || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1
}
( sleep $((DEADLINE_H * 3600)); selfkill "deadline ${DEADLINE_H}h" ) &
log "start: $(nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader); $(nproc) cores; $(free -g | awk '/Mem:/{print $2}') GB RAM"

step() {   # step <name> <timeout s> <command...>
    local name=$1 to=$2; shift 2
    log "step $name: $*"
    ( time timeout $to "$@" ) > results/$name.out 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then log "$(grep -h 'RESULT\|GB/s\|-----\|iterations  \|^real' results/$name.out | tr '\n' ' ')"
    else log "STEP FAILED $name (rc $rc): $(grep -v '^\[W' results/$name.out | tail -3 | tr '\n' ' ')"; fi
}
B="python tests/bench_stream.py --its 3"

# real SDP full runs: resident vs everything streamed (validates the chunked arithmetic: GramAcc, panels)
P="--precision 768 --findPrimalFeasible --findDualFeasible --detectPrimalFeasibleJump --detectDualFeasibleJump"
R=runs/mixed_n10_ising
step real_whole  3600 python -m sdpvibe -s $R/sdp $P -o $R/gpu_whole
SDPB_GPU_STREAM=all SDPB_GPU_CHUNK_GB=0.3 step real_stream 5400 python -m sdpvibe -s $R/sdp $P -o $R/gpu_stream
python tools/compare_iterations.py $R/gpu_whole/iterations.json $R/gpu_stream/iterations.json > results/real_compare.txt 2>&1
log "real compare: $(tail -3 results/real_compare.txt | tr '\n' ' ')"
# iteration times (no timers), then one timed run per size for the breakdown
step n8_whole    1800 $B --nmax 8 --stream none
step n8_b32      1800 $B --nmax 8 --budget 32
step n10_b96     3600 $B --nmax 10
step n10_b32     3600 $B --nmax 10 --budget 32
step n12_b96     5400 $B --nmax 12
step n12_b32     5400 $B --nmax 12 --budget 32
step n14_b96     7200 $B --nmax 14 --its 2
step n14_b32     7200 $B --nmax 14 --budget 32 --its 2
step n16_b96     9000 $B --nmax 16 --its 2
step n16_b32     9000 $B --nmax 16 --budget 32 --its 2
SDPB_GPU_TIMING=1 step t_n12_b32 5400 $B --nmax 12 --budget 32
SDPB_GPU_TIMING=1 step t_n14_b32 7200 $B --nmax 14 --budget 32 --its 2
log DONE
for i in $(seq 240); do [ -f results/PULLED ] && break; sleep 30; done
selfkill "finished"
