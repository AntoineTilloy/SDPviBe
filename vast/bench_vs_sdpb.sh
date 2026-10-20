#!/usr/bin/env bash
# Run ON a GPU instance (detached) from ~/SDPviBe after vast/dropin_box.sh's setup: synthetic stress-tensor-like
# SDPs (tools/synth_sdp.py) at nmax in $SIZES, solved for a few iterations by official SDPB (all cores) and by
# SDPviBe; also SDPviBe's start-up time. Log results/bench_vs_sdpb.log (last line DONE); self-destroys.
cd ~/SDPviBe
DEADLINE_H=${DEADLINE_H:-5}; HOLD_H=${HOLD_H:-1}; PREC=${PREC:-832}
[ -n "${CONTAINER_API_KEY:-}" ] || eval "$(tr '\0' '\n' < /proc/1/environ | grep '^CONTAINER_\(ID\|API_KEY\)=' | sed 's/^/export /')"
LOG=results/bench_vs_sdpb.log
log() { echo "$(date +%T) $*" >> $LOG; }
selfkill() { log "shutting down ($1)"; vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> $LOG 2>&1 \
    || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1; }
( sleep $((DEADLINE_H * 3600)); selfkill "deadline ${DEADLINE_H}h" ) &
R=$PWD/tools/sdpb_root
export OPAL_PREFIX=$R/usr OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
MPI="$R/usr/bin/mpirun --allow-run-as-root"
mt() { python3 -c "
import json,sys; a=json.load(open(sys.argv[1])); t=[r['iter_time'] for r in a]; print('iteration times', t, 'median(2+):', sorted(t[1:])[len(t[1:])//2] if len(t)>1 else t)" $1 2>&1; }
# start-up overhead of a short run (kernel caches warm after the suite)
S=runs/dropin/ising_n6_allowed/sdp
( time bin/sdpb -s $S --precision 768 --maxIterations 1 -o runs/startup -c runs/startup.ck --noFinalCheckpoint --verbosity 0 ) > results/startup.log 2>&1
log "startup (1 iteration of ising n6, warm caches): $(grep real results/startup.log)"
for n in ${SIZES:-6 8 10}; do
    D=runs/synth_n$n
    ( time python3 tools/synth_sdp.py --nmax $n --precision $PREC --out $D ) > results/synth_n$n.log 2>&1 || { log "FAILED synth n$n: $(tail -3 results/synth_n$n.log | tr '\n' ' ')"; continue; }
    log "$(head -1 results/synth_n$n.log)"
    its=${SDPB_ITS:-3}
    ( time timeout 7200 $MPI --bind-to core $R/usr/local/bin/sdpb -s $D --precision $PREC --maxIterations $its -o $D.sdpb \
        -c $D.sdpb.ck --noFinalCheckpoint --verbosity 1 ) > results/n${n}_sdpb.log 2>&1
    log "n$n SDPB ($(nproc) threads): $(mt $D.sdpb/iterations.json); $(grep real results/n${n}_sdpb.log)"
    ( time timeout 7200 bin/sdpb -s $D --precision $PREC --maxIterations $((its + 1)) -o $D.gpu -c $D.gpu.ck --noFinalCheckpoint ) > results/n${n}_gpu.log 2>&1
    log "n$n SDPviBe: $(mt $D.gpu/iterations.json); $(grep real results/n${n}_gpu.log)"
    python3 tools/compare_iterations.py $D.sdpb/iterations.json $D.gpu/iterations.json 2>&1 | tail -2 | tr '\n' ' ' >> $LOG; echo >> $LOG
    rm -rf $D                                                       # disk
done
log DONE
for i in $(seq $((HOLD_H * 120))); do [ -f results/PULLED ] && break; sleep 30; done
selfkill "finished"
