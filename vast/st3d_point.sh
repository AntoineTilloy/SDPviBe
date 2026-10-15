#!/usr/bin/env bash
# Run ON a many-core instance (detached) from ~/SDPviBe:
#   1. build the stress-tensors-3d pipeline (hyperion) with its sdpb call parked (SDPB_WRAPPER=block)
#   2. for each program in $PROGRAMS (one feasibility point each, see vast/st3d_programs.patch):
#      blocks + PMP + binary SDP (hyperion, timed) -> JSON SDP -> sdpvibe cache + shape (prepare)
#      -> official SDPB on all cores for $SDPB_ITS iterations (per-iteration time, first iterates)
#      -> results/<program>_READY (a GPU machine then copies <sdp_json> from here)
#   3. log DONE; wait for results/PULLED (<= $HOLD_H h), then destroy the instance. Hard deadline DEADLINE_H.
# results/st3d_point.log: one line per step; failures logged as FAILED.
set -u
cd ~/SDPviBe
DEADLINE_H=${DEADLINE_H:-16}; HOLD_H=${HOLD_H:-10}
PROGRAMS=${PROGRAMS:-"SDPviBe_nmax10_point SDPviBe_nmax14_point"}
SDPB_ITS=${SDPB_ITS:-4}
NP=$(nproc)
[ -n "${CONTAINER_API_KEY:-}" ] || eval "$(tr '\0' '\n' < /proc/1/environ | grep '^CONTAINER_\(ID\|API_KEY\)=' | sed 's/^/export /')"
mkdir -p results
LOG=results/st3d_point.log
log() { echo "$(date +%T) $*" >> $LOG; }
selfkill() {
    log "shutting down ($1)"
    vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> $LOG 2>&1 \
      || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1
}
( sleep $((DEADLINE_H * 3600)); selfkill "deadline ${DEADLINE_H}h" ) &
log "start: $NP cores, $(free -g | awk '/Mem:/{print $2}') GB RAM, $(df -h ~ | awk 'NR==2{print $4}') free disk"

# ---------------------------------------------------------------- 1. setup
t0=$(date +%s)
bash vast/remote_setup.sh > results/setup_python.log 2>&1 || log "FAILED python setup: $(tail -2 results/setup_python.log)"
bash vast/remote_sdpb_setup.sh > results/setup_sdpb.log 2>&1 || log "FAILED sdpb setup: $(tail -2 results/setup_sdpb.log)"
[ -d ext/stress-tensors-3d ] || git clone -q https://gitlab.com/davidsd/stress-tensors-3d ext/stress-tensors-3d
( cd ext/stress-tensors-3d && git apply ../../vast/st3d_programs.patch ) || log "FAILED patch"
SDPB_WRAPPER=block bash vast/st3d_setup.sh > results/setup_st3d.log 2>&1 || log "FAILED st3d setup: $(tail -2 results/setup_st3d.log)"
rm -f /lib/ld-musl-x86_64.so.1; rm -rf ~/.stack/programs/x86_64-linux/ghc-musl*     # stack must not see musl
( cd ext/stress-tensors-3d && time bash ./build.sh ) > results/build.log 2>&1
ln -sf ~/SDPviBe/tools/sdpb_root/lib/ld-musl-x86_64.so.1 /lib/ld-musl-x86_64.so.1  # official binaries need it
EXE=$(find ~/SDPviBe/ext/stress-tensors-3d/.stack-work -type f -name stress-tensors-3d-exec -perm -u+x | head -1)
[ -x "$EXE" ] || { log "FAILED build: $(tail -5 results/build.log | tr '\n' ' ')"; selfkill "build failed"; exit 1; }
bash vast/fake_slurm.sh > results/fake_slurm.log 2>&1 || log "FAILED fake slurm: $(tail -3 results/fake_slurm.log | tr '\n' ' ')"
log "setup done in $(( ($(date +%s) - t0) / 60 )) min (exe $EXE)"

R=$PWD/tools/sdpb_root
export OPAL_PREFIX=$R/usr OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
MPI="$R/usr/bin/mpirun --allow-run-as-root"
stop_hyperion() {
    for p in $(pgrep -x stress-tensors-); do kill $p 2>/dev/null; done
    pkill -f '^sleep infinity$' 2>/dev/null            # parked sdpb calls (not the watchdog's sleep)
    for p in $(pgrep -x blocks_3d) $(pgrep -x pmp2sdp); do kill $p 2>/dev/null; done
    sleep 5
}

# ---------------------------------------------------------------- 2. points
for PROG in $PROGRAMS; do
    log "== $PROG"
    rm -f /root/hyp/sdpb_calls.log
    t0=$(date +%s)
    ( cd ext/stress-tensors-3d && setsid nohup $EXE master -p $PROG > ~/SDPviBe/results/${PROG}_master.out 2>&1 < /dev/null & )
    ok=0
    while true; do
        sleep 60
        [ -s /root/hyp/sdpb_calls.log ] && { ok=1; break; }
        pgrep -x stress-tensors- > /dev/null || break
    done
    t1=$(date +%s)
    if [ $ok = 0 ]; then
        log "FAILED $PROG: hyperion exited before the SDP was built: $(tail -5 results/${PROG}_master.out | tr '\n' ' ')"
        continue
    fi
    ARGS=$(head -1 /root/hyp/sdpb_calls.log)
    stop_hyperion
    SDP=$(echo "$ARGS" | grep -o -- '-s [^ ]*' | head -1 | cut -c4-); SDP=${SDP%/}
    [ -n "$SDP" ] || SDP=$(echo "$ARGS" | grep -o -- '--sdpDir[= ][^ ]*' | head -1 | sed 's/--sdpDir[= ]//')
    PREC=$(echo "$ARGS" | grep -o -- '--precision[= ][0-9]*' | head -1 | grep -o '[0-9]*$')
    OBJ=$(dirname "$SDP")
    NSV=$(ls $OBJ/json/obj/*.nsv 2>/dev/null | head -1)
    log "$PROG: SDP built in $(( (t1 - t0) / 60 )) min (blocks + PMP + pmp2sdp): $SDP, precision $PREC, nsv $NSV"
    log "$PROG: sdpb args: $ARGS"
    cp -r /root/hyp/logs results/${PROG}_hyp_logs 2>/dev/null
    W=$OBJ/sdpvibe; mkdir -p $W
    ( time $MPI -n $NP $R/usr/local/bin/pmp2sdp --precision $PREC --input $NSV --output $W/sdp_json --outputFormat json ) \
        > results/${PROG}_pmp2sdp_json.log 2>&1 || { log "FAILED $PROG pmp2sdp json: $(tail -3 results/${PROG}_pmp2sdp_json.log | tr '\n' ' ')"; continue; }
    log "$PROG: json SDP $(du -sh $W/sdp_json | cut -f1) ($(grep real results/${PROG}_pmp2sdp_json.log))"
    ( time python3 -m sdpvibe.prepare -s $W/sdp_json --precision $PREC --workers $NP --json results/${PROG}_shape.json ) \
        > results/${PROG}_prepare.log 2>&1 || { log "FAILED $PROG prepare: $(tail -3 results/${PROG}_prepare.log | tr '\n' ' ')"; continue; }
    log "$PROG: shape $(head -1 results/${PROG}_prepare.log)"
    P="--precision $PREC --findPrimalFeasible --findDualFeasible --detectPrimalFeasibleJump --detectDualFeasibleJump"
    ( time $MPI --bind-to core -n $NP $R/usr/local/bin/sdpb -s $SDP -o $W/sdpb_out $P --maxIterations $SDPB_ITS \
        --noFinalCheckpoint --verbosity 1 ) > results/${PROG}_sdpb.log 2>&1 || log "FAILED $PROG sdpb: $(tail -3 results/${PROG}_sdpb.log | tr '\n' ' ')"
    cp $W/sdpb_out/iterations.json results/${PROG}_sdpb_iterations.json 2>/dev/null
    echo "$W/sdp_json $PREC" > results/${PROG}_READY                 # after SDPB: GPU runs get the CPUs
    log "$PROG: SDPB $SDPB_ITS iterations on $NP ranks: $(grep real results/${PROG}_sdpb.log) $(python3 -c "
import json,sys
a=json.load(open('results/${PROG}_sdpb_iterations.json')); t=[r['iter_time'] for r in a]
print('iteration times', t)" 2>&1 | tail -1)"
done
log DONE
[ "${NO_SELFKILL:-0}" = 1 ] && exit 0                # same machine runs vast/gpu_point.sh, which shuts down
for i in $(seq $((HOLD_H * 120))); do [ -f results/PULLED ] && break; sleep 30; done
selfkill "finished"
