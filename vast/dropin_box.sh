#!/usr/bin/env bash
# Run ON a GPU instance (detached) from ~/SDPviBe, with testdata/ uploaded: official SDPB 3.1.0 images,
# then tests/dropin_suite.py (SDPviBe vs SDPB on every case + feature tests). Log: results/dropin.log
# (last line DONE). Waits <= HOLD_H h for results/PULLED, then destroys the instance; hard deadline DEADLINE_H.
cd ~/SDPviBe
DEADLINE_H=${DEADLINE_H:-8}; HOLD_H=${HOLD_H:-1}
[ -n "${CONTAINER_API_KEY:-}" ] || eval "$(tr '\0' '\n' < /proc/1/environ | grep '^CONTAINER_\(ID\|API_KEY\)=' | sed 's/^/export /')"
mkdir -p results; LOG=results/dropin.log
log() { echo "$(date +%T) $*" >> $LOG; }
selfkill() { log "shutting down ($1)"; vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> $LOG 2>&1 \
    || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1; }
( sleep $((DEADLINE_H * 3600)); selfkill "deadline ${DEADLINE_H}h" ) &
log "start: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); $(nproc) cores; $(free -g | awk '/Mem:/{print $2}') GB RAM"
bash vast/remote_setup.sh > results/setup.log 2>&1 || log "FAILED setup: $(tail -3 results/setup.log | tr '\n' ' ')"
bash vast/remote_sdpb_setup.sh > results/sdpb_setup.log 2>&1 || log "FAILED sdpb setup: $(tail -3 results/sdpb_setup.log | tr '\n' ' ')"
R=$PWD/tools/sdpb_root; NP=$(nproc)
export OPAL_PREFIX=$R/usr OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
MPI="$R/usr/bin/mpirun --allow-run-as-root"
log "setup done; running suite"
python3 tests/dropin_suite.py --work runs/dropin --sdpb "$MPI --bind-to core -n $NP $R/usr/local/bin/sdpb" \
    --pmp2sdp "$MPI -n $NP $R/usr/local/bin/pmp2sdp" --spectrum "$MPI -n 1 $R/usr/local/bin/spectrum" \
    --sdpvibe "bin/sdpb" > results/dropin_suite.out 2>&1 || log "FAILED suite: $(tail -5 results/dropin_suite.out | tr '\n' ' ')"
cp runs/dropin/report.json runs/dropin/report.md results/ 2>/dev/null
log "suite finished: $(grep -c '^{' results/dropin_suite.out) result lines"
log DONE
for i in $(seq $((HOLD_H * 120))); do [ -f results/PULLED ] && break; sleep 30; done
selfkill "finished"
