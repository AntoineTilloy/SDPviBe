#!/usr/bin/env bash
# Run ON the bench box after vast/bench_vs_sdpb.sh: SDPB again on the synthetic SDPs (regenerated, same seed)
# whose run failed (MPI shared window larger than /dev/shm), with --maxSharedMemory; compares with the
# existing SDPviBe runs runs/synth_n<n>.gpu.  Log: results/bench_rerun.log (last line DONE).
cd ~/SDPviBe
R=$PWD/tools/sdpb_root; PREC=${PREC:-832}
export OPAL_PREFIX=$R/usr OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
MPI="$R/usr/bin/mpirun --allow-run-as-root"; LOG=results/bench_rerun.log
log() { echo "$(date +%T) $*" >> $LOG; }
mt() { python3 -c "
import json,sys; a=json.load(open(sys.argv[1])); t=[r['iter_time'] for r in a]; print('iteration times', t, 'median(2+):', sorted(t[1:])[len(t[1:])//2] if len(t)>1 else t)" $1 2>&1; }
for n in ${SIZES:-8 10}; do
    D=runs/synth_n$n; its=$([ $n -ge 10 ] && echo 2 || echo 3)
    python3 tools/synth_sdp.py --nmax $n --precision $PREC --out $D > results/synth_n${n}_rerun.log 2>&1 || { log "FAILED synth n$n"; continue; }
    ( time timeout 5400 $MPI --bind-to core $R/usr/local/bin/sdpb -s $D --precision $PREC --maxIterations $its -o $D.sdpb \
        --maxSharedMemory ${SDPB_MSM:-16G} -c $D.sdpb.ck --noFinalCheckpoint --verbosity 1 ) > results/n${n}_sdpb_rerun.log 2>&1
    log "n$n SDPB ($(nproc) threads, maxSharedMemory ${SDPB_MSM:-16G}): $(mt $D.sdpb/iterations.json); $(grep real results/n${n}_sdpb_rerun.log)"
    log "n$n SDPviBe (earlier run): $(mt $D.gpu/iterations.json)"
    python3 tools/compare_iterations.py $D.sdpb/iterations.json $D.gpu/iterations.json 2>&1 | tail -2 | tr '\n' ' ' >> $LOG; echo >> $LOG
    rm -rf $D
done
log DONE
