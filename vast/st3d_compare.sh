#!/usr/bin/env bash
# Run ON the instance: take the PMP built by stress-tensors-3d (hyperion object dir), convert it,
# run official SDPB on all cores and the GPU solver with identical parameters, compare.
#   bash vast/st3d_compare.sh <object_dir> [precision] [ranks]
# Writes results/st3d_compare.log; last line is DONE or FAILED.
OBJ=$1; PREC=${2:-1024}; NP=${3:-64}
cd ~/SDPviBe
LOG=results/st3d_compare.log; mkdir -p results
log() { echo "$(date +%T) $*" | tee -a $LOG; }
fail() { log "ERROR: $*"; log FAILED; exit 1; }
# hyperion's (fake) Slurm environment confuses OpenMPI -> drop it
for v in $(env | grep -o '^SLURM[A-Z_]*'); do unset $v; done
R=$PWD/tools/sdpb_root
export OPAL_PREFIX=$R/usr OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
MPI="$R/usr/bin/mpirun --allow-run-as-root"
NSV=$(ls $OBJ/json/obj/*.nsv | head -1); [ -f "$NSV" ] || fail "no nsv in $OBJ"
W=$PWD/runs/st3d_$(basename $OBJ | cut -c1-14); mkdir -p $W
log "object $OBJ -> $W ; nsv $(wc -c < $NSV) bytes"
PARAMS="--precision $PREC --findPrimalFeasible --findDualFeasible --detectPrimalFeasibleJump --detectDualFeasibleJump"
if [ ! -f $W/sdp_bin/control.json ]; then
  ( time $MPI -n $NP $R/usr/local/bin/pmp2sdp --precision $PREC --input $NSV --output $W/sdp_bin ) > $W/pmp2sdp_bin.log 2>&1 || fail "pmp2sdp bin: $(tail -3 $W/pmp2sdp_bin.log)"
fi
if [ ! -f $W/sdp_json/control.json ]; then
  ( time $MPI -n $NP $R/usr/local/bin/pmp2sdp --precision $PREC --input $NSV --output $W/sdp_json --outputFormat json ) > $W/pmp2sdp_json.log 2>&1 || fail "pmp2sdp json: $(tail -3 $W/pmp2sdp_json.log)"
fi
log "pmp2sdp done: $(grep -h 'Processed' $W/pmp2sdp_*.log | head -1)"
python3 - $W/sdp_json >> $LOG <<'EOF'
import json, glob, sys, collections
d = sys.argv[1]
dims = collections.Counter(); pts = []
for f in glob.glob(d + "/block_info_*.json"):
    i = json.load(open(f)); dims[i["dim"]] += 1; pts.append(i["num_points"])
b = json.load(open(d + "/objectives.json"))["b"]
print(f"SDP: N={len(b)}, blocks={sum(dims.values())}, matrix dims {dict(sorted(dims.items()))}, points {min(pts)}..{max(pts)}")
EOF
log "GPU solver"
( time python -m sdpvibe -s $W/sdp_json $PARAMS -o $W/gpu_out ) > $W/gpu.log 2>&1 || fail "gpu: $(tail -5 $W/gpu.log)"
log "GPU: $(grep -E 'read SDP|-----|iterations|real' $W/gpu.log | tr '\n' ' ')"
log "SDPB $NP ranks"
( time $MPI --bind-to core -n $NP $R/usr/local/bin/sdpb -s $W/sdp_bin -o $W/sdpb_out $PARAMS --noFinalCheckpoint --verbosity 1 ) > $W/sdpb.log 2>&1 || fail "sdpb: $(tail -5 $W/sdpb.log)"
log "SDPB: $(grep -E 'terminateReason|real' $W/sdpb.log $W/sdpb_out/out.txt | tr '\n' ' ')"
python3 tools/compare_iterations.py $W/sdpb_out/iterations.json $W/gpu_out/iterations.json | tail -3 >> $LOG
python3 - $W >> $LOG <<'EOF'
import json, sys
w = sys.argv[1]
a = json.load(open(w + "/sdpb_out/iterations.json")); b = json.load(open(w + "/gpu_out/iterations.json"))
ta = sorted(r["iter_time"] for r in a[2:]); tb = sorted(r["iter_time"] for r in b[2:])
print(f"per-iteration (median after warm-up): SDPB {ta[len(ta)//2]:.2f} s on ranks, GPU {tb[len(tb)//2]:.2f} s")
EOF
cp $W/gpu.log $W/sdpb.log results/ 2>/dev/null
log DONE
