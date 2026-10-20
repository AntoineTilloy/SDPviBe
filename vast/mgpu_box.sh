#!/usr/bin/env bash
# Run ON a multi-GPU instance (detached) from ~/SDPviBe, with SDPs uploaded to testdata/sdp_<name>/:
# multi-GPU checks of SDPviBe (--gpus 1 vs 2 on real SDPs, resident and streamed/lean; 2-GPU checkpoint
# resumed on 1 GPU) and per-iteration times on synthetic SDPs.  Log: results/mgpu.log (last line DONE).
# Waits <= HOLD_H h for results/PULLED, then destroys the instance; hard deadline DEADLINE_H.
cd ~/SDPviBe
DEADLINE_H=${DEADLINE_H:-2}; HOLD_H=${HOLD_H:-0.5}
[ -n "${CONTAINER_API_KEY:-}" ] || eval "$(tr '\0' '\n' < /proc/1/environ | grep '^CONTAINER_\(ID\|API_KEY\)=' | sed 's/^/export /')"
mkdir -p results runs/mgpu; LOG=results/mgpu.log
log() { echo "$(date +%T) $*" >> $LOG; }
selfkill() { log "shutting down ($1)"; vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> $LOG 2>&1 \
    || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1; }
( sleep $(python3 -c "print(int($DEADLINE_H * 3600))"); selfkill "deadline ${DEADLINE_H}h" ) &
log "start: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ';') $(nproc) cores; $(free -g | awk '/Mem:/{print $2}') GB RAM"
nvidia-smi topo -m >> results/mgpu_topo.txt 2>&1
bash vast/remote_setup.sh > results/setup.log 2>&1 || log "FAILED setup: $(tail -3 results/setup.log | tr '\n' ' ')"
. ~/venv/bin/activate
W=runs/mgpu
sv() {  # tag gpus sdp extra-args...
    local tag=$1 g=$2 sdp=$3; shift 3
    local t0=$(date +%s)
    python3 -m sdpvibe -s $sdp -o $W/$tag -c $W/$tag.ck --precision 768 --gpus $g --verbosity 2 "$@" > $W/$tag.log 2>&1
    log "$tag (gpus=$g): rc=$? $(( $(date +%s) - t0 )) s; $(grep -m1 'chunks' $W/$tag.log | sed 's/^\s*//' | cut -c1-220); $(grep -m1 'terminateReason' $W/$tag/out.txt 2>/dev/null)"
}
cmp_it() {  # a b
    python3 - "$W/$1/iterations.json" "$W/$2/iterations.json" >> $LOG 2>&1 <<'EOF'
import json, sys, mpmath as mp
mp.mp.prec = 1200
a, b = (json.load(open(f)) for f in sys.argv[1:3])
ks = ["mu", "P-obj", "D-obj", "P-step", "D-step", "beta"]
def dig(x, y):
    x, y = mp.mpf(x), mp.mpf(y)
    return 99.0 if x == y else float(-mp.log10(abs(x - y) / max(abs(x), abs(y))))
per = [min(dig(x[k], y[k]) for k in ks) for x, y in zip(a, b)]
same = sum(all(x[k] == y[k] for k in ks) for x, y in zip(a, b))
t = lambda r: sum(float(x["iter_time"]) for x in r[2:]) / max(1, len(r) - 2)
print(f"  compare {sys.argv[1].split('/')[-2]} vs {sys.argv[2].split('/')[-2]}: iterations {len(a)} / {len(b)}, "
      f"bitwise-equal rows {same}, worst digits {min(per):.1f}, median {sorted(per)[len(per) // 2]:.1f}; "
      f"s/it {t(a):.3f} / {t(b):.3f}")
EOF
}
F="--findPrimalFeasible --findDualFeasible --detectPrimalFeasibleJump --detectDualFeasibleJump --noFinalCheckpoint"
log "setup done"
for c in mixed_n6_in mixed_n10_ising; do
    S=testdata/sdp_$c
    sv ${c}_g1 1 $S $F; sv ${c}_g2 2 $S $F; cmp_it ${c}_g1 ${c}_g2
    SDPB_GPU_CHUNK_GB=0.3 sv ${c}_lean_g1 1 $S $F; SDPB_GPU_CHUNK_GB=0.3 sv ${c}_lean_g2 2 $S $F; cmp_it ${c}_lean_g1 ${c}_lean_g2
done
# checkpoint written on 2 GPUs, resumed on 1 GPU (and the 2-GPU run continued) -> same as the 2-GPU run to the end
S=testdata/sdp_mixed_n6_in
sv ck_a 2 $S $F --maxIterations 20 --writeSolution x,y
rm -rf $W/ck_b.ck; cp -r $W/ck_a.ck $W/ck_b.ck
sv ck_b 1 $S $F -i $W/ck_b.ck
python3 - $W >> $LOG 2>&1 <<'EOF'
import json, sys
W = sys.argv[1]
a = json.load(open(f"{W}/mixed_n6_in_g2/iterations.json")); b = json.load(open(f"{W}/ck_b/iterations.json"))
print(f"  resume on 1 GPU from 2-GPU checkpoint: {len(b)} more iterations (2-GPU run: {len(a)} total); "
      f"final out.txt: {open(W + '/ck_b/out.txt').read().splitlines()[0]}")
EOF
# synthetic per-iteration times
for n in 8 10; do
    python3 tools/synth_sdp.py --nmax $n --precision 768 --out $W/synth_n$n >> $LOG 2>&1
    sv synth_n${n}_g1 1 $W/synth_n$n --maxIterations 4 --noFinalCheckpoint
    sv synth_n${n}_g2 2 $W/synth_n$n --maxIterations 4 --noFinalCheckpoint
    cmp_it synth_n${n}_g1 synth_n${n}_g2
    rm -rf $W/synth_n$n
done
log "DONE"
t=0; while [ ! -f results/PULLED ] && [ $t -lt $(python3 -c "print(int($HOLD_H * 3600))") ]; do sleep 60; t=$((t + 60)); done
selfkill "finished"
