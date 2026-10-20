#!/usr/bin/env bash
# Run ON the drop-in box after tests/dropin_suite.py (uses its SDPs in runs/dropin): SDPB's approx_objective
# on solutions written by sdpb and by SDPviBe (text x, y, X, Y), Ising nmax=6 solved to optimality, new SDP =
# the same structure at another Delta_epsilon.  Log: results/approx.log (last line DONE).
cd ~/SDPviBe
R=$PWD/tools/sdpb_root
export OPAL_PREFIX=$R/usr OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
MPI="$R/usr/bin/mpirun --allow-run-as-root"
W=runs/approx; mkdir -p $W results; LOG=results/approx.log
SDP=runs/dropin/ising_n6_allowed/sdp; NEW=runs/dropin/ising_n6_excluded/sdp
for tag in sdpb sdpvibe; do
    exe=$([ $tag = sdpb ] && echo "$MPI --bind-to core $R/usr/local/bin/sdpb" || echo "bin/sdpb")
    $exe -s $SDP -o $W/out_$tag -c $W/ck_$tag --precision 768 --writeSolution=x,y,X,Y --noFinalCheckpoint > $W/$tag.log 2>&1
    echo "$tag: $(head -1 $W/out_$tag/out.txt)" >> $LOG
    $MPI -n 8 $R/usr/local/bin/approx_objective --sdp $SDP --newSdp $NEW --solutionDir $W/out_$tag --precision 768 \
        > $W/approx_$tag.json 2> $W/approx_$tag.log
    echo "approx_objective on $tag solution: rc=$? $(tr -d '\n ' < $W/approx_$tag.json | cut -c1-300)" >> $LOG
done
python3 - $W >> $LOG 2>&1 <<'EOF'
import json, re, sys, mpmath as mp
mp.mp.prec = 1200
W = sys.argv[1]
def load(t):
    s = open(f"{W}/approx_{t}.json").read()
    return json.loads(s[s.index("["):])
a, b = load("sdpb"), load("sdpvibe")
for x, y in zip(a, b):
    for k in x:
        if k == "path": continue
        u, v = mp.mpf(x[k]), mp.mpf(y[k])
        d = 99.0 if u == v else float(-mp.log10(abs(u - v) / max(abs(u), abs(v))))
        print(f"  {k}: {mp.nstr(u, 12)} vs {mp.nstr(v, 12)}: {d:.1f} digits")
EOF
echo DONE >> $LOG
