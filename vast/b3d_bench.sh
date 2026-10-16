#!/usr/bin/env bash
# Run ON the instance after a stress-tensors-3d point has been built (vast/st3d_point.sh): time single
# blocks_3d tasks at the parameters of larger runs, to extrapolate the block cost of nmax=14/22/26.
# Uses parameter files that hyperion wrote for the point (one per four-point structure family) as
# templates, and replaces lambda/order/kept-pole-order/precision and the spin list.
#   bash vast/b3d_bench.sh  -> results/b3d_bench.log (one line per run)
cd ~/SDPviBe
B3=$PWD/tools/b3d_root/usr/local/bin/blocks_3d
LOG=results/b3d_bench.log
T=${THREADS:-32}
SPINS=${SPINS:-"0,20,40"}
OBJ=$(ls -d /root/hyp/data/*/*/Object_* | head -1)
pick() {   # first param file whose j-external matches
    for f in $OBJ/blocks/*/params*.ini /root/hyp/data/*/*/Object_*/blocks/*/params*.ini; do
        [ -f "$f" ] && grep -q "^j-external=$1\$" "$f" && { echo "$f"; return; }
    done
}
mkdir -p /tmp/b3d
for FAM in "2,2,2,2" "2,2,2,0"; do
    TPL=$(pick "$FAM")
    [ -n "$TPL" ] || { echo "$(date +%T) no template for j-external=$FAM" >> $LOG; continue; }
    for cfg in "19 60 512" "27 106 832" "43 130 832" "51 152 896"; do
        set -- $cfg; LAM=$1; ORD=$2; PREC=$3
        P=/tmp/b3d/p_${FAM//,/}_$LAM.ini; O=/tmp/b3d/o_${FAM//,/}_$LAM; mkdir -p $O
        sed -e "s/^lambda=.*/lambda=$LAM/" -e "s/^order=.*/order=$ORD/" -e "s/^kept-pole-order=.*/kept-pole-order=$ORD/" \
            -e "s/^precision=.*/precision=$PREC/" -e "s/^num-threads=.*/num-threads=$T/" -e "s/^j-internal=.*/j-internal=$SPINS/" \
            -e "s#^out-template=.*#out-template=$O/spin_{:.1f}.json#" $TPL > $P
        t0=$(date +%s.%N)
        timeout ${TMAX:-3600} $B3 --param-file $P > $O.log 2>&1; rc=$?
        t1=$(date +%s.%N)
        POLES=$(python3 -c "
import json, glob
f = sorted(glob.glob('$O/spin_*.json'))
print(len(json.load(open(f[0]))['pole_list_x']) if f else -1)" 2>/dev/null)
        echo "$(date +%T) j-ext $FAM lambda $LAM order $ORD prec $PREC spins $SPINS threads $T: rc $rc, $(python3 -c "print(round($t1-$t0,1))") s wall, $(du -sh $O | cut -f1) output, poles $POLES" >> $LOG
    done
done
echo "$(date +%T) DONE" >> $LOG
