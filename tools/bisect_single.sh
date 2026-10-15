#!/usr/bin/env bash
# bisect the single-correlator upper bound on Delta_eps at fixed Delta_sigma with the GPU solver
DS=$1; LO=${2:-1.3}; HI=${3:-1.6}; STEPS=${4:-7}; NMAX=${5:-6}
source "$(dirname "$0")/sdpb_env.sh"
for i in $(seq $STEPS); do
  MID=$(python3 -c "print(round(($LO+$HI)/2, 6))")
  R=runs/bisect_n${NMAX}_ds${DS}_de${MID}; mkdir -p $R
  .venv/bin/python tools/ising_pmp.py --nmax $NMAX --ds $DS --de $MID --lmax ${LMAX:-30} --out $R/pmp.json > /dev/null
  pmp2sdp --precision 768 -i $R/pmp.json -o $R/sdp --outputFormat json > /dev/null 2>&1
  python -m sdpvibe -s $R/sdp --precision 768 --findPrimalFeasible --findDualFeasible > $R/gpu.log 2>&1
  if grep -q "dual feasible" $R/gpu.log; then HI=$MID; v=excluded; else LO=$MID; v=allowed; fi
  echo "De=$MID $v  -> bound in [$LO, $HI]"
done
