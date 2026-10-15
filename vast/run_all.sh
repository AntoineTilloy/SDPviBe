#!/usr/bin/env bash
# Run ON the instance (detached) from ~/SDPviBe.  Does everything, then shuts the instance down:
#   1. nmax=14 mixed-Ising blocks + PMP on the CPUs   (background)
#   2. nmax=6 island scan on the GPU                   (meanwhile)
#   3. nmax=14: GPU solver, then SDPB on all cores (20 iterations)  -- clean timings
#   4. touch results/DONE; wait (<= 2 h) for the client to pull and `touch results/PULLED`;
#      then destroy the instance (fallback: stop).  Hard deadline: DEADLINE_H hours overall.
cd ~/SDPviBe
CORES=${CORES:-$(nproc)}; DEADLINE_H=${DEADLINE_H:-7}
mkdir -p results
selfkill() {
    echo "$(date) shutting down ($1)" >> results/run_all.log
    pip install -q --break-system-packages vastai > /dev/null 2>&1
    vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> results/run_all.log 2>&1 \
      || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> results/run_all.log 2>&1
}
( sleep $((DEADLINE_H * 3600)); selfkill "deadline ${DEADLINE_H}h" ) &
WATCHDOG=$!
. tools/sdpb_env_native.sh
R14=runs/mixed_n14_0.5182_1.4127
mkdir -p $R14
echo "$(date) start, $CORES cores" >> results/run_all.log
( time python tools/ising_mixed_pmp.py --nmax 14 --lmax 50 --order 90 --poles 24 --prec 800 --ds 0.5182 --de 1.4127 \
    --out $R14/pmp.json ) > $R14/gen.log 2>&1 &
GEN=$!
python tools/scan_mixed.py --nmax 6 --deltas=-0.885,-0.895,-0.905,-0.915 --ds 0.516:0.523:0.001 \
    --out results/scan_n6.csv > results/scan_n6.log 2>&1
echo "$(date) scan done" >> results/run_all.log
wait $GEN
echo "$(date) n14 PMP done: $(tail -3 $R14/gen.log | tr '\n' ' ')" >> results/run_all.log
bash vast/scale_test.sh 14 0.5182 1.4127 50 $CORES 896 20 > results/scale14.log 2>&1
echo "$(date) scale test done" >> results/run_all.log
touch results/DONE
for i in $(seq 1 120); do [ -f results/PULLED ] && break; sleep 60; done
kill $WATCHDOG 2>/dev/null
selfkill "finished"
