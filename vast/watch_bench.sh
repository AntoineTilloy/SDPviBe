#!/usr/bin/env bash
# local: follow results/bench_stream.log on instance $1; on DONE pull results to results/stream_bench/ and
# touch PULLED (the instance then destroys itself). Exits if the instance disappears.
ID=$1; LOGF=${2:-bench_stream.log}; DEST=${3:-results/stream_bench}; cd "$(dirname "$0")/.."
seen=0
while true; do
  URL=$(~/.local/bin/vastai ssh-url "$ID" 2>/dev/null) || { echo "instance gone"; exit 1; }
  HP=${URL#ssh://}; H=${HP%:*}; PT=${HP##*:}
  out=$(ssh -p $PT -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new $H "cat SDPviBe/results/$LOGF" 2>/dev/null)
  if [ -n "$out" ]; then
    n=$(echo "$out" | wc -l)
    [ $n -gt $seen ] && echo "$out" | tail -n $((n - seen)) && seen=$n
    if echo "$out" | grep -q "DONE\|shutting down"; then
      mkdir -p $DEST
      ssh -p $PT $H 'tar czf - -C SDPviBe results' 2>/dev/null | tar xzf - -C $DEST
      ssh -p $PT $H 'touch SDPviBe/results/PULLED' 2>/dev/null
      echo "PULLED"; exit 0
    fi
  else
    ~/.local/bin/vastai show instance $ID --raw >/dev/null 2>&1 || { echo "instance gone"; exit 1; }
  fi
  sleep 120
done
