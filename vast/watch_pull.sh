#!/usr/bin/env bash
# local: every PERIOD s pull results/ (and extra paths) from instance $1 into $2; print new lines of the
# log $3 (relative to ~/SDPviBe/results); on DONE/shutdown: final pull, touch PULLED, exit.
ID=$1; DEST=$2; LOGF=$3; PERIOD=${PERIOD:-1200}; cd "$(dirname "$0")/.."
seen=0; mkdir -p $DEST
while true; do
  URL=$(~/.local/bin/vastai ssh-url "$ID" 2>/dev/null) || { echo "instance gone"; exit 1; }
  HP=${URL#ssh://}; H=${HP%:*}; PT=${HP##*:}
  ssh -p $PT -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new $H "tar czf - -C SDPviBe --exclude=sdp --exclude=*.ck --exclude=sdp.zip --exclude=\"*.pt\" results ${EXTRA:-}" 2>/dev/null | tar xzf - -C $DEST 2>/dev/null
  if [ -f $DEST/results/$LOGF ]; then
    n=$(wc -l < $DEST/results/$LOGF); [ $n -gt $seen ] && tail -n $((n - seen)) $DEST/results/$LOGF && seen=$n
    if grep -q "DONE\|shutting down" $DEST/results/$LOGF; then
      [ "${NO_PULLED:-0}" = 1 ] && { echo "DONE (not signalling PULLED)"; exit 0; }
      ssh -p $PT $H 'touch SDPviBe/results/PULLED' 2>/dev/null; echo PULLED; exit 0
    fi
  fi
  ~/.local/bin/vastai show instance $ID --raw > /dev/null 2>&1 || { echo "instance gone"; exit 1; }
  sleep $PERIOD
done
