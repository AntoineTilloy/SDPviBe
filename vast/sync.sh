#!/usr/bin/env bash
# Copy the project to / results from a running vast instance.
#   bash vast/sync.sh push <instance_id>     # code -> instance:~/SDPviBe
#   bash vast/sync.sh pull <instance_id>     # instance:~/SDPviBe/results -> ./results
#   bash vast/sync.sh ssh  <instance_id>     # interactive shell
set -euo pipefail
cd "$(dirname "$0")/.."
ACT=$1; ID=$2
URL=$(~/.local/bin/vastai ssh-url "$ID")             # ssh://root@host:port
HOSTPORT=${URL#ssh://}; HOST=${HOSTPORT%:*}; PORT=${HOSTPORT##*:}
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
case $ACT in
  push) tar czf - --exclude .venv --exclude archive --exclude __pycache__ --exclude results --exclude ./runs --exclude ./data --exclude ./tools/sdpb_root --exclude ./tools/sb_root --exclude ./tools/sdpb_src --exclude ./ext --exclude ./tools/b3d_root \
            --exclude 'vast/o_*.json' --exclude 'vast/offers_*' --exclude cpu_baseline/mpf_baseline . |
        $SSH "$HOST" 'mkdir -p SDPviBe && tar xzf - -C SDPviBe' ;;   # tar: images may lack rsync
  pull) mkdir -p results && $SSH "$HOST" 'tar czf - -C SDPviBe results' | tar xzf - ;;
  ssh)  $SSH "$HOST" ;;
esac
