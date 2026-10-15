#!/usr/bin/env bash
# Run ON the instance (root). Installs minimal stand-ins for the Slurm commands hyperion uses
# (single machine, jobs run as local background processes), and passwordless ssh to localhost.
# A real slurmd cannot run here: the container's cgroup filesystem is read-only.
set -euo pipefail
for p in $(pgrep -x slurmctld) $(pgrep -x slurmd); do kill $p; done 2>/dev/null || true
mkdir -p /var/lib/fakeslurm
HN=$(hostname)

cat > /usr/local/bin/sbatch <<'EOF'
#!/usr/bin/env bash
# fake sbatch: script on stdin, options as sbatch; runs the job locally in the background
nodes=1; ntpn=1; out=""; dir=$PWD; name=job
while [ $# -gt 0 ]; do
  case "$1" in
    --nodes) nodes=$2; shift;; --nodes=*) nodes=${1#*=};;
    --ntasks-per-node) ntpn=$2; shift;; --ntasks-per-node=*) ntpn=${1#*=};;
    --output) out=$2; shift;; --output=*) out=${1#*=};;
    -D|--chdir) dir=$2; shift;; --chdir=*) dir=${1#*=};;
    --job-name) name=$2; shift;; --job-name=*) name=${1#*=};;
    --time|--mem|--mail-type|--mail-user|--partition|--constraint|--account|--qos) shift;;
    *) ;;
  esac; shift
done
d=/var/lib/fakeslurm
id=$(( $(cat $d/last 2>/dev/null || echo 1000) + 1 )); echo $id > $d/last
script=$d/job_$id.sh; cat > $script; chmod +x $script
out=${out:-$dir/slurm-%j.out}; out=${out//%j/$id}; out=${out//%x/$name}
mkdir -p "$(dirname "$out")" "$dir"
HN=$(hostname)
cd "$dir"
# NB: no subshell around the background job -- it would hold sbatch's stdout pipe open until the
# job ends, and hyperion blocks reading sbatch's output.
SLURM_JOB_ID=$id SLURM_JOBID=$id SLURM_JOB_NAME=$name SLURM_JOB_NODELIST=$HN SLURM_NODELIST=$HN \
  SLURM_JOB_NUM_NODES=$nodes SLURM_NNODES=$nodes SLURM_NTASKS=$((nodes*ntpn)) SLURM_NTASKS_PER_NODE=$ntpn \
  SLURMD_NODENAME=$HN SLURM_CPUS_ON_NODE=$(nproc) setsid nohup bash $script > "$out" 2>&1 < /dev/null &
echo $! > $d/job_$id.pid
echo "$(date +%T) job $id ($name) nodes=$nodes ntpn=$ntpn out=$out" >> $d/log
echo "Submitted batch job $id"
EOF

cat > /usr/local/bin/scontrol <<'EOF'
#!/usr/bin/env bash
if [ "$1" = show ] && [ "$2" = hostnames ]; then
  for h in ${3:-$SLURM_JOB_NODELIST}; do echo "$h"; done | tr ',' '\n'
else echo "fake scontrol: unsupported: $*" >&2; fi
EOF

cat > /usr/local/bin/scancel <<'EOF'
#!/usr/bin/env bash
for id in "$@"; do p=$(cat /var/lib/fakeslurm/job_$id.pid 2>/dev/null) && kill -- -$p 2>/dev/null; done; true
EOF

cat > /usr/local/bin/squeue <<'EOF'
#!/usr/bin/env bash
echo "JOBID NAME STATE"
for f in /var/lib/fakeslurm/job_*.pid; do [ -f "$f" ] || continue; id=${f##*job_}; id=${id%.pid}
  kill -0 $(cat $f) 2>/dev/null && echo "$id job RUNNING"; done
EOF
chmod +x /usr/local/bin/sbatch /usr/local/bin/scontrol /usr/local/bin/scancel /usr/local/bin/squeue

# passwordless ssh to this machine (hyperion starts workers with: ssh -f <host> sh -c 'nohup ... &')
mkdir -p ~/.ssh && chmod 700 ~/.ssh
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub > ~/.ssh/authorized_keys2      # vast rewrites authorized_keys
chmod 600 ~/.ssh/authorized_keys2
grep -q "StrictHostKeyChecking" ~/.ssh/config 2>/dev/null || printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n' >> ~/.ssh/config
grep -q " $HN\$" /etc/hosts || echo "127.0.1.1 $HN" >> /etc/hosts
# the hostname must resolve to ONE address (docker also maps it to the container IP; hyperion's
# master and workers otherwise disagree about where to connect)
grep -v "^172\.17\." /etc/hosts > /tmp/hosts.new && cat /tmp/hosts.new > /etc/hosts
ssh -o BatchMode=yes $HN 'echo ssh-to-self ok'
echo "Submitted check: $(printf 'echo hi $SLURM_NTASKS\n' | sbatch --nodes 1 --ntasks-per-node 4 --output /tmp/fs_test.out)"; sleep 1; cat /tmp/fs_test.out
scontrol show hostnames $HN
