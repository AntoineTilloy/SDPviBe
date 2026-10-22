#!/usr/bin/env bash
# Run ON the instance (as root) from ~/SDPviBe after remote_setup.sh + remote_sdpb_setup.sh.
# Prepares the stress-tensors-3d pipeline (Simmons-Duffin et al.) on a single machine:
#   system libs, Haskell stack, single-node Slurm (hyperion needs it), blocks_3d binary,
#   a 'vast' hyperion config whose scripts call the unpacked official binaries.
set -uo pipefail
cd ~/SDPviBe
export DEBIAN_FRONTEND=noninteractive
log() { echo "[$(date +%T)] $*"; }

log "apt packages"
apt-get update -qq
apt-get install -y -qq slurm-wlm munge libmpfr-dev libgmp-dev libopenblas-dev liblapack-dev zlib1g-dev \
    libtinfo-dev libncurses-dev pkg-config git curl xz-utils build-essential libffi-dev libnuma-dev \
    qhull-bin time netbase > /tmp/apt.log 2>&1 || { tail -20 /tmp/apt.log; }
# some Haskell deps link -lblas -llapack
ln -sf /usr/lib/x86_64-linux-gnu/libopenblas.so /usr/local/lib/libblas.so 2>/dev/null || true

# NB: stack treats the machine as musl-based if /lib/ld-musl-x86_64.so.1 exists (installed by
# remote_sdpb_setup.sh for the official SDPB binaries) and then fetches a musl GHC that fails.
# Remove it for the Haskell build; restore it afterwards with:
#   ln -sf ~/SDPviBe/tools/sdpb_root/lib/ld-musl-x86_64.so.1 /lib/ld-musl-x86_64.so.1
rm -f /lib/ld-musl-x86_64.so.1
log "stack"
command -v stack > /dev/null || curl -sSL https://get.haskellstack.org/ | sh > /tmp/stack_install.log 2>&1
stack --version

log "blocks_3d image"
[ -x tools/b3d_root/usr/local/bin/blocks_3d ] || python3 tools/pull_image.py bootstrapcollaboration/blocks_3d master tools/b3d_root
B3=$PWD/tools/b3d_root
echo "$B3/usr/local/lib" >> /etc/ld-musl-x86_64.path; echo "$B3/usr/lib" >> /etc/ld-musl-x86_64.path
ls $B3/usr/local/bin
# musl's malloc makes the official (Alpine) blocks_3d image 20-40x slower with threads
# (docs/BLOCKS_PLAN.md); preload Alpine's jemalloc (identical outputs)
JE=$PWD/tools/apk/usr/lib/libjemalloc.so.2
[ -f $JE ] || { mkdir -p tools/apk && curl -s https://dl-cdn.alpinelinux.org/alpine/v3.18/main/x86_64/jemalloc-5.3.0-r2.apk \
    -o tools/apk/jemalloc.apk && tar -xzf tools/apk/jemalloc.apk -C tools/apk 2> /dev/null; }

log "single-node slurm: NOT possible in vast containers (read-only cgroupfs) -> use vast/fake_slurm.sh"
if false; then
HN=$(hostname); NCPU=$(nproc); MEMMB=$(( $(free -m | awk '/Mem:/{print $2}') * 9 / 10 ))
grep -q "$HN" /etc/hosts || echo "127.0.1.1 $HN" >> /etc/hosts
mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /run/munge
cat > /etc/slurm/slurm.conf <<EOF
ClusterName=vast
SlurmctldHost=$HN
AuthType=auth/munge
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
SelectType=select/cons_tres
SelectTypeParameters=CR_Core
SchedulerType=sched/backfill
ReturnToService=2
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
SlurmdSpoolDir=/var/spool/slurmd
StateSaveLocation=/var/spool/slurmctld
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
MpiDefault=none
JobAcctGatherType=jobacct_gather/none
NodeName=$HN CPUs=$NCPU RealMemory=$MEMMB State=UNKNOWN
PartitionName=debug Nodes=$HN Default=YES MaxTime=INFINITE State=UP OverSubscribe=YES
EOF
echo "CgroupPlugin=disabled" > /etc/slurm/cgroup.conf
chown -R munge:munge /etc/munge /var/lib/munge /var/log/munge /run/munge 2>/dev/null
[ -f /etc/munge/munge.key ] || /usr/sbin/mungekey -c 2>/dev/null || dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null
chown munge:munge /etc/munge/munge.key; chmod 400 /etc/munge/munge.key
pgrep munged > /dev/null || su -s /bin/sh munge -c /usr/sbin/munged
pgrep slurmctld > /dev/null || slurmctld
pgrep slurmd > /dev/null || slurmd
sleep 3; sinfo; srun -n 2 hostname
fi

log "hyperion config 'vast'"
ST=$PWD/ext/stress-tensors-3d
C=$ST/hyperion-config/vast
mkdir -p $C /root/hyp
R=$PWD/tools/sdpb_root
cat > $C/Config.hs <<'EOF'
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell   #-}
module Config where
import           Data.FileEmbed (makeRelativeToProject, strToExp)
import           Hyperion
scriptsDir :: FilePath
scriptsDir = $(makeRelativeToProject "vast" >>= strToExp)
config :: HyperionConfig
config = (defaultHyperionConfig "/root/hyp") { emailAddr = Nothing }
staticConfig :: HyperionStaticConfig
staticConfig = defaultHyperionStaticConfig
EOF
NP=${SDPB_RANKS:-$(nproc)}
cat > $C/blocks_3d.sh <<EOF
#!/bin/bash
LD_PRELOAD=$JE exec $B3/usr/local/bin/blocks_3d "\$@"
EOF
cat > $C/scalar_blocks.sh <<EOF
#!/bin/bash
LD_PRELOAD=$JE exec $PWD/tools/sb_root/usr/local/bin/scalar_blocks "\$@"
EOF
for s in pmp2sdp srun_pmp2sdp; do cat > $C/$s.sh <<EOF
#!/bin/bash
export OPAL_PREFIX=$R/usr OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
echo pmp2sdp "\$@" >> /root/hyp/pmp2sdp_calls.log
$R/usr/bin/mpirun --allow-run-as-root -n $NP $R/usr/local/bin/pmp2sdp "\$@"
EOF
done
# SDPB_WRAPPER=sdpvibe: hyperion's sdpb calls go to SDPviBe (bin/sdpb); block: hyperion only builds the SDP,
# its sdpb call is logged and parked (solved outside hyperion, see vast/st3d_point.sh); default: official SDPB
for s in sdpb srun_sdpb; do
if [ "${SDPB_WRAPPER:-run}" = sdpvibe ]; then cat > $C/$s.sh <<EOF
#!/bin/bash
# SDPviBe on the GPU, same command line as sdpb (no MPI needed)
for v in \$(env | grep -o '^SLURM[A-Z_]*'); do unset \$v; done
echo "\$(date +%s) sdpvibe \$@" >> /root/hyp/sdpb_calls.log
exec $PWD/bin/sdpb "\$@"
EOF
elif [ "${SDPB_WRAPPER:-run}" = block ]; then cat > $C/$s.sh <<EOF
#!/bin/bash
echo "\$(date +%s) sdpb \$@" >> /root/hyp/sdpb_calls.log
exec sleep infinity
EOF
else cat > $C/$s.sh <<EOF
#!/bin/bash
export OPAL_PREFIX=$R/usr OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
echo sdpb "\$@" >> /root/hyp/sdpb_calls.log
$R/usr/bin/mpirun --allow-run-as-root --bind-to core -n $NP $R/usr/local/bin/sdpb "\$@"
EOF
fi
done
cat > $C/qdelaunay.sh <<'EOF'
#!/bin/bash
qdelaunay "$@"
EOF
chmod +x $C/*.sh
cd $ST && python3 configure.py -c hyperion-config/vast > /tmp/configure.log 2>&1; tail -3 /tmp/configure.log
log "setup done"
