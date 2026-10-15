#!/usr/bin/env bash
# Run ON the instance (as root) from ~/SDPviBe: unpack official SDPB 3.1.0 + scalar_blocks images and
# make their musl binaries runnable natively (so mpirun can start orted / sdpb ranks):
#   /lib/ld-musl-x86_64.so.1 -> unpacked loader, library search path in /etc/ld-musl-x86_64.path
set -euo pipefail
cd ~/SDPviBe
[ -x tools/sdpb_root/usr/local/bin/sdpb ] || python3 tools/pull_image.py bootstrapcollaboration/sdpb 3.1.0-amd64 tools/sdpb_root
[ -x tools/sb_root/usr/local/bin/scalar_blocks ] || python3 tools/pull_image.py bootstrapcollaboration/scalar_blocks master tools/sb_root
R=$PWD/tools/sdpb_root
ln -sf $R/lib/ld-musl-x86_64.so.1 /lib/ld-musl-x86_64.so.1
printf '%s\n' "$R/usr/local/lib" "$R/usr/lib" "$R/lib" "$PWD/tools/sb_root/usr/local/lib" "$PWD/tools/sb_root/usr/lib" > /etc/ld-musl-x86_64.path
cat > tools/sdpb_env_native.sh <<EOF
export OPAL_PREFIX=$R/usr PATH=$R/usr/local/bin:\$PATH:$R/usr/bin
export OMPI_MCA_plm_rsh_agent= OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpisdpb() { $R/usr/bin/mpirun --allow-run-as-root --bind-to core -n "\$1" $R/usr/local/bin/sdpb "\${@:2}"; }
mpipmp2sdp() { $R/usr/bin/mpirun --allow-run-as-root -n "\$1" $R/usr/local/bin/pmp2sdp "\${@:2}"; }
EOF
. tools/sdpb_env_native.sh
nproc
$R/usr/bin/mpirun --allow-run-as-root -n 2 $R/usr/local/bin/sdpb --version 2>&1 | head -3
tools/sb_root/usr/local/bin/scalar_blocks --version 2>&1 | head -2
