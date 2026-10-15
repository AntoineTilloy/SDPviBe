# source this: functions pmp2sdp / sdpb running the official SDPB 3.1.0 binaries from the unpacked image
_T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OPAL_PREFIX=$_T/sdpb_root/usr OMPI_MCA_ess_singleton_isolated=1
pmp2sdp() { "$_T/run_in_image.sh" "$_T/sdpb_root" pmp2sdp "$@"; }
sdpb() { "$_T/run_in_image.sh" "$_T/sdpb_root" sdpb "$@"; }
