#!/usr/bin/env bash
# run a binary from an unpacked Alpine image root through its musl loader:
#   tools/run_in_image.sh <root> <binary> args...
R=$(cd "$1" && pwd); shift; B=$1; shift
[ -x "$B" ] || B=$R/usr/local/bin/$B
exec "$R/lib/ld-musl-x86_64.so.1" --library-path "$R/usr/local/lib:$R/usr/lib:$R/lib" "$B" "$@"
