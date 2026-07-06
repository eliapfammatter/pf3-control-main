#!/usr/bin/env bash
# Run any command inside a Linux x86-64 container on Apple Silicon.
#
# The SimSEN FMU ships only a linux64 .so, so it can't load on macOS natively;
# this gives it a real Linux userland (emulated via QEMU on M-series chips).
#
# Usage:
#   ./run-on-linux                                  # interactive shell
#   ./run-on-linux python compare_simulation_models.py --stop-time 60
#   ./run-on-linux pip-compile requirements.in -o requirements-linux.txt
#   ./run-on-linux bash -c 'nc -vz ${RFMI_SERVER_SIMSEN% *} ${RFMI_SERVER_SIMSEN#* }'
#
# Rebuild after editing requirements.in or the Dockerfile:
#   REBUILD=1 ./run-on-linux ...
#
# Ctrl-C: --init (below) runs a real init as PID 1 so a single Ctrl-C is
# forwarded to the process and the container is torn down (--rm). Without it,
# your process runs as PID 1, whose default SIGINT is ignored by the kernel,
# which is why Ctrl-C previously needed spamming.
set -euo pipefail

IMAGE=pf3-fmu

# The linux64 FMU is an RFMI client/proxy; it reads the SimsenRFMIServer address
# from the env var RFMI_SERVER_SIMSEN, formatted as "<IP> <port>" (space-separated,
# NOT a colon). Name is case-sensitive on Linux. See SIMSEN FMI docs, RFMI chapter.
#
# Set it once in your shell (e.g. ~/.zshrc) on the Mac:
#   export RFMI_SERVER_SIMSEN="192.168.64.2 6090"
# This wrapper reads it from your environment and forwards it into the container
if [[ -z "${RFMI_SERVER_SIMSEN:-}" ]]; then
    echo "error: RFMI_SERVER_SIMSEN is not set." >&2
    echo "       export it first, e.g.:" >&2
    echo '       export RFMI_SERVER_SIMSEN="192.168.64.2 6090"' >&2
    exit 1
fi

# Build if the image is missing or REBUILD=1 was set.
if [[ "${REBUILD:-0}" == "1" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo ">> building $IMAGE (linux/amd64)..." >&2
    docker build --platform linux/amd64 -t "$IMAGE" .
fi

# If no command was given, drop into an interactive shell.
if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec docker run --rm -it \
    --init \
    --platform linux/amd64 \
    --add-host host.docker.internal:host-gateway \
    -e MPLBACKEND=Agg \
    -e RFMI_SERVER_SIMSEN \
    -v "$PWD":/work \
    -w /work \
    "$IMAGE" \
    "$@"
