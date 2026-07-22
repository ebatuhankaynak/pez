#!/usr/bin/env bash
# Thin wrapper around `docker run` for the pezevid transition pipeline.
# Usage (pipeline stages live in src/; build_report.py + serve.py are at the root):
#   ./docker-run.sh src/detect_transitions.py --qa
#   ./docker-run.sh src/split_clips.py --workers 8
#   ./docker-run.sh build_report.py --out report.html
#
# Add GPU=0 to force CPU:  GPU=0 ./docker-run.sh src/detect_transitions.py --limit 5
set -euo pipefail

IMAGE="${IMAGE:-pezevid-transitions:latest}"
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU="${GPU:-1}"

GPU_FLAG=()
if [[ "$GPU" == "1" ]]; then GPU_FLAG=(--gpus all); fi

# Run as the host user so outputs aren't root-owned; mount the whole repo (live code +
# data), with the source videos re-mounted read-only on top.
exec docker run --rm "${GPU_FLAG[@]}" --user "$(id -u):$(id -g)" \
  -v "$HERE:/app" \
  -v "$HERE/freckled_spike_tiktok:/app/freckled_spike_tiktok:ro" \
  "$IMAGE" "$@"
