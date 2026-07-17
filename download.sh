#!/usr/bin/env bash
# Dockerized instaloader wrapper. Builds a tiny fetcher image on first use, then
# runs instaloader with the repo mounted at /data so downloaded profile folders
# land straight in the repo (matching the freckled_spike_tiktok/ layout).
#
# Usage (anonymous = just omit --login; reels only, clean mp4-only folders):
#   ./download.sh --reels --no-pictures --no-metadata-json --no-captions \
#       --dirname-pattern '{profile}_instagram' PROFILE
#
# Any instaloader flag/arg passes straight through.
set -euo pipefail

IMAGE="${IMAGE:-pezevenk-instaloader:latest}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -f "$HERE/Dockerfile.download" -t "$IMAGE" "$HERE"
fi

# Run as the host user (outputs aren't root-owned); HOME=/data keeps any
# instaloader config writeable without a real home dir.
exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/data \
  -v "$HERE:/data" \
  "$IMAGE" "$@"
