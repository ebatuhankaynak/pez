#!/usr/bin/env bash
# Serve this folder so the UI can load JSON + videos (browsers block file:// for both).
cd "$(dirname "$0")"
PORT="${1:-8000}"
echo "app      ->  http://localhost:${PORT}/            (merged workbench; re-run pipeline then ↻ reload)"
echo "verify   ->  http://localhost:${PORT}/verify.html  (static fallback)"
echo "report   ->  http://localhost:${PORT}/report.html  (static; opens with a double-click too)"
exec python3 -m http.server "$PORT"
