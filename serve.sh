#!/usr/bin/env bash
# Serve the UI through Docker (app / cut editor / report / verify + the batu-GT save
# endpoint). Everything runs in the container — nothing local. Ctrl-C to stop.
#
#   ./serve.sh            # -> http://localhost:8000/
#
# Under the hood: `docker compose up serve`, which bind-mounts the repo and publishes
# port 8000. To change the port, edit the `serve` service in docker-compose.yml.
cd "$(dirname "$0")"
echo "app      ->  http://localhost:8000/            (workbench; pick claude/batu truth)"
echo "editor   ->  http://localhost:8000/editor.html  (input cuts to ms precision; autosaves batu GT)"
echo "report   ->  http://localhost:8000/report.html"
echo "verify   ->  http://localhost:8000/verify.html"
exec docker compose up serve
