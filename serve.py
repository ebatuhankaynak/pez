#!/usr/bin/env python3
"""
Static file server for the pezevenk workbench + a tiny save endpoint so the cut
editor (editor.html) can write batu's ground truth straight to disk.

    python serve.py            # http://localhost:8000/   (app, editor, report, verify)
    python serve.py 8080       # custom port

Everything under this folder is served like `python3 -m http.server`. The ONE
extra route is:

    POST /api/save/<name>      body = JSON  ->  writes transitions/<name>

`<name>` is whitelisted (only ground_truth_batu.json) so a stray request can't
overwrite arbitrary files. If you serve with plain `python3 -m http.server`
instead, the editor still works — it just falls back to a manual JSON download.
"""

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Only these files may be written by the save endpoint.
SAVE_WHITELIST = {"ground_truth_batu.json"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.startswith("/api/save/"):
            self._json(404, {"error": "unknown endpoint"})
            return
        name = self.path[len("/api/save/"):]
        if name not in SAVE_WHITELIST:
            self._json(403, {"error": f"'{name}' is not writable"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            doc = json.loads(raw)                      # validate it's JSON
        except Exception as e:
            self._json(400, {"error": f"bad body: {e}"})
            return
        dst = HERE / "transitions" / name
        dst.write_text(json.dumps(doc, indent=2))
        self._json(200, {"ok": True, "wrote": str(dst.relative_to(HERE)),
                         "clips": len(doc.get("clips", []))})

    def log_message(self, fmt, *args):               # quieter logs
        if self.command == "POST":
            super().log_message(fmt, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"app     ->  http://localhost:{port}/            (workbench; pick claude/batu truth)")
    print(f"editor  ->  http://localhost:{port}/editor.html  (input cuts, autosaves to batu GT)")
    print(f"report  ->  http://localhost:{port}/report.html")
    print(f"verify  ->  http://localhost:{port}/verify.html")
    print(f"save endpoint: POST /api/save/ground_truth_batu.json")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
