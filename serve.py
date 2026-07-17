#!/usr/bin/env python3
"""
Static file server for the pezevenk workbench + a tiny save endpoint so the cut
editor (editor.html) can write batu's ground truth straight to disk.

    python serve.py            # http://localhost:8000/   (app, editor, report)
    python serve.py 8080       # custom port

Everything under this folder is served like `python3 -m http.server`. The ONE
extra route is:

    POST /api/save/<name>      body = JSON  ->  writes transitions/<name>

`<name>` is whitelisted (only ground_truth_batu.json) so a stray request can't
overwrite arbitrary files. If you serve with plain `python3 -m http.server`
instead, the editor still works — it just falls back to a manual JSON download.
"""

import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Only these files may be written by the save endpoint.
SAVE_WHITELIST = {"ground_truth_batu.json"}
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)\s*$")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p == "/healthz":                       # uptime probe (pc_home etc.)
            return self._json(200, {"ok": True, "app": "pezevenk"})
        # never serve dotfiles (.git/.env/…) even though we sit at the repo root
        if any(seg.startswith(".") for seg in p.split("/") if seg):
            return self._json(404, {"error": "not found"})
        return super().do_GET()

    # --- HTTP Range support (the stdlib handler has none) -------------------
    # Without this, <video> can't seek to an unbuffered position: the seek
    # stalls, no `seeked`/`timeupdate` fires, and the cut editor's playhead
    # freezes ("can't select the frame — have to refresh"). We answer byte
    # ranges with 206 Partial Content and advertise Accept-Ranges.

    def handle_one_request(self):
        # per-request state (a keep-alive connection reuses the instance)
        self._range_remaining = None
        self._accept_ranges_sent = False
        super().handle_one_request()

    def end_headers(self):
        if not getattr(self, "_accept_ranges_sent", False):
            self.send_header("Accept-Ranges", "bytes")
            self._accept_ranges_sent = True
        super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        m = _RANGE_RE.match(rng.strip())
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()          # dirs / malformed -> stdlib
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        try:
            st = os.fstat(f.fileno())
            size = st.st_size
            start_s, end_s = m.group(1), m.group(2)
            if start_s == "":                    # suffix range: last N bytes
                start = max(0, size - int(end_s or 0))
                end = size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            if size == 0 or start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                f.close()
                return None
            length = end - start + 1
            f.seek(start)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Last-Modified", self.date_time_string(st.st_mtime))
            self.end_headers()
            self._range_remaining = length
            return f
        except Exception:
            f.close()
            raise

    def copyfile(self, source, outputfile):
        if getattr(self, "_range_remaining", None) is None:
            return super().copyfile(source, outputfile)
        remaining = self._range_remaining
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

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
        # atomic write: readers (editor reload / workbench) never see a
        # half-written file if an autosave lands mid-GET
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2))
        os.replace(tmp, dst)
        self._json(200, {"ok": True, "wrote": str(dst.relative_to(HERE)),
                         "clips": len(doc.get("clips", []))})

    def log_message(self, fmt, *args):               # quieter logs
        if self.command == "POST":
            super().log_message(fmt, *args)


def main():
    # PORT/HOST env win (so a supervisor can place it), then argv, then default.
    port = int(os.environ.get("PORT") or (sys.argv[1] if len(sys.argv) > 1 else 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"app     ->  http://localhost:{port}/            (workbench; pick claude/batu truth)")
    print(f"editor  ->  http://localhost:{port}/editor.html  (input cuts, autosaves to batu GT)")
    print(f"report  ->  http://localhost:{port}/report.html")
    print(f"save endpoint: POST /api/save/ground_truth_batu.json")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
