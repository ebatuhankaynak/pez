#!/usr/bin/env python3
"""Render a debug overlay video per clip: every detected face boxed with its cosine
similarity to the enrolled creator, color-coded by the match threshold. Lets you see
WHY a shot was labeled meme — is the face missed, or detected-but-under-threshold?

  green  sim >= --face-threshold (0.35)   -> counts as the creator
  yellow within 0.10 below threshold       -> "right there but under the line"
  red    below that                        -> no match

Writes debug/<stem>.mp4 (H.264, browser-playable). Shown next to the original in app.html.

    python tools/debug_faces.py                 # all clips (Docker/GPU: ~5-7 min)
    python tools/debug_faces.py --only 0b9bf76f76fa --limit 1
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import cv2
from decord import VideoReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from relabel_faces import (load_face_app, enroll_creator, normed,
                           CLIPS_DIR, TRANSITIONS, short)

DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"   # repo root/debug (this file lives in tools/)
YELLOW_BAND = 0.10   # sim within this much below threshold: "right there but under the line"


def color(sim, thr):
    if sim >= thr:                  return (80, 220, 80)      # green  (BGR)
    if sim >= thr - YELLOW_BAND:    return (60, 210, 230)     # yellow
    return (70, 70, 235)                                      # red


def render(app, src, out, centroid, thr, stride):
    vr = VideoReader(str(src))
    fps = vr.get_avg_fps() or 25.0
    total = len(vr)
    h, w = vr[0].shape[:2]
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    faces = []
    for i in range(total):
        bgr = np.ascontiguousarray(vr[i].asnumpy()[:, :, ::-1])
        if i % stride == 0:
            faces = app.get(bgr)
        best = 0.0
        for f in faces:
            sim = float(normed(f.normed_embedding) @ centroid)
            best = max(best, sim)
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            c = color(sim, thr)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), c, 3)
            cv2.putText(bgr, f"{sim:.2f}", (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2, cv2.LINE_AA)
        t = i / fps
        lab = "CREATOR" if best >= thr else "meme"
        hud = f"t={t:4.2f}s  max={best:.2f}  thr={thr:.2f}  -> {lab}"
        cv2.putText(bgr, hud, (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (240, 240, 240), 2, cv2.LINE_AA)
        ff.stdin.write(bgr.tobytes())
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", default=str(TRANSITIONS))
    ap.add_argument("--face-threshold", type=float, default=0.35)
    ap.add_argument("--stride", type=int, default=2, help="detect every Nth frame (boxes held between)")
    ap.add_argument("--only", nargs="*", default=None, help="short ids to render (default: all)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    DEBUG_DIR.mkdir(exist_ok=True)
    records = json.load(open(args.transitions))
    app = load_face_app()
    print("Enrolling creator...", flush=True)
    centroid = enroll_creator(app, records)

    todo = [c for c in records if not args.only or short(c["clip"]) in args.only]
    if args.limit:
        todo = todo[:args.limit]
    for i, c in enumerate(todo, 1):
        src = CLIPS_DIR / c["clip"]
        if not src.exists():
            continue
        out = DEBUG_DIR / (c["clip"][:-4] + ".mp4")
        render(app, src, out, centroid, args.face_threshold, args.stride)
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)
    print(f"Wrote {len(todo)} debug videos to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
