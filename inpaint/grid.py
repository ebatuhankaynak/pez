#!/usr/bin/env python3
"""Grid of evenly-sampled frames from one clip so you can eyeball its content.
  python /app/inpaint/grid.py 62a28230430b 12   # 12 frames, prefer FIX output
"""
import glob, os, sys
import cv2, numpy as np

DIRS = ["/app/inpaint/eval/out_fix", "/app/inpaint/eval/out", "/app/split/meme"]
key = sys.argv[1] if len(sys.argv) > 1 else "62a28230430b"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
COLS, TW = 4, 300

path = None
for d in DIRS:
    hit = glob.glob(os.path.join(d, f"*{key}*.mp4"))
    if hit:
        path = hit[0]; break
if not path:
    print("no clip"); sys.exit(1)
print(f"[grid] {os.path.basename(path)} from {os.path.dirname(path)}")

cap = cv2.VideoCapture(path); n = int(cap.get(7)) or 1
idx = sorted(set(np.linspace(0, n - 1, N).astype(int).tolist()))
tiles = []
for i in idx:
    cap.set(1, i); ok, f = cap.read()
    if not ok:
        continue
    H, W = f.shape[:2]; sc = TW / W
    r = cv2.resize(f, (TW, int(H * sc)))
    bar = np.full((22, TW, 3), 25, np.uint8)
    cv2.putText(bar, f"f{i}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    tiles.append(np.vstack([bar, r]))
cap.release()

th = max(t.shape[0] for t in tiles)
tiles = [np.vstack([t, np.full((th - t.shape[0], TW, 3), 12, np.uint8)]) if t.shape[0] < th else t for t in tiles]
rows = []
for k in range(0, len(tiles), COLS):
    chunk = tiles[k:k + COLS]
    while len(chunk) < COLS:
        chunk.append(np.full((th, TW, 3), 12, np.uint8))
    rows.append(np.hstack(chunk))
grid = np.vstack(rows)
out = f"/app/inpaint/eval/grid_{key}.png"
cv2.imwrite(out, grid); print(f"-> {out}  ({len(tiles)} frames of {n})")
