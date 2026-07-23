#!/usr/bin/env python3
"""Compare ONE clip across every variant (orig + each output dir) side by side.
  python /app/inpaint/cmp_one.py 62a28230430b
"""
import glob, json, os, sys
import cv2, numpy as np
import inpaint_text as it

SRC = "/app/split/meme"
LAMAN = "/app/inpaint/eval/manifest.json"
COLS = [("ORIG", SRC), ("LaMa-OLD", "/app/inpaint/eval/out"),
        ("LaMa-FIX", "/app/inpaint/eval/out_fix"), ("MiniMax", "/app/inpaint/eval/out_minimax")]
TILE_W, BAR, RING, KD = 340, 30, 25, 3


def band(rects, H, W, pad=8):
    return max(0, min(r[1] for r in rects) - pad), min(H, max(r[3] for r in rects) + pad)


def metrics(orig, out, rects):
    m = it.glyph_mask(orig, rects)
    md = cv2.dilate(m, np.ones((2 * KD + 1,) * 2, np.uint8))
    ring = (cv2.dilate(md, np.ones((2 * RING + 1,) * 2, np.uint8)) > 0) & (md == 0)
    strokes = md > 0
    if strokes.sum() < 30 or ring.sum() < 50:
        return None, None
    og = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ou = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = float(og[ring].mean()); contrast = float(np.abs(og[strokes] - bg).mean())
    changed = float(np.abs(ou[strokes] - og[strokes]).mean())
    lap = cv2.Laplacian(ou, cv2.CV_32F); rl = float(lap[ring].var())
    return round(changed / (contrast + 1e-6), 2), (round(float(lap[strokes].var()) / (rl + 1e-6), 2) if rl >= 3 else None)


def grab(path, idx):
    c = cv2.VideoCapture(path); c.set(1, idx); ok, f = c.read(); c.release()
    return f if ok else None


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "62a28230430b"
    # rects can be absent in a manifest that marked the clip no-caption; take them from
    # whichever variant actually detected the caption (prefer the fixed run).
    name, rects = None, []
    for mp in ("/app/inpaint/eval/out_fix/manifest.json", LAMAN,
               "/app/inpaint/eval/out_minimax/manifest.json"):
        if not os.path.exists(mp):
            continue
        man = {r["name"]: r for r in json.load(open(mp))}
        hit = next((n for n in man if key in n), None)
        if hit:
            name = hit
            if man[hit].get("rects"):
                rects = [tuple(map(int, r)) for r in man[hit]["rects"]]
                break
    if not rects:
        print("no rects in any manifest"); return
    # busiest band frame on the original
    src = os.path.join(SRC, name)
    c = cv2.VideoCapture(src); n = int(c.get(7)) or 1
    idx = sorted(set(np.linspace(0, n - 1, 8).astype(int).tolist())); bi, bs = idx[0], -1
    for i in idx:
        c.set(1, i); ok, f = c.read()
        if not ok:
            continue
        H, W = f.shape[:2]; y1, y2 = band(rects, H, W)
        s = float(cv2.Laplacian(cv2.cvtColor(f[y1:y2], cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
        if s > bs:
            bs, bi = s, i
    c.release()
    orig = grab(src, bi)

    tiles = []
    for lab, d in COLS:
        f = grab(os.path.join(d, name), bi)
        if f is None:
            continue
        rm, sh = (None, None) if lab == "ORIG" else metrics(orig, f, rects)
        H, W = f.shape[:2]; sc = TILE_W / W
        r = cv2.resize(f, (TILE_W, int(H * sc)))
        y1, y2 = band(rects, H, W)
        cv2.rectangle(r, (1, int(y1 * sc)), (TILE_W - 2, int(y2 * sc)), (60, 220, 60), 1)
        bar = np.full((BAR, TILE_W, 3), 28, np.uint8)
        txt = lab if lab == "ORIG" else f"{lab} rm={rm} sh={sh}"
        cv2.putText(bar, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, r]))
        print(f"  {lab:9} rm={rm} sh={sh}")
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((h - t.shape[0], TILE_W, 3), 12, np.uint8)]) if t.shape[0] < h else t for t in tiles]
    sep = np.full((h, 3, 3), 60, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, sep, t])
    out = f"/app/inpaint/eval/cmp_{key}.png"
    cv2.imwrite(out, row); print(f"-> {out}  frame {bi}/{n}")


if __name__ == "__main__":
    main()
