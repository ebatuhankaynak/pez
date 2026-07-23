#!/usr/bin/env python3
"""Visual A/B contact sheet: three columns, one PNG per clip, full frames.

Generic over the two things being compared -- pass any two output dirs. Defaults
compare the two ENGINES (LaMa vs MiniMax); pass --mid/--right to compare, say, the
OLD detection vs the FIXED detection (both LaMa) instead:

  # engines
  python /app/inpaint/compare_frames.py --n 30
  # before/after a detection change (baseline out/ vs re-run out_fix/)
  python /app/inpaint/compare_frames.py --n 30 \
      --mid /app/inpaint/eval/out      --midlab "LaMa OLD" \
      --right /app/inpaint/eval/out_fix --rightlab "LaMa FIX" \
      --dir /app/inpaint/eval/compare_fix

Picks clips whose caption sits over REAL CONTENT (busy background), grabs each
clip's busiest frame, crops nothing (full frame), outlines the caption band, and
computes removed/sharp live on the shown frame so the tile labels match the pixels.
Writes <dir>/<id>.png + index.html.
"""
import argparse, html, json, os
import cv2, numpy as np
import inpaint_text as it

SRC   = "/app/split/meme"
LAMAN = "/app/inpaint/eval/manifest.json"
TILE_W = 380
BAR = 30
SAMPLES = 8
RING = 25
KD = 3


def band(rects, H, W, pad=8):
    y1 = max(0, min(r[1] for r in rects) - pad)
    y2 = min(H, max(r[3] for r in rects) + pad)
    return y1, y2


def metrics(orig, out, rects):
    """removed (~1 erased, ~0 untouched) + sharp (~1 clean, <<1 blurry) on ONE frame."""
    m = it.glyph_mask(orig, rects)
    md = cv2.dilate(m, np.ones((2 * KD + 1,) * 2, np.uint8))
    ring = (cv2.dilate(md, np.ones((2 * RING + 1,) * 2, np.uint8)) > 0) & (md == 0)
    strokes = md > 0
    if strokes.sum() < 30 or ring.sum() < 50:
        return None, None
    og = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ou = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = float(og[ring].mean())
    contrast = float(np.abs(og[strokes] - bg).mean())
    changed = float(np.abs(ou[strokes] - og[strokes]).mean())
    lap = cv2.Laplacian(ou, cv2.CV_32F)
    ring_lv = float(lap[ring].var())
    removed = changed / (contrast + 1e-6)
    sharp = float(lap[strokes].var()) / (ring_lv + 1e-6) if ring_lv >= 3.0 else None
    return round(removed, 2), (round(sharp, 2) if sharp is not None else None)


def score_clip(path, rects):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = sorted(set(np.linspace(0, n - 1, min(SAMPLES, n)).astype(int).tolist()))
    best_i, best_s = idx[0], -1.0
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if not ok:
            continue
        H, W = f.shape[:2]; y1, y2 = band(rects, H, W)
        s = float(cv2.Laplacian(cv2.cvtColor(f[y1:y2], cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
        if s > best_s:
            best_s, best_i = s, i
    cap.release()
    return best_i, best_s, n


def grab(path, idx):
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read(); cap.release()
    return f if ok else None


def col(img, text, rects):
    H, W = img.shape[:2]; s = TILE_W / W
    r = cv2.resize(img, (TILE_W, max(1, int(H * s))))
    y1, y2 = band(rects, H, W)
    cv2.rectangle(r, (1, int(y1 * s)), (TILE_W - 2, int(y2 * s)), (60, 220, 60), 1)
    bar = np.full((BAR, TILE_W, 3), 28, np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([bar, r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dir", default="/app/inpaint/eval/compare")
    ap.add_argument("--mid", default="/app/inpaint/eval/out"); ap.add_argument("--midlab", default="LaMa")
    ap.add_argument("--right", default="/app/inpaint/eval/out_minimax"); ap.add_argument("--rightlab", default="MiniMax")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)

    man = {r["name"]: r for r in json.load(open(LAMAN))}
    # candidates: have rects + a file in BOTH compared dirs
    cand = []
    for name, rec in man.items():
        rects = [tuple(map(int, r)) for r in rec.get("rects", [])]
        if not rects:
            continue
        po, pm, pr = os.path.join(SRC, name), os.path.join(a.mid, name), os.path.join(a.right, name)
        if all(os.path.exists(p) for p in (po, pm, pr)):
            cand.append((name, rects))
    print(f"[compare] {len(cand)} clips in both dirs; scoring texture ...")
    scored = []
    for name, rects in cand:
        bi, bs, n = score_clip(os.path.join(a.mid, name), rects)
        scored.append((bs, name, rects, bi, n))
    scored.sort(reverse=True)
    keep = scored[:a.n]

    made = []
    for rank, (bs, name, rects, bi, n) in enumerate(keep, 1):
        fo = grab(os.path.join(SRC, name), bi)
        fm = grab(os.path.join(a.mid, name), bi)
        fr = grab(os.path.join(a.right, name), bi)
        if fo is None or fm is None or fr is None:
            continue
        rm_m, sh_m = metrics(fo, fm, rects)
        rm_r, sh_r = metrics(fo, fr, rects)
        short = name.split("_meme")[0].split("_")[-1][:12] if "_meme" in name else name[:12]
        c1 = col(fo, f"ORIG {short} f{bi}/{n}", rects)
        c2 = col(fm, f"{a.midlab}  rm={rm_m} sh={sh_m}", rects)
        c3 = col(fr, f"{a.rightlab}  rm={rm_r} sh={sh_r}", rects)
        h = max(c1.shape[0], c2.shape[0], c3.shape[0])
        pad = lambda c: np.vstack([c, np.full((h - c.shape[0], TILE_W, 3), 12, np.uint8)]) if c.shape[0] < h else c
        sep = np.full((h, 3, 3), 60, np.uint8)
        fn = f"{rank:02d}_{short}.png"
        cv2.imwrite(os.path.join(a.dir, fn), np.hstack([pad(c1), sep, pad(c2), sep, pad(c3)]))
        made.append((fn, short, rm_m, rm_r))
        print(f"  {fn}  texture={bs:.0f}  {a.midlab} rm={rm_m} | {a.rightlab} rm={rm_r}")

    cards = "\n".join(
        f'<figure><figcaption>{html.escape(s)} &mdash; {html.escape(a.midlab)} rm{mm} '
        f'vs {html.escape(a.rightlab)} rm{rr}</figcaption><img src="{fn}" loading="lazy"></figure>'
        for fn, s, mm, rr in made)
    doc = ("<!doctype html><meta charset=utf-8><title>compare</title>"
           "<style>body{background:#111;color:#ddd;font:14px system-ui;margin:0;padding:16px}"
           "figure{margin:0 0 28px}figcaption{margin:0 0 6px;color:#9ad}"
           "img{width:100%;max-width:1200px;display:block;border:1px solid #333}h1{font-size:16px}</style>"
           f"<h1>ORIG | {html.escape(a.midlab)} | {html.escape(a.rightlab)} &mdash; busiest frame per clip "
           "(green box = caption band)</h1>" + cards)
    with open(os.path.join(a.dir, "index.html"), "w") as f:
        f.write(doc)
    print(f"[compare] -> {a.dir}/  ({len(made)} PNGs + index.html)")


if __name__ == "__main__":
    main()
