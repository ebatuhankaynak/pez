#!/usr/bin/env python3
"""Cheap no-reference quality metrics for the caption-removal outputs.

We have NO clean ground truth (the original has the caption burned in), so this scores
each output on its own, restricted to the region we actually inpainted -- the glyph
mask + caption band recorded in the batch manifest. Four fast per-clip numbers, each
computed on ~8 sampled frames and reduced by median. The first three are model-free
(pure pixels); OCR is optional and demoted -- it answers "text present?" not "clean?":

  removed    How far the OUTPUT moved the text-stroke pixels away from the ORIGINAL,
             scaled by how much the text stood out from its background to begin with.
             ~1 = strokes overwritten with background (text gone) ; ~0 = output ~=
             original there (text likely still sitting). This is the frame-diff idea:
             big diff = removed, small diff = maybe still there -- but NOTE a big diff
             can also be a big smear, so read it WITH sharp/resid below.

  sharp      Laplacian variance at the stroke pixels / a background ring just outside
             them, in the OUTPUT. A smeared/blurry fill is much softer than its
             surroundings (<<1); a fill that matches its texture is ~1. This is the
             blur number.

  resid_edge Gradient (Sobel) energy at the stroke pixels / the ring, in the OUTPUT. A
             clean fill matches its surroundings (~1); a leftover glyph, ghost, or hard
             seam leaves abnormal edges there (>>1). This is the weird-distortion number.

  leftover   (optional, --ocr) OCR still finds text in the output band. Fraction of
             frames with a box conf >= 0.5. Kept only as a cross-check on `removed`.

The mask is rebuilt from the ORIGINAL frame (which still has the text) using the rects
the batch stored, then measured on the aligned OUTPUT frame (inpaint preserves frame
count + fps, so frame i <-> frame i). Near-flat bands (letterbox / solid) are marked
`flat` -- the ratios are trivially ~1 there and not meaningful, so they don't pollute
the averages.

Run INSIDE the pezevid docker (needs cv2; OCR only if --ocr):
  python /app/inpaint/eval_quality.py                 # both engines, pure-CV (no model)
  python /app/inpaint/eval_quality.py --engine lama
  python /app/inpaint/eval_quality.py --ocr           # + OCR leftover cross-check
Writes inpaint/eval/quality.json and prints a worst-first table + per-engine summary.
"""
import argparse, glob, json, os, statistics as st
import cv2, numpy as np

import inpaint_text as it   # sibling module (glyph_mask, temporal_max, PAD-style band)

SRC = "/app/split/meme"
ENGINES = {
    "lama":    ("/app/inpaint/eval/out",         "/app/inpaint/eval/manifest.json"),
    "minimax": ("/app/inpaint/eval/out_minimax", "/app/inpaint/eval/out_minimax/manifest.json"),
}
QUALITY = "/app/inpaint/eval/quality.json"

FRAMES = 8          # frames sampled per clip
RING = 25           # background-ring width (px) just outside the dilated mask
MASK_DILATE = 3     # grow the stroke mask a touch so we sit ON the glyph, not beside it
FLAT_LAPVAR = 3.0   # ring Laplacian variance below this = flat band (ratios not meaningful)


def _read_frames(path, idx):
    cap = cv2.VideoCapture(path)
    out = {}
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out[i] = f
    cap.release()
    return out


def _band_bbox(rects, W, H, pad=8):
    x1 = max(0, min(r[0] for r in rects) - pad); y1 = max(0, min(r[1] for r in rects) - pad)
    x2 = min(W, max(r[2] for r in rects) + pad); y2 = min(H, max(r[3] for r in rects) + pad)
    return x1, y1, x2, y2


def clip_metrics(name, out_dir, rec, ocr=None):
    """Return dict of metrics for one output clip, or None if unmeasurable."""
    src, outp = os.path.join(SRC, name), os.path.join(out_dir, name)
    rects = [tuple(map(int, r)) for r in rec.get("rects", [])]
    if not rects or not os.path.exists(src) or not os.path.exists(outp):
        return None
    n = int(cv2.VideoCapture(outp).get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = sorted(set(np.linspace(0, n - 1, min(FRAMES, n)).astype(int).tolist()))
    of, oo = _read_frames(src, idx), _read_frames(outp, idx)

    resid, sharp, removed, flatframes, ocr_hits, ocr_n = [], [], [], 0, 0, 0
    kd = np.ones((2 * MASK_DILATE + 1, 2 * MASK_DILATE + 1), np.uint8)
    kr = np.ones((2 * RING + 1, 2 * RING + 1), np.uint8)
    for i in idx:
        if i not in of or i not in oo:
            continue
        og, ou = of[i], oo[i]
        H, W = ou.shape[:2]
        m = it.glyph_mask(og, rects)                     # strokes located on the ORIGINAL
        md = cv2.dilate(m, kd)
        ring = (cv2.dilate(md, kr) > 0) & (md == 0)
        strokes = md > 0
        if strokes.sum() < 30 or ring.sum() < 50:
            continue
        g = cv2.cvtColor(ou, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        ring_lv = float(lap[ring].var())
        if ring_lv < FLAT_LAPVAR:
            flatframes += 1                              # flat band: ratios not meaningful
            continue
        resid.append(float(grad[strokes].mean()) / (float(grad[ring].mean()) + 1e-6))
        sharp.append(float(lap[strokes].var()) / (ring_lv + 1e-6))
        # removed: how far the OUTPUT moved the text-stroke pixels away from the ORIGINAL,
        # scaled by how much those strokes stood out from their local background to begin
        # with. ~1 = strokes overwritten with background (text gone) ; ~0 = output ~= original
        # there (text likely still sitting). Pure pixels, no OCR -- the user's frame-diff idea.
        og_gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = float(og_gray[ring].mean())                       # background color, ORIGINAL
        contrast = float(np.abs(og_gray[strokes] - bg).mean()) # text-vs-bg contrast to erase
        changed = float(np.abs(g[strokes] - og_gray[strokes]).mean())   # output vs original
        removed.append(changed / (contrast + 1e-6))
        if ocr is not None:
            ocr_n += 1
            bx1, by1, bx2, by2 = _band_bbox(rects, W, H)
            crop = ou[by1:by2, bx1:bx2]
            if crop.size:
                res, _ = ocr(crop)
                if any(score >= 0.5 for _b, _t, score in (res or [])):
                    ocr_hits += 1

    if not resid and flatframes == 0:
        return None
    flat = len(resid) == 0 and flatframes > 0
    return {
        "name": name, "frames": rec.get("frames"), "cov": rec.get("cov"),
        "flat": flat,
        "removed": round(st.median(removed), 3) if removed else None,
        "resid_edge": round(st.median(resid), 3) if resid else None,
        "sharp": round(st.median(sharp), 3) if sharp else None,
        "leftover": round(ocr_hits / ocr_n, 3) if ocr_n else None,
    }


def score_engine(engine, ocr):
    out_dir, manifest = ENGINES[engine]
    if not os.path.exists(manifest):
        return []
    recs = json.load(open(manifest))
    done = [r for r in recs if r.get("status") == "done"]
    rows = []
    for k, r in enumerate(done):
        m = clip_metrics(r["name"], out_dir, r, ocr=ocr)
        if m:
            m["engine"] = engine
            rows.append(m)
        if (k + 1) % 25 == 0:
            print(f"  [{engine}] {k + 1}/{len(done)}")
    return rows


def summarize(rows, engine):
    meas = [r for r in rows if not r["flat"]]
    nflat = sum(r["flat"] for r in rows)
    def mean(key):
        v = [r[key] for r in meas if r.get(key) is not None]
        return st.mean(v) if v else float("nan")
    rm = [r for r in meas if r.get("removed") is not None]
    not_removed = sum(r["removed"] < 0.4 for r in rm)
    lo = [r for r in meas if r.get("leftover") is not None]
    print(f"\n=== {engine}: {len(rows)} clips ({len(meas)} measurable, {nflat} flat) ===")
    print(f"  removed   mean {mean('removed'):.3f}   (~1 text erased to bg, ~0 barely changed; "
          f"<0.4 on {not_removed}/{len(rm)} clips = text maybe still there)")
    print(f"  sharp     mean {mean('sharp'):.3f}   (~1 clean, <<1 blurry/smeared fill)")
    print(f"  resid_edge mean {mean('resid_edge'):.3f}  (~1 clean, >>1 weird edges/ghost/seam)")
    if lo:
        left_bad = sum(r["leftover"] >= 0.5 for r in lo)
        print(f"  leftover  mean {mean('leftover'):.3f}   (OCR; >=0.5 on {left_bad}/{len(lo)} clips)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("lama", "minimax", "both"), default="both")
    ap.add_argument("--ocr", action="store_true",
                    help="also run the OCR leftover cross-check (loads RapidOCR; off by default)")
    a = ap.parse_args()

    ocr = None
    if a.ocr:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()

    engines = ("lama", "minimax") if a.engine == "both" else (a.engine,)
    engines = [e for e in engines if os.path.exists(ENGINES[e][1])]
    all_rows = []
    for e in engines:
        print(f"[quality] scoring {e} ...")
        all_rows += score_engine(e, ocr)

    with open(QUALITY, "w") as f:
        json.dump(all_rows, f, indent=1)

    for e in engines:
        rows = [r for r in all_rows if r["engine"] == e]
        summarize(rows, e)
        nonflat = [r for r in rows if not r["flat"]]
        least = sorted([r for r in nonflat if r.get("removed") is not None],
                       key=lambda r: r["removed"])[:5]
        if least:
            print(f"  least removed {e} (low = text maybe still there):")
            for r in least:
                print(f"    removed={r['removed']:.2f} sharp={r['sharp']} "
                      f"resid={r['resid_edge']}  {r['name'][-26:]}")
        worst = sorted([r for r in nonflat if r.get("sharp") is not None],
                       key=lambda r: r["sharp"])[:5]
        if worst:
            print(f"  blurriest {e} (low sharp = smeared/soft fill):")
            for r in worst:
                print(f"    sharp={r['sharp']:.2f} removed={r['removed']} "
                      f"resid={r['resid_edge']}  {r['name'][-26:]}")

    # side-by-side when both engines scored the same clips
    if len(engines) == 2:
        by = {e: {r["name"]: r for r in all_rows if r["engine"] == e} for e in engines}
        common = set(by["lama"]) & set(by["minimax"])
        cmp = [(n, by["lama"][n], by["minimax"][n]) for n in common
               if not by["lama"][n]["flat"] and not by["minimax"][n]["flat"]]
        if cmp:
            sh = sum(l["sharp"] >= m["sharp"] for _n, l, m in cmp)
            rm = sum((l.get("removed") or 0) >= (m.get("removed") or 0) for _n, l, m in cmp)
            print(f"\n=== LaMa vs MiniMax on {len(cmp)} shared non-flat clips ===")
            print(f"  sharper fill:   LaMa {sh} / MiniMax {len(cmp) - sh}")
            print(f"  more removed:   LaMa {rm} / MiniMax {len(cmp) - rm}")
    print(f"\n[quality] -> {QUALITY}")


if __name__ == "__main__":
    main()
