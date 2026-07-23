#!/usr/bin/env python3
"""Why did clip X come out bad? Overlay what the pipeline actually saw:
  * blue boxes  = OCR rects stored in the manifest (what we told it to remove)
  * red pixels  = glyph_mask (the strokes we actually inpaint inside those rects)
  * also re-runs RapidOCR live on the frame and prints every box it finds now,
    so we can see if a whole text LINE was never detected in the first place.
Run in docker: python /app/inpaint/diag_mask.py 0fda06d8f518 ff6923a7578e
"""
import json, os, sys
import cv2, numpy as np
import inpaint_text as it

SRC   = "/app/split/meme"
LAMAN = "/app/inpaint/eval/manifest.json"
OUTD  = "/app/inpaint/eval/compare"


def grab(path, idx=0):
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read(); cap.release()
    return f if ok else None


def main():
    keys = sys.argv[1:] or ["0fda06d8f518", "ff6923a7578e"]
    man = {r["name"]: r for r in json.load(open(LAMAN))}
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    os.makedirs(OUTD, exist_ok=True)

    for key in keys:
        hit = [n for n in man if key in n]
        if not hit:
            print(f"{key}: not in manifest"); continue
        name = hit[0]; rec = man[name]
        rects = [tuple(map(int, r)) for r in rec.get("rects", [])]
        f = grab(os.path.join(SRC, name), 0)
        if f is None:
            print(f"{key}: no frame"); continue
        H, W = f.shape[:2]
        # LIVE rects from the CURRENT auto_rects code (reflects the cheap fix), vs the
        # OLD rects baked in the manifest -- so we can see the fix without re-batching.
        newr = [tuple(map(int, r)) for r in it.auto_rects(os.path.join(SRC, name), W, H, ocr=ocr)]
        m = it.glyph_mask(f, newr or rects)
        cov = (m > 0).mean() * 100
        rectpx = sum((y2 - y1) * (x2 - x1) for x1, y1, x2, y2 in (newr or rects)) or 1
        fillfrac = (m > 0).sum() / rectpx * 100

        vis = f.copy()
        vis[m > 0] = (0, 0, 255)                          # red = mask from NEW rects
        vis = cv2.addWeighted(f, 0.45, vis, 0.55, 0)
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 140, 0), 1)   # thin blue = OLD manifest rects
        for x1, y1, x2, y2 in newr:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 60), 2)    # green = NEW live rects

        res, _ = ocr(f)
        print(f"\n=== {name}  {W}x{H} ===")
        print(f"  OLD rects {len(rects)} -> NEW rects {len(newr)} ; NEW mask fills "
              f"{fillfrac:.1f}% of rect area ({cov:.2f}% of frame)")
        print(f"  live OCR sees {len(res or [])} text box(es) (covered by NEW rects?):")
        for box, txt, sc in (res or []):
            ys = [p[1] for p in box]; xs = [p[0] for p in box]
            inside = any(x1 <= (min(xs)+max(xs))/2 <= x2 and y1 <= (min(ys)+max(ys))/2 <= y2
                         for x1, y1, x2, y2 in newr)
            tag = "COVERED" if inside else "MISSED "
            print(f"    [{tag} conf {sc:.2f}] y{int(min(ys))}-{int(max(ys))}  {txt!r}")
            if not inside:
                cv2.rectangle(vis, (int(min(xs)), int(min(ys))), (int(max(xs)), int(max(ys))),
                              (0, 255, 255), 2)   # yellow = OCR text NEW rects still miss

        out = os.path.join(OUTD, f"diag_{key}.png")
        cv2.imwrite(out, vis)
        print(f"  -> {out}  (thin-blue=OLD rect, green=NEW rect, red=NEW mask, yellow=still-missed)")


if __name__ == "__main__":
    main()
