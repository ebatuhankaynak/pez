#!/usr/bin/env python3
"""Find a clip by the words burned into its caption. OCRs a few frames of every
meme clip and prints the ones whose text matches any given keyword.

  python /app/inpaint/find_clip.py german shirt tshirt haul
"""
import glob, os, re, sys
import cv2
import numpy as np

SRC = "/app/split/meme"
FRAMES = 4


def main():
    kws = [k.lower() for k in (sys.argv[1:] or ["german", "shirt", "tshirt", "haul"])]
    pat = re.compile("|".join(re.escape(k) for k in kws))
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    clips = sorted(glob.glob(os.path.join(SRC, "*.mp4")))
    print(f"[find] {len(clips)} clips ; keywords={kws}")
    hits = []
    for ci, path in enumerate(clips):
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idx = sorted(set(np.linspace(0, n - 1, min(FRAMES, n)).astype(int).tolist()))
        texts = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if not ok:
                continue
            res, _ = ocr(f)
            for _b, t, sc in (res or []):
                if sc >= 0.4 and t:
                    texts.append(t.strip())
        cap.release()
        blob = " | ".join(dict.fromkeys(texts))     # dedup, keep order
        if pat.search(blob.lower()):
            hits.append((os.path.basename(path), blob))
        if (ci + 1) % 25 == 0:
            print(f"  scanned {ci + 1}/{len(clips)} ; {len(hits)} hit(s) so far")
    print(f"\n[find] {len(hits)} match(es):")
    for name, blob in hits:
        print(f"\n  {name}\n    {blob[:300]}")


if __name__ == "__main__":
    main()
