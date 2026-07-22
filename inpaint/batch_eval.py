#!/usr/bin/env python3
"""Batch text-removal over the pezevid meme clips for the eval tab.

Loads RapidOCR (locate) and big-LaMa (inpaint) ONCE and reuses them across every
clip -- so N clips cost ~one model load, not N. For each clip it auto-locates the
caption(s), builds a per-frame glyph mask, LaMa-inpaints, and re-encodes with the
original audio to inpaint/eval/out/<name>.mp4. Writes a manifest the tab reads.

Run INSIDE the pezevid docker:
  python /app/inpaint/batch_eval.py --count 30
Resumable: clips whose output already exists are skipped.
"""
import argparse, glob, json, os, shutil, time
import cv2, numpy as np

import inpaint_text as it   # sibling module in the same dir (not a package)

SRC = "/app/split/meme"
OUT = "/app/inpaint/eval/out"
MANIFEST = "/app/inpaint/eval/manifest.json"

FEATHER = 5         # mask-edge feather (px) for the LaMa composite
MASK_TEMPORAL = 3   # temporal-max window (frames) to stabilize the glyph mask


def process(path, out_path, lama, ocr):
    """Locate + mask + LaMa one clip. Returns a manifest record."""
    name = os.path.basename(path)
    W, H, fps = it.probe(path)
    rects = it.auto_rects(path, W, H, ocr=ocr)
    if not rects:
        return {"name": name, "status": "no-caption", "rects": [], "w": W, "h": H}

    work = os.path.join(OUT, ".work_" + os.path.splitext(name)[0])
    origdir, final = os.path.join(work, "orig"), os.path.join(work, "final")
    for d in (origdir, final):
        os.makedirs(d, exist_ok=True)
    try:
        it.sh(["ffmpeg", "-y", "-v", "error", "-i", path, os.path.join(origdir, "%05d.png")])
        origs = sorted(glob.glob(os.path.join(origdir, "*.png")))
        masks = it.temporal_max([it.glyph_mask(cv2.imread(of), rects) for of in origs],
                                MASK_TEMPORAL)
        cov = float(np.mean([(m > 0).mean() for m in masks])) if masks else 0.0
        it.lama_frames(origs, masks, final, lama=lama, feather=FEATHER, log=False)
        it.sh(["ffmpeg", "-y", "-v", "error", "-framerate", f"{fps:.6f}",
               "-i", os.path.join(final, "%05d.png"), "-i", path,
               "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
               "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy",
               "-shortest", out_path])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"name": name, "status": "done", "rects": rects,
            "w": W, "h": H, "frames": len(origs), "cov": round(cov * 100, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30, help="clips to process (evenly sampled)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="process every clip")
    ap.add_argument("--random", type=int, default=0, help="process N random clips")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --random")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    clips = sorted(glob.glob(os.path.join(SRC, "*.mp4")))
    if a.random > 0:
        rng = np.random.default_rng(a.seed)
        pick = rng.choice(len(clips), size=min(a.random, len(clips)), replace=False)
        sel = [clips[i] for i in sorted(pick)]
    elif a.all or a.count >= len(clips):
        sel = clips[a.offset:]
    else:
        idx = np.linspace(0, len(clips) - 1, a.count).astype(int)
        sel = [clips[i] for i in sorted(set(idx))][a.offset:]
    print(f"[batch] {len(sel)}/{len(clips)} clips")

    from rapidocr_onnxruntime import RapidOCR
    from simple_lama_inpainting import SimpleLama
    ocr, lama = RapidOCR(), SimpleLama()

    records = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            records = {r["name"]: r for r in json.load(f)}

    for k, path in enumerate(sel):
        name = os.path.basename(path)
        out_path = os.path.join(OUT, name)
        if os.path.exists(out_path) and records.get(name, {}).get("status") == "done":
            print(f"[{k+1}/{len(sel)}] skip (done) {name}")
            continue
        t = time.time()
        try:
            rec = process(path, out_path, lama, ocr)
        except Exception as e:
            rec = {"name": name, "status": f"error: {e.__class__.__name__}: {e}"}
            print(f"[{k+1}/{len(sel)}] ERROR {name}: {e}")
        rec["secs"] = round(time.time() - t, 1)
        records[name] = rec
        with open(MANIFEST, "w") as f:
            json.dump(list(records.values()), f, indent=1)
        print(f"[{k+1}/{len(sel)}] {rec['status']:12} {name}  "
              f"({rec.get('frames','?')}f, {rec['secs']}s, "
              f"{len(rec.get('rects',[]))} rect)")
    print(f"[batch] done -> {MANIFEST}")


if __name__ == "__main__":
    main()
