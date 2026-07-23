#!/usr/bin/env python3
"""Batch text-removal over the pezevid meme clips for the eval tab.

Loads RapidOCR (locate) and big-LaMa (inpaint) ONCE and reuses them across every
clip -- so N clips cost ~one model load, not N. For each clip it auto-locates the
caption(s), builds a per-frame glyph mask, LaMa-inpaints, and re-encodes with the
original audio to inpaint/eval/out/<name>.mp4. Writes a manifest the tab reads.

Throughput: the per-clip work is a chain of CPU (ffmpeg extract/encode, OpenCV
masking) and GPU (LaMa) stages. Running clips strictly one-at-a-time leaves the GPU
idle during every CPU stage. So clips are pipelined across a small worker pool: while
one worker holds the single GPU model and inpaints, the others run ffmpeg/OpenCV for
other clips. The GPU is used by AT MOST ONE worker at a time (a lock), so VRAM cost is
unchanged -- there is still exactly one LaMa model. Worker count auto-scales to the
host and floors at 1, so a small/CPU-only box runs it as a plain serial loop.

Run INSIDE the pezevid docker:
  python /app/inpaint/batch_eval.py --all                     # LaMa -> inpaint/eval/out
  python /app/inpaint/batch_eval.py --all --engine minimax    # MiniMax -> inpaint/eval/out_minimax
  python /app/inpaint/batch_eval.py --all --workers 1         # force serial (low RAM/cores)
The two engines write to SEPARATE folders (out/ vs out_minimax/), each with its own
manifest, so a MiniMax run never overwrites the LaMa outputs. --engine minimax needs the
image built with MINIMAX=1 (else it errors with a rebuild hint).
Resumable: clips whose output already exists are skipped.
"""
import argparse, glob, json, os, shutil, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2, numpy as np

import inpaint_text as it   # sibling module in the same dir (not a package)

SRC = "/app/split/meme"

# Per-engine output dir + manifest. LaMa keeps the original paths unchanged; MiniMax
# writes to a SEPARATE folder so the two engines' outputs never overwrite each other and
# the memes tab can show them side by side (it reads out/ + out_minimax/).
ENGINE_CFG = {
    "lama":    ("/app/inpaint/eval/out",         "/app/inpaint/eval/manifest.json"),
    "minimax": ("/app/inpaint/eval/out_minimax", "/app/inpaint/eval/out_minimax/manifest.json"),
}
OUT, MANIFEST = ENGINE_CFG["lama"]   # reassigned per --engine in main()

FEATHER = 5         # mask-edge feather (px) for the composite
MASK_TEMPORAL = 3   # temporal-max window (frames) to stabilize the glyph mask
PAD = 200           # vertical context (px) around the caption band for MiniMax band-crop


def auto_workers():
    """Conservative default: half the cores, capped at 4, floor 1. Small enough to keep
    RAM (each worker holds one clip's masks) and parallel-ffmpeg load bounded; a 1-2 core
    host runs a single worker = plain serial."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def preprocess(path, ocr, ocr_lock):
    """CPU stage: locate caption(s), extract frames, build the glyph masks. Returns
    (skip_record, meta). skip_record is set (and meta None) when there is no caption to
    remove; otherwise meta carries everything the GPU + encode stages need. RapidOCR is
    shared across workers, so its calls are serialized by ocr_lock (only ~8 sampled
    frames per clip -- negligible contention)."""
    name = os.path.basename(path)
    W, H, fps = it.probe(path)
    with ocr_lock:
        rects = it.auto_rects(path, W, H, ocr=ocr)
    if not rects:
        return {"name": name, "status": "no-caption", "rects": [], "w": W, "h": H}, None

    work = os.path.join(OUT, ".work_" + os.path.splitext(name)[0])
    origdir, final = os.path.join(work, "orig"), os.path.join(work, "final")
    for d in (origdir, final):
        os.makedirs(d, exist_ok=True)
    it.sh(["ffmpeg", "-y", "-v", "error", "-i", path, os.path.join(origdir, "%05d.png")])
    origs = sorted(glob.glob(os.path.join(origdir, "*.png")))
    masks = it.temporal_max([it.glyph_mask(cv2.imread(of), rects) for of in origs],
                            MASK_TEMPORAL)
    cov = float(np.mean([(m > 0).mean() for m in masks])) if masks else 0.0
    # caption band (rects +- PAD, snapped to /8) -- MiniMax inpaints only this band
    y1 = max(0, min(r[1] for r in rects) - PAD)
    y2 = min(H, max(r[3] for r in rects) + PAD)
    by1 = it.snap8(y1); by2 = by1 + it.snap8(y2 - by1)
    meta = {"name": name, "work": work, "final": final, "origs": origs, "masks": masks,
            "rects": rects, "w": W, "h": H, "fps": fps, "cov": cov,
            "by1": by1, "by2": by2, "out_path": os.path.join(OUT, name)}
    return None, meta


def inpaint_stage(meta, model, gpu_lock, engine):
    """GPU stage: inpaint every frame -> final/ with the chosen engine. Serialized by
    gpu_lock so only one worker touches the (single, shared) model at a time -- VRAM stays
    at one model's cost regardless of --workers. Masks are dropped afterwards to release
    RAM. MiniMax reloads the BGR frames here (it needs them in memory for the band-crop);
    LaMa streams them from disk itself."""
    with gpu_lock:
        if engine == "minimax":
            frames = [cv2.imread(of) for of in meta["origs"]]
            it.minimax_frames(frames, meta["masks"], meta["final"],
                              meta["by1"], meta["by2"], meta["w"],
                              feather=FEATHER, pipe=model)
        else:
            it.lama_frames(meta["origs"], meta["masks"], meta["final"],
                           lama=model, feather=FEATHER, log=False)
    meta["masks"] = None


def encode_stage(meta):
    """CPU stage: re-encode final/ at native fps + copy original audio, then drop the
    work dir. Returns the manifest record."""
    it.sh(["ffmpeg", "-y", "-v", "error", "-framerate", f"{meta['fps']:.6f}",
           "-i", os.path.join(meta["final"], "%05d.png"), "-i",
           os.path.join(SRC, meta["name"]),
           "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
           "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy",
           "-shortest", meta["out_path"]])
    shutil.rmtree(meta["work"], ignore_errors=True)
    return {"name": meta["name"], "status": "done", "rects": meta["rects"],
            "w": meta["w"], "h": meta["h"], "frames": len(meta["origs"]),
            "cov": round(meta["cov"] * 100, 3)}


def process_clip(path, model, ocr, ocr_lock, gpu_lock, engine):
    """One clip through all three stages. preprocess + encode run on CPU (parallel across
    workers); inpaint holds gpu_lock (serial). A worker doing GPU work overlaps other
    workers' ffmpeg/OpenCV, which is where the speedup comes from."""
    _skip, meta = preprocess(path, ocr, ocr_lock)
    if meta is None:
        return _skip
    try:
        inpaint_stage(meta, model, gpu_lock, engine)
        return encode_stage(meta)
    finally:
        shutil.rmtree(meta["work"], ignore_errors=True)   # no-op if encode already cleaned


def main():
    global OUT, MANIFEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30, help="clips to process (evenly sampled)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="process every clip")
    ap.add_argument("--random", type=int, default=0, help="process N random clips")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --random")
    ap.add_argument("--engine", choices=("lama", "minimax"), default="lama",
                    help="lama (default) -> inpaint/eval/out + manifest.json ; minimax -> "
                         "inpaint/eval/out_minimax (SEPARATE folder, never overwrites LaMa; "
                         "needs the MINIMAX=1 image build).")
    ap.add_argument("--tag", default="",
                    help="suffix the output dir/manifest (e.g. --tag fix -> out_fix/) so a "
                         "re-run with changed detection never clobbers the baseline outputs.")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel CPU workers feeding the single GPU model "
                         "(0 = auto: min(4, cores/2), floor 1). Use 1 for serial on "
                         "low-RAM/low-core hosts.")
    a = ap.parse_args()

    OUT, MANIFEST = ENGINE_CFG[a.engine]
    if a.tag:
        OUT = f"{OUT}_{a.tag}"
        MANIFEST = os.path.join(OUT, "manifest.json")
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
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

    records = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            records = {r["name"]: r for r in json.load(f)}

    # skip already-done clips up front (resume)
    todo = [p for p in sel
            if not (os.path.exists(os.path.join(OUT, os.path.basename(p)))
                    and records.get(os.path.basename(p), {}).get("status") == "done")]
    workers = a.workers if a.workers > 0 else auto_workers()
    workers = max(1, min(workers, len(todo) or 1))
    print(f"[batch] engine={a.engine} -> {OUT} ; {len(sel)}/{len(clips)} clips ; "
          f"{len(todo)} to do ; workers={workers} (cores={os.cpu_count()})")

    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    if a.engine == "minimax":
        model = it.load_minimax()          # raises a clear rebuild hint if image lacks MiniMax
    else:
        from simple_lama_inpainting import SimpleLama
        model = SimpleLama()
    ocr_lock, gpu_lock = threading.Lock(), threading.Lock()

    def save_manifest():
        with open(MANIFEST, "w") as f:
            json.dump(list(records.values()), f, indent=1)

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_clip, p, model, ocr, ocr_lock, gpu_lock, a.engine):
                os.path.basename(p) for p in todo}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"name": name, "status": f"error: {e.__class__.__name__}: {e}"}
            done += 1
            records[name] = rec
            save_manifest()                                  # incremental (main thread only)
            avg = (time.time() - t0) / done
            print(f"[{done}/{len(todo)}] {rec['status']:12} {name}  "
                  f"({rec.get('frames', '?')}f, {len(rec.get('rects', []))} rect, "
                  f"avg {avg:.1f}s/clip)")
    print(f"[batch] done in {time.time() - t0:.1f}s -> {MANIFEST}")


if __name__ == "__main__":
    main()
