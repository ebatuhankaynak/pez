#!/usr/bin/env python3
"""
Split each clip into ALL of its detected segments (multi-cut), one file per segment.

Reads transitions/segments.json (produced by `face_cut.py --split segments`) -- the
refined, eval-scored person/meme segmentation. EVERY segment becomes its own
frame-accurate file, numbered per label. A clip that goes person->meme->person yields
three files:

    split/person/<clip>_person_1.mp4   [seg0.start, seg0.end]
    split/meme/<clip>_meme_1.mp4       [seg1.start, seg1.end]
    split/person/<clip>_person_2.mp4   [seg2.start, seg2.end]

so a meme segment starts exactly where the creator stops -- no creator tail bleeding
into the meme clip.

A single-segment clip (whole thing is one label) is stream-copied whole (lossless) to
its one labeled file. Multi-segment clips are frame-accurate re-encoded per segment.

Clips absent from segments.json are skipped with a warning (segmentation should cover
every clip, so this normally never fires).

Usage:
    python src/split_clips.py                 # split everything per segments.json
    python src/split_clips.py --workers 8
    python src/split_clips.py --dry-run
"""

import argparse
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent   # repo root (this file lives in src/)
CLIPS_DIR = SCRIPT_DIR / "freckled_spike_tiktok"
SEGMENTS = SCRIPT_DIR / "transitions" / "segments.json"
OUT_DIR = SCRIPT_DIR / "split"

FFMPEG = ["ffmpeg", "-y", "-v", "error"]


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return p.returncode, p.stderr.decode("utf-8", "ignore")[-400:]


def cut_segment(src, dst, start=None, end=None):
    """Frame-accurate re-encode of [start, end]. None means clip boundary."""
    cmd = list(FFMPEG)
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if end is not None:
        # -t is relative to the (post-seek) input start, so subtract start
        dur = end - (start or 0.0)
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-movflags", "+faststart", str(dst)]
    return run(cmd)


def copy_whole(src, dst):
    return run([*FFMPEG, "-i", str(src), "-c", "copy",
                "-movflags", "+faststart", str(dst)])


def plan_clip(rec, clips_dir, out_dir):
    """Return (actions, meta) for one clip.

    actions: list of (label, src, dst, start, end, mode) -- one per segment.
    meta:    manifest record describing the clip and every file it produced.
    """
    name = rec["clip"]
    stem = name[:-4] if name.lower().endswith(".mp4") else Path(name).stem
    src = clips_dir / name
    segs = rec.get("segments", []) or []

    if not src.exists() or not segs:
        disp = "skip_missing_src" if not src.exists() else "skip_no_segments"
        return [], {"clip": name, "short": rec.get("short"),
                    "pattern": rec.get("pattern"), "n_segments": len(segs),
                    "disposition": disp, "outputs": []}

    single = len(segs) == 1
    counters = Counter()
    actions, outputs = [], []
    for s in segs:
        label = s["label"]
        counters[label] += 1
        k = counters[label]
        dst = out_dir / label / f"{stem}_{label}_{k}.mp4"
        if single:
            # whole clip is one label -> lossless stream copy, no re-encode
            start = end = None
            mode = "copy"
        else:
            start, end = float(s["start"]), float(s["end"])
            mode = "cut"
        actions.append((label, src, dst, start, end, mode))
        outputs.append({"label": label, "index": k, "file": dst.name,
                        "start": s.get("start"), "end": s.get("end"),
                        "dur": s.get("dur"), "mode": mode})

    disp = "single_segment" if single else f"multi_{len(segs)}seg"
    meta = {"clip": name, "short": rec.get("short"), "pattern": rec.get("pattern"),
            "n_segments": len(segs), "disposition": disp, "outputs": outputs}
    return actions, meta


def execute(actions):
    errs = []
    for label, src, dst, start, end, mode in actions:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            rc, err = copy_whole(src, dst)
        else:
            rc, err = cut_segment(src, dst, start, end)
        if rc != 0:
            errs.append(f"{dst.name} [{label}]: {err}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default=str(SEGMENTS),
                    help="refined segmentation JSON from face_cut --split segments")
    ap.add_argument("--clips-dir", default=str(CLIPS_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir)
    out_dir = Path(args.out)

    with open(args.segments) as f:
        records = json.load(f)
    (out_dir / "person").mkdir(parents=True, exist_ok=True)
    (out_dir / "meme").mkdir(parents=True, exist_ok=True)

    disposition = Counter()
    seg_labels = Counter()
    all_actions = []
    manifest = []
    for rec in records:
        actions, meta = plan_clip(rec, clips_dir, out_dir)
        disposition[meta["disposition"]] += 1
        for o in meta["outputs"]:
            seg_labels[o["label"]] += 1
        all_actions.append((rec["clip"], actions))
        manifest.append(meta)

    print("Plan:")
    for disp, n in disposition.most_common():
        print(f"  {n:3d}  {disp}")
    total_out = sum(len(a) for _, a in all_actions)
    print(f"  => {total_out} segment files "
          f"({seg_labels.get('meme', 0)} meme, {seg_labels.get('person', 0)} person) "
          f"into {out_dir}/person and {out_dir}/meme")

    if args.dry_run:
        print("\n(dry run, nothing written)")
        return

    all_errs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(execute, actions): name
                for name, actions in all_actions if actions}
        done = 0
        for fut in as_completed(futs):
            done += 1
            all_errs.extend(fut.result())
            if done % 20 == 0 or done == len(futs):
                print(f"  processed {done}/{len(futs)} clips")

    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_person = len(list((out_dir / "person").glob("*.mp4")))
    n_meme = len(list((out_dir / "meme").glob("*.mp4")))
    print(f"\nDone. person/={n_person} files, meme/={n_meme} files.")
    if all_errs:
        print(f"{len(all_errs)} errors:")
        for e in all_errs[:10]:
            print("  ", e)


if __name__ == "__main__":
    main()
