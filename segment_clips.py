#!/usr/bin/env python3
"""
Multi-segment labeling: instead of one person->meme cut, emit the FULL sequence
of creator / meme segments per clip (person -> meme -> person -> meme -> ...).

It reuses the per-shot creator-face score already stored in transitions.json:
  - face_sim   : similarity of the shot to the creator's face

Knobs to play with:
  --face-threshold 0.35       face_sim >= threshold -> "person", else "meme"
  --min-seg 1.0               segments shorter than this (s) are absorbed into
                              their neighbours -> suppresses single-shot label
                              noise (a stray face dropout won't create a segment)

Outputs transitions/segments.json:  [{clip, segments:[{start,end,dur,label}], ...}]
With --split, also cuts each clip into segments/<clip>/NN_<label>.mp4 in order.

    conda run -n wedia_telif python segment_clips.py --min-seg 1.0
    conda run -n wedia_telif python segment_clips.py --min-seg 0.5 --split
"""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CLIPS_DIR = SCRIPT_DIR / "freckled_spike_tiktok"
TRANSITIONS = SCRIPT_DIR / "transitions" / "transitions.json"


def label_shots(shots, thr):
    return ["person" if (s.get("face_sim", 0.0) or 0.0) >= thr else "meme" for s in shots]


def segments_from(shots, labels):
    """Merge consecutive same-label shots into segments."""
    segs = []
    i = 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        segs.append({
            "start": round(shots[i]["start_sec"], 3),
            "end": round(shots[j]["end_sec"], 3),
            "label": labels[i],
        })
        i = j + 1
    for s in segs:
        s["dur"] = round(s["end"] - s["start"], 3)
    return segs


def smooth(shots, labels, min_seg):
    """Absorb any segment shorter than min_seg into its neighbours (repeat)."""
    labels = labels[:]
    while True:
        # build (start_idx, end_idx, label) runs
        runs, i = [], 0
        while i < len(labels):
            j = i
            while j + 1 < len(labels) and labels[j + 1] == labels[i]:
                j += 1
            runs.append((i, j, labels[i]))
            i = j + 1
        if len(runs) <= 1:
            break
        # find the shortest run under min_seg; flip it toward its neighbours
        flipped = False
        for k, (a, b, lab) in enumerate(runs):
            dur = shots[b]["end_sec"] - shots[a]["start_sec"]
            if dur < min_seg:
                nb = runs[k - 1][2] if k > 0 else runs[k + 1][2]
                for idx in range(a, b + 1):
                    labels[idx] = nb
                flipped = True
                break
        if not flipped:
            break
    return labels


def cut(src, start, end, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src),
                    "-t", f"{end - start:.3f}", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-c:a", "aac", "-movflags", "+faststart", str(dst)],
                   stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", default=str(TRANSITIONS))
    ap.add_argument("--out", default=str(SCRIPT_DIR / "transitions" / "segments.json"))
    ap.add_argument("--face-threshold", type=float, default=0.35)
    ap.add_argument("--min-seg", type=float, default=1.0,
                    help="Absorb segments shorter than this many seconds (noise suppression)")
    ap.add_argument("--split", action="store_true", help="Also cut clips into segments/<clip>/NN_label.mp4")
    ap.add_argument("--seg-dir", default=str(SCRIPT_DIR / "segments"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    records = json.load(open(args.transitions))
    out, multi, cut_jobs = [], [], []
    for c in records:
        shots = c.get("shots", [])
        if not shots:
            continue
        labels = smooth(shots, label_shots(shots, args.face_threshold), args.min_seg)
        segs = segments_from(shots, labels)
        n_person = sum(1 for s in segs if s["label"] == "person")
        n_meme = sum(1 for s in segs if s["label"] == "meme")
        rec = {"clip": c["clip"], "n_segments": len(segs),
               "pattern": "→".join(s["label"][:4] for s in segs), "segments": segs}
        out.append(rec)
        if len(segs) >= 3:
            multi.append(rec)
        if args.split:
            stem = c["clip"][:-4]
            for k, s in enumerate(segs, 1):
                dst = Path(args.seg_dir) / stem / f"{k:02d}_{s['label']}.mp4"
                cut_jobs.append((CLIPS_DIR / c["clip"], s["start"], s["end"], dst))

    Path(args.out).write_text(json.dumps(out, indent=2))

    from collections import Counter
    dist = Counter(r["n_segments"] for r in out)
    print(f"threshold={args.face_threshold} min-seg={args.min_seg}s")
    print("segments-per-clip distribution:")
    for k in sorted(dist):
        print(f"  {k} segment(s): {dist[k]} clips")
    print(f"\nmulti-segment clips (3+ segments): {len(multi)}")
    for r in multi:
        spans = "  ".join(f"{s['label'][0]}:{s['start']}-{s['end']}" for s in r["segments"])
        print(f"  {r['clip'][:-4][-12:]}  {r['pattern']:28}  {spans}")
    print(f"\nWrote {args.out}")

    if args.split and cut_jobs:
        print(f"Cutting {len(cut_jobs)} segment files -> {args.seg_dir}/ ...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(lambda j: cut(*j), cut_jobs))
        print("Done.")


if __name__ == "__main__":
    main()
