#!/usr/bin/env python3
"""
Stage 1 - shot boundary detection (TransNetV2).

Splits every clip into shots and writes transitions/transitions.json holding the
raw shots. Labeling ("is this shot the creator?") and the person->meme cut are
decided in stage 2 by relabel_faces.py, using the creator's face.

This stage runs ONLY TransNetV2 - no OpenCLIP / no CLIP.

    python detect_transitions.py            # all clips
    python detect_transitions.py --limit 5
"""

import argparse
import json
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CLIPS = SCRIPT_DIR / "freckled_spike_tiktok"
DEFAULT_OUT = SCRIPT_DIR / "transitions"


def load_transnet(device):
    from transnetv2_pytorch import TransNetV2
    model = TransNetV2(device=device)
    model.eval()
    return model


def _project_to_true_clock(video_path, shots):
    """TransNetV2 decodes at a constant NOMINAL rate, so on variable-frame-rate clips its
    shot times sit on the wrong clock (and its frame indices in a different space). Re-project
    every boundary onto the real per-frame PTS — the clock the player and the ground truth use —
    via the nearest real frame, so stage 2, the segments, and the UI all agree."""
    import bisect
    from decord import VideoReader
    vr = VideoReader(str(video_path))
    n = len(vr)
    try:
        pts = [float(x[0]) for x in vr.get_frame_timestamp(list(range(n)))]
    except Exception:
        pts = [float(vr.get_frame_timestamp(i)[0]) for i in range(n)]

    def near(t):
        i = bisect.bisect_left(pts, t)
        if i <= 0:
            return 0
        return n - 1 if i >= n else (i if (pts[i] - t) < (t - pts[i - 1]) else i - 1)

    for s in shots:
        si, ei = near(s["start_sec"]), near(s["end_sec"])
        s["start_frame"], s["end_frame"] = si, ei
        s["start_sec"], s["end_sec"] = round(pts[si], 3), round(pts[ei], 3)
    return shots


def detect_shots(transnet, video_path, threshold):
    """TransNetV2 shot boundaries -> list of shot dicts (times on the true per-frame clock)."""
    scenes = transnet.detect_scenes(str(video_path), threshold=threshold)
    shots = []
    for s in scenes:
        shots.append({
            "start_sec": round(float(s["start_time"]), 3),
            "end_sec": round(float(s["end_time"]), 3),
            "start_frame": int(s["start_frame"]),
            "end_frame": int(s["end_frame"]),
        })
    return _project_to_true_clock(video_path, shots)


def main():
    ap = argparse.ArgumentParser(description="Shot boundary detection (stage 1)")
    ap.add_argument("--clips-dir", default=str(DEFAULT_CLIPS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--pattern", default="*.mp4")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="TransNetV2 cut threshold (lower = more cuts)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    clips_dir = Path(args.clips_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(clips_dir.glob(args.pattern))
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        print(f"No clips matching {args.pattern} in {clips_dir}")
        return

    print(f"Device: {device} | clips: {len(clips)}")
    print("Loading TransNetV2 (once)...")
    transnet = load_transnet(device)

    results = []
    t0 = time.time()
    for i, clip in enumerate(clips, 1):
        try:
            shots = detect_shots(transnet, clip, args.threshold)
            results.append({
                "clip": clip.name,
                "transition_sec": None,          # filled by relabel_faces.py
                "method": "shots_detected",       # ""
                "num_shots": len(shots),
                "shots": shots,
            })
            print(f"[{i}/{len(clips)}] {clip.name[:48]:48s} shots={len(shots)}")
        except Exception as e:
            print(f"[{i}/{len(clips)}] {clip.name} FAILED: {e}")
            results.append({"clip": clip.name, "error": str(e)})

    out_json = out_dir / "transitions.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s. {len(results)} clips -> {out_json}")
    print("Next: relabel_faces.py  (label shots by the creator's face + pick the cut)")


if __name__ == "__main__":
    main()
