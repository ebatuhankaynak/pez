#!/usr/bin/env python3
"""
Split each clip at its detected person->meme transition into two folders:

    split/person/<clip>.mp4   the creator-talking-to-camera part  [0, transition]
    split/meme/<clip>.mp4     the meme part                        [transition, end]

Reads transitions/transitions.json (produced by detect_transitions.py).

No-transition clips are routed whole to the side that matches their method:
    single_shot_creator / all_creator_no_transition  -> person/ only
    all_meme_no_creator                              -> meme/ only

Cuts are frame-accurate (re-encoded). Whole-clip copies are stream-copied (lossless).

Usage:
    python src/split_clips.py                 # split everything per transitions.json
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
TRANSITIONS = SCRIPT_DIR / "transitions" / "transitions.json"
OUT_DIR = SCRIPT_DIR / "split"

FFMPEG = ["ffmpeg", "-y", "-v", "error"]

# No-transition method names emitted by the face-based labeler (relabel_faces.py).
PERSON_ONLY = {"single_shot_creator", "all_creator_no_transition"}
MEME_ONLY = {"all_meme_no_creator"}


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
        # -to is relative to the (post-seek) input start, so subtract start
        dur = end - (start or 0.0)
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-movflags", "+faststart", str(dst)]
    return run(cmd)


def copy_whole(src, dst):
    return run([*FFMPEG, "-i", str(src), "-c", "copy",
                "-movflags", "+faststart", str(dst)])


def plan_clip(rec, clips_dir, out_dir):
    """Return list of (kind, src, dst, start, end, mode) actions for one clip."""
    name = rec["clip"]
    src = clips_dir / name
    t = rec.get("transition_sec")
    method = rec.get("method")
    actions = []
    if "error" in rec or not src.exists():
        return actions, "skip_missing_or_error"
    if t is not None:
        actions.append(("person", src, out_dir / "person" / name, None, t, "cut"))
        actions.append(("meme", src, out_dir / "meme" / name, t, None, "cut"))
        return actions, "split"
    if method in PERSON_ONLY:
        actions.append(("person", src, out_dir / "person" / name, None, None, "copy"))
        return actions, "person_only"
    if method in MEME_ONLY:
        actions.append(("meme", src, out_dir / "meme" / name, None, None, "copy"))
        return actions, "meme_only"
    return actions, "skip_unknown"


def execute(actions):
    errs = []
    for kind, src, dst, start, end, mode in actions:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            rc, err = copy_whole(src, dst)
        else:
            rc, err = cut_segment(src, dst, start, end)
        if rc != 0:
            errs.append(f"{dst.name} [{kind}]: {err}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", default=str(TRANSITIONS))
    ap.add_argument("--clips-dir", default=str(CLIPS_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir)
    out_dir = Path(args.out)

    with open(args.transitions) as f:
        records = json.load(f)
    (out_dir / "person").mkdir(parents=True, exist_ok=True)
    (out_dir / "meme").mkdir(parents=True, exist_ok=True)

    disposition = Counter()
    all_actions = []
    manifest = []
    for rec in records:
        actions, disp = plan_clip(rec, clips_dir, out_dir)
        disposition[disp] += 1
        all_actions.append((rec["clip"], actions))
        manifest.append({"clip": rec["clip"], "disposition": disp,
                         "transition_sec": rec.get("transition_sec"),
                         "method": rec.get("method"),
                         "person": any(a[0] == "person" for a in actions),
                         "meme": any(a[0] == "meme" for a in actions)})

    print("Plan:")
    for disp, n in disposition.most_common():
        print(f"  {n:3d}  {disp}")
    total_out = sum(len(a) for _, a in all_actions)
    print(f"  => {total_out} output files into {out_dir}/person and {out_dir}/meme")

    if args.dry_run:
        print("\n(dry run, nothing written)")
        return

    all_errs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(execute, actions): name for name, actions in all_actions if actions}
        done = 0
        for fut in as_completed(futs):
            done += 1
            errs = fut.result()
            all_errs.extend(errs)
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
