#!/usr/bin/env python3
"""EXPERIMENT: ignore TransNetV2 entirely. Derive cuts purely from the dense face-sim
curve — sample the clip, mark each frame creator/meme by sim>=thr, merge short runs,
and cut at every present<->absent boundary. Writes a transitions-shaped file so
evaluate.py can score it. Does NOT touch the canonical pipeline.

    python attic/face_only.py --sample-fps 8 --out <scratch>/faceonly.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from decord import VideoReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # import the pipeline modules
from relabel_faces import load_face_app, enroll_creator, normed, CLIPS_DIR, TRANSITIONS, short


def sim_curve(app, src, centroid, sample_fps):
    vr = VideoReader(str(src)); fps = vr.get_avg_fps() or 25.0; total = len(vr); dur = total / fps
    n = max(2, int(dur * sample_fps))
    times = [dur * (k + 0.5) / n for k in range(n)]
    sims = []
    for t in times:
        idx = min(int(t * fps), total - 1)
        bgr = np.ascontiguousarray(vr[idx].asnumpy()[:, :, ::-1])
        best = 0.0
        for f in app.get(bgr):
            best = max(best, float(normed(f.normed_embedding) @ centroid))
        sims.append(best)
    return times, sims, dur


def merge_runs(present, min_frames):
    segs = []
    for i, p in enumerate(present):
        if segs and segs[-1][0] == p:
            segs[-1][2] = i
        else:
            segs.append([p, i, i])
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for s in segs:
            if s[2] - s[1] + 1 < min_frames:
                s[0] = not s[0]; changed = True; break
        if changed:
            new = []
            for s in segs:
                if new and new[-1][0] == s[0]:
                    new[-1][2] = s[2]
                else:
                    new.append(list(s))
            segs = new
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--transitions", default=str(TRANSITIONS))
    ap.add_argument("--face-threshold", type=float, default=0.35)
    ap.add_argument("--sample-fps", type=float, default=8.0)
    ap.add_argument("--min-seg", type=float, default=0.5, help="merge runs shorter than this (s)")
    ap.add_argument("--evidence-floor", type=float, default=0.15)
    args = ap.parse_args()

    records = json.load(open(args.transitions))
    app = load_face_app()
    print("Enrolling creator...", flush=True)
    centroid = enroll_creator(app, records)
    min_frames = max(1, int(args.min_seg * args.sample_fps))

    out = []
    for i, c in enumerate(records, 1):
        src = CLIPS_DIR / c["clip"]
        if not src.exists():
            out.append({**c, "face_error": True}); continue
        times, sims, dur = sim_curve(app, src, centroid, args.sample_fps)
        present = [s >= args.face_threshold for s in sims]
        segs = merge_runs(present, min_frames)
        shots = []
        for lab, a, b in segs:
            shots.append({"start_sec": round(times[a], 3), "end_sec": round(times[b], 3),
                          "label": "person" if lab else "meme",
                          "face_sim": round(float(np.median(sims[a:b + 1])), 3)})
        # transition = first person->meme boundary
        trans, method = None, "all_creator_no_transition"
        if all(not p for p, *_ in segs):
            method = "all_meme_no_creator" if max(sims) < args.evidence_floor else "single_shot_creator"
        else:
            for j in range(1, len(segs)):
                if segs[j - 1][0] and not segs[j][0]:
                    trans = round(times[segs[j][1]], 3); method = "creator_to_meme"; break
        out.append({"clip": c["clip"], "transition_sec": trans, "method": method,
                    "num_shots": len(shots), "shots": shots})
        if i % 20 == 0 or i == len(records):
            print(f"  {i}/{len(records)}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
