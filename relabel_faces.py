#!/usr/bin/env python3
"""
Re-label shots by CREATOR FACE instead of CLIP, then re-pick the transition.

Why: TransNetV2 finds the cuts correctly; the failures come from the CLIP
"creator vs meme" labeler being fooled by memes that contain a person (or dark
footage). Since the SAME creator appears in every clip, we can instead ask a much
sharper question per shot: "is *her* face in this shot?" A meme showing a
different person, or no person, no longer looks like the creator.

Pipeline (reuses the cached shots in transitions.json — no TransNetV2 re-run):
  1. ENROLL: sample ~1s into every clip, embed the largest face (ArcFace), and
     take the dominant recurring identity as the creator.
  2. RELABEL: for each shot, sample a few frames; shot is "creator" iff any face
     matches the creator embedding (cosine >= --face-threshold), else "meme".
  3. PICK: transition = start of the first meme shot after the leading creator run.

Writes transitions_face.json and prints accuracy vs transitions/verification.json.

    conda run -n wedia_telif python relabel_faces.py
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from decord import VideoReader

SCRIPT_DIR = Path(__file__).resolve().parent
CLIPS_DIR = SCRIPT_DIR / "freckled_spike_tiktok"
TRANSITIONS = SCRIPT_DIR / "transitions" / "transitions.json"
VERIFICATION = SCRIPT_DIR / "transitions" / "verification.json"


def short(name):
    return name[:-4][-12:]


def save_qa(video_path, transition_sec, out_dir, stem):
    """Dump the frames ~0.15s before/after the cut, side by side, for eyeballing."""
    if transition_sec is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    before = max(0.0, transition_sec - 0.15)
    after = transition_sec + 0.15
    tmp_b, tmp_a = out_dir / f"{stem}_b.png", out_dir / f"{stem}_a.png"
    combo = out_dir / f"{stem}_transition.png"
    for t, dst in [(before, tmp_b), (after, tmp_a)]:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video_path),
                        "-frames:v", "1", "-vf", "scale=240:-1", str(dst)], check=False)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp_b), "-i", str(tmp_a),
                    "-filter_complex", "hstack", str(combo)], check=False)
    tmp_b.unlink(missing_ok=True)
    tmp_a.unlink(missing_ok=True)
    return combo.name if combo.exists() else None


def normed(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_face_app():
    from insightface.app import FaceAnalysis
    import onnxruntime as ort
    root = os.environ.get("INSIGHTFACE_HOME", os.path.expanduser("~/.insightface"))
    gpu = "CUDAExecutionProvider" in ort.get_available_providers()
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu
                 else ["CPUExecutionProvider"])
    app = FaceAnalysis(name="buffalo_l", root=root, providers=providers)
    app.prepare(ctx_id=0 if gpu else -1, det_size=(384, 384))
    print(f"[face] running on {'GPU (CUDA)' if gpu else 'CPU'}", flush=True)
    return app


def faces_at(app, vr, fps, t, total):
    idx = min(int(t * fps), total - 1)
    if idx < 0:
        return []
    rgb = vr[idx].asnumpy()
    bgr = rgb[:, :, ::-1]
    return app.get(bgr)


def largest_face_emb(faces):
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return normed(f.normed_embedding)


def enroll_creator(app, records):
    """Dominant recurring face across clip intros = the creator."""
    embs = []
    for c in records:
        src = CLIPS_DIR / c["clip"]
        if not src.exists():
            continue
        try:
            vr = VideoReader(str(src))
            fps = vr.get_avg_fps()
            total = len(vr)
            e = largest_face_emb(faces_at(app, vr, fps, 1.0, total))
            if e is not None:
                embs.append(e)
        except Exception:
            continue
    embs = np.array(embs)
    print(f"  enrollment faces collected: {len(embs)}")
    # Robust centroid: start from mean, keep inliers, refeat.
    centroid = normed(embs.mean(axis=0))
    for _ in range(4):
        sims = embs @ centroid
        inliers = embs[sims >= 0.35]
        if len(inliers) < 5:
            break
        centroid = normed(inliers.mean(axis=0))
    frac = float((embs @ centroid >= 0.35).mean())
    print(f"  creator identity covers {frac*100:.0f}% of intros (sanity: should be high)")
    return centroid


def label_shots(app, src, shots, centroid, thr, frames_per_shot=5):
    vr = VideoReader(str(src))
    fps = vr.get_avg_fps()
    total = len(vr)
    out = []
    for s in shots:
        a, b = s["start_sec"], s["end_sec"]
        ts = [a + (b - a) * (k + 0.5) / frames_per_shot for k in range(frames_per_shot)]
        # Per-frame best match to the creator, then take the MEDIAN across the shot
        # so a face merely *lingering* into a dissolve/text-card doesn't keep the
        # whole shot labeled "creator" (that pushed the cut one shot late).
        frame_sims = []
        for t in ts:
            faces = faces_at(app, vr, fps, t, total)
            fb = 0.0
            for f in faces:
                fb = max(fb, float(normed(f.normed_embedding) @ centroid))
            frame_sims.append(fb)
        out.append(float(np.median(frame_sims)))
    return out


def pick(labels_creator, sims=None, evidence_floor=0.15):
    """labels_creator: list of bool (is creator). Return (transition_idx, method).

    DOMAIN RULE: the creator is ALWAYS present and opens every clip; a clip can be
    creator-only but never meme-only. So the leading shot is always the creator, and
    the transition is where she first leaves. If she never clearly leaves -> all-creator.

    Max-agreement split (maximizes #creator-before + #meme-after) tolerates an isolated
    face dropout mid-intro; a result of "no leading creator" usually means her dark/occluded
    intro fell just under the face threshold, so we treat the first cut as the transition.

    The rule is a strong prior for *resolving ambiguity*, not a hard override: if the whole
    clip shows essentially NO face of her (max sim < evidence_floor, e.g. pure reposted
    footage), we respect that and report no creator rather than inventing one.
    """
    n = len(labels_creator)
    if n == 1:
        # One shot -> no boundary to split on. She opens every clip, so a lone shot
        # is her *unless* there's essentially no face evidence at all (a genuine
        # creator-less repost), which the evidence floor lets us respect.
        if sims is not None and max(sims) < evidence_floor:
            return None, "all_meme_no_creator"
        return None, "single_shot_creator"        # she's on screen throughout
    best_i, best = 0, -1
    for i in range(0, n + 1):
        score = sum(labels_creator[:i]) + sum(1 for x in labels_creator[i:] if not x)
        if score > best:
            best, best_i = score, i
    if best_i == n:
        return None, "all_creator_no_transition"  # never leaves the creator
    if best_i == 0:
        if sims is not None and max(sims) < evidence_floor:
            return None, "all_meme_no_creator"    # she is genuinely absent (no face anywhere)
        return 1, "creator_to_meme_prior"         # face missed her intro; she opens -> first cut
    return best_i, "creator_to_meme"


def dense_transition(app, src, centroid, thr, sample_fps=4.0):
    """Soft-cut fallback: TransNetV2 found no hard cut, so scan the WHOLE clip for
    where the creator's face presence drops. Returns the transition time or None.

    Sample the clip densely, mark each frame present/absent (her face), then take
    the same max-agreement split on the per-frame presence curve. Only fires when
    there is a clear present-prefix -> absent-suffix (a real fade/dissolve away
    from her); returns None if she is never present or never leaves.
    """
    vr = VideoReader(str(src))
    fps = vr.get_avg_fps()
    total = len(vr)
    dur = total / fps if fps else 0
    n = max(2, int(dur * sample_fps))
    times = [dur * (k + 0.5) / n for k in range(n)]
    present = []
    for t in times:
        best = 0.0
        for f in faces_at(app, vr, fps, t, total):
            best = max(best, float(normed(f.normed_embedding) @ centroid))
        present.append(best >= thr)

    m = len(present)
    # Conservative guards: only trust a soft cut when she clearly OPENS the clip
    # (else we invent cuts in all-meme clips) and the meme tail is clearly not her
    # (else we cut early on flickery detections). These clips have poor face
    # detection, so without the guards the dense scan does more harm than good.
    lead = max(2, m // 5)
    if sum(present[:lead]) < 0.7 * lead:
        return None

    best_i, best = 0, -1
    for i in range(m + 1):
        score = sum(present[:i]) + sum(1 for x in present[i:] if not x)
        if score > best:
            best, best_i = score, i
    if best_i == 0 or best_i == m:
        return None
    if times[-1] - times[best_i] < 1.0:          # meme tail < 1s -> not a real meme; she's on
        return None                              #   screen the whole clip (all-creator)
    tail = present[best_i:]
    if sum(1 for x in tail if not x) / len(tail) < 0.8:
        return None
    return round(times[best_i], 3)


def refine_within_shot(app, src, a, b, centroid, thr, sample_fps=6.0, hi=0.45):
    """A hard cut was picked at the END of the leading creator shot [a,b], but a
    SOFT fade to the meme may sit INSIDE that shot (TransNetV2 didn't split it).
    Scan [a,b] densely and find where her CONFIDENT presence run ends. We key on a
    high similarity (>= hi) so that stray faces in the meme (e.g. a red-carpet
    crowd) that weakly match her can't hold the run open. Returns that time, else None.
    """
    vr = VideoReader(str(src))
    fps = vr.get_avg_fps()
    total = len(vr)
    n = max(3, int((b - a) * sample_fps))
    times = [a + (b - a) * (k + 0.5) / n for k in range(n)]
    sims = []
    for t in times:
        best = 0.0
        for f in faces_at(app, vr, fps, t, total):
            best = max(best, float(normed(f.normed_embedding) @ centroid))
        sims.append(best)
    lead = max(2, n // 5)
    # Only trust an in-shot refine when she STRONGLY opens the shot. If her face
    # only weakly matches (~0.4) throughout, the confident threshold below would
    # cut too early, so we bail and keep the hard-cut boundary instead.
    if sum(sims[:lead]) / lead < 0.55:
        return None
    present = [s >= hi for s in sims]            # confident-her (hi), so meme crowd faces don't count
    # Max-agreement split: robust to brief mid-talk dips (a momentary drop with
    # presence resuming won't beat the real leave point).
    best_i, best = 0, -1
    for i in range(n + 1):
        score = sum(present[:i]) + sum(1 for x in present[i:] if not x)
        if score > best:
            best, best_i = score, i
    if best_i == 0 or best_i == n:               # she never clearly leaves this shot
        return None
    tail = present[best_i:]
    if sum(1 for x in tail if not x) / len(tail) < 0.6:
        return None
    return round(times[best_i], 3)


_TRANSNET = None


def redetect_lowthr(src, threshold, device="auto"):
    """Last resort: a clip collapsed to a SINGLE shot (TransNetV2 merged a fast
    match-cut) and the face pass found no cut either. Re-run TransNetV2 on just
    this one clip at a lower threshold to recover the boundary. Loaded lazily so
    the common path never pays the TransNetV2 import/model cost."""
    global _TRANSNET
    if _TRANSNET is None:
        import torch
        from transnetv2_pytorch import TransNetV2
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        _TRANSNET = TransNetV2(device=dev)
        _TRANSNET.eval()
        print(f"  [lowthr fallback] loaded TransNetV2 on {dev}", flush=True)
    scenes = _TRANSNET.detect_scenes(str(src), threshold=threshold)
    return [{"start_sec": round(float(s["start_time"]), 3),
             "end_sec": round(float(s["end_time"]), 3),
             "start_frame": int(s["start_frame"]), "end_frame": int(s["end_frame"])}
            for s in scenes]


def label_and_pick(app, src, shots, centroid, args):
    """Face-label the shots, pick the person->meme cut, and apply the soft-cut
    fallbacks. Returns (trans, method, sims, is_creator)."""
    sims = label_shots(app, src, shots, centroid, args.face_threshold, args.frames_per_shot)
    is_creator = [sim >= args.face_threshold for sim in sims]
    idx, method = pick(is_creator, sims)
    trans = shots[idx]["start_sec"] if idx is not None else None
    if not args.no_soft_fallback:
        if trans is None:
            # No hard-cut boundary at all -> scan the whole clip.
            ft = dense_transition(app, src, centroid, args.face_threshold)
            if ft is not None:
                trans, method = ft, "creator_to_meme_soft"
        elif idx >= 1:
            # A hard cut was picked, but the real (soft) cut may sit inside the
            # last creator shot, which TransNetV2 failed to split. Refine earlier.
            a, b = shots[idx - 1]["start_sec"], shots[idx - 1]["end_sec"]
            if b - a > 0.8:
                rt = refine_within_shot(app, src, a, b, centroid, args.face_threshold)
                # Accept only a MODERATE earlier move: a missed soft cut sits just
                # before the next hard cut. A big jump (>2.5s) means she occluded
                # her own face mid-talk (drinking, mirror), not a real soft cut.
                if rt is not None and trans - 2.5 <= rt < trans - 0.4:
                    trans, method = rt, "creator_to_meme_softcut"
    return trans, method, sims, is_creator


def main():
    ap = argparse.ArgumentParser()
    # Reads the shots produced by detect_transitions.py and rewrites transitions.json
    # in place with face-based labels + picks (the canonical final step of the pipeline).
    ap.add_argument("--transitions", default=str(TRANSITIONS))
    ap.add_argument("--out", default=str(TRANSITIONS))
    ap.add_argument("--face-threshold", type=float, default=0.35)
    ap.add_argument("--frames-per-shot", type=int, default=3)
    ap.add_argument("--qa", action="store_true",
                    help="Also dump before/after-cut frames to transitions/qa/")
    ap.add_argument("--no-soft-fallback", action="store_true",
                    help="Disable the dense face-presence fallback for soft cuts")
    ap.add_argument("--lowthr-redetect", type=float, default=0.4,
                    help="When a clip is a single shot with NO cut found, re-detect "
                         "just that clip at this lower TransNetV2 threshold to recover "
                         "a merged match-cut.")
    ap.add_argument("--no-lowthr-fallback", action="store_true",
                    help="Disable the single-shot low-threshold re-detect fallback")
    ap.add_argument("--device", default="auto",
                    help="Device for the low-threshold re-detect (auto/cuda/cpu)")
    args = ap.parse_args()

    qa_dir = Path(args.out).parent / "qa"

    records = json.load(open(args.transitions))
    print("Loading InsightFace (buffalo_l)...")
    app = load_face_app()

    print("Enrolling creator face from clip intros...")
    centroid = enroll_creator(app, records)

    print(f"Re-labeling {len(records)} clips by face (threshold {args.face_threshold})...")
    out_records = []
    for i, c in enumerate(records, 1):
        src = CLIPS_DIR / c["clip"]
        shots = c.get("shots", [])
        if "error" in c or not src.exists() or not shots:
            out_records.append({**c, "face_error": True})
            continue
        trans, method, sims, is_creator = label_and_pick(app, src, shots, centroid, args)
        # LAST RESORT: the clip is a single shot AND neither the boundaries nor the
        # face pass found a cut. That's the fast-match-cut case (TransNetV2 merged
        # the person->meme cut and her intro fell under the face threshold). Re-detect
        # THIS clip at a lower threshold and re-pick. Genuine creator-less / no-
        # transition clips stay silent here (the evidence floor still applies), so
        # this can only recover a real merged cut, never invent one.
        if trans is None and len(shots) == 1 and not args.no_lowthr_fallback:
            fb_shots = redetect_lowthr(src, args.lowthr_redetect, args.device)
            if len(fb_shots) > 1:
                t2, m2, s2, c2 = label_and_pick(app, src, fb_shots, centroid, args)
                shots, sims, is_creator = fb_shots, s2, c2
                trans, method = t2, m2
                if t2 is not None:
                    print(f"    [lowthr] {short(c['clip'])}: recovered cut at {t2}s "
                          f"({len(fb_shots)} shots @ {args.lowthr_redetect})", flush=True)
        # Make the shot LABELS honor the pick/domain-prior, not just the raw face
        # threshold. When the prior concludes she opens the clip (her intro scored
        # below the face threshold but she is present), the leading shots up to the
        # cut are the creator — so a clip is never shown as pure 'meme' when we've
        # decided she's the one talking at the top.
        labels_creator = list(is_creator)
        if method == "creator_to_meme_prior" and trans is not None:
            labels_creator = [s["start_sec"] < trans - 1e-6 for s in shots]
        elif method in ("single_shot_creator", "all_creator_no_transition"):
            labels_creator = [True] * len(shots)
        new_shots = []
        for s, sim, cr in zip(shots, sims, labels_creator):
            ns = dict(s)
            ns["face_sim"] = round(sim, 3)
            ns["label"] = "person" if cr else "meme"
            new_shots.append(ns)
        rec = {
            "clip": c["clip"],
            "transition_sec": round(trans, 3) if trans is not None else None,
            "method": method,
            "num_shots": len(shots),
            "shots": new_shots,
        }
        if args.qa:
            rec["qa_image"] = save_qa(src, rec["transition_sec"], qa_dir, short(c["clip"]))
        out_records.append(rec)
        if i % 20 == 0 or i == len(records):
            print(f"  {i}/{len(records)}")

    Path(args.out).write_text(json.dumps(out_records, indent=2))
    print(f"Wrote {args.out}")

    # Evaluate vs verified truth
    try:
        ver = json.load(open(VERIFICATION))["by_clip"]
    except Exception:
        ver = {}
    if ver:
        tol = 0.5
        ok = close = bad = 0
        flips = []
        base = {c["clip"]: c.get("transition_sec") for c in records}
        for c in out_records:
            name = c["clip"]
            d = c.get("transition_sec")
            t = ver.get(short(name), {}).get("true_sec")
            b = base.get(name)
            if d is None and t is None:
                ok += 1
            elif d is not None and t is not None:
                e = abs(d - t)
                if e <= tol: ok += 1
                elif e <= 1.5: close += 1
                else: bad += 1;
            else:
                bad += 1
            # track clips the new method changed materially
            if (b is None) != (d is None) or (b is not None and d is not None and abs(b - d) > 0.5):
                flips.append((short(name), b, d, t))
        tot = len(out_records)
        print(f"\n=== FACE labeler vs verified truth ===")
        print(f"  exact (<={tol}s): {ok}/{tot} = {ok/tot*100:.1f}%")
        print(f"  within 1.5s:    {(ok+close)/tot*100:.1f}%")
        print(f"  wrong:          {bad}")
        print(f"\n  clips changed vs CLIP picker (short, clip_time, face_time, true):")
        for s, b, d, t in flips[:40]:
            print(f"    {s:14} clip={str(b):>6}  face={str(d):>6}  true={str(t):>5}")


if __name__ == "__main__":
    main()
