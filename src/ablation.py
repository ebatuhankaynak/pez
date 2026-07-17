#!/usr/bin/env python3
"""Ablation: score CLIP-only / face-only / CLIP+face / +fallback against the ground truth.

All configs use the SAME shots (TransNetV2) and the SAME max-agreement picker, so the
only thing that varies is the labeling signal (and the soft-cut fallback for the last).
Signals are read from stored results:
  p_person  -> transitions_clip.json (the old OpenCLIP labeler)
  face_sim  -> transitions.json      (InsightFace)
The +fallback column grafts the face-based soft-cut recoveries from the shipped run.
"""
import json
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "transitions"   # repo root/transitions (this file lives in src/)
clip = {c["clip"]: c for c in json.load(open(P / "transitions_clip.json"))}
face = {c["clip"]: c for c in json.load(open(P / "transitions.json"))}
gt = {c["short"]: c for c in json.load(open(P / "ground_truth.json"))["clips"]}
short = lambda n: n[:-4][-12:]


def pick(isc):
    """Max-agreement split -> transition shot index, or None."""
    n = len(isc)
    if n == 1:
        return None
    best_i, best = 0, -1
    for i in range(n + 1):
        s = sum(isc[:i]) + sum(1 for x in isc[i:] if not x)
        if s > best:
            best, best_i = s, i
    return None if best_i in (0, n) else best_i


def trans_for(name, mode):
    shots = face[name]["shots"]
    pp = [s.get("p_person", 0) for s in clip[name]["shots"]]   # CLIP
    fs = [s.get("face_sim", 0) for s in shots]                 # face
    if mode == "clip":
        isc = [p >= 0.5 for p in pp]
    elif mode == "face":
        isc = [f >= 0.35 for f in fs]
    else:  # clip+face: creator if EITHER signal says so
        isc = [(p >= 0.5) or (f >= 0.35) for p, f in zip(pp, fs)]
    idx = pick(isc)
    return round(shots[idx]["start_sec"], 3) if idx is not None else None


def verdict(det, tru, tol=0.5):
    if det is None and tru is None: return "ok"
    if det is None or tru is None: return "bad"
    e = abs(det - tru)
    return "ok" if e <= tol else ("close" if e <= 1.5 else "bad")


def score(get_det, tol):
    ok = close = 0
    for name in face:
        v = verdict(get_det(name), gt.get(short(name), {}).get("cut_sec"), tol)
        ok += v == "ok"; close += v == "close"
    n = len(face)
    return ok / n * 100, (ok + close) / n * 100


# +fallback: clip+face pick, but use the shipped face-based soft-cut recovery where it fired
def clipface_fallback(name):
    m = face[name].get("method", "")
    if m in ("creator_to_meme_soft", "creator_to_meme_softcut"):
        return face[name]["transition_sec"]
    return trans_for(name, "both")

configs = [
    ("OpenCLIP only",            lambda n: trans_for(n, "clip")),
    ("face only",            lambda n: trans_for(n, "face")),
    ("OpenCLIP + face",          lambda n: trans_for(n, "both")),
    ("OpenCLIP + face + fallback", clipface_fallback),
    ("face + fallback  (shipped)", lambda n: face[n]["transition_sec"]),
    ("(ref) OpenCLIP, its own picker", lambda n: clip[n].get("transition_sec")),
]

print(f"{'configuration':30}  {'exact ≤0.5s':>11}  {'exact ≤0.7s':>11}  {'within 1.5s':>11}")
print("-" * 70)
for label, fn in configs:
    e5, w = score(fn, 0.5)
    e7, _ = score(fn, 0.7)
    print(f"{label:30}  {e5:9.1f}%  {e7:9.1f}%  {w:9.1f}%")
