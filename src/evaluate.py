#!/usr/bin/env python3
"""
Score a transitions.json against a ground truth — MULTI-CUT aware.

A clip can have more than one cut (creator->meme->creator->meme). The ground truth
carries the whole sequence in its 'cuts' array (each {sec, to}); older single-cut
labels fall back to 'cut_sec'. The MODEL's cut sequence is the boundaries between
its segments (segments.json — consecutive same-label shots merged; each segment
after the first starts at a cut), falling back to shot-label flips in the
transitions file when segments.json isn't present.

Every cut is scored, not just the first: each GT cut is matched one-to-one to the
nearest predicted cut within --tol; we report how many clips get their WHOLE
sequence right, plus cut-level precision / recall.

    python src/evaluate.py                                   # transitions.json vs the MANUAL GT (batu, multi-cut)
    python src/evaluate.py --gt transitions/ground_truth.json  # vs the Claude/agent GT (single-cut labels)
    python src/evaluate.py --tol 1.5                          # looser (within 1.5s)

Per-clip verdicts:
  correct        whole cut sequence right (all GT cuts matched, none missed, none extra)
  correct_none   GT has no cut and the model predicted none
  partial        some GT cuts matched, but at least one missed and/or extra
  missed         GT has cut(s), model predicted none
  false_positive GT has no cut, model invented one or more
"""

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

from pezutil import short

SCRIPT_DIR = Path(__file__).resolve().parent.parent   # repo root (this file lives in src/)
# Default to the manual cut-editor GT: it's the only label set with multi-cut ('cuts')
# arrays, so it's the correct reference for this multi-cut eval. Pass --gt for the Claude one.
GT = SCRIPT_DIR / "transitions" / "ground_truth_batu.json"
TRANSITIONS = SCRIPT_DIR / "transitions" / "transitions.json"
SEGMENTS = SCRIPT_DIR / "transitions" / "segments.json"




def gt_cuts(g):
    """Full GT cut sequence: the 'cuts' array when present (multi-cut clips), else
    the single 'cut_sec', else [] for a no-transition clip."""
    cuts = g.get("cuts") or []
    if cuts:
        return [c["sec"] for c in cuts]
    return [g["cut_sec"]] if g.get("cut_sec") is not None else []


def pred_cuts(sid, segs, recs):
    """Model cut sequence: segment boundaries (segments.json), else shot-label flips."""
    if sid in segs:
        return [s["start"] for s in segs[sid].get("segments", [])[1:]]
    r = recs.get(sid, {})
    shots = r.get("shots", [])
    return [b["start_sec"] for a, b in zip(shots, shots[1:]) if a.get("label") != b.get("label")]


def match(gts, preds, tol):
    """Greedy one-to-one nearest match. Returns (matched, missed, extra, abs_errors)."""
    preds = list(preds)
    tp, errs = 0, []
    for g in gts:
        best = None
        for i, p in enumerate(preds):
            if abs(p - g) <= tol and (best is None or abs(p - g) < abs(preds[best] - g)):
                best = i
        if best is not None:
            tp += 1
            errs.append(abs(preds[best] - g))
            preds.pop(best)
    return tp, len(gts) - tp, len(preds), errs


def verdict(n_gt, tp, missed, extra):
    if n_gt == 0:
        return "correct_none" if extra == 0 else "false_positive"
    if tp == 0:
        return "missed"
    return "correct" if (missed == 0 and extra == 0) else "partial"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transitions", nargs="?", default=str(TRANSITIONS), help="a transitions.json to score")
    ap.add_argument("--gt", default=str(GT))
    ap.add_argument("--segments", default=str(SEGMENTS),
                    help="model's multi-cut output; falls back to shot-label flips if missing")
    ap.add_argument("--tol", type=float, default=0.5)
    args = ap.parse_args()

    with open(args.gt) as f:
        gt = {c["short"]: c for c in json.load(f)["clips"]}
    with open(args.transitions) as f:
        recs = {short(c["clip"]): c for c in json.load(f)}
    try:
        with open(args.segments) as f:
            segs = {r["short"]: r for r in json.load(f)}
    except Exception:
        segs = {}

    cnt = Counter()
    TP = MISS = EXTRA = 0
    firstcut_ok = 0
    errs_all, problems = [], []
    for sid, g in gt.items():
        gc = gt_cuts(g)
        pc = pred_cuts(sid, segs, recs)
        tp, missed, extra, errs = match(gc, pc, args.tol)
        TP += tp; MISS += missed; EXTRA += extra; errs_all += errs
        v = verdict(len(gc), tp, missed, extra)
        cnt[v] += 1
        # first-cut-only reference (what the old evaluate.py measured)
        if (not gc and not pc) or (gc and pc and abs(pc[0] - gc[0]) <= args.tol):
            firstcut_ok += 1
        if v not in ("correct", "correct_none"):
            problems.append((sid, [round(x, 2) for x in gc], [round(x, 2) for x in pc], v,
                             (g.get("notes") or "")[:45]))

    tot = sum(cnt.values())
    full_ok = cnt["correct"] + cnt["correct_none"]
    prec = TP / (TP + EXTRA) * 100 if TP + EXTRA else 100.0
    rec = TP / (TP + MISS) * 100 if TP + MISS else 100.0
    print(f"\n{args.transitions}  vs  {Path(args.gt).name}   ({tot} clips, tol {args.tol}s, MULTI-CUT)")
    print(f"  clips with FULL sequence correct: {full_ok}/{tot} = {full_ok/tot*100:.1f}%")
    print(f"  cut-level: matched={TP} missed={MISS} extra={EXTRA}  "
          f"precision={prec:.1f}%  recall={rec:.1f}%"
          + (f"  mean|Δ|={st.mean(errs_all):.3f}s" if errs_all else ""))
    print(f"  (ref) first-cut-only correct: {firstcut_ok}/{tot} = {firstcut_ok/tot*100:.1f}%")
    for k in ["correct", "correct_none", "partial", "missed", "false_positive"]:
        if cnt[k]:
            print(f"    {cnt[k]:3d}  {k}")
    if problems:
        print("  problems (gt cuts vs predicted cuts):")
        for sid, gc, pc, v, n in problems:
            print(f"    {sid}  {v:14} gt={gc} pred={pc}  {n}")


if __name__ == "__main__":
    main()
