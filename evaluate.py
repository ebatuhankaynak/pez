#!/usr/bin/env python3
"""
Score a transitions.json against the canonical ground truth (transitions/ground_truth.json).

    python evaluate.py transitions/transitions.json
    python evaluate.py transitions/transitions_clip.json

Verdicts per clip (tol = 0.7s exact, 1.5s close):
  correct        detected within tol of the true cut
  close          within 1.5s
  wrong_time     off by >1.5s
  correct_none   truth has no transition AND detector said none
  missed         truth has a cut, detector said none
  false_positive truth has no cut, detector invented one
"""

import argparse
import json
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GT = SCRIPT_DIR / "transitions" / "ground_truth.json"


def short(name):
    return name[:-4][-12:] if name.endswith(".mp4") else name


def verdict(det, true, tol=0.7, close=1.5):
    if det is None and true is None:
        return "correct_none"
    if det is None:
        return "missed"
    if true is None:
        return "false_positive"
    e = abs(det - true)
    return "correct" if e <= tol else ("close" if e <= close else "wrong_time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transitions", help="a transitions.json to score")
    ap.add_argument("--gt", default=str(GT))
    ap.add_argument("--tol", type=float, default=0.5)
    args = ap.parse_args()

    gt = {c["short"]: c for c in json.load(open(args.gt))["clips"]}
    recs = json.load(open(args.transitions))
    det = {short(c["clip"]): c.get("transition_sec") for c in recs}

    cnt = Counter()
    problems = []
    for sid, g in gt.items():
        d = det.get(sid)
        v = verdict(d, g["cut_sec"], args.tol)
        cnt[v] += 1
        if v in ("wrong_time", "missed", "false_positive"):
            problems.append((sid, d, g["cut_sec"], v, g["notes"][:55]))

    tot = sum(cnt.values())
    exact = cnt["correct"] + cnt["correct_none"]
    within = exact + cnt["close"]
    print(f"\n{args.transitions}  vs  {Path(args.gt).name}   ({tot} clips)")
    print(f"  exact (<= {args.tol}s or correct-none): {exact}/{tot} = {exact/tot*100:.1f}%")
    print(f"  within 1.5s:                           {within/tot*100:.1f}%")
    for k in ["correct", "correct_none", "close", "wrong_time", "missed", "false_positive"]:
        if cnt[k]:
            print(f"    {cnt[k]:3d}  {k}")
    if problems:
        print("  problems:")
        for sid, d, t, v, n in problems:
            print(f"    {sid}  {v:14} detected={d} true={t}  {n}")


if __name__ == "__main__":
    main()
