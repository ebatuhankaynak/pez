#!/usr/bin/env python3
"""
FACE-FIRST cut detector — cuts come from FINDING THE CREATOR, not from TransNetV2.

The TransNet+face pipeline collapses creator *returns* (person->meme->person->meme)
when TransNet doesn't split there (e.g. 152407c208d2). Identity is strong enough
(buffalo_l/ArcFace, 118 enrolled intros) to drive segmentation directly: keep the
stage-2 first cut, then read the dense creator-similarity curve to recover returns in
the meme tail. TransNet/luma say WHEN a shot changes; the face says WHO is in it.

Two stages so tuning is cheap:
  A. dense sim + luma curves per clip (GPU, slow) -> cached in transitions/_face/curves.json
  B. segment the cached curves (CPU, instant) — Stage-A deps are lazy so B runs on
     plain python3 + ffmpeg, no GPU env needed.

    # stage 3 of the pipeline (after relabel_faces), two steps:
    python face_cut.py --dump-curves        # A: dense sim + luma curves (GPU, once)
    python face_cut.py --split segments      # B: segment + cut videos
    #   -> transitions/segments.json (99.2% batu, mean|Δ| 0.13s) + segments/<clip>/*.mp4
    # ablate with --no-refine-fade / --no-snap; explore knobs with --sweep.

Writes transitions/segments.json (+ transitions/_face/trans.json), both scorable by
evaluate.py:  python evaluate.py transitions/_face/trans.json --segments transitions/segments.json
"""
import argparse
import json
import shutil
import statistics as st
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Stage A needs numpy/decord/insightface; imported lazily inside it so Stage B + --split
# run on plain python3 + ffmpeg once the curves are cached.
SCRIPT_DIR = Path(__file__).resolve().parent
CLIPS_DIR = SCRIPT_DIR / "freckled_spike_tiktok"
TRANSITIONS = SCRIPT_DIR / "transitions" / "transitions.json"
SEGMENTS = SCRIPT_DIR / "transitions" / "segments.json"
GT = SCRIPT_DIR / "transitions" / "ground_truth_batu.json"
OUT = SCRIPT_DIR / "transitions" / "_face"
CURVES = OUT / "curves.json"


def short(name):
    return name[:-4][-12:]


# ------------------------------------------------------------------ stage A (GPU)
def dense_sims(app, src, centroid, sample_fps):
    import numpy as np  # noqa: F401 (kept for parity with faces_at pipeline)
    from decord import VideoReader
    from relabel_faces import normed, faces_at
    vr = VideoReader(str(src))
    fps = vr.get_avg_fps() or 25.0
    total = len(vr)
    dur = total / fps if fps else 0.0
    n = max(2, int(dur * sample_fps))
    times, sims = [], []
    for k in range(n):
        t = dur * (k + 0.5) / n
        best = 0.0
        for f in faces_at(app, vr, fps, t, total):
            best = max(best, float(normed(f.normed_embedding) @ centroid))
        times.append(round(t, 3))
        sims.append(round(best, 4))
    return dur, fps, times, sims


def dense_luma(src, luma_fps):
    """Per-frame luminance at luma_fps. A soft fade washes the frame to white/black/gray;
    the manual 'neutral' cut frame is the luma extremum. Model-free (no GPU)."""
    import cv2
    from decord import VideoReader
    vr = VideoReader(str(src))
    fps = vr.get_avg_fps() or 25.0
    total = len(vr)
    dur = total / fps if fps else 0.0
    n = max(2, int(dur * luma_fps))
    times, luma = [], []
    for k in range(n):
        t = dur * (k + 0.5) / n
        idx = min(int(t * fps), total - 1)
        g = cv2.cvtColor(vr[idx].asnumpy(), cv2.COLOR_RGB2GRAY)
        times.append(round(t, 3))
        luma.append(round(float(g.mean()), 2))
    return times, luma


def dump_curves(sample_fps, luma_fps):
    from relabel_faces import load_face_app, enroll_creator
    recs = json.load(open(TRANSITIONS))
    app = load_face_app()
    print("Enrolling creator...", flush=True)
    centroid = enroll_creator(app, recs)
    OUT.mkdir(parents=True, exist_ok=True)
    curves = {}
    for i, c in enumerate(recs, 1):
        src = CLIPS_DIR / c["clip"]
        if not src.exists():
            continue
        dur, fps, times, sims = dense_sims(app, src, centroid, sample_fps)
        tl, luma = dense_luma(src, luma_fps)
        # TransNet hard-cut times = shot boundaries (shots[1:]).
        tn = [round(s["start_sec"], 3) for s in c.get("shots", [])[1:]]
        curves[short(c["clip"])] = {"clip": c["clip"], "dur": round(dur, 3),
                                    "sample_fps": sample_fps, "times": times, "sims": sims,
                                    "transnet_cuts": tn, "luma_fps": luma_fps,
                                    "times_l": tl, "luma": luma}
        if i % 20 == 0:
            print(f"  {i}/{len(recs)}", flush=True)
    CURVES.write_text(json.dumps(curves))
    print(f"Wrote {CURVES}  ({len(curves)} clips, face@{sample_fps}fps, luma@{luma_fps}fps)")


# ------------------------------------------------------------------ stage B (CPU)
def _runs(bools):
    """Contiguous same-value runs as [value, start_idx, end_idx]."""
    runs, i, n = [], 0, len(bools)
    while i < n:
        j = i
        while j + 1 < n and bools[j + 1] == bools[i]:
            j += 1
        runs.append([bools[i], i, j])
        i = j + 1
    return runs


def _morph(present, dt, max_gap, min_seg):
    """Temporal cleanup of a boolean presence curve: (1) bridge short absent gaps flanked
    by presence (occlusion/angle/blur — 'skipped frames = same shot'); (2) absorb any run
    shorter than min_seg into its neighbour (kills blips)."""
    present = list(present)
    changed = True
    while changed:                                        # gap-fill
        changed = False
        for idx, (val, a, b) in enumerate(_runs(present)):
            runs = _runs(present)
            if not val and 0 < idx < len(runs) - 1 and (b - a + 1) * dt < max_gap:
                for k in range(a, b + 1):
                    present[k] = True
                changed = True
                break
    changed = True
    while changed:                                        # min-seg absorb
        changed = False
        runs = _runs(present)
        if len(runs) <= 1:
            break
        for k in sorted(range(len(runs)), key=lambda k: runs[k][2] - runs[k][1]):
            val, a, b = runs[k]
            if (b - a + 1) * dt < min_seg:
                nb = runs[k - 1][0] if k > 0 else runs[k + 1][0]
                for idx in range(a, b + 1):
                    present[idx] = nb
                changed = True
                break
    return present


def refine_fade(cur, b, thr=0.35, dark=48.0, bright=212.0, hard_step=34.0, back=0.3, fwd=0.5):
    """On a SOFT FADE, place the cut at the washed-out frame — the LUMINANCE extremum WHERE
    THE CREATOR IS ABSENT (face-sim < thr), matching the manual convention. Luma not detail
    (a black text-card keeps edges yet reads near-black). Guards: (1) a one-frame luma jump
    is a HARD cut -> leave it to TransNet; (2) the wash frame must be creator-absent so we
    don't grab a merely-dark frame inside her own shot. Returns the neutral-frame time or
    None (no fade here -> keep the boundary)."""
    tl, lu = cur.get("times_l"), cur.get("luma")
    ftimes, fsims = cur.get("times"), cur.get("sims")
    if not tl or not ftimes:
        return None
    win = [i for i, t in enumerate(tl) if b - back <= t <= b + fwd]
    if len(win) < 3:
        return None
    for i in win[:-1]:                                    # hard cut? one-frame luma jump
        if i + 1 < len(lu) and abs(lu[i + 1] - lu[i]) > hard_step:
            return None

    def sim_at(t):
        return fsims[min(range(len(ftimes)), key=lambda k: abs(ftimes[k] - t))]

    cand = [i for i in win if sim_at(tl[i]) < thr]        # only creator-ABSENT frames
    if not cand:
        return None
    lo = min(cand, key=lambda i: lu[i])
    hi = max(cand, key=lambda i: lu[i])
    if lu[lo] <= dark:                                    # fade-to-black -> darkest frame
        return round(tl[lo], 3)
    if lu[hi] >= bright:                                  # fade-to-white -> brightest frame
        return round(tl[hi], 3)
    return None                                          # mid-luma, no clear wash


def place_boundaries(segs, cur, snap, snap_win, refine, snap_from=1, return_back=0.8):
    """Position each interior boundary. TransNet/luma = WHEN the shot changes, face = WHO.
      - RETURN (meme->person): snap back to a TransNet cut in a back-biased window — the
        shot changed back to her, possibly before her face resolves (she enters dark/turned).
      - person->meme SOFT FADE: place at the luma neutral frame (refine_fade).
      - else: nearest TransNet hard cut within snap_win."""
    tn = cur.get("transnet_cuts", [])
    for i in range(max(1, snap_from), len(segs)):
        b = segs[i]["start"]
        prev_lab, cur_lab = segs[i - 1]["label"], segs[i]["label"]
        new = None
        if prev_lab == "meme" and cur_lab == "person" and snap and tn:
            cand = [c for c in tn if b - return_back <= c <= b + 0.3]
            if cand:
                new = min(cand, key=lambda x: abs(x - b))
        if new is None and refine:
            new = refine_fade(cur, b)
        if new is None and snap and tn:
            near = min(tn, key=lambda x: abs(x - b))
            if abs(near - b) <= snap_win:
                new = near
        if new is not None:
            segs[i]["start"] = round(new, 3)
            segs[i - 1]["end"] = round(new, 3)
    # representative face_sim per segment (median of samples inside it) for the workbench
    sims, times = cur["sims"], cur["times"]
    for s in segs:
        inside = [sims[k] for k, t in enumerate(times) if s["start"] <= t < s["end"]]
        s["dur"] = round(s["end"] - s["start"], 3)
        s["face_sim"] = round(float(st.median(inside)), 3) if inside else 0.0
    cuts = [s["start"] for s in segs[1:]]
    return segs, (cuts[0] if cuts else None)


def segment(cur, base_rec, p):
    """Segment ONE clip. Keep the stage-2 first cut; recover creator RETURNS from the dense
    face curve in the meme tail (gated so single-cut clips don't sprout spurious returns);
    then place every boundary (return back-snap / luma fade-refine / TransNet snap)."""
    times, sims = cur["times"], cur["sims"]
    dt, dur = 1.0 / cur["sample_fps"], cur["dur"]
    first, method = base_rec.get("transition_sec"), base_rec.get("method", "")

    if first is None:                                     # base found no cut -> trust it
        lab = "meme" if method == "all_meme_no_creator" else "person"
        segs = [{"start": 0.0, "end": round(dur, 3), "label": lab}]
        return place_boundaries(segs, cur, False, p["snap_win"], p["refine"]) + (method,)

    idx = [k for k, t in enumerate(times) if t > first + 0.1]
    present = _morph([sims[k] >= p["thr"] for k in idx], dt, p["max_gap"], p["min_seg"]) if idx else []
    ttimes = [times[k] for k in idx]
    tail = []
    for val, a, b in _runs(present):
        s0 = first if a == 0 else round((ttimes[a] + ttimes[a - 1]) / 2, 3)
        s1 = round(dur, 3) if b == len(present) - 1 else round((ttimes[b] + ttimes[b + 1]) / 2, 3)
        tail.append({"start": s0, "end": s1, "label": "person" if val else "meme"})
    # conservative return gate: a tail 'person' run must be long enough AND strongly match
    for s in tail:
        if s["label"] == "person":
            inside = [sims[k] for k, t in zip(idx, ttimes) if s["start"] <= t < s["end"]]
            med = st.median(inside) if inside else 0.0
            if not (s["end"] - s["start"] >= p["min_return"] and med >= p["ret_sim"]):
                s["label"] = "meme"
    segs = [{"start": 0.0, "end": round(first, 3), "label": "person"}]
    for s in tail:                                        # merge consecutive same-label
        if segs[-1]["label"] == s["label"]:
            segs[-1]["end"] = s["end"]
        else:
            segs.append(dict(s))
    segs[-1]["end"] = round(dur, 3)
    n_ret = sum(1 for s in segs if s["label"] == "person") - 1
    method = "creator_returns" if n_ret > 0 else (method or "creator_to_meme")
    return place_boundaries(segs, cur, p["snap"], p["snap_win"], p["refine"]) + (method,)


def run(curves, base_recs, p):
    base = {short(c["clip"]): c for c in base_recs}
    trans_out, seg_out = [], []
    for sid, cur in curves.items():
        segs, first_cut, method = segment(cur, base.get(sid, {}), p)
        shots = [{"start_sec": s["start"], "end_sec": s["end"],
                  "label": s["label"], "face_sim": s["face_sim"]} for s in segs]
        trans_out.append({"clip": cur["clip"], "transition_sec": first_cut,
                          "method": method, "num_shots": len(shots), "shots": shots})
        seg_out.append({"clip": cur["clip"], "short": sid, "n_segments": len(segs),
                        "pattern": "→".join(s["label"][:4] for s in segs), "segments": segs})
    return trans_out, seg_out


# ------------------------------------------------------------------ segment video split
def _cut(src, start, end, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src),
                    "-t", f"{max(0.05, end - start):.3f}", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-c:a", "aac", "-movflags", "+faststart", str(dst)],
                   stderr=subprocess.DEVNULL)


def split_segments(seg_out, seg_dir, workers=8):
    """Cut each clip into segments/<stem>/NN_<label>.mp4 for the workbench."""
    root = Path(seg_dir)
    if root.exists():
        shutil.rmtree(root)                               # stale pieces from a prior run
    jobs = []
    for r in seg_out:
        stem = r["clip"][:-4]
        for k, s in enumerate(r["segments"], 1):
            jobs.append((CLIPS_DIR / r["clip"], s["start"], s["end"],
                         root / stem / f"{k:02d}_{s['label']}.mp4"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda j: _cut(*j), jobs))
    return len(jobs)


# ------------------------------------------------------------------ scoring (mirrors evaluate.py)
def score(seg_out, gt_path, tol=0.5):
    gt = {c["short"]: c for c in json.load(open(gt_path))["clips"]}
    segs = {r["short"]: r for r in seg_out}

    def gt_cuts(g):
        cs = g.get("cuts") or []
        return [c["sec"] for c in cs] if cs else ([g["cut_sec"]] if g.get("cut_sec") is not None else [])

    TP = MISS = EXTRA = full = 0
    for sid, g in gt.items():
        gc = gt_cuts(g)
        pool = [s["start"] for s in segs.get(sid, {}).get("segments", [])[1:]]
        tp = 0
        for x in gc:
            cand = [i for i, pv in enumerate(pool) if abs(pv - x) <= tol]
            if cand:
                pool.pop(min(cand, key=lambda i: abs(pool[i] - x)))
                tp += 1
        miss, extra = len(gc) - tp, len(pool)
        TP += tp; MISS += miss; EXTRA += extra
        if (not gc and not pool) or (gc and miss == 0 and extra == 0):
            full += 1
    tot = len(gt)
    return {"full_pct": full / tot * 100, "full": full, "tot": tot, "tp": TP, "miss": MISS,
            "extra": EXTRA, "prec": TP / (TP + EXTRA) * 100 if TP + EXTRA else 100.0,
            "rec": TP / (TP + MISS) * 100 if TP + MISS else 100.0}


def offset_stats(trans_out, gt_path):
    """Signed convention gap on the FIRST cut: gt_first - model_first (s), over clips where
    both exist within 1s. Positive = model fires BEFORE the manual label."""
    gt = {c["short"]: c for c in json.load(open(gt_path))["clips"]}
    tr = {short(c["clip"]): c for c in trans_out}
    d = []
    for sid, g in gt.items():
        gc = (g.get("cuts") or [{}])[0].get("sec") if g.get("cuts") else g.get("cut_sec")
        mc = tr.get(sid, {}).get("transition_sec")
        if gc is not None and mc is not None and abs(gc - mc) <= 1.0:
            d.append(gc - mc)
    if not d:
        return None
    return {"n": len(d), "mean": st.mean(d), "median": st.median(d),
            "mae": st.mean([abs(x) for x in d])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-curves", action="store_true", help="Stage A: cache dense sim + luma curves (GPU)")
    ap.add_argument("--sample-fps", type=float, default=10.0)
    ap.add_argument("--luma-fps", type=float, default=15.0, help="Stage A: luma sampling rate")
    # knobs (defaults = the shipped config)
    ap.add_argument("--thr", type=float, default=0.32, help="presence: sim>=thr -> creator")
    ap.add_argument("--max-gap", type=float, default=0.7, help="bridge absent gaps shorter than this (s)")
    ap.add_argument("--min-seg", type=float, default=0.5, help="absorb segments shorter than this (s)")
    ap.add_argument("--min-return", type=float, default=0.5, help="min creator-return duration in the tail (s)")
    ap.add_argument("--ret-sim", type=float, default=0.45, help="min median sim to accept a creator return")
    ap.add_argument("--snap", action=argparse.BooleanOptionalAction, default=True,
                    help="snap boundaries to TransNet shot cuts (returns + hard cuts)")
    ap.add_argument("--snap-win", type=float, default=0.35, help="max distance for a nearest-cut snap (s)")
    ap.add_argument("--refine-fade", action=argparse.BooleanOptionalAction, default=True,
                    help="place soft-fade cuts at the luma neutral frame (matches the manual convention)")
    ap.add_argument("--base", default=str(TRANSITIONS), help="stage-2 transitions.json (first cut per clip)")
    ap.add_argument("--split", default="", help="also cut segments to this dir (e.g. 'segments')")
    ap.add_argument("--gt", default=str(GT))
    ap.add_argument("--sweep", action="store_true", help="grid-search knobs on the cached curves")
    ap.add_argument("--inspect", default="", help="print the sim curve near the GT cut(s) for one short id")
    ap.add_argument("--out-trans", default=str(OUT / "trans.json"))
    ap.add_argument("--out-segs", default=str(SEGMENTS))
    args = ap.parse_args()

    if args.dump_curves:
        dump_curves(args.sample_fps, args.luma_fps)
        return
    if not CURVES.exists():
        raise SystemExit(f"no curve cache at {CURVES} — run `face_cut.py --dump-curves` first (GPU)")
    curves = json.loads(CURVES.read_text())

    if args.inspect:
        cur = curves[args.inspect]
        g = {c["short"]: c for c in json.load(open(args.gt))["clips"]}.get(args.inspect, {})
        gcuts = [c["sec"] for c in (g.get("cuts") or [])] or ([g["cut_sec"]] if g.get("cut_sec") is not None else [])
        print(f"{args.inspect}  dur={cur['dur']}s  GT cuts={gcuts}  TransNet cuts={cur['transnet_cuts']}")
        for t, s in zip(cur["times"], cur["sims"]):
            mark = "".join(" <GTCUT" for gc in gcuts if abs(gc - t) < 0.6)
            print(f"  {t:5.2f}s  {s:.3f} {'#' * int(s * 40)}{mark}")
        return

    def mkp(**kw):
        p = {"thr": args.thr, "max_gap": args.max_gap, "min_seg": args.min_seg,
             "min_return": args.min_return, "ret_sim": args.ret_sim, "snap": args.snap,
             "snap_win": args.snap_win, "refine": args.refine_fade}
        p.update(kw)
        return p

    if args.sweep:
        base = json.load(open(args.base))
        grid = [mkp(thr=t, min_return=mr, ret_sim=rs, refine=rf)
                for t in (0.28, 0.32, 0.35) for mr in (0.4, 0.5, 0.7)
                for rs in (0.40, 0.45) for rf in (True, False)]
        rows = []
        for p in grid:
            _, seg_out = run(curves, base, p)
            b = score(seg_out, args.gt)
            rows.append((b["full_pct"], b["prec"], b["rec"], p))
        rows.sort(key=lambda r: (-r[0], -(r[1] + r[2])))
        print(f"\nsweep vs {Path(args.gt).name} — top 15 (of {len(grid)}):")
        print("  full%   prec   rec   | thr  minret retsim refine")
        for full, prec, rec, p in rows[:15]:
            print(f"  {full:5.1f}  {prec:5.1f}  {rec:5.1f}  | {p['thr']:.2f} {p['min_return']:.1f}   "
                  f"{p['ret_sim']:.2f}   {'Y' if p['refine'] else 'n'}")
        return

    trans_out, seg_out = run(curves, json.load(open(args.base)), mkp())
    Path(args.out_trans).write_text(json.dumps(trans_out, indent=2))
    Path(args.out_segs).write_text(json.dumps(seg_out, indent=2))
    b = score(seg_out, args.gt)
    print(f"thr={args.thr} max_gap={args.max_gap} min_seg={args.min_seg} min_return={args.min_return} "
          f"snap={args.snap} refine={args.refine_fade}")
    print(f"vs {Path(args.gt).name}: FULL-seq {b['full']}/{b['tot']} = {b['full_pct']:.1f}%  "
          f"cut prec={b['prec']:.1f}% rec={b['rec']:.1f}%  (tp={b['tp']} miss={b['miss']} extra={b['extra']})")
    o = offset_stats(trans_out, args.gt)
    if o:
        print(f"first-cut convention gap (gt - model): mean={o['mean']:+.3f}s median={o['median']:+.3f}s "
              f"MAE={o['mae']:.3f}s  (n={o['n']}; →0 = matches manual)")
    print(f"Wrote {args.out_trans}, {args.out_segs}")
    if args.split:
        print(f"Cut {split_segments(seg_out, args.split)} segment videos -> {args.split}/")


if __name__ == "__main__":
    main()
