#!/usr/bin/env python3
"""
FACE-FIRST cut detector — cuts come from FINDING THE CREATOR, not from TransNetV2.

Motivation: the TransNet+face pipeline collapses creator *returns* (person->meme->
person->meme) when TransNet doesn't split there, e.g. 152407c208d2. Identity is strong
enough (buffalo_l/ArcFace, 118 enrolled intros) to drive segmentation directly: mark
every frame present/absent by the creator's face, bridge short occlusion/angle dips,
and cut at each present<->absent boundary. TransNet is demoted to an *after-stitch* that
snaps each face-derived boundary onto the nearest hard-cut for frame accuracy.

Two stages so tuning is cheap:
  A. dense sim curve per clip (GPU, slow)  -> cached in transitions/_face/curves.json
  B. segment the cached curves (CPU, instant, all the knobs)

    # stage 3 of the pipeline (after relabel_faces). Two steps:
    python face_cut.py --dump-curves                 # A: dense sim + luma curves (GPU, once)
    python face_cut.py --split segments              # B: segment (hybrid+refine ON) + cut videos
    #   -> writes transitions/segments.json (98.3% batu, mean|Δ| 0.13s) + segments/<clip>/*.mp4
    # ablate with --no-hybrid / --no-refine-fade / --no-snap; explore with --sweep.

Writes transitions/_face/{trans,segs}.json (transitions- & segments-shaped, so
evaluate.py scores them directly):
    python evaluate.py transitions/_face/trans.json --segments transitions/_face/segs.json
"""
import argparse
import json
import shutil
import statistics as st
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Stage A (dump_curves) needs numpy/decord/insightface; they're imported lazily inside
# it so Stage B (segment from the cache) + --split run on plain python3 + ffmpeg — no GPU
# env needed once the curves are cached.
SCRIPT_DIR = Path(__file__).resolve().parent
CLIPS_DIR = SCRIPT_DIR / "freckled_spike_tiktok"
TRANSITIONS = SCRIPT_DIR / "transitions" / "transitions.json"
OUT = SCRIPT_DIR / "transitions" / "_face"
CURVES = OUT / "curves.json"


def short(name):
    return name[:-4][-12:]


# ------------------------------------------------------------------ stage A
def dense_sims(app, src, centroid, sample_fps):
    import numpy as np
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
    """Per-frame luminance + detail (Laplacian variance) at luma_fps. A soft fade
    washes the frame to white/black/gray: luma goes extreme AND detail collapses
    (a flat frame has no edges). The user's 'neutral' cut frame == the detail MINIMUM
    (deepest wash). Model-free, so no GPU needed."""
    import cv2
    from decord import VideoReader
    vr = VideoReader(str(src))
    fps = vr.get_avg_fps() or 25.0
    total = len(vr)
    dur = total / fps if fps else 0.0
    n = max(2, int(dur * luma_fps))
    times, luma, detail = [], [], []
    for k in range(n):
        t = dur * (k + 0.5) / n
        idx = min(int(t * fps), total - 1)
        g = cv2.cvtColor(vr[idx].asnumpy(), cv2.COLOR_RGB2GRAY)
        h, w = g.shape
        g = cv2.resize(g, (320, max(1, int(320 * h / w))))   # normalize the detail scale
        times.append(round(t, 3))
        luma.append(round(float(g.mean()), 2))
        detail.append(round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 2))
    return times, luma, detail


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
        tl, luma, detail = dense_luma(src, luma_fps)
        # TransNet hard-cut times for the after-stitch = shot boundaries (shots[1:]).
        tn = [round(s["start_sec"], 3) for s in c.get("shots", [])[1:]]
        curves[short(c["clip"])] = {"clip": c["clip"], "dur": round(dur, 3),
                                    "sample_fps": sample_fps, "times": times,
                                    "sims": sims, "transnet_cuts": tn,
                                    "luma_fps": luma_fps, "times_l": tl,
                                    "luma": luma, "detail": detail}
        if i % 20 == 0:
            print(f"  {i}/{len(recs)}", flush=True)
    CURVES.write_text(json.dumps(curves))
    print(f"Wrote {CURVES}  ({len(curves)} clips, face@{sample_fps}fps, luma@{luma_fps}fps)")


# ------------------------------------------------------------------ stage B
def _runs(bools):
    runs, i, n = [], 0, len(bools)
    while i < n:
        j = i
        while j + 1 < n and bools[j + 1] == bools[i]:
            j += 1
        runs.append([bools[i], i, j])
        i = j + 1
    return runs


def _morph(present, dt, max_gap, min_seg):
    """Temporal cleanup on a boolean presence curve: (1) bridge short absent gaps
    flanked by presence (occlusion/angle/blur — 'skipped frames = same shot'), then
    (2) absorb any run shorter than min_seg into its neighbour (kills blips)."""
    present = list(present)
    changed = True
    while changed:                                        # gap-fill
        changed = False
        runs = _runs(present)
        for idx, (val, a, b) in enumerate(runs):
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


def segment_curve(cur, thr, max_gap, min_seg, evidence_floor, snap, snap_win,
                  refine=False, fade_frac=0.4):
    times, sims = cur["times"], cur["sims"]
    dt = 1.0 / cur["sample_fps"]
    n = len(sims)
    dur = cur["dur"]

    # creator-less repost: no real face evidence anywhere -> all meme, no cut.
    if max(sims) < evidence_floor:
        segs = [{"start": 0.0, "end": round(dur, 3), "label": "meme"}]
        return _finalize(segs, cur, "all_meme_no_creator", snap, snap_win, refine, fade_frac)

    present = _morph([s >= thr for s in sims], dt, max_gap, min_seg)

    # 3) DOMAIN PRIOR: the creator ALWAYS opens (never meme-only). If segmentation
    #    starts on meme (dark/occluded intro under threshold), flip the leading meme
    #    run to creator so we don't invent an opening meme.
    runs = _runs(present)
    if runs and not runs[0][0]:
        _, a, b = runs[0]
        for k in range(a, b + 1):
            present[k] = True

    # build segments in time, boundary = midpoint between the two straddling samples
    runs = _runs(present)
    segs = []
    for val, a, b in runs:
        start = 0.0 if a == 0 else round((times[a] + times[a - 1]) / 2, 3)
        end = round(dur, 3) if b == n - 1 else round((times[b] + times[b + 1]) / 2, 3)
        segs.append({"start": start, "end": end, "label": "person" if val else "meme"})
    segs[0]["start"] = 0.0
    segs[-1]["end"] = round(dur, 3)

    method = ("single_shot_creator" if len(segs) == 1
              else "creator_to_meme_faces")
    return _finalize(segs, cur, method, snap, snap_win, refine, fade_frac)


def refine_fade(cur, b, thr=0.35, dark=48.0, bright=212.0, hard_step=34.0, back=0.3, fwd=0.5):
    """Match the manual convention: on a SOFT FADE, cut at the washed-out frame — the
    LUMINANCE extremum WHERE THE CREATOR IS ABSENT (face-sim < thr). Luma not detail (a
    black text-card keeps edges yet reads near-black). Two guards: (1) a HARD cut is a
    one-frame luma jump -> leave it to TransNet; (2) the wash frame must be creator-absent
    so we don't grab a merely-dark frame inside her own shot. Returns the neutral-frame
    time, or None (no fade here -> keep the boundary)."""
    tl, lu = cur.get("times_l"), cur.get("luma")
    ftimes, fsims = cur.get("times"), cur.get("sims")
    if not tl or not ftimes:
        return None
    win = [i for i, t in enumerate(tl) if b - back <= t <= b + fwd]
    if len(win) < 3:
        return None
    for i in win[:-1]:                        # hard cut? single-frame luma jump
        if i + 1 < len(lu) and abs(lu[i + 1] - lu[i]) > hard_step:
            return None

    def sim_at(t):
        return fsims[min(range(len(ftimes)), key=lambda k: abs(ftimes[k] - t))]

    cand = [i for i in win if sim_at(tl[i]) < thr]    # only creator-ABSENT frames
    if not cand:
        return None
    lo = min(cand, key=lambda i: lu[i])
    hi = max(cand, key=lambda i: lu[i])
    if lu[lo] <= dark:                        # fade-to-black -> darkest frame
        return round(tl[lo], 3)
    if lu[hi] >= bright:                      # fade-to-white -> brightest frame
        return round(tl[hi], 3)
    return None                              # mid-luma, no clear wash -> keep as-is


def _finalize(segs, cur, method, snap, snap_win, refine=False, fade_frac=0.4,
              snap_fwd=1.0, snap_from=1):
    # Place each interior boundary. Two regimes:
    #  - SOFT FADE (refine): snap to the luma/detail NEUTRAL frame (the washed-out frame
    #    the manual labeler picks — creator gone, meme not yet in). Model-free.
    #  - HARD CUT (TransNet snap, DIRECTIONAL): the face signal dies mid-fade, so on a
    #    person->meme snap to the LATER TransNet edge, meme->person the EARLIER edge.
    # refine wins when a real wash exists; else fall back to TransNet; else keep face time.
    dur, sims, times, tn_cuts = cur["dur"], cur["sims"], cur["times"], cur.get("transnet_cuts", [])
    snapped = fades = 0
    if len(segs) > 1:
        for i in range(max(1, snap_from), len(segs)):
            b = segs[i]["start"]
            prev_lab, cur_lab = segs[i - 1]["label"], segs[i]["label"]
            new = refine_fade(cur, b) if refine else None
            if new is not None:
                fades += 1
            elif snap and tn_cuts:
                if refine:
                    # refine already handles fade timing; snap only nudges HARD cuts to
                    # the NEAREST TransNet boundary — no directional overshoot.
                    near = min(tn_cuts, key=lambda x: abs(x - b))
                    new = near if abs(near - b) <= snap_win else None
                else:
                    # no refine: directional snap compensates for face-death firing
                    # mid-fade (person->meme -> LATER edge, meme->person -> EARLIER edge).
                    if prev_lab == "person" and cur_lab == "meme":
                        cand = [c for c in tn_cuts if b - 0.25 <= c <= b + snap_fwd]
                        new = max(cand) if cand else None
                    elif prev_lab == "meme" and cur_lab == "person":
                        cand = [c for c in tn_cuts if b - snap_fwd <= c <= b + 0.25]
                        new = min(cand) if cand else None
                    if new is None:
                        near = min(tn_cuts, key=lambda x: abs(x - b))
                        new = near if abs(near - b) <= snap_win else None
                if new is not None:
                    snapped += 1
            if new is not None:
                segs[i]["start"] = round(new, 3)
                segs[i - 1]["end"] = round(new, 3)
    # representative face_sim per segment (median of samples inside it) for the workbench
    for s in segs:
        inside = [sims[k] for k, t in enumerate(times) if s["start"] <= t < s["end"]]
        s["dur"] = round(s["end"] - s["start"], 3)
        s["face_sim"] = round(float(st.median(inside)), 3) if inside else 0.0
    cuts = [s["start"] for s in segs[1:]]
    suffix = ("_fade" if fades else "") + ("_snap" if snapped else "")
    return segs, (cuts[0] if cuts else None), (method + suffix), snapped + fades


def hybrid_segment(cur, base_rec, p):
    """HYBRID: keep the baseline pipeline's (fade-accurate) FIRST cut, then add
    creator RETURNS detected from the dense face curve in the meme tail. A return is
    accepted only if the tail creator-run is long enough AND strongly matches the
    creator — so the 118 single-cut clips don't sprout spurious returns."""
    times, sims = cur["times"], cur["sims"]
    dt, dur, tn = 1.0 / cur["sample_fps"], cur["dur"], cur.get("transnet_cuts", [])
    first = base_rec.get("transition_sec")
    method = base_rec.get("method", "")
    if first is None:                                     # base found no cut -> trust base
        lab = "meme" if method == "all_meme_no_creator" else "person"
        segs = [{"start": 0.0, "end": round(dur, 3), "label": lab}]
        return _finalize(segs, cur, method, False, p["snap_win"],
                         p.get("refine", False), p.get("fade_frac", 0.4))

    idx = [k for k, t in enumerate(times) if t > first + 0.1]
    present = _morph([sims[k] >= p["thr"] for k in idx], dt, p["max_gap"], p["min_seg"]) if idx else []
    ttimes = [times[k] for k in idx]
    tsegs = []
    for val, a, b in _runs(present):
        s0 = first if a == 0 else round((ttimes[a] + ttimes[a - 1]) / 2, 3)
        s1 = round(dur, 3) if b == len(present) - 1 else round((ttimes[b] + ttimes[b + 1]) / 2, 3)
        tsegs.append({"start": s0, "end": s1, "label": "person" if val else "meme"})
    # conservative return gate
    for s in tsegs:
        if s["label"] == "person":
            inside = [sims[k] for k, t in zip(idx, ttimes) if s["start"] <= t < s["end"]]
            med = st.median(inside) if inside else 0.0
            if not (s["end"] - s["start"] >= p["min_return"] and med >= p["ret_sim"]):
                s["label"] = "meme"
    segs = [{"start": 0.0, "end": round(first, 3), "label": "person"}]
    for s in tsegs:
        if segs[-1]["label"] == s["label"]:
            segs[-1]["end"] = s["end"]
        else:
            segs.append({"start": s["start"], "end": s["end"], "label": s["label"]})
    segs[-1]["end"] = round(dur, 3)
    n_ret = sum(1 for s in segs if s["label"] == "person") - 1
    method = "creator_returns_hybrid" if n_ret > 0 else (method or "creator_to_meme")
    # refine the baseline first cut to the neutral fade-frame too (convention match),
    # and snap/refine the returns; nothing is "protected" now that refine is smarter.
    return _finalize(segs, cur, method, p["snap"], p["snap_win"],
                     p.get("refine", False), p.get("fade_frac", 0.4), snap_from=1)


def run_hybrid(curves, base_recs, p):
    base = {short(c["clip"]): c for c in base_recs}
    trans_out, seg_out, snapped_total = [], [], 0
    for sid, cur in curves.items():
        segs, first_cut, method, snapped = hybrid_segment(cur, base.get(sid, {}), p)
        snapped_total += snapped
        shots = [{"start_sec": s["start"], "end_sec": s["end"],
                  "label": s["label"], "face_sim": s["face_sim"]} for s in segs]
        trans_out.append({"clip": cur["clip"], "transition_sec": first_cut,
                          "method": method, "num_shots": len(shots), "shots": shots})
        seg_out.append({"clip": cur["clip"], "short": sid, "n_segments": len(segs),
                        "pattern": "→".join(s["label"][:4] for s in segs), "segments": segs})
    return trans_out, seg_out, snapped_total


def run_segmentation(curves, p):
    trans_out, seg_out, snapped_total = [], [], 0
    for sid, cur in curves.items():
        segs, first_cut, method, snapped = segment_curve(
            cur, p["thr"], p["max_gap"], p["min_seg"], p["evidence_floor"],
            p["snap"], p["snap_win"], p.get("refine", False), p.get("fade_frac", 0.4))
        snapped_total += snapped
        shots = [{"start_sec": s["start"], "end_sec": s["end"],
                  "label": s["label"], "face_sim": s["face_sim"]} for s in segs]
        trans_out.append({"clip": cur["clip"], "transition_sec": first_cut,
                          "method": method, "num_shots": len(shots), "shots": shots})
        seg_out.append({"clip": cur["clip"], "short": sid, "n_segments": len(segs),
                        "pattern": "→".join(s["label"][:4] for s in segs),
                        "segments": segs})
    return trans_out, seg_out, snapped_total


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
        shutil.rmtree(root)                      # stale pieces from a prior segmentation
    jobs = []
    for r in seg_out:
        stem = r["clip"][:-4]
        for k, s in enumerate(r["segments"], 1):
            jobs.append((CLIPS_DIR / r["clip"], s["start"], s["end"],
                         root / stem / f"{k:02d}_{s['label']}.mp4"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda j: _cut(*j), jobs))
    return len(jobs)


# ------------------------------------------------------------------ eval helper (mirrors evaluate.py)
def score(seg_out, gt_path, tol=0.5):
    gt = {c["short"]: c for c in json.load(open(gt_path))["clips"]}
    segs = {r["short"]: r for r in seg_out}

    def gt_cuts(g):
        cuts = g.get("cuts") or []
        if cuts:
            return [c["sec"] for c in cuts]
        return [g["cut_sec"]] if g.get("cut_sec") is not None else []

    TP = MISS = EXTRA = full = 0
    for sid, g in gt.items():
        gc = gt_cuts(g)
        pc = [s["start"] for s in segs.get(sid, {}).get("segments", [])[1:]]
        pool = list(pc)
        tp = 0
        for x in gc:
            cand = [i for i, pv in enumerate(pool) if abs(pv - x) <= tol]
            if cand:
                i = min(cand, key=lambda i: abs(pool[i] - x))
                pool.pop(i)
                tp += 1
        miss, extra = len(gc) - tp, len(pool)
        TP += tp; MISS += miss; EXTRA += extra
        if (not gc and not pc) or (gc and miss == 0 and extra == 0):
            full += 1
    tot = len(gt)
    prec = TP / (TP + EXTRA) * 100 if TP + EXTRA else 100.0
    rec = TP / (TP + MISS) * 100 if TP + MISS else 100.0
    return {"full_pct": full / tot * 100, "full": full, "tot": tot,
            "tp": TP, "miss": MISS, "extra": EXTRA, "prec": prec, "rec": rec}


def offset_stats(trans_out, gt_path):
    """Signed convention gap on the FIRST cut: gt_first - model_first (s), over clips
    where both exist within 1s. Positive = model fires BEFORE the manual label."""
    gt = {c["short"]: c for c in json.load(open(gt_path))["clips"]}
    tr = {c["clip"][:-4][-12:]: c for c in trans_out}
    d = []
    for sid, g in gt.items():
        gc = (g.get("cuts") or [{}])[0].get("sec") if g.get("cuts") else g.get("cut_sec")
        mc = tr.get(sid, {}).get("transition_sec")
        if gc is not None and mc is not None and abs(gc - mc) <= 1.0:
            d.append(gc - mc)
    if not d:
        return None
    d.sort()
    return {"n": len(d), "mean": st.mean(d), "median": st.median(d),
            "mae": st.mean([abs(x) for x in d])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-curves", action="store_true", help="Stage A: compute+cache dense sim curves (GPU)")
    ap.add_argument("--sample-fps", type=float, default=10.0)
    ap.add_argument("--luma-fps", type=float, default=15.0, help="Stage A: luma/detail sampling rate")
    # The canonical method is hybrid + TransNet-snap + luma fade-refine; all default ON
    # (use --no-hybrid / --no-snap / --no-refine-fade to ablate).
    ap.add_argument("--refine-fade", action=argparse.BooleanOptionalAction, default=True,
                    help="place soft-fade cuts at the neutral (washed-out) frame — matches the manual convention")
    ap.add_argument("--fade-frac", type=float, default=0.4, help="(reserved) fade-wash sensitivity")
    ap.add_argument("--thr", type=float, default=0.32, help="presence: sim>=thr -> creator")
    ap.add_argument("--max-gap", type=float, default=0.7, help="bridge absent gaps shorter than this (s)")
    ap.add_argument("--min-seg", type=float, default=0.5, help="absorb segments shorter than this (s)")
    ap.add_argument("--evidence-floor", type=float, default=0.15, help="max sim below this -> creator-less clip")
    ap.add_argument("--snap", action=argparse.BooleanOptionalAction, default=True,
                    help="snap hard cuts to the nearest TransNet boundary")
    ap.add_argument("--snap-win", type=float, default=0.35, help="max distance for a TransNet snap (s)")
    ap.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=True,
                    help="keep the baseline first cut, add face-first RETURNS on the meme tail")
    ap.add_argument("--base", default=str(TRANSITIONS), help="baseline transitions.json for --hybrid")
    ap.add_argument("--min-return", type=float, default=0.5, help="min creator-return duration in the tail (s)")
    ap.add_argument("--ret-sim", type=float, default=0.45, help="min median sim to accept a creator return")
    ap.add_argument("--split", default="", help="also cut segments to this dir (e.g. 'segments')")
    ap.add_argument("--gt", default=str(Path(TRANSITIONS).parent / "ground_truth_batu.json"))
    ap.add_argument("--sweep", action="store_true", help="grid-search knobs on the cached curves")
    ap.add_argument("--inspect", default="", help="print the sim curve near the GT cut(s) for one short id")
    ap.add_argument("--out-trans", default=str(OUT / "trans.json"))
    ap.add_argument("--out-segs", default=str(Path(TRANSITIONS).parent / "segments.json"))
    args = ap.parse_args()

    if args.dump_curves:
        dump_curves(args.sample_fps, args.luma_fps)
        return

    if not CURVES.exists():
        raise SystemExit(f"no curve cache at {CURVES} — run `face_cut.py --dump-curves` first (GPU/Docker)")
    curves = json.loads(CURVES.read_text())

    if args.inspect:
        cur = curves[args.inspect]
        g = {c["short"]: c for c in json.load(open(args.gt))["clips"]}.get(args.inspect, {})
        gcuts = [c["sec"] for c in (g.get("cuts") or [])] or ([g["cut_sec"]] if g.get("cut_sec") is not None else [])
        print(f"{args.inspect}  dur={cur['dur']}s  GT cuts={gcuts}  TransNet cuts={cur['transnet_cuts']}")
        for t, s in zip(cur["times"], cur["sims"]):
            bar = "#" * int(s * 40)
            mark = "".join(" <GTCUT" for gc in gcuts if abs(gc - t) < 0.6)
            print(f"  {t:5.2f}s  {s:.3f} {bar}{mark}")
        return

    if args.sweep:
        grid = []
        for thr in (0.15, 0.20, 0.25, 0.28, 0.32, 0.35):
            for gap in (0.5, 0.7, 1.0):
                for ms in (0.4, 0.5, 0.7):
                    for snap in (True, False):
                        grid.append({"thr": thr, "max_gap": gap, "min_seg": ms,
                                     "evidence_floor": args.evidence_floor,
                                     "snap": snap, "snap_win": args.snap_win})
        rows = []
        for p in grid:
            _, seg_out, _ = run_segmentation(curves, p)
            b = score(seg_out, args.gt)
            rows.append((b["full_pct"], b["prec"], b["rec"], p))
        rows.sort(key=lambda r: (-r[0], -(r[1] + r[2])))
        print(f"\nsweep vs {Path(args.gt).name} — top 15 (of {len(grid)}):")
        print("  full%   prec   rec   | thr  gap  minseg snap")
        for full, prec, rec, p in rows[:15]:
            print(f"  {full:5.1f}  {prec:5.1f}  {rec:5.1f}  | {p['thr']:.2f} {p['max_gap']:.1f}  "
                  f"{p['min_seg']:.1f}   {'Y' if p['snap'] else 'n'}")
        return

    p = {"thr": args.thr, "max_gap": args.max_gap, "min_seg": args.min_seg,
         "evidence_floor": args.evidence_floor, "snap": args.snap, "snap_win": args.snap_win,
         "min_return": args.min_return, "ret_sim": args.ret_sim,
         "refine": args.refine_fade, "fade_frac": args.fade_frac}
    if args.hybrid:
        trans_out, seg_out, snapped = run_hybrid(curves, json.load(open(args.base)), p)
        mode = f"HYBRID (base first cut + returns; min_return={args.min_return} ret_sim={args.ret_sim})"
    else:
        trans_out, seg_out, snapped = run_segmentation(curves, p)
        mode = "pure face-first"
    Path(args.out_trans).write_text(json.dumps(trans_out, indent=2))
    Path(args.out_segs).write_text(json.dumps(seg_out, indent=2))
    b = score(seg_out, args.gt)
    print(f"[{mode}]  thr={args.thr} max_gap={args.max_gap} min_seg={args.min_seg} snap={args.snap} "
          f"refine={args.refine_fade} (moved {snapped} boundaries)")
    print(f"vs {Path(args.gt).name}: FULL-seq {b['full']}/{b['tot']} = {b['full_pct']:.1f}%  "
          f"cut prec={b['prec']:.1f}% rec={b['rec']:.1f}%  (tp={b['tp']} miss={b['miss']} extra={b['extra']})")
    o = offset_stats(trans_out, args.gt)
    if o:
        print(f"first-cut convention gap (gt - model): mean={o['mean']:+.3f}s median={o['median']:+.3f}s "
              f"MAE={o['mae']:.3f}s  (n={o['n']}; →0 = matches manual)")
    print(f"Wrote {args.out_trans}, {args.out_segs}")
    if args.split:
        n = split_segments(seg_out, args.split)
        print(f"Cut {n} segment videos -> {args.split}/")


if __name__ == "__main__":
    main()
