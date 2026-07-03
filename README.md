# pezevenk — person → meme transition detection

Find the timestamp where each `freckled_spike_tiktok` clip cuts from the creator
talking to the camera over to a meme (still image / meme video / text card).

## Dataset

121 vertical clips (720×1280, ~8 s each). Frame rate varies per clip (typical of
TikTok re-encodes), so the pipeline reads each clip's native fps and works in
seconds throughout — cut timestamps are comparable across clips regardless of fps.

| fps | clips |
|-----|-------|
| 30.00 | 57 |
| 26.42 | 38 |
| 28.83 | 11 |
| 24.00 | 11 |
| 19.17 | 3 |
| 16.83 | 1 |

## Approach

Started as a reuse of the `wedia_telif` pipeline (TransNetV2 + OpenCLIP). The cut
detection (TransNetV2) is rock-solid; the weak link was the CLIP "creator vs meme"
labeling — memes that contain a person (or dark footage) fooled it. Since it's the
**same creator in every clip**, labeling is now done by **her face**. OpenCLIP has
been removed entirely — it is no longer used.

| stage | model | job |
|-------|-------|-----|
| 1. shot boundaries | **TransNetV2** | split each clip into shots (the cut is always one of these) |
| 2. label each shot | **InsightFace (buffalo_l)** | is *the creator's* face in this shot? |
| 3. pick the cut | max-agreement | best "creator prefix → meme suffix" split, tolerant of a stray dropout |
| 3b. soft-cut recovery | dense face scan | TransNetV2 sometimes misses a soft fade, so it either finds *no* cut (scan the whole clip) or *merges* creator+meme into one shot (scan that shot for where her face leaves) |

**Accuracy vs the canonical ground truth (`transitions/ground_truth.json`):**
**86.8 % exact (≤0.5 s), 96.7 % within 1.5 s** — and at a ≤0.7 s bar, **94.2 %**.
The 0.5 vs 0.7 gap is the ground truth's own **0.5 s reading granularity** (agents
read 2 fps filmstrips), not detector error. Up from **84 % / 88 %** for the old
CLIP labeler. Remaining errors: **3 misses** (creator occluded/dark in her intro,
so her face isn't detected), **1 false positive** (an all-meme clip whose figure
weakly matches her face), **0 wrong-time**. Score any approach with `evaluate.py`.

**Soft-cut recovery (`relabel_faces.py`, on by default, `--no-soft-fallback` to disable):**
TransNetV2 only fires on hard cuts, so soft fades slip through two ways, both handled
by scanning the creator's face presence:
- *no cut found* → scan the whole clip for where she leaves (fires only when she
  clearly opens and the tail is clearly not her);
- *creator+meme merged into one shot* → scan that shot; if she strongly opens and
  then leaves, cut there — accepted only for a *moderate* move (≤2.5 s), so a
  mid-talk self-occlusion (drinking, mirror) can't drag the cut early.

This eliminated all wrong-time errors and lifted the ≤0.7 s score from 91.7 % to 94.2 %.

The core insight is unchanged: the person→meme transition **is a shot boundary**,
so TransNetV2 detects it directly; labeling only decides *which* boundary it is.

## The pipeline

| script | does | writes |
|--------|------|--------|
| [`detect_transitions.py`](detect_transitions.py) | **stage 1** — TransNetV2 shot boundaries (no CLIP) | `transitions/transitions.json` (raw shots) |
| [`relabel_faces.py`](relabel_faces.py) | **stage 2** — label shots by the creator's face + pick the cut | rewrites `transitions.json` (+ `transitions/qa/` with `--qa`) |
| [`split_clips.py`](split_clips.py) | cut each clip at its transition | `split/person/*.mp4`, `split/meme/*.mp4` |
| [`evaluate.py`](evaluate.py) | score any transitions.json against `ground_truth.json` | prints exact / within-1.5s |
| [`build_report.py`](build_report.py) | QA dashboard (click a thumbnail to enlarge) | `report.html` |
| [`build_verify_ui.py`](build_verify_ui.py) | review UI (cut frame + original/person/meme videos per row) | `verify.html` |

**Main UI: `app.html` (the merged workbench).** `./serve.sh` → `http://localhost:8000/`
(the root redirects there). It reads `transitions.json` + `ground_truth.json` live —
after a pipeline run just hit ↻ reload, no regenerating HTML. Per clip it shows the
timeline (every TransNetV2 cut as grey ticks, green=creator/blue=meme, orange=picked,
white=ground-truth), the before|after cut frame (click to enlarge), the playable
ORIGINAL/PERSON/MEME videos, a **live tolerance slider** (recomputes verdicts client-side),
filters, and ✗-flagging with export. Vue is vendored in `vendor/` so it runs offline.
`verify.html` and `report.html` remain as static fallbacks (regenerate with
`build_verify_ui.py` / `build_report.py`; `report.html` even opens with a double-click).

`transitions_clip.json` keeps the older CLIP-only result for comparison.

**Beyond the single cut:** [`segment_clips.py`](segment_clips.py) emits the *full*
creator/meme segment sequence per clip (person→meme→person→…) instead of one cut.
Knobs: `--face-threshold`, `--min-seg` (absorb segments shorter than N seconds —
suppresses stray-dropout noise). In this dataset only ~1–2 clips are genuinely
multi-segment (e.g. she returns after the meme); `--split` cuts them into
`segments/<clip>/NN_<label>.mp4`.

## Run it — Docker (recommended, reproducible)

The environment is pinned across [`requirements-torch.txt`](requirements-torch.txt)
(torch cu128), [`requirements.txt`](requirements.txt) (TransNetV2 + I/O), and
[`requirements-face.txt`](requirements-face.txt) (InsightFace). TransNetV2 and the
buffalo_l face model are baked into the image, so runs are offline. Uses the GPU
automatically when present, falls back to CPU otherwise.

```bash
docker compose build                      # one-time
docker compose run --rm all               # detect → relabel → split → report
# or stage by stage:
docker compose run --rm detect
docker compose run --rm relabel
docker compose run --rm split
docker compose run --rm report            # -> transitions/report.html
```

No compose? Use the helper: `./docker-run.sh detect_transitions.py`
(set `GPU=0` to force CPU).

## Run it — local (existing conda env)

```bash
conda run -n wedia_telif python detect_transitions.py        # stage 1: shots (~40s)
conda run -n wedia_telif python relabel_faces.py --qa        # stage 2: face labels + cut (~2-3 min, CPU)
conda run -n wedia_telif python split_clips.py --workers 8
conda run -n wedia_telif python build_report.py              # -> report.html
```

### Key flags
- `detect_transitions.py --threshold` (default 0.5) — TransNetV2 cut sensitivity; lower = more cuts.
- `detect_transitions.py --limit N` — process only the first N clips (smoke test).
- `relabel_faces.py --face-threshold` (default 0.35) — cosine cutoff for matching her face.
- `relabel_faces.py --frames-per-shot` (default 3) — frames sampled per shot for the face check.
- `relabel_faces.py --qa` — dumps `transitions/qa/<id>_transition.png` (frame before | after the cut).

## Accuracy (independent verification)

Two independent multi-agent passes reviewed all 121 clips frame-by-frame:

- Algorithm: **86.8 %** exact (≤0.5 s) / **94.2 %** (≤0.7 s), **96.7 %** within 1.5 s.
- A blind *manual* pass (agents finding cuts from filmstrips) scored **98.3 %** vs
  the corrected truth and found **all** of the cuts the algorithm misses — because a
  human reads *content change*, not just hard-cut + face-identity. But that pass is
  not free of error either: cross-checking the two passes surfaced **8** ground-truth
  mistakes in the first one. So manual review is more capable but still ~2–6 %
  noisy at this scale, and some soft dissolves are genuinely ± a few frames.

Open [`report.html`](report.html) for the full visual breakdown (clean / multi-cut /
no-transition / problems), and `transitions/verification.json` for the per-clip verdicts.

## Reviewing / fixing splits — `verify.html`

To eyeball each split (e.g. to catch "the meme seeped into the person half"):

```bash
python3 build_verify_ui.py        # -> verify.html  (pure stdlib, no GPU/env needed)
./serve.sh                        # -> http://localhost:8000/verify.html
```

Each row puts, side by side: the **shot timeline** (green = creator, red = meme, white
mark = the picked cut), the **before | after cut frame**, and the **PERSON** and **MEME**
videos, both playable. Rows are ordered problems-first. Flag rows `✓`/`✗` (saved in your
browser) and hit **Export wrong** to download `wrong_splits.json`.

When the meme bleeds into the person half it's almost always the **wrong cut being
picked**: CLIP mislabels the first meme shot as the creator, so the transition jumps to a
later cut. You can hand-fix `transition_sec` for those clips in `transitions/transitions.json`
and re-run `split_clips.py` to regenerate the folders.

## Output

`transitions/transitions.json` — one record per clip:

```json
{
  "clip": "tikcdn_..._8a5c21466d03.mp4",
  "transition_sec": 3.33,
  "method": "person_to_meme",
  "num_shots": 2,
  "shots": [ { "start_sec": 0.0, "end_sec": 3.33, "p_person": 0.98, "label": "person" }, ... ]
}
```

`method` values:
- `person_to_meme` — confident: leading creator shot(s), then a meme shot.
- `no_leading_person_fallback_first_cut` — CLIP did **not** label the opening shot as the creator, so the first cut is reported as a weak fallback (the timestamp is often still right; only the labeling failed — see limitation below).
- `single_shot_meme` / `single_shot_person` — TransNetV2 found no cut.
- `all_person_no_transition` — every shot labeled creator.

## Known limitation (the honest part)

TransNetV2 cut detection is very reliable here. The weak link is the **zero-shot
CLIP person/meme labeling**, exactly as `wedia_telif/APPROACH.md` predicted.
When the creator wears themed clothing (e.g. a basketball jersey) or sits over a
busy background, CLIP can mislabel her shot as "meme", triggering the fallback.
The cut *time* is usually still correct; only the reasoning is off.

Two ways to harden it if needed:
1. **Better prompts / a tiny linear probe** on frozen CLIP embeddings (seconds to
   train, no GPU) — the same fallback `wedia_telif` documents.
2. **Face-based** (InsightFace, `wedia_telif`'s stage 3): the creator is the *same
   person* in every clip. Enroll her face once, then the transition is simply
   where her face stops appearing. This is the most robust option for this dataset.
