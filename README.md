# pezevenk — person → meme transition detection

Find the timestamp where each `freckled_spike_tiktok` clip cuts from the creator
talking to camera over to a meme (still image / meme video / text card).

## Dataset

121 vertical clips (720×1280, ~8 s each). Frame rate varies per clip (TikTok
re-encodes), so the pipeline reads each clip's native fps and works in seconds
throughout — timestamps are comparable across clips regardless of fps.

| fps | 30.00 | 26.42 | 28.83 | 24.00 | 19.17 | 16.83 |
|-----|-------|-------|-------|-------|-------|-------|
| clips | 57 | 38 | 11 | 11 | 3 | 1 |

## Approach

The core insight: the person→meme transition **is a shot boundary**, so TransNetV2
detects it directly; labeling only decides *which* boundary it is. It's the **same
creator in every clip**, so each shot is labeled by **her face** (InsightFace) rather
than by content (an earlier CLIP labeler was fooled by memes containing people / dark
footage, and is removed).

| stage | model | job |
|-------|-------|-----|
| 1. shot boundaries | **TransNetV2** | split each clip into shots (the cut is one of these) |
| 2. label each shot | **InsightFace (buffalo_l)** | is *the creator's* face in this shot? |
| 3. pick the cut | max-agreement | best "creator prefix → meme suffix" split, tolerant of a stray face dropout |
| 3b. soft-cut recovery | dense face scan | TransNetV2 only fires on hard cuts; when it finds *no* cut, scan the whole clip for where she leaves — or when it *merges* creator+meme into one shot, scan that shot (accepted only for a ≤2.5 s move, so a mid-talk self-occlusion can't drag the cut early) |
| 3c. low-threshold re-detect | TransNetV2 retry | last resort: if a clip is a *single shot* and **neither** boundaries nor the face scan found a cut, re-detect just that clip at a lower TransNetV2 threshold (`--lowthr-redetect`, default 0.4) to recover a fast **match-cut** it merged away, then re-pick. Genuine creator-less clips stay silent (the evidence floor still applies), so it can't invent a cut |

**Domain prior.** The creator is *always* present and opens every clip — a clip can be
creator-only but never meme-only. So when the face pass finds no leading creator (her
dark/occluded intro fell just under threshold), the first cut is taken as the transition.
This resolves *ambiguity*, it's not a hard override: if the whole clip shows essentially
no face of her (max similarity below a small evidence floor — e.g. reposted stadium
footage), that's respected as genuinely creator-less. See `pick()` in `relabel_faces.py`.

**Accuracy** vs the canonical ground truth (`transitions/ground_truth.json`):
**89.3 % exact (≤0.5 s), 100 % within 1.5 s, 0 false positives, 0 wrong-time, 0 missed**
(up from 84 % / 88 % for the old CLIP labeler). The last remaining error — a fast
match-cut TransNetV2 merged into one shot while her walking intro also fell under the
face threshold — is recovered by the stage-3c low-threshold re-detect. Score anything
with `evaluate.py`; compare labelers with `ablation.py`.

## The pipeline

| script | does | writes |
|--------|------|--------|
| [`detect_transitions.py`](detect_transitions.py) | **stage 1** — TransNetV2 shot boundaries | `transitions/transitions.json` (raw shots) |
| [`relabel_faces.py`](relabel_faces.py) | **stage 2** — label shots by the creator's face + pick the cut | rewrites `transitions.json` (+ `transitions/qa/` with `--qa`) |
| [`segment_clips.py`](segment_clips.py) | **stage 3 (default)** — full creator/meme segment sequence per clip | `transitions/segments.json`, `segments/<clip>/NN_<label>.mp4` |
| [`split_clips.py`](split_clips.py) | binary cut at the transition | `split/person/*.mp4`, `split/meme/*.mp4` |
| [`evaluate.py`](evaluate.py) / [`ablation.py`](ablation.py) | score / compare approaches | prints tables |
| [`build_report.py`](build_report.py) / [`build_verify_ui.py`](build_verify_ui.py) | static `report.html` / `verify.html` | HTML |

**Segments are the general representation:** merge consecutive same-label shots →
`creator→meme` (2 segments), `creator→meme→creator…` (returns), or "no transition"
(1 segment). `segment_clips.py` honors the stage-2 first cut so the leading boundary keeps
the shipped accuracy. Knobs: `--face-threshold`, `--min-seg` (absorbs sub-N-second
segments — suppresses stray face-dropout noise).

**Main UI: `app.html`.** `./serve.sh` → `http://localhost:8000/` (root redirects there).
Reads `transitions.json` + `ground_truth.json` + `segments.json` live — after a run just
hit ↻ reload. Per clip: the timeline (every TransNetV2 cut as grey ticks,
green=creator/blue=meme, orange=picked, white=ground-truth), the before|after cut frame
(click to enlarge), the segment pattern with each piece playable next to the original, a
live tolerance slider (client-side verdicts), filters, and ✗-flagging with export. Vue is
vendored in `vendor/` so it runs offline. `report.html` / `verify.html` are static
fallbacks.

## Run it — Docker (recommended, reproducible)

The env is pinned across [`requirements-torch.txt`](requirements-torch.txt) (torch cu128),
[`requirements.txt`](requirements.txt) (TransNetV2 + I/O), and
[`requirements-face.txt`](requirements-face.txt) (InsightFace on **`onnxruntime-gpu`**).
TransNetV2 and buffalo_l are baked in, so runs are offline. **Both** stages use the GPU
when started with the NVIDIA runtime (TransNetV2 via torch, InsightFace via
onnxruntime-gpu — `relabel_faces.py` auto-detects CUDA, else CPU). `onnxruntime-gpu` is
pinned to the CUDA-12 line (1.22.x) to match the `nvidia/cuda:12.8-cudnn` base; 1.23+
moved to CUDA 13 and won't load here.

```bash
docker compose build                      # one-time
docker compose run --rm all               # detect → relabel → split → segment → report
# or a single stage: docker compose run --rm {detect|relabel|split|segment|report}
```

No compose? `./docker-run.sh detect_transitions.py` (set `GPU=0` to force CPU).

## Run it — local (conda)

```bash
conda run -n wedia_telif python detect_transitions.py     # stage 1: shots (~40 s, GPU)
conda run -n wedia_telif python relabel_faces.py --qa     # stage 2: face + cut (~2-3 min, CPU here*)
conda run -n wedia_telif python segment_clips.py --split  # stage 3: segments
```

\* the `wedia_telif` env's onnxruntime has no CUDA provider, so the face stage runs on CPU
there — use Docker for a GPU face stage.

**Key flags:** `detect_transitions.py --threshold` (0.5, cut sensitivity) `--limit N`
(smoke test); `relabel_faces.py --face-threshold` (0.35, cosine cutoff) `--frames-per-shot`
(3) `--qa` (dump before|after cut frames) `--no-soft-fallback`.

## Output

`transitions/transitions.json` — one record per clip:

```json
{
  "clip": "tikcdn_..._8a5c21466d03.mp4",
  "transition_sec": 3.331,
  "method": "creator_to_meme",
  "shots": [ { "start_sec": 0.0, "end_sec": 3.331, "face_sim": 0.71, "label": "person" }, ... ]
}
```

`method`: `creator_to_meme` (leading creator shot(s) → meme) · `creator_to_meme_prior`
(face missed her intro; domain prior takes the first cut) · `creator_to_meme_soft` /
`creator_to_meme_softcut` (soft-cut recovery) · `single_shot_creator` /
`all_creator_no_transition` (all creator, no cut) · `all_meme_no_creator` (genuinely no
creator anywhere).

## Verification

Two independent multi-agent passes reviewed all 121 clips frame-by-frame. A blind *manual*
pass (agents reading filmstrips) scored **98.3 %** and found every cut the algorithm
misses — a human reads *content change*, not just hard-cut + face-identity — but is itself
~2–6 % noisy at this scale (cross-checking the passes surfaced 8 ground-truth mistakes,
since corrected). Some soft dissolves are genuinely ± a few frames. Per-clip verdicts are
in `transitions/verification.json`; open `report.html` for the visual breakdown.
