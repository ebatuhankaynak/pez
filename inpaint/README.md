# inpaint — burned-in text / caption removal

Removes hardcoded captions/watermarks from the meme clips and re-encodes with the
original audio. Auto-locates the caption, masks the glyph strokes, and fills them in.

## Engine — per-frame big-LaMa (spatial), not temporal
These captions are **static centred text over near-static scenery**, so the pixels
behind the text are rarely revealed by motion. A temporal/flow method (ProPainter)
has nothing reliable to propagate and drags moving subjects into the band as a dark
blob. So the default is **big-LaMa** (`simple-lama-inpainting`): a per-frame spatial
inpaint that hallucinates a plausible background even where it's never revealed.
`--engine propainter` is kept for the opposite case (camera motion genuinely reveals
the real background) but is not used for these clips.

### Fill (`inpaint_composite`)
1. **One-sided feather** — full opacity everywhere the mask is set, soft falloff only
   *outward*. A two-sided feather leaves the fill semi-transparent over thin strokes
   and the original bright text bleeds back through as a ghost (worst on black).
2. **Flat-snap on a uniform surround** — measure background uniformity in a tight ring
   just outside the text. Flat (black band / wall) → snap to a Telea-diffused colour
   (exact, no muddy bar). Busy → keep LaMa's plausible texture. Only the surround is
   trusted — the block interior is unknowable from one frame.
3. **Flatness-scaled dilation** — swallow the glyph's anti-alias/shadow halo wide on a
   flat surround (the fill is one colour, so it's free), tight on texture (keep detail).

Result: flat/black backgrounds come out clean; textured ones are soft but artifact-free.

## Auto-location — RapidOCR
Rects are **optional**; omit them and the caption is located automatically
(`rapidocr-onnxruntime`, ONNX, no torch dep). It samples frames, clusters OCR boxes
across them by IoU, keeps the screen-static ones (burned-in captions persist; transient
background text doesn't), and emits a tight box per line. Pass `-r x1,y1,x2,y2`
(repeatable) to override.

## Usage
Runs in the pezevenk docker (`pez-inpaint`; venv `/opt/venv`, repo at `/app`):
```bash
python inpaint/inpaint_text.py -i split/meme/<clip>.mp4 -o out.mp4
```
- `-r x1,y1,x2,y2`  caption bbox override (repeatable; default = auto-locate).
- `--engine`        `lama` (default) | `propainter`.
- `--feather N`     mask edge blur sigma (default 5).
- `--mask-preview P` write a mask montage and exit (no inpaint).

## Batch eval
```bash
python inpaint/batch_eval.py --random 10 --seed 0   # or --all / --count N / --offset
```
Loads RapidOCR + LaMa once, processes the picked clips (~18–45 s each on the RTX 5070 Ti),
writes `eval/out/<name>.mp4` + `eval/manifest.json`. The **"text removal" tab of
`memes.html`** (served by `serve.py` on :8000) shows the synced IN/OUT pairs.

## Notes
- Output re-encodes video at CRF 16 (visually lossless) and copies the original audio.
- Model weights land under `.cache/` (git-ignored) via `TORCH_HOME=/app/inpaint/.cache`.
