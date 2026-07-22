# inpaint — burned-in text / caption removal

Removes hardcoded captions/watermarks from the meme clips and re-encodes with the
original audio. Auto-locates the caption, masks the glyph strokes, and fills them in.

## Engines
These captions are **static centred text over near-static scenery**, so the pixels
behind the text are rarely revealed by motion — a temporal/flow method has nothing
reliable to propagate. Three engines, picked per clip by `--engine auto` (default):

- **`solid`** *(auto-selected)* — when the band is near-uniform (letterbox, flat wall)
  the strokes are just replaced with the ring's median colour. Exact, instant, no smudge.
- **`minimax`** *(auto default for textured bands)* — band-cropped MiniMax-Remover
  (distilled Wan2.1 video DiT). Runs on the caption band only, so few latent tokens and
  no whole-frame softening. Needs the vendored repo + weights (see setup below).
- **`lama`** — per-frame big-LaMa (`simple-lama-inpainting`), a spatial inpaint that
  hallucinates a plausible background even where motion never reveals it. Self-contained
  (weights auto-download to `.cache/`); good fallback when minimax isn't provisioned.

`--engine auto` computes band flatness per clip: flat → `solid`, else → `minimax`.
Force any engine explicitly with `--engine {auto,lama,minimax}`.

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
```bash
python inpaint/inpaint_text.py -i split/meme/<clip>.mp4 -o out.mp4
```
- `-r x1,y1,x2,y2`  caption bbox override (repeatable; default = auto-locate).
- `--engine`        `auto` (default) | `lama` | `minimax`.
- `--flat-thr F`    band-flatness cutoff for auto → solid-fill (default 0.6).
- `--pad N`         vertical context around the caption band (default 200).
- `--feather N`     mask edge blur sigma (default 5).
- `--mask-preview P` write a mask montage and exit (no inpaint).

### MiniMax engine setup (not baked into the Docker image)
`minimax` (and therefore `auto` on textured bands) needs a vendored third-party repo
plus its HuggingFace weights, **provisioned separately** — the pezevid Docker image
does *not* install `diffusers`/download these, and `inpaint/` is not a compose service.
They live under `inpaint/_minimax/` (git-ignored):
```bash
# from inpaint/
git clone https://github.com/zibojia/MiniMax-Remover _minimax
pip install -r _minimax/requirements.txt          # diffusers==0.33.1 etc.
huggingface-cli download zibojia/minimax-remover --local-dir _minimax/weights
```
Without this, use `--engine lama` (self-contained). Running the pipeline inside the
transitions container also requires `pip install rapidocr-onnxruntime simple-lama-inpainting`
— these are likewise not in the image's requirements yet.

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
