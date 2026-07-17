#!/usr/bin/env python3
"""Remove burned-in text/captions from a video with ProPainter.

Strategy for speed + quality on a 16GB GPU:
  * Crop only the horizontal band containing the caption (full width + vertical
    context) and inpaint THAT at native resolution -- few pixels, fits VRAM, no
    upscaling.
  * Mask only the actual TEXT STROKES (+ emoji flag), not a big rectangle, so the
    real background between/around the letters is preserved and never hallucinated.
    A rectangle mask is what makes ProPainter output look blurry; a glyph-tight
    per-frame mask keeps the untouched scenery sharp.
  * Composite the inpainted band back onto the pristine full-res original.

Usage:
  inpaint_text.py -i in.mp4 -o out.mp4 -r X1,Y1,X2,Y2 [-r ...] [--pad 200]
  inpaint_text.py ... --mask-preview preview.png   # verify masks, don't inpaint
  inpaint_text.py ... --rect-mask                  # old behaviour (mask whole box)

Each -r is the SEARCH region where the caption may appear (generous is fine); the
glyph detector tightens the actual mask within it per frame.
"""
import argparse, glob, json, os, shutil, subprocess, time
import cv2, numpy as np

PROP_DIR = os.path.expanduser("~/dev/ProPainter")
PY = os.path.join(PROP_DIR, "venv", "bin", "python")


def sh(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path])
    s = json.loads(out)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(n) / float(d)


def snap8(v):
    return int(round(v / 8) * 8)


def _fill_holes(mask):
    """Fill interior holes (letter counters, emoji interiors) via border flood fill."""
    h, w = mask.shape
    ff = mask.copy()
    m2 = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m2, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff))


def caption_mask(bgr, white_thr=200, black_thr=60, margin=5):
    """General, background-INDEPENDENT mask of one outlined caption line (+ any
    inline trailing emoji) in the crop `bgr`. Returns uint8 (255 = remove).

    Core cue: outlined caption text is the ONLY place a near-white fill pixel and a
    near-black outline pixel occur within a few px of each other. Smooth sky, cloud,
    road and mountain never pair bright-white directly against near-black, so they
    are rejected with NO colour/hue, "cloud" or "flag" special-casing. The cue is a
    property of the text itself, so it transfers to arbitrary clips/backgrounds and
    to either polarity (light text/dark outline or dark text/light outline).

    The inline trailing emoji is NOT outlined text; it is found generically as
    same-line foreground that deviates from the vertically-continuous scenery just
    above and below the text band (an emoji pasted on top stands out; sky/road do
    not), reinforced by local edge energy -- again colour-agnostic. It is filled as
    a band-clamped box so no anti-aliased emoji fringe can survive."""
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1) outlined-text co-occurrence -> the caption letters
    white = (gray >= white_thr).astype(np.uint8)
    black = (gray <= black_thr).astype(np.uint8)
    prox = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cooc = cv2.bitwise_and(cv2.dilate(white, prox), cv2.dilate(black, prox)) * 255
    cooc = cv2.morphologyEx(cooc, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cooc = cv2.dilate(cooc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cooc, 8)
    txt = np.zeros_like(cooc)
    for i in range(1, n):                                    # drop micro-texture specks
        if stats[i, cv2.CC_STAT_AREA] >= 80 and stats[i, cv2.CC_STAT_HEIGHT] >= 12:
            txt[lab == i] = 255
    txt = _fill_holes(txt)
    if not txt.any():
        return np.zeros((H, W), np.uint8)

    ys = np.where(txt.any(axis=1))[0]
    xs = np.where(txt.any(axis=0))[0]
    y0, y1 = int(ys.min()), int(ys.max())
    x_right = int(xs.max())
    band_h = y1 - y0 + 1
    out = txt.copy()

    # 2) inline trailing emoji: same-band foreground vs. scenery above/below
    by0, by1 = max(0, y0 - 2), min(H, y1 + 3)
    strip = max(5, band_h // 2)
    t0, b1 = max(0, by0 - strip), min(H, by1 + strip)
    lab_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    salband = np.zeros((H, W), np.uint8)
    refs = []
    if by0 - t0 >= 3:
        refs.append(np.median(lab_img[t0:by0], axis=0))
    if b1 - by1 >= 3:
        refs.append(np.median(lab_img[by1:b1], axis=0))
    if refs:
        pred = np.mean(refs, axis=0)                          # per-column bg colour
        resid = np.abs(lab_img[by0:by1] - pred[None]).sum(axis=2)
        salband[by0:by1] |= (resid > 35).astype(np.uint8) * 255
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    salband[by0:by1] |= (grad[by0:by1] > 70).astype(np.uint8) * 255
    salband = cv2.morphologyEx(salband, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    wx1 = min(W, x_right + int(2.3 * band_h))                 # don't reach far scenery
    colfg = salband[by0:by1].mean(axis=0) / 255.0
    thr, GAP = 0.12, max(6, int(0.4 * band_h))
    start = end = None
    gap = 0
    c = x_right + 1
    while c < wx1:                                            # walk right until a real gap
        if colfg[c] >= thr:
            if start is None:
                start = c
            end = c
            gap = 0
        elif start is not None:
            gap += 1
            if gap > GAP:
                break
        c += 1
    if start is not None and end is not None:                # height clamped to text band
        pad = max(3, int(0.12 * band_h))                     # generous: LaMa fills big masks cleanly,
        out[by0:by1, max(0, start - pad):min(W, end + pad)] = 255  # so never leave a flag remnant

    # 3) small uniform margin so no thin anti-alias halo survives
    out = _fill_holes(out)
    k = max(3, 2 * margin - 1)
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return out


def glyph_mask(bgr, rects):
    """Full-frame caption mask (255 = remove). OCR already LOCATED the caption line(s)
    in `rects`; inside each box we mask only the actual TEXT STROKES so LaMa fills a
    thin, easy region (plausible) instead of a whole band (foggy smear).

    Cue = morphological top-hat + black-hat with a kernel ~ the line height. Top-hat
    isolates BRIGHT structures thinner than the kernel (the letter fill) and is ~zero
    on large smooth bright regions (fog / sky / a lit subject), so it does NOT mask the
    background -- the failure of the earlier "bright-cover" mask, which keyed on raw
    HSV-V and obliterated bright subjects (e.g. a shoebill's pale beak) into fog.
    Black-hat catches the thin dark outline. Union, then a small dilate+close scaled to
    the line height swallows the outline + anti-alias halo. Background-independent, so
    it neither leaves a ghost (co-occurrence's failure on bright bg) nor smears the
    scene (bright-cover's failure)."""
    H, W = bgr.shape[:2]
    m = np.zeros((H, W), np.uint8)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for x1, y1, x2, y2 in rects:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        bh, bw = y2 - y1, x2 - x1
        # morphology with a line-height kernel is O(kernel) and gets very slow on a
        # tall (multi-line) box; do it on a downscaled crop -- the kernel-vs-content
        # size RATIO (what selects thin strokes over big blobs) is scale-invariant --
        # then upscale the stroke map and finish the small dilate/close at full res.
        sc = min(1.0, 140.0 / bh)
        cw, ch = max(1, int(bw * sc)), max(1, int(bh * sc))
        crop = cv2.resize(g[y1:y2, x1:x2], (cw, ch), interpolation=cv2.INTER_AREA)
        se = max(9, int(0.35 * ch)) | 1                       # > stroke width: excludes big bright blobs
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se, se))
        th = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, k)      # thin bright fill
        bl = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, k)    # thin dark outline
        small = (((th > 20) | (bl > 25)).astype(np.uint8)) * 255
        sub = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
        kd = max(5, int(0.10 * bh)) | 1                       # swallow outline + AA halo
        se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
        sub = cv2.dilate(sub, se2)
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, se2)
        sub = _fill_holes(sub)
        m[y1:y2, x1:x2] = cv2.max(m[y1:y2, x1:x2], sub)
    return m


def auto_rects(path, W, H, ocr=None):
    """Auto-locate caption search regions with NO manual coordinates.

    Primary: RapidOCR (ONNX, no torch/torchvision dep) LOCATES text robustly on
    arbitrary backgrounds -- far more reliable than any hand-tuned pixel heuristic.
    We sample frames, keep PERSISTENT text boxes (a burned-in caption sits at the
    same place across the clip; transient background text/signs do not), merge
    per-line and pad generously (extra to the right for a trailing inline emoji).
    caption_mask then tightens the actual per-frame stroke mask inside each rect.

    Falls back to the co-occurrence locator if RapidOCR is unavailable. Pass a
    preloaded `ocr` (RapidOCR instance) to avoid re-init when batching."""
    try:
        return _auto_rects_ocr(path, W, H, ocr=ocr)
    except Exception as e:                                    # rapidocr missing / failed
        print(f"[auto] OCR locate unavailable ({e.__class__.__name__}: {e}); "
              f"falling back to co-occurrence")
        return _auto_rects_cooc(path, W, H)


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _cooc_density(gray_box, white_thr=200, black_thr=60):
    """Outlined-text cue: fraction of pixels where a near-white FILL and a near-black
    OUTLINE occur within a few px of each other -- the signature of a burned-in caption.
    Printed real-world text (a sign, a phone/app UI label, a shipping label on a held
    box) has no black outline, so this reads ~0 for it. Used to tell an added caption
    from text that belongs to the footage."""
    if gray_box.size == 0:
        return 0.0
    w = (gray_box >= white_thr).astype(np.uint8)
    b = (gray_box <= black_thr).astype(np.uint8)
    pr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return float(cv2.bitwise_and(cv2.dilate(w, pr), cv2.dilate(b, pr)).mean())


def _auto_rects_ocr(path, W, H, ocr=None, sample=8, persist=0.34, min_score=0.5,
                    style_thr=0.12, drift_thr=3.0, txtvar_thr=0.9):
    """OCR-driven caption locator returning TIGHT per-line boxes.

    A burned-in caption is screen-static, so each text LINE's OCR box recurs at the
    same place across frames. We cluster boxes across sampled frames by overlap (IoU),
    keep clusters seen in >= `persist` of the frames (rejects transient background
    text/signs), and emit each cluster's MEDIAN box with a small pad -- a little extra
    to the RIGHT for a trailing inline emoji the OCR can't read. Tight per-line boxes
    keep the downstream stroke mask OFF the subject; an earlier ballooned block box let
    the mask fire on the scene (a shoebill's beak, a taped package) and smear it.

    Persistence alone is NOT enough: static real-world text -- a sign in a locked shot,
    a held phone/app screen, a shipping label on a slowly-moving box -- also recurs and
    was being masked (erasing footage the viewer expects, and, when it moved through a
    region, smearing a whole band via the +-N temporal mask union). So a persistent
    cluster is rejected as scene text when it is NOT outlined caption (`style` co-occ
    below `style_thr`) AND is either moving with an object (centroid `drift` over
    `drift_thr` px) or garbled/unstable (`txtvar` = distinct strings / occurrences, at
    or above `txtvar_thr`). Outlined captions clear it on style; plain captions on a
    flat band clear it by being screen-locked with stable text. Thresholds were
    measured on 40 clips: every rejected cluster was real scene/UI text, none a caption."""
    if ocr is None:
        from rapidocr_onnxruntime import RapidOCR       # ONNX, no torch dep
        ocr = RapidOCR()
    cap = cv2.VideoCapture(path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = np.linspace(0, N - 1, min(sample, N)).astype(int)
    clusters = []                                        # each: [list of {b,text,style}, set of frame idxs]
    n = 0
    for fi, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        n += 1
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        res, _ = ocr(f)
        for box, _text, score in (res or []):
            if score < min_score:
                continue
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            b = (max(0, int(min(xs))), max(0, int(min(ys))),
                 min(W, int(max(xs))), min(H, int(max(ys))))
            if b[2] - b[0] < 10 or b[3] - b[1] < 6:
                continue
            best, bj = 0.0, -1
            for j, (boxes, _fr) in enumerate(clusters):
                v = _iou(b, boxes[-1]["b"])
                if v > best:
                    best, bj = v, j
            rec = {"b": b, "text": (_text or "").strip(),
                   "style": _cooc_density(g[b[1]:b[3], b[0]:b[2]])}
            if best >= 0.3:
                clusters[bj][0].append(rec); clusters[bj][1].add(fi)
            else:
                clusters.append([[rec], {fi}])
    cap.release()
    if n == 0:
        return []
    need = max(2, int(round(persist * n)))
    rects = []
    for boxes, frames in clusters:
        if len(frames) < need:                          # not persistent -> transient bg text
            continue
        arr = np.array([r["b"] for r in boxes])
        x0, y0, x1, y1 = (int(np.median(arr[:, k])) for k in range(4))
        h = y1 - y0
        if h < 0.008 * H or (x1 - x0) < 20:
            continue
        cx = (arr[:, 0] + arr[:, 2]) / 2.0; cy = (arr[:, 1] + arr[:, 3]) / 2.0
        drift = float(np.hypot(cx.std(), cy.std()))
        style = float(np.mean([r["style"] for r in boxes]))
        txts = [r["text"] for r in boxes]
        txtvar = len(set(txts)) / max(1, len(txts))
        if style < style_thr and (drift > drift_thr or txtvar >= txtvar_thr):
            print(f"[auto] skip scene-text (style={style:.3f} drift={drift:.1f} "
                  f"txtvar={txtvar:.2f}): '{txts[len(txts) // 2][:30]}'")
            continue
        pv, pl = int(0.22 * h), int(0.12 * h)           # tight pad; extra right for inline emoji
        rects.append((max(0, x0 - pl), max(0, y0 - pv),
                      min(W, x1 + int(0.7 * h)), min(H, y1 + pv)))
    return rects


def _auto_rects_cooc(path, W, H, sample=48, persist=0.10, white_thr=200, black_thr=60):
    """Fallback co-occurrence locator (no OCR): a burned-in caption is a PERSISTENT,
    screen-static overlay of outlined text, so the white-fill/near-black-outline
    co-occurrence fires at the SAME place across frames. Less robust than OCR on
    non-outlined text, but needs no extra packages."""
    cap = cv2.VideoCapture(path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = np.linspace(0, N - 1, min(sample, N)).astype(int)
    prox = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    heat = np.zeros((H, W), np.float32)
    n = 0
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        white = (g >= white_thr).astype(np.uint8)
        black = (g <= black_thr).astype(np.uint8)
        heat += cv2.bitwise_and(cv2.dilate(white, prox), cv2.dilate(black, prox))
        n += 1
    cap.release()
    if n == 0:
        return []
    persistent = (heat / n >= persist).astype(np.uint8)
    # merge glyphs on a line into one region (wide horizontal close), drop specks
    persistent = cv2.morphologyEx(persistent, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (41, 9)))
    persistent = cv2.morphologyEx(persistent, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(persistent, 8)
    rects = []
    for i in range(1, nlab):
        x, y, w, h, area = st[i]
        fill = area / float(w * h)
        # a caption line is WIDE-and-SHORT outlined text: reject tall blocks (busy
        # background), narrow verticals, and dense solid blobs (not glyph-sparse text)
        if h > 0.16 * H or w < 40 or h < 12 or w < 1.2 * h or fill > 0.55:
            continue
        px, py = int(0.6 * h), int(0.5 * h)                 # pad; extra right for emoji
        rects.append((max(0, int(x - px)), max(0, int(y - py)),
                      min(W, int(x + w + 1.6 * h)), min(H, int(y + h + py))))
    return rects


def _ring_flat(bgr, mask, ring=25):
    """How uniform is the REAL background in the annulus just OUTSIDE the text mask.
    Returns w in [0,1]: 1 when the surround is essentially one colour (black band,
    flat wall), 0 when busy. Sampled only from unmasked pixels, so it is never
    'starved' inside a thick block the way a per-pixel window is. Single-frame stats
    on the surround are safe here BY CONSTRUCTION: we only act on them when they say
    'flat', which is exactly the case where a flat fill is the correct answer -- if a
    textured clip goes momentarily flat (a dark frame), the flat fill still matches.
    The ring is TIGHT (~25px): a wide annulus reaches unrelated bright content (frame
    edges, watermarks) far from the caption and spikes the std, mislabelling a black
    band as textured."""
    k = int(ring) | 1
    ann = (cv2.dilate(mask, np.ones((k, k), np.uint8)) > 0) & (mask == 0)
    if int(ann.sum()) < 50:
        return 0.0
    std = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)[ann].std())
    LO, HI = 4.0, 9.0                                        # flat <=LO ; textured >=HI
    return float(np.clip((HI - std) / (HI - LO), 0.0, 1.0))


def inpaint_composite(full, mask, lama, feather=5, win=121):
    """Fill the masked text. LaMa is the default -- its plausible texture reads fine on
    busy backgrounds. Its ONE failure is a faint colour-speckle cloud on a flat region
    (a ghost on pure black), so ONLY where the immediate surround is near-uniform
    (`_ring_flat`) do we snap instead to a Telea-diffused background colour, which is
    exact on flat/one-colour regions and, following edges from the boundary inward,
    never pulls a distant dark region into a hard bar the way a box mean-fill did.
    No background routing on the block INTERIOR (unknowable from one frame, and the
    source of every misroute) -- only on the trustworthy surround. Only masked pixels
    change."""
    from PIL import Image
    H, W = full.shape[:2]
    w = _ring_flat(full, mask)                              # tight fixed ring, not `win`
    # A glyph's anti-aliased outline/drop-shadow extends a few px past the thresholded
    # stroke and survives as a faint ghost -- most visible on black. On a FLAT surround
    # (w high) the fill is one colour, so dilating to swallow that halo costs nothing;
    # on texture keep it tight to avoid erasing real detail.
    d = 2 + int(round(6 * w))
    md = cv2.dilate(mask, np.ones((2 * d + 1, 2 * d + 1), np.uint8))
    out = lama(Image.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB)),
               Image.fromarray(md))
    lo = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    if lo.shape[:2] != (H, W):                               # lama pads to /8
        lo = cv2.resize(lo, (W, H), interpolation=cv2.INTER_LANCZOS4)
    if w > 0.01:
        diff = cv2.inpaint(full, (md > 0).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        fill = diff.astype(np.float32) * w + lo.astype(np.float32) * (1.0 - w)
    else:
        fill = lo.astype(np.float32)
    # ONE-SIDED feather: full opacity everywhere the mask is set (so a thin glyph
    # stroke is FULLY replaced, never left semi-transparent -- a two-sided Gaussian
    # feather let the original bright text bleed through the thin strokes and show as
    # a readable ghost, worst on black bg), soft falloff ONLY outward past the edge.
    fe = cv2.GaussianBlur(md, (0, 0), feather).astype(np.float32) / 255.0
    fe = np.maximum(fe, (md > 0).astype(np.float32))[..., None]
    comp = full.astype(np.float32) * (1 - fe) + fill * fe
    return comp.clip(0, 255).astype(np.uint8)


def lama_frames(origs, full_masks, final, lama=None, feather=5, log=True, win=121):
    """Per-frame background reconstruction where motion NEVER reveals the truth
    (static centred captions) -- the case ProPainter can't handle. Uses a Telea/LaMa
    hybrid (see `inpaint_composite`) so flat backgrounds fill clean and textured ones
    stay plausible. Writes full-frame results to `final/%05d.png`. Pass a preloaded
    `lama` (SimpleLama) to share the model across clips when batching."""
    if lama is None:
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
    for i, (of, m) in enumerate(zip(origs, full_masks)):
        full = cv2.imread(of)
        if m.any():
            full = inpaint_composite(full, m, lama, feather=feather, win=win)
        cv2.imwrite(os.path.join(final, f"{i:05d}.png"), full)
        if log and i % 40 == 0:
            print(f"[lama] {i + 1}/{len(origs)}")
    return lama


def temporal_max(masks, radius):
    if radius <= 0:
        return masks
    out = []
    n = len(masks)
    for i in range(n):
        acc = masks[i].copy()
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            acc = cv2.max(acc, masks[j])
        out.append(acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output")
    ap.add_argument("-r", "--rect", action="append",
                    help="caption search region 'x1,y1,x2,y2' (px); repeatable. "
                         "Omit to auto-locate the caption(s) (see --auto).")
    ap.add_argument("--auto", action="store_true",
                    help="auto-locate caption region(s) from persistent outlined-text "
                         "(implied when no -r is given). General across memes.")
    ap.add_argument("--engine", choices=("lama", "propainter"), default="lama",
                    help="lama = per-frame spatial inpaint (best when the background behind "
                         "the caption is never revealed, e.g. static centred text; runs in the "
                         "pezevenk docker). propainter = temporal (best when camera motion "
                         "reveals the real background across frames).")
    ap.add_argument("--pad", type=int, default=200, help="vertical context around band (propainter)")
    ap.add_argument("--pp-scale", type=float, default=1.0,
                    help="downscale the caption band by this factor before ProPainter, upscale "
                         "after (0.5 = quarter the pixels: far less VRAM -> larger subvideo_length "
                         "-> much faster; the temporal inpaint is soft anyway so the loss is small)")
    ap.add_argument("--feather", type=int, default=5, help="mask edge blur sigma")
    ap.add_argument("--rect-mask", action="store_true", help="mask the whole rect (no glyph detect)")
    ap.add_argument("--mask-temporal", type=int, default=3, help="+-N frame mask union (stability)")
    ap.add_argument("--mask-preview", help="write a mask montage PNG and exit (no inpaint)")
    ap.add_argument("--subvideo_length", type=int, default=80)
    ap.add_argument("--neighbor_length", type=int, default=10)
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()

    t0 = time.time(); T = {}
    W, H, fps = probe(a.input)
    if a.rect:
        rects = [tuple(map(int, r.split(","))) for r in a.rect]
    else:                                                # auto-locate caption(s)
        rects = auto_rects(a.input, W, H)
        print(f"[auto] found {len(rects)} caption region(s): {rects}")
        if not rects:
            print("[auto] no caption detected -> copying input unchanged")
            if a.output:
                shutil.copyfile(a.input, a.output)
            return
    y1 = max(0, min(r[1] for r in rects) - a.pad)
    y2 = min(H, max(r[3] for r in rects) + a.pad)
    by1 = snap8(y1); bh = snap8(y2 - by1); by2 = by1 + bh
    print(f"[i] {W}x{H} @ {fps:.3f}fps ; engine={a.engine} ; band y[{by1}:{by2}] = {W}x{bh} ; "
          f"{'RECT' if a.rect_mask else 'glyph'} mask")

    work = os.path.join(os.path.dirname(os.path.abspath(a.output or a.input)),
                        ".work_" + os.path.splitext(os.path.basename(a.input))[0])
    bandin, origdir, maskdir, res, final = (
        os.path.join(work, d) for d in ("band_in", "orig", "masks", "results", "final"))
    for d in (bandin, origdir, maskdir, res, final):
        os.makedirs(d, exist_ok=True)

    # 1) extract full-res frames (+ native caption band, only needed for propainter)
    t = time.time()
    if a.engine == "propainter":
        sh(["ffmpeg", "-y", "-v", "error", "-i", a.input,
            "-vf", f"crop={W}:{bh}:0:{by1}", os.path.join(bandin, "%05d.png")])
    sh(["ffmpeg", "-y", "-v", "error", "-i", a.input, os.path.join(origdir, "%05d.png")])
    T["extract"] = time.time() - t
    origs = sorted(glob.glob(os.path.join(origdir, "*.png")))

    # 2) build per-frame glyph masks (full-frame), then band-crop them for ProPainter
    t = time.time()
    if a.rect_mask:
        base = np.zeros((H, W), np.uint8)
        for x1, ry1, x2, ry2 in rects:
            cv2.rectangle(base, (x1, ry1), (x2, ry2), 255, -1)
        full_masks = [base] * len(origs)
    else:
        full_masks = [glyph_mask(cv2.imread(of), rects) for of in origs]
        full_masks = temporal_max(full_masks, a.mask_temporal)
    T["mask"] = time.time() - t

    if a.mask_preview:
        idx = np.linspace(0, len(origs) - 1, min(12, len(origs))).astype(int)
        rows = []
        for i in idx:
            im = cv2.imread(origs[i]); ov = im.copy(); ov[full_masks[i] > 0] = (0, 0, 255)
            vis = cv2.addWeighted(im, 0.5, ov, 0.5, 0)[by1:by2]
            cv2.putText(vis, f"f{i} cov{100*(full_masks[i]>0).mean():.2f}%",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            rows.append(cv2.resize(vis, (W // 2, bh // 2)))
        cv2.imwrite(a.mask_preview, np.vstack(rows))
        print(f"[preview] {a.mask_preview}  (mean cov "
              f"{100*np.mean([(m>0).mean() for m in full_masks]):.2f}%)  | {time.time()-t0:.1f}s")
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        return

    # 3) inpaint the masked caption region -> full-frame results in `final/`
    if a.engine == "lama":
        t = time.time()
        win = int(np.clip(2.5 * np.median([r[3] - r[1] for r in rects]), 61, 221))
        lama_frames(origs, full_masks, final, feather=a.feather, win=win)
        T["lama"] = time.time() - t
    else:
        for i, m in enumerate(full_masks):
            cv2.imwrite(os.path.join(maskdir, f"{i:05d}.png"), m[by1:by2])
        # optional downscale of the band (+masks) so ProPainter fits a larger temporal
        # window in VRAM and runs much faster; composite upscales the result back.
        if a.pp_scale < 1.0:
            sw = snap8(max(64, int(W * a.pp_scale))); sh_ = snap8(max(64, int(bh * a.pp_scale)))
            for d in (bandin, maskdir):
                interp = cv2.INTER_AREA if d == bandin else cv2.INTER_NEAREST
                for f in glob.glob(os.path.join(d, "*.png")):
                    cv2.imwrite(f, cv2.resize(cv2.imread(f), (sw, sh_), interpolation=interp))
            print(f"[propainter] band downscaled {W}x{bh} -> {sw}x{sh_} (pp_scale={a.pp_scale})")
        # ProPainter on the band, per-frame masks. Retry with smaller temporal chunks on
        # CUDA OOM (VRAM headroom varies with whatever else is on the GPU), robust w/o tuning.
        t = time.time()
        env = dict(os.environ, PYTORCH_ALLOC_CONF="expandable_segments:True")
        ladder = sorted({s for s in (a.subvideo_length, 40, 24, 12) if s <= a.subvideo_length},
                        reverse=True) or [a.subvideo_length]
        err = None
        for sv in ladder:
            try:
                sh([PY, os.path.join(PROP_DIR, "inference_propainter.py"),
                    "-i", bandin, "-m", maskdir, "-o", res,
                    "--subvideo_length", str(sv),
                    "--neighbor_length", str(min(a.neighbor_length, sv)),
                    "--fp16", "--save_frames", "--save_fps", str(int(round(fps)))],
                   cwd=PROP_DIR, env=env)
                print(f"[propainter] ok at subvideo_length={sv}")
                err = None
                break
            except subprocess.CalledProcessError as e:
                err = e
                print(f"[propainter] subvideo_length={sv} failed (likely OOM); retrying smaller")
                for d in glob.glob(os.path.join(res, "*")):
                    shutil.rmtree(d, ignore_errors=True)
        if err is not None:
            raise err
        T["propainter"] = time.time() - t
        ip = sorted(glob.glob(os.path.join(res, "*", "frames", "*.png")))
        assert ip and len(ip) == len(origs), f"frame mismatch: {len(ip)} vs {len(origs)}"

        # composite band back onto full-res original, feathered per-frame mask
        t = time.time()
        for i, (of, ipf) in enumerate(zip(origs, ip)):
            full = cv2.imread(of)
            band = cv2.imread(ipf)
            if band.shape[:2] != (bh, W):
                band = cv2.resize(band, (W, bh), interpolation=cv2.INTER_LANCZOS4)
            mb = full_masks[i][by1:by2]
            fe = cv2.GaussianBlur(mb, (0, 0), a.feather).astype(np.float32) / 255.0
            fe = np.maximum(fe, (mb > 0).astype(np.float32))[..., None]  # full opacity over mask
            reg = full[by1:by2].astype(np.float32)
            full[by1:by2] = (reg * (1 - fe) + band.astype(np.float32) * fe).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(final, f"{i:05d}.png"), full)
        T["composite"] = time.time() - t

    # 5) encode at native fps + copy original audio
    t = time.time()
    sh(["ffmpeg", "-y", "-v", "error", "-framerate", f"{fps:.6f}",
        "-i", os.path.join(final, "%05d.png"), "-i", a.input,
        "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16",
        "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", a.output])
    T["encode"] = time.time() - t

    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"[✓] {a.output}")
    print("    " + "  ".join(f"{k}={v:.1f}s" for k, v in T.items())
          + f"  | TOTAL {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
