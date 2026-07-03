#!/usr/bin/env python3
"""
Build a single self-contained report.html (base64-embedded thumbnails) that shows
every clip grouped by outcome:

  1. Clean person->meme      (a clear 2-3 shot creator->meme cut)
  2. Multi-cut / complex     (>=4 shots: the meme itself is a multi-shot video)
  3. No transition           (single continuous clip, nothing to split)
  4. Problems / needs review (flagged by the independent verification pass)

For split clips the thumbnail shows creator frames (GREEN border) then meme frames
(RED border). For "problem" clips it shows the whole original clip (GREY) so you see
the real content next to the detected-vs-true timestamps.

Usage:
    python build_report.py                 # -> report.html
    python build_report.py --out report.html
"""

import argparse
import base64
import json
import subprocess
import tempfile
from html import escape
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CLIPS_DIR = SCRIPT_DIR / "freckled_spike_tiktok"
SPLIT_DIR = SCRIPT_DIR / "split"
TRANSITIONS = SCRIPT_DIR / "transitions" / "transitions.json"
VERIFICATION = SCRIPT_DIR / "transitions" / "verification.json"


def short(name):
    return name[:-4][-12:]


def dur(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def strip(video, n, color, out, h=260):
    d = dur(video)
    if d <= 0:
        return False
    fps = max(0.25, n / d)
    vf = f"fps={fps},scale=-2:{h},tile={n}x1:padding=3:margin=3:color={color}"
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video),
                        "-vf", vf, "-frames:v", "1", str(out)],
                       stderr=subprocess.DEVNULL)
    return r.returncode == 0 and out.exists()


def hstack(a, b, out):
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(a), "-i", str(b),
                        "-filter_complex", "[0][1]hstack", "-q:v", "4", str(out)],
                       stderr=subprocess.DEVNULL)
    return r.returncode == 0 and out.exists()


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


GREEN = "0x1a7f37"
RED = "0xcf222e"
GREY = "0x57606a"


def make_thumb(clip, section, tmp):
    """Return base64 JPEG for one clip's thumbnail, or None."""
    name = clip["clip"]
    sid = short(name)
    person = SPLIT_DIR / "person" / name
    meme = SPLIT_DIR / "meme" / name
    p_png = tmp / f"{sid}_p.png"
    m_png = tmp / f"{sid}_m.png"
    combo = tmp / f"{sid}.jpg"

    if section == "problem":
        # Show the whole ORIGINAL clip so the true content is visible.
        src = CLIPS_DIR / name
        if strip(src, 8, GREY, combo):
            return b64(combo)
        return None

    has_p = person.exists()
    has_m = meme.exists()
    if has_p and has_m:
        ok_p = strip(person, 3, GREEN, p_png)
        ok_m = strip(meme, 4, RED, m_png)
        if ok_p and ok_m and hstack(p_png, m_png, combo):
            return b64(combo)
        # fall back to whichever worked
        if ok_p:
            return b64(p_png)
        if ok_m:
            return b64(m_png)
        return None
    if has_p:
        if strip(person, 7, GREEN, combo):
            return b64(combo)
    if has_m:
        if strip(meme, 7, RED, combo):
            return b64(combo)
    return None


VERDICT_STYLE = {
    "correct": ("#1a7f37", "correct"),
    "correct_none": ("#1a7f37", "correct (no transition)"),
    "close": ("#9a6700", "close (<1.5s)"),
    "wrong_time": ("#cf222e", "wrong time"),
    "missed": ("#cf222e", "missed"),
    "false_positive": ("#cf222e", "false positive"),
    "unreadable": ("#57606a", "unreadable"),
}


def fmt(t):
    return "none" if t is None else f"{t:.2f}s"


def card_html(clip, ver, thumb_b64):
    name = clip["clip"]
    sid = short(name)
    det = clip.get("transition_sec")
    method = clip.get("method", "")
    nshots = clip.get("num_shots", "?")
    vd = (ver or {}).get("verdict", "")
    vtrue = (ver or {}).get("true_sec")
    vnote = (ver or {}).get("notes", "")
    color, label = VERDICT_STYLE.get(vd, ("#57606a", vd or "—"))

    img = (f'<img class="thumb" src="data:image/jpeg;base64,{thumb_b64}" loading="lazy">'
           if thumb_b64 else '<div class="noimg">no thumbnail</div>')
    return f"""
    <div class="card">
      {img}
      <div class="meta">
        <div class="idrow"><span class="sid">{escape(sid)}</span>
          <span class="badge" style="background:{color}">{escape(label)}</span></div>
        <div class="times">
          <span class="t det">detected <b>{fmt(det)}</b></span>
          <span class="t tru">verified <b>{fmt(vtrue)}</b></span>
          <span class="t sh">{nshots} shots</span>
        </div>
        <div class="method">{escape(method)}</div>
        <div class="note">{escape(vnote)}</div>
      </div>
    </div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCRIPT_DIR / "report.html"))
    ap.add_argument("--transitions", default=str(TRANSITIONS))
    ap.add_argument("--verification", default=str(VERIFICATION))
    args = ap.parse_args()

    records = json.load(open(args.transitions))
    try:
        ver_by = json.load(open(args.verification)).get("by_clip", {})
        ver_summary = json.load(open(args.verification)).get("summary", {})
    except Exception:
        ver_by, ver_summary = {}, {}

    # Assign each clip to exactly one section (priority order).
    sections = {"clean": [], "multicut": [], "none": [], "problem": []}
    for x in records:
        v = ver_by.get(short(x["clip"]), {})
        vd = v.get("verdict")
        if vd in ("wrong_time", "missed", "false_positive"):
            sections["problem"].append(x)
        elif x.get("transition_sec") is None:
            sections["none"].append(x)
        elif (x.get("num_shots") or 0) >= 4:
            sections["multicut"].append(x)
        else:
            sections["clean"].append(x)

    # sort each section by detected time (None last)
    for k in sections:
        sections[k].sort(key=lambda x: (x.get("transition_sec") is None,
                                        x.get("transition_sec") or 0))

    SECTION_META = [
        ("clean", "✅ Clean person → meme", "One clear cut from the creator to the meme (2–3 shots)."),
        ("multicut", "🔀 Multi-cut / complex", "≥4 shots — the meme itself is a multi-shot video, or there are extra cuts."),
        ("none", "⏹ No transition", "A single continuous clip — nothing to split (correctly reported as none)."),
        ("problem", "⚠️ Problems / needs review", "Flagged by the independent verification pass — detected time is wrong, missed, or a false split."),
    ]

    tmp = Path(tempfile.mkdtemp(prefix="pezreport_"))
    body = []
    for key, title, desc in SECTION_META:
        clips = sections[key]
        cards = []
        for i, clip in enumerate(clips, 1):
            thumb = make_thumb(clip, key, tmp)
            cards.append(card_html(clip, ver_by.get(short(clip["clip"])), thumb))
        body.append(f"""
        <section>
          <h2>{title} <span class="count">{len(clips)}</span></h2>
          <p class="desc">{desc}</p>
          <div class="grid">{''.join(cards)}</div>
        </section>""")
        print(f"  {title}: {len(clips)} clips rendered")

    bv = ver_summary.get("byVerdict", {})
    strict = ver_summary.get("strict_correct_pct", "?")
    within = ver_summary.get("within_1p5s_pct", "?")
    summary_html = f"""
      <div class="summary">
        <div class="stat big"><b>{strict}%</b><span>exactly correct</span></div>
        <div class="stat big"><b>{within}%</b><span>within 1.5s</span></div>
        <div class="stat"><b>{len(records)}</b><span>clips</span></div>
        <div class="stat"><b>{bv.get('correct',0)+bv.get('correct_none',0)}</b><span>correct</span></div>
        <div class="stat"><b>{bv.get('close',0)}</b><span>close</span></div>
        <div class="stat warn"><b>{bv.get('wrong_time',0)+bv.get('missed',0)+bv.get('false_positive',0)}</b><span>problems</span></div>
      </div>"""

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pezevenk — person→meme transitions</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#0d1117; color:#e6edf3; font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }}
  header {{ padding:24px 28px 8px; border-bottom:1px solid #21262d; }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  .lede {{ color:#8b949e; margin:0 0 16px; max-width:760px; }}
  .summary {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:8px; }}
  .stat {{ background:#161b22; border:1px solid #21262d; border-radius:10px; padding:10px 16px; text-align:center; min-width:70px; }}
  .stat b {{ display:block; font-size:20px; }}
  .stat.big b {{ font-size:26px; color:#3fb950; }}
  .stat.warn b {{ color:#f85149; }}
  .stat span {{ color:#8b949e; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  .legend {{ color:#8b949e; font-size:12px; margin:10px 0 0; }}
  .legend .sw {{ display:inline-block; width:11px; height:11px; border-radius:2px; vertical-align:middle; margin:0 3px 0 12px; }}
  section {{ padding:20px 28px; border-bottom:1px solid #21262d; }}
  h2 {{ font-size:17px; margin:0 0 2px; }}
  h2 .count {{ background:#21262d; border-radius:20px; padding:1px 10px; font-size:13px; color:#8b949e; margin-left:6px; }}
  .desc {{ color:#8b949e; margin:0 0 14px; font-size:12.5px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:14px; }}
  .card {{ background:#161b22; border:1px solid #21262d; border-radius:10px; overflow:hidden; }}
  .card img {{ display:block; width:100%; background:#000; cursor:zoom-in; }}
  #lb {{ position:fixed; inset:0; background:#000000f2; z-index:100; display:none; overflow:auto; }}
  #lb.on {{ display:block; }}
  #lbstage {{ min-width:100%; min-height:100%; display:flex; align-items:center; justify-content:center; box-sizing:border-box; padding:60px 24px 24px; }}
  #lbimg {{ --z:41vh; height:var(--z); width:auto; max-width:none; border-radius:6px; }}
  #lbbar {{ position:fixed; top:10px; left:50%; transform:translateX(-50%); z-index:101; display:flex; gap:6px; align-items:center;
           background:#161b22ee; border:1px solid #30363d; border-radius:22px; padding:5px 8px; }}
  #lbbar button {{ width:30px; height:30px; border-radius:50%; border:1px solid #30363d; background:#21262d; color:#e6edf3; font-size:16px; line-height:1; cursor:pointer; }}
  #lbbar button:hover {{ border-color:#8b949e; }}
  #lbbar .zl {{ min-width:46px; text-align:center; color:#8b949e; font-size:12px; }}
  .card .noimg {{ padding:30px; text-align:center; color:#8b949e; background:#010409; }}
  .meta {{ padding:9px 11px; }}
  .idrow {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
  .sid {{ font-family:ui-monospace,monospace; font-size:12px; color:#8b949e; }}
  .badge {{ color:#fff; font-size:11px; padding:2px 8px; border-radius:20px; white-space:nowrap; }}
  .times {{ display:flex; gap:6px; flex-wrap:wrap; margin:7px 0 4px; }}
  .t {{ font-size:11px; padding:2px 7px; border-radius:6px; background:#21262d; }}
  .t.det {{ border:1px solid #388bfd55; }}
  .t.tru {{ border:1px solid #3fb95055; }}
  .t b {{ color:#e6edf3; }}
  .method {{ font-family:ui-monospace,monospace; font-size:10.5px; color:#6e7681; }}
  .note {{ font-size:12px; color:#adbac7; margin-top:5px; }}
  footer {{ padding:18px 28px; color:#6e7681; font-size:12px; }}
</style></head><body>
<header>
  <h1>pezevenk — person → meme transition detection</h1>
  <p class="lede">Each TikTok clip cuts from the creator talking to camera over to a meme. Detector = TransNetV2 (shot boundaries) + InsightFace (is this shot <em>the creator's</em> face?). Accuracy below is from an independent verification pass (6 ground-truth entries corrected after visual re-check).</p>
  {summary_html}
  <p class="legend">Thumbnails:
    <span class="sw" style="background:#1a7f37"></span>creator part
    <span class="sw" style="background:#cf222e"></span>meme part
    <span class="sw" style="background:#57606a"></span>whole clip (problem cases) &nbsp;•&nbsp; frames shown left→right in time order.</p>
</header>
{''.join(body)}
<footer>Generated by build_report.py — detected times from transitions.json, verified times from a blind multi-agent verification pass.</footer>
<div id="lb">
  <div id="lbbar">
    <button onclick="zoomBy(-1,event)" title="zoom out (-)">&minus;</button>
    <span class="zl" id="lbzl">100%</span>
    <button onclick="zoomBy(1,event)" title="zoom in (+)">+</button>
    <button onclick="lbReset(event)" title="reset size">&#8862;</button>
    <button onclick="lbClose()" title="close (Esc)">&times;</button>
  </div>
  <div id="lbstage"><img id="lbimg" alt=""></div>
</div>
<script>
  const lb=document.getElementById('lb'), lbimg=document.getElementById('lbimg'),
        lbzl=document.getElementById('lbzl'), lbstage=document.getElementById('lbstage');
  const BASE=41; let p=100;  // 100% == the comfortable default size; +/- steps by 25%
  function setZ(){{ lbimg.style.setProperty('--z', (p/100*BASE)+'vh'); lbzl.textContent=p+'%'; }}
  function zoomBy(d,e){{ if(e) e.stopPropagation(); p=Math.min(600, Math.max(40, p+d*25)); setZ(); }}
  function lbReset(e){{ if(e) e.stopPropagation(); p=100; setZ(); }}
  function lbClose(){{ lb.classList.remove('on'); }}
  function lbOpen(src){{ lbimg.src=src; p=100; setZ(); lb.classList.add('on'); }}
  lb.addEventListener('click', e=>{{ if(e.target===lb || e.target===lbstage) lbClose(); }});
  lbimg.addEventListener('click', e=>e.stopPropagation());
  lb.addEventListener('wheel', e=>{{ if(e.ctrlKey){{ e.preventDefault(); zoomBy(e.deltaY<0?1:-1); }} }}, {{passive:false}});
  document.querySelectorAll('.card img.thumb').forEach(im=> im.addEventListener('click', ()=>lbOpen(im.src)));
  document.addEventListener('keydown', e=>{{ if(!lb.classList.contains('on')) return;
    if(e.key==='Escape') lbClose(); else if(e.key==='+'||e.key==='=') zoomBy(1); else if(e.key==='-') zoomBy(-1); }});
  setZ();
</script>
</body></html>"""

    Path(args.out).write_text(html)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nWrote {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
