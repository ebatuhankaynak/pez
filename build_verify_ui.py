#!/usr/bin/env python3
"""
Build verify.html — a local verification UI. One row per clip, side by side:

  [ shot timeline + meta + ok/wrong buttons ] [ transition.png ] [ PERSON video ] [ MEME video ]

Play the two halves next to the cut frame to spot "meme seeped into the person
half" (usually the wrong cut was picked). Flags are saved in your browser
(localStorage); "Export wrong" downloads the list so you can fix those clips.

The HTML references the real files by relative path, so open it FROM this folder.
Because some browsers block file:// video, the reliable way is a tiny local server:

    python3 -m http.server 8000        # then open http://localhost:8000/verify.html

Usage:
    python build_verify_ui.py            # -> verify.html
"""

import argparse
import json
from html import escape
from pathlib import Path
from urllib.parse import quote

SCRIPT_DIR = Path(__file__).resolve().parent
TRANSITIONS = SCRIPT_DIR / "transitions" / "transitions.json"
VERIFICATION = SCRIPT_DIR / "transitions" / "verification.json"


def short(name):
    return name[:-4][-12:]


VERDICT_COLOR = {
    "correct": "#1a7f37", "correct_none": "#1a7f37", "close": "#9a6700",
    "wrong_time": "#cf222e", "missed": "#cf222e", "false_positive": "#cf222e",
    "unreadable": "#57606a",
}


def category(clip, ver):
    vd = (ver or {}).get("verdict")
    if vd in ("wrong_time", "missed", "false_positive"):
        return "problem"
    if clip.get("transition_sec") is None:
        return "none"
    if (clip.get("num_shots") or 0) >= 4:
        return "multicut"
    return "clean"


CAT_ORDER = {"problem": 0, "multicut": 1, "clean": 2, "none": 3}
CAT_LABEL = {"problem": "problem", "multicut": "multi-cut", "clean": "clean", "none": "no-transition"}


def fmt(t):
    return "—" if t is None else f"{t:.2f}s"


def timeline_html(clip):
    shots = clip.get("shots", [])
    if not shots:
        return ""
    dur = max((s["end_sec"] for s in shots), default=0) or 1
    trans = clip.get("transition_sec")
    segs = []
    for s in shots:
        w = max(0.5, (s["end_sec"] - s["start_sec"]) / dur * 100)
        col = "#1a7f37" if s.get("label") == "person" else "#4184e4"
        tip = f'{s["start_sec"]:.2f}-{s["end_sec"]:.2f}s  {s.get("label")}  face_sim={s.get("face_sim")}'
        segs.append(f'<div class="seg" style="width:{w}%;background:{col}" title="{escape(tip)}"></div>')
    marker = ""
    if trans is not None:
        marker = f'<div class="mark" style="left:{trans/dur*100:.2f}%" title="picked cut {trans:.2f}s"></div>'
    return f'<div class="tl">{"".join(segs)}{marker}</div><div class="tlabel">0s ← {len(shots)} shots → {dur:.1f}s</div>'


def video_html(rel_path, exists, label, sublabel, klass):
    if not exists:
        return f'<div class="vid empty"><div class="vlabel {klass}">{label}</div><div class="none">(none)</div></div>'
    src = quote(rel_path)
    return (f'<div class="vid"><div class="vlabel {klass}">{label} '
            f'<span class="sub">{sublabel}</span></div>'
            f'<video controls preload="metadata" playsinline src="{src}"></video></div>')


def row_html(clip, ver):
    name = clip["clip"]
    sid = short(name)
    det = clip.get("transition_sec")
    method = clip.get("method", "")
    nshots = clip.get("num_shots", "?")
    cat = category(clip, ver)
    vd = (ver or {}).get("verdict", "")
    vtrue = (ver or {}).get("true_sec")
    vnote = (ver or {}).get("notes", "")
    vcolor = VERDICT_COLOR.get(vd, "#57606a")

    qa_rel = f"transitions/qa/{sid}_transition.png"
    qa_exists = (SCRIPT_DIR / qa_rel).exists()
    person_rel = f"split/person/{name}"
    meme_rel = f"split/meme/{name}"
    person_exists = (SCRIPT_DIR / person_rel).exists()
    meme_exists = (SCRIPT_DIR / meme_rel).exists()

    qa_html = (f'<a class="qa" href="{quote(qa_rel)}" target="_blank">'
               f'<img src="{quote(qa_rel)}" loading="lazy">'
               f'<span class="qcap">before | after cut</span></a>'
               if qa_exists else '<div class="qa empty">no cut frame</div>')

    person_dur = f"[0–{det:.2f}s]" if det is not None else "[full]"
    meme_dur = f"[{det:.2f}s–end]" if det is not None else "[full]"

    return f"""
    <div class="row" data-cat="{cat}" data-clip="{escape(name)}">
      <div class="meta">
        <div class="idline"><span class="sid">{escape(sid)}</span>
          <span class="catb cat-{cat}">{CAT_LABEL[cat]}</span></div>
        {timeline_html(clip)}
        <div class="times">
          <span class="t det">detected <b>{fmt(det)}</b></span>
          <span class="t sh">{nshots} shots</span>
          <span class="t vb" style="border-color:{vcolor}" title="Independent reviewer's true cut = {fmt(vtrue)}. Verdict '{escape(vd)}' = how the detected time compared.">verified <b>{fmt(vtrue)}</b> {escape(vd)}</span>
        </div>
        <div class="method">{escape(method)}</div>
        <div class="note">{escape(vnote)}</div>
        <div class="review">
          <button class="ok" onclick="mark(this,'ok')">✓ looks right</button>
          <button class="bad" onclick="mark(this,'wrong')">✗ wrong split</button>
        </div>
      </div>
      {qa_html}
      {video_html(f"freckled_spike_tiktok/{name}", (SCRIPT_DIR / "freckled_spike_tiktok" / name).exists(), "ORIGINAL", "[full]", "o")}
      {video_html(person_rel, person_exists, "PERSON", person_dur, "p")}
      {video_html(meme_rel, meme_exists, "MEME", meme_dur, "m")}
    </div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCRIPT_DIR / "verify.html"))
    ap.add_argument("--transitions", default=str(TRANSITIONS))
    ap.add_argument("--verification", default=str(VERIFICATION))
    args = ap.parse_args()

    records = json.load(open(args.transitions))
    try:
        ver_by = json.load(open(args.verification)).get("by_clip", {})
    except Exception:
        ver_by = {}

    rows = sorted(records, key=lambda c: (CAT_ORDER[category(c, ver_by.get(short(c["clip"])))],
                                          (c.get("transition_sec") is None),
                                          c.get("transition_sec") or 0))
    rows_html = "".join(row_html(c, ver_by.get(short(c["clip"]))) for c in rows)

    from collections import Counter
    cats = Counter(category(c, ver_by.get(short(c["clip"]))) for c in records)

    verdicts = Counter((ver_by.get(short(c["clip"])) or {}).get("verdict") for c in records)
    n = len(records)
    good = verdicts["correct"] + verdicts["correct_none"]
    review = verdicts["wrong_time"] + verdicts["missed"] + verdicts["false_positive"]
    banner_html = (
        f'<b>{good}/{n} match ground truth ({round(good/n*100)}%)</b> · '
        f'{verdicts["close"]} within 1.5s · <span class="warn">{review} to review</span> &nbsp;—&nbsp; '
        'green = creator, blue = meme (both are normal; a clip has both halves). '
        'only the <span class="warn">problems</span> filter needs attention.'
    )

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>verify — person/meme splits</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#0d1117; color:#e6edf3; font:13px/1.45 -apple-system,Segoe UI,Roboto,sans-serif; }}
  .bar {{ position:sticky; top:0; z-index:10; background:#010409ee; backdrop-filter:blur(6px);
         border-bottom:1px solid #21262d; padding:10px 16px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
  .bar h1 {{ font-size:15px; margin:0 10px 0 0; }}
  .bar button {{ background:#21262d; color:#e6edf3; border:1px solid #30363d; border-radius:16px;
                padding:4px 12px; cursor:pointer; font-size:12px; }}
  .bar button.active {{ background:#1f6feb; border-color:#1f6feb; }}
  .bar .prog {{ margin-left:auto; color:#8b949e; }}
  .bar .exp {{ background:#238636; border-color:#238636; }}
  .row {{ display:flex; gap:12px; align-items:flex-start; padding:12px 16px;
         border-bottom:1px solid #21262d; }}
  .row.st-ok {{ background:#0e2a16; }}
  .row.st-wrong {{ background:#2a0e12; }}
  .meta {{ width:280px; flex:0 0 280px; }}
  .idline {{ display:flex; justify-content:space-between; align-items:center; }}
  .sid {{ font-family:ui-monospace,monospace; color:#8b949e; }}
  .catb {{ font-size:10.5px; padding:1px 8px; border-radius:12px; }}
  .cat-problem {{ background:#cf222e; }} .cat-multicut {{ background:#8957e5; }}
  .cat-clean {{ background:#1a7f37; }} .cat-none {{ background:#57606a; }}
  .tl {{ position:relative; display:flex; height:16px; border-radius:4px; overflow:hidden; margin:8px 0 2px; background:#161b22; }}
  .tl .seg {{ height:100%; border-right:1px solid #0d1117; }}
  .tl .mark {{ position:absolute; top:-2px; width:2px; height:20px; background:#f0f6fc; box-shadow:0 0 3px #000; }}
  .tlabel {{ font-size:10px; color:#6e7681; }}
  .times {{ display:flex; gap:5px; flex-wrap:wrap; margin:6px 0; }}
  .t {{ font-size:10.5px; padding:2px 6px; border-radius:5px; background:#21262d; }}
  .t.vb {{ border:1px solid #57606a; }}
  .method {{ font-family:ui-monospace,monospace; font-size:10px; color:#6e7681; }}
  .note {{ font-size:11.5px; color:#adbac7; margin:4px 0; }}
  .review button {{ margin-right:6px; margin-top:4px; border:1px solid #30363d; border-radius:6px;
                   background:#161b22; color:#e6edf3; padding:3px 8px; cursor:pointer; font-size:11.5px; }}
  .review button.ok:hover {{ border-color:#238636; }} .review button.bad:hover {{ border-color:#cf222e; }}
  .qa {{ flex:0 0 auto; text-decoration:none; }}
  .qa img {{ height:230px; border-radius:6px; display:block; background:#000; }}
  .qa .qcap {{ font-size:10px; color:#8b949e; }}
  .qa.empty, .vid.empty .none {{ color:#6e7681; font-size:11px; padding:20px; }}
  .vid {{ flex:0 0 auto; }}
  .vid video {{ height:230px; border-radius:6px; display:block; background:#000; }}
  .vlabel {{ font-size:11px; font-weight:600; margin-bottom:3px; }}
  .vlabel.p {{ color:#3fb950; }} .vlabel.m {{ color:#6cb6ff; }} .vlabel.o {{ color:#c9d1d9; }}
  .banner {{ padding:9px 16px; background:#0e2a16; border-bottom:1px solid #1a7f37; color:#adbac7; font-size:13px; }}
  .banner b {{ color:#3fb950; }} .banner .warn {{ color:#e3b341; }}
  .vlabel .sub {{ color:#8b949e; font-weight:400; }}
  .hidden {{ display:none; }}
</style></head><body>
<nav id="peznav" style="position:sticky;top:0;z-index:50;display:flex;gap:4px;align-items:center;background:#010409ee;backdrop-filter:blur(6px);border-bottom:1px solid #21262d;padding:8px 16px;font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif"><span style="font-weight:600;margin-right:10px;color:#e6edf3">pezevenk</span><a href="app.html" style="padding:4px 12px;border-radius:13px;text-decoration:none;color:#adbac7">workbench</a><a href="editor.html" style="padding:4px 12px;border-radius:13px;text-decoration:none;color:#adbac7">cut editor</a><a href="report.html" style="padding:4px 12px;border-radius:13px;text-decoration:none;color:#adbac7">report</a><a href="verify.html" style="padding:4px 12px;border-radius:13px;text-decoration:none;background:#1f6feb;color:#fff">verify</a></nav>
<div class="bar">
  <h1>verify splits</h1>
  <button data-f="all" class="active" onclick="filt(this,'all')">all {len(records)}</button>
  <button data-f="problem" onclick="filt(this,'problem')">problems {cats['problem']}</button>
  <button data-f="multicut" onclick="filt(this,'multicut')">multi-cut {cats['multicut']}</button>
  <button data-f="clean" onclick="filt(this,'clean')">clean {cats['clean']}</button>
  <button data-f="none" onclick="filt(this,'none')">no-transition {cats['none']}</button>
  <button data-f="wrong" onclick="filt(this,'wrong')">flagged ✗</button>
  <span class="prog" id="prog"></span>
  <button class="exp" onclick="exportWrong()">Export wrong</button>
</div>
<div class="banner">{banner_html}</div>
<div id="rows">{rows_html}</div>
<script>
const KEY='pez_review_v1';
function load(){{ try{{return JSON.parse(localStorage.getItem(KEY))||{{}}}}catch(e){{return {{}}}} }}
function save(s){{ localStorage.setItem(KEY, JSON.stringify(s)); }}
function apply(){{
  const s=load();
  document.querySelectorAll('.row').forEach(r=>{{
    const st=s[r.dataset.clip];
    r.classList.remove('st-ok','st-wrong');
    if(st==='ok') r.classList.add('st-ok');
    if(st==='wrong') r.classList.add('st-wrong');
  }});
  const vals=Object.values(s);
  const wrong=vals.filter(v=>v==='wrong').length, ok=vals.filter(v=>v==='ok').length;
  document.getElementById('prog').textContent=`reviewed ${{vals.length}}/${{document.querySelectorAll('.row').length}} · ✓${{ok}} ✗${{wrong}}`;
}}
function mark(btn,state){{
  const row=btn.closest('.row'); const s=load();
  if(s[row.dataset.clip]===state){{ delete s[row.dataset.clip]; }} else {{ s[row.dataset.clip]=state; }}
  save(s); apply();
}}
function filt(btn,f){{
  document.querySelectorAll('.bar button[data-f]').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  const s=load();
  document.querySelectorAll('.row').forEach(r=>{{
    let show = (f==='all') || (f==='wrong' ? s[r.dataset.clip]==='wrong' : r.dataset.cat===f);
    r.classList.toggle('hidden', !show);
  }});
}}
function exportWrong(){{
  const s=load(); const wrong=Object.keys(s).filter(k=>s[k]==='wrong');
  const blob=new Blob([JSON.stringify(wrong,null,2)],{{type:'application/json'}});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='wrong_splits.json'; a.click();
}}
apply();
</script>
</body></html>"""

    Path(args.out).write_text(html)
    print(f"Wrote {args.out} ({Path(args.out).stat().st_size/1024:.0f} KB, {len(records)} rows)")
    print("Open via a local server for video playback:")
    print("  cd", SCRIPT_DIR, "&& python3 -m http.server 8000  ->  http://localhost:8000/verify.html")


if __name__ == "__main__":
    main()
