#!/usr/bin/env python3
"""
One-off migration: ground_truth_batu.json single `cut_sec` -> `cuts` array.

Old per-clip shape had one scalar cut time:
    {"has_transition": true, "cut_sec": 3.742, "pattern": "person->meme", ...}

New shape adds an ordered list of cuts (and keeps cut_sec = cuts[0].sec so the
workbench / evaluate.py keep working unchanged):
    {"has_transition": true, "cut_sec": 3.742,
     "cuts": [{"sec": 3.742, "to": "meme"}], ...}

`to` = what the clip cuts TO at that moment, alternating from the creator:
cut 0 -> meme, cut 1 -> person, cut 2 -> meme, ...

Idempotent: clips that already have `cuts` are left alone. A backup is written
next to the file before anything is overwritten.

    python attic/migrate_batu_cuts.py
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # repo root (this file lives in attic/)
GT = HERE / "transitions" / "ground_truth_batu.json"
BAK = HERE / "transitions" / "ground_truth_batu.json.bak"

# how many cuts each pattern implies (None = unknown / free-form)
PATTERN_CUTS = {
    "person->meme": 1,
    "person->meme->person": 2,
    "person->meme->person->meme": 3,
    "all person": 0,
    "all meme": 0,
    "": None,
}


def cuts_from(clip):
    """Build the cuts array from the legacy single cut_sec."""
    if clip.get("has_transition") is False:
        return []
    cs = clip.get("cut_sec")
    if cs is None:
        return []
    return [{"sec": round(float(cs), 3), "to": "meme"}]


def main():
    doc = json.loads(GT.read_text())
    clips = doc.get("clips", [])

    if not BAK.exists():
        shutil.copy2(GT, BAK)
        print(f"backup -> {BAK.relative_to(HERE)}")
    else:
        print(f"backup already exists ({BAK.relative_to(HERE)}) — left as-is")

    migrated, already = 0, 0
    needs_attention = []
    for c in clips:
        if isinstance(c.get("cuts"), list):
            already += 1
        else:
            cuts = cuts_from(c)
            c["cuts"] = cuts
            c["cut_sec"] = cuts[0]["sec"] if cuts else None
            migrated += 1

        # flag clips whose pattern implies more cuts than we could recover
        want = PATTERN_CUTS.get(c.get("pattern", ""))
        if want is not None and len(c.get("cuts", [])) < want:
            needs_attention.append((c["short"], c.get("pattern"), len(c["cuts"]), want))

    doc["precision_note"] = (
        "cut times entered by hand to 3-decimal (ms) precision via editor.html; "
        "multi-cut clips stored in `cuts` (cut_sec mirrors cuts[0])"
    )
    GT.write_text(json.dumps(doc, indent=2))

    print(f"migrated {migrated} clip(s), {already} already had cuts, {len(clips)} total")
    if needs_attention:
        print(f"\n⚠  {len(needs_attention)} clip(s) whose pattern implies MORE cuts than "
              "were recorded (only the first cut existed in the old schema).")
        print("   Re-open these in the editor and add the missing cut(s):")
        for short, pat, have, want in needs_attention:
            print(f"     {short}  pattern={pat!r}  has {have} cut(s), needs {want}")


if __name__ == "__main__":
    main()
