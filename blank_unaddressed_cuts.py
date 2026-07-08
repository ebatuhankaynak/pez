#!/usr/bin/env python3
"""
Clear the inherited (claude-seeded) cut on UNADDRESSED clips in
ground_truth_batu.json, so the editor presents a blank canvas for clips you
haven't labeled yet instead of a pre-placed person->meme cut.

Only clips with `edited: false` are touched. Your real labels (`edited: true`)
are left exactly as-is. For each unaddressed clip we clear the *cut* fields
(cuts, cut_sec, has_transition, pattern, returns_to_creator) but KEEP notes and
confidence — claude's description stays as a reference hint, and claude's cut
still shows as the grey reference marker in the editor.

A fresh backup (with all your current labels) is written before anything is
overwritten. Idempotent: re-running only affects clips that still carry a cut.

    python blank_unaddressed_cuts.py
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
GT = HERE / "transitions" / "ground_truth_batu.json"
BAK = HERE / "transitions" / "ground_truth_batu.json.bak2"   # pre-blank snapshot


def main():
    doc = json.loads(GT.read_text())
    clips = doc.get("clips", [])

    if not BAK.exists():
        shutil.copy2(GT, BAK)
        print(f"backup -> {BAK.relative_to(HERE)} (current labels, pre-blank)")
    else:
        print(f"backup already exists ({BAK.relative_to(HERE)}) — left as-is")

    edited = sum(1 for c in clips if c.get("edited"))
    cleared = 0
    for c in clips:
        if c.get("edited"):
            continue                      # your work — never touch
        if c.get("cuts") or c.get("cut_sec") is not None or c.get("has_transition") \
           or c.get("pattern") or c.get("returns_to_creator"):
            cleared += 1
        c["cuts"] = []
        c["cut_sec"] = None
        c["has_transition"] = False
        c["pattern"] = ""
        c["returns_to_creator"] = False
        # notes + confidence kept as a claude reference hint

    GT.write_text(json.dumps(doc, indent=2))
    print(f"kept {edited} edited clip(s) untouched; "
          f"blanked the cut on {cleared} unaddressed clip(s); {len(clips)} total")


if __name__ == "__main__":
    main()
