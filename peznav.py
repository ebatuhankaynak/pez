"""Shared top-nav for the generated report.html.

The canonical styles live in peznav.css (also linked by editor.html + app.html).
The generators inline css() + nav() so the produced HTML stays self-contained
while still matching the rest of the app byte-for-byte.
"""

from pathlib import Path

_CSS = Path(__file__).resolve().parent / "peznav.css"

# order + labels must match editor.html / app.html
LINKS = [
    ("app.html", "workbench"),
    ("editor.html", "cut editor"),
    ("report.html", "report"),
    ("memes.html", "text removal"),
]


def css():
    """peznav.css contents, to drop inside a <style> block."""
    return _CSS.read_text() if _CSS.exists() else ""


def nav(active, sticky=True):
    """<nav> markup with `active` (e.g. 'report.html') highlighted.

    sticky=True makes the nav its own sticky header (report). Pass sticky=False
    when the nav is wrapped in a .pezhdr alongside a toolbar."""
    parts = []
    for href, label in LINKS:
        cls = ' class="on"' if href == active else ""
        parts.append(f'<a href="{href}"{cls}>{label}</a>')
    wrap = " sticky" if sticky else ""
    return (f'<nav class="peznav{wrap}"><span class="brand">pezevenk</span>'
            f'{"".join(parts)}</nav>')
