"""Write the pipekit-evaluate dependency figure as light and dark SVGs.

The layout is pipekit-array's (``pipekit_array_ecosystem/``): the package at the
top, the packages that import it below, and what it builds on underneath. No
package imports pipekit-evaluate on its ``main``: a grep of every repository of
the stack for ``from pipekit_evaluate`` and ``import pipekit_evaluate`` outside
tests finds only geostack's report notebooks 14, 15 and 17, which are not
libraries. So the figure draws the empty state in place of the layers. geostack's
``docs/api/dependency_graph.md`` has no edge into pipekit-evaluate either. What it
builds on is its own ``import`` lines on ``main``: ``pipekit_ops.saving`` at
module level, ``pipekit_cycle.protocols`` under ``TYPE_CHECKING``, and nothing
from NumPy or SciPy, which ``pyproject.toml`` declares. ``_icon.svg`` is
pipekit's ``docs/assets/packages/pipekit-evaluate/icon.svg``, copied unchanged.
Re-run after editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pipekit_evaluate_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the pipekit-evaluate dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

TAGLINE = "tagged scores, evaluation reports and the benchmark ladder"
TIER = "Core · tier 0"
EMPTY = "no package imports pipekit-evaluate on main yet"
EMPTY_NOTE = (
    "geostack's report notebooks 14, 15 and 17 import its benchmark and scoring "
    "areas; they are not libraries"
)
# The "built on" strip: (lead, bold name, rest of the line), one per line.
BASE = [
    ("pipekit-evaluate builds on ", "pipekit-ops", " (its content hash)"),
    ("and ", "pipekit-cycle", " through its cycle extra, for type checking only"),
    (
        "declared but imported by no module: ",
        "NumPy · SciPy",
        ", and xskillscore and properscoring behind two extras",
    ),
]

THEMES = {
    "light": {
        "accent": "#BE123C",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#9CA3AF"),
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#FB7185",
        "card": "#111827",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "box": ("#0F172A", "#4B5563"),
        "base": ("#1F2937", "#374151"),
    },
}

WIDTH = 960
BOX_X, BOX_W, EMPTY_Y, EMPTY_H = 20, 920, 120, 78


def icon(x: float, y: float, theme: str) -> str:
    """pipekit-evaluate's icon at 48 px, with its gradient id made unique per theme."""
    src = (HERE / "_icon.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    body = body.replace("pipekitevaluate-i", f"pke-{theme}").strip()
    return (
        f'<svg x="{x}" y="{y}" width="48" height="48" viewBox="0 0 128 128">'
        f"{body}</svg>"
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    fill, stroke = t["box"]
    out = [
        f'<rect x="235" y="18" width="490" height="74" rx="14" fill="{t["card"]}" '
        f'stroke="{t["accent"]}" stroke-width="2"/>',
        icon(251, 31, theme),
        f'<text x="313" y="52" font-family="{SANS}" font-size="22" '
        f'font-weight="700" fill="{t["ink"]}">pipekit-evaluate</text>',
        f'<text x="313" y="74" font-family="{SANS}" font-size="13" '
        f'fill="{t["muted"]}">{escape(TAGLINE)}</text>',
        f'<text x="709" y="52" text-anchor="end" font-family="{SANS}" '
        f'font-size="12" fill="{t["muted"]}">{escape(TIER)}</text>',
        # the empty state, where the layers of importing packages would be
        f'<rect x="{BOX_X}" y="{EMPTY_Y}" width="{BOX_W}" height="{EMPTY_H}" rx="14" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" '
        'stroke-dasharray="3 5"/>',
        f'<text x="480" y="{EMPTY_Y + 33}" text-anchor="middle" '
        f'font-family="{SANS}" font-size="15" font-weight="700" '
        f'fill="{t["ink"]}">{escape(EMPTY)}</text>',
        f'<text x="480" y="{EMPTY_Y + 56}" text-anchor="middle" '
        f'font-family="{SANS}" font-size="12.5" fill="{t["muted"]}">'
        f"{escape(EMPTY_NOTE)}</text>",
    ]
    # what pipekit-evaluate builds on
    by = EMPTY_Y + EMPTY_H + 22
    bfill, bstroke = t["base"]
    base_h = 22 + 22 * len(BASE)
    out.append(
        f'<rect x="{BOX_X}" y="{by}" width="{BOX_W}" height="{base_h}" rx="10" '
        f'fill="{bfill}" stroke="{bstroke}"/>'
    )
    for i, (lead, name, rest) in enumerate(BASE):
        size = 14 if i == 0 else 13
        out.append(
            f'<text x="480" y="{by + 27 + 22 * i}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="{size}" fill="{t["muted"]}">'
            f'{escape(lead)}<tspan font-weight="700" fill="{t["ink"]}">'
            f"{escape(name)}</tspan>{escape(rest)}</text>"
        )
    height = by + base_h + 20
    label = (
        "No package imports pipekit-evaluate on its main branch yet; geostack's "
        "report notebooks 14, 15 and 17 do, and they are not libraries. "
        "pipekit-evaluate builds on pipekit-ops, and on pipekit-cycle through its "
        "cycle extra for type checking only; NumPy and SciPy are declared but "
        "imported by no module."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>pipekit-evaluate and the packages that import it</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``pipekit_evaluate_ecosystem-light.svg`` and ``-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"pipekit_evaluate_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
