"""Write the pipekit-experiment dependency figure as light and dark SVGs.

The layout is pipekit-array's (``pipekit_array_ecosystem/``): the package at
the top, the packages that import it below in their stacked layers, and what it
builds on underneath. The edges and the modules each package uses are its
``pipekit_experiment`` imports on ``main``, outside its tests. One package
imports it: pipekit-train, through its ``experiment`` extra, with
``import_module("pipekit_experiment.artifacts")`` in
``TrainingLoop._build_artifact``, which ``run`` calls,
so its card and the arrow are dashed (as in ``vardax_ecosystem/``, where every
import is through an extra). pipekit-jax and the ``pipekit`` umbrella import
nothing from it, so they get a note and no card. No package outside the
workspace imports it: vardax lists it in its ``persist`` extra and imports
nothing, so it is left out. geostack's ``docs/api/dependency_graph.md`` folds
pipekit's packages into one node and has no edge for this package.
``_icon.svg`` is pipekit's ``docs/assets/packages/pipekit-experiment/icon.svg``,
copied unchanged, and the accent is its slate. Re-run after editing the data
below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pipekit_experiment_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the pipekit-experiment dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

TAGLINE = "a content-addressed model registry and run records"
TIER = "Core · tier 0"
# (package, what it is, the modules it imports, its extra)
WORKSPACE = [("pipekit-train", "training loops", "artifacts", "experiment")]
# What the rest of the workspace does with it without importing it.
WORKSPACE_NOTE = (
    "pipekit-jax imports nothing from it: JaxModelOp.from_registry reads weights "
    "from any ModelRegistry by shape. pipekit-train's Checkpoint and "
    "LogToExperiment take any ModelRegistry or ExperimentTracker the same way, "
    "and the pipekit umbrella installs it without importing it."
)
OUTSIDE = "no package outside the pipekit workspace imports pipekit-experiment on main"
BASE = "pipekit-ops"
BASE_EXTRAS = (
    "and, through its extras, fsspec and s3fs, DVC, Hydra and hydra-zen, and Metaflow"
)

THEMES = {
    "light": {
        "accent": "#334155",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "grey": "#9CA3AF",
        "uses": "#334155",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#94A3B8",
        "card": "#111827",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "box": ("#0F172A", "#1F2937"),
        "grey": "#6B7280",
        "uses": "#CBD5E1",
        "base": ("#1F2937", "#374151"),
    },
}

WIDTH = 960
BOX_X, BOX_W, TOP_Y = 20, 920, 140
PER_ROW, CARD_GAP, CARD_H, LAYER_GAP = 4, 16, 92, 16


def wrap(text: str, width: int) -> list[str]:
    """Split ``text`` into lines of at most about ``width`` characters."""
    line, lines = "", []
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line]


def icon(x: float, y: float, theme: str) -> str:
    """pipekit-experiment's icon at 48 px, its gradient id made unique per theme."""
    src = (HERE / "_icon.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    body = body.replace("pipekitexperiment-i", f"pke-{theme}").strip()
    return (
        f'<svg x="{x}" y="{y}" width="48" height="48" viewBox="0 0 128 128">'
        f"{body}</svg>"
    )


def card(x: float, y: float, w: float, pkg: tuple, t: dict) -> str:
    """One package: its name, what it is, its extra, and the modules it imports."""
    name, what, modules, extra = pkg
    dash = ' stroke-dasharray="6 4"' if extra else ""
    stroke = t["grey"] if extra else t["accent"]
    cx = x + w / 2
    about = [what] + ([f"through its {extra} extra"] if extra else [])
    out = [
        f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{CARD_H}" rx="10" '
        f'fill="{t["card"]}" stroke="{stroke}" stroke-width="1.6"{dash}/>'
        f'<text x="{cx:.1f}" y="{y + 22}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="13.5" font-weight="700" fill="{t["ink"]}">{escape(name)}</text>'
    ]
    ty = y + 40
    for line in about:
        out.append(
            f'<text x="{cx:.1f}" y="{ty}" text-anchor="middle" font-family="{SANS}" '
            f'font-size="12" fill="{t["muted"]}">{escape(line)}</text>'
        )
        ty += 17
    out.append(
        f'<text x="{cx:.1f}" y="{ty + 3}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="10.5" fill="{t["uses"]}">{escape(modules)}</text>'
    )
    return "".join(out)


def layer_box(y: float, h: float, name: str, caption: str, t: dict) -> str:
    """One stack layer: a box with its name on the left and a caption on the right."""
    fill, stroke = t["box"]
    return (
        f'<rect x="{BOX_X}" y="{y}" width="{BOX_W}" height="{h}" rx="14" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        f'<text x="{BOX_X + 18}" y="{y + 26}" font-family="{SANS}" '
        f'font-size="15" font-weight="700" fill="{t["ink"]}">{name}</text>'
        f'<text x="{BOX_X + BOX_W - 18}" y="{y + 26}" text-anchor="end" '
        f'font-family="{SANS}" font-size="12.5" fill="{t["muted"]}">'
        f"{escape(caption)}</text>"
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    out = [
        f'<defs><marker id="ea-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["accent"]}"/></marker></defs>',
        f'<rect x="250" y="18" width="460" height="74" rx="14" fill="{t["card"]}" '
        f'stroke="{t["accent"]}" stroke-width="2"/>',
        icon(266, 31, theme),
        f'<text x="328" y="52" font-family="{SANS}" font-size="22" '
        f'font-weight="700" fill="{t["ink"]}">pipekit-experiment</text>',
        f'<text x="328" y="74" font-family="{SANS}" font-size="13" '
        f'fill="{t["muted"]}">{escape(TAGLINE)}</text>',
        f'<text x="694" y="52" text-anchor="end" font-family="{SANS}" '
        f'font-size="12" fill="{t["muted"]}">{escape(TIER)}</text>',
        # dashed: the one import is through an extra
        f'<path d="M480 {TOP_Y} L480 95" fill="none" stroke="{t["accent"]}" '
        f'stroke-width="2.2" stroke-dasharray="7 5" marker-end="url(#ea-{theme})"/>',
        f'<text x="492" y="122" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imported through an optional extra</text>',
    ]
    # the rest of the workspace: the one card, and a note on what uses it by shape
    card_w = (BOX_W - 36 - CARD_GAP * (PER_ROW - 1)) / PER_ROW
    y = TOP_Y
    box_h = 44 + CARD_H + 18
    out.append(layer_box(y, box_h, "pipekit", "the rest of the workspace", t))
    for i, pkg in enumerate(WORKSPACE):
        out.append(card(BOX_X + 18 + i * (card_w + CARD_GAP), y + 44, card_w, pkg, t))
    note_x = BOX_X + 18 + len(WORKSPACE) * (card_w + CARD_GAP) + 8
    lines = wrap(WORKSPACE_NOTE, 92)
    ny = y + 44 + CARD_H / 2 - 9 * (len(lines) - 1) + 4
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{note_x:.1f}" y="{ny + 18 * i:.1f}" font-family="{SANS}" '
            f'font-size="12.5" fill="{t["muted"]}">{escape(line)}</text>'
        )
    y += box_h + LAYER_GAP
    # outside the workspace: an honest empty state
    out_h = 82
    out.append(
        layer_box(y, out_h, "outside", "GeoStack, GeoNumerics, GeoML, GeoModels", t)
        + f'<text x="480" y="{y + 58}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="13" font-style="italic" fill="{t["muted"]}">'
        f"{escape(OUTSIDE)}</text>"
    )
    y += out_h + LAYER_GAP
    # what pipekit-experiment builds on
    by = y + 6
    bfill, bstroke = t["base"]
    out.append(
        f'<rect x="{BOX_X}" y="{by}" width="{BOX_W}" height="62" rx="10" '
        f'fill="{bfill}" stroke="{bstroke}"/>'
        f'<text x="480" y="{by + 26}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="14" fill="{t["muted"]}">pipekit-experiment builds on '
        f'<tspan font-weight="700" fill="{t["ink"]}">{escape(BASE)}</tspan></text>'
        f'<text x="480" y="{by + 48}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="13" fill="{t["muted"]}">{escape(BASE_EXTRAS)}</text>'
    )
    # legend: only the mark the figure uses
    ly = by + 92
    out.append(
        f'<rect x="40" y="{ly - 12}" width="26" height="16" rx="4" fill="none" '
        f'stroke="{t["grey"]}" stroke-width="1.6" stroke-dasharray="6 4"/>'
        f'<text x="74" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imports pipekit-experiment through an optional extra '
        "(the module it imports)</text>"
    )
    height = ly + 22
    label = (
        "The packages that import pipekit-experiment on their main branches: "
        "pipekit-train, through its experiment extra, for its artifacts module. "
        "pipekit-jax and the pipekit umbrella import nothing from it, and no "
        "package outside the pipekit workspace imports it. pipekit-experiment "
        "builds on pipekit-ops, with fsspec and s3fs, DVC, Hydra and hydra-zen, "
        "and Metaflow as extras."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>pipekit-experiment and the packages that import it</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``pipekit_experiment_ecosystem-light.svg`` and ``-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"pipekit_experiment_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
