"""Write the pipekit-train dependency figure as light and dark SVGs.

The layout is pipekit-ops's (``pipekit_ops_ecosystem/``): the package at the
top, the packages that import it below in their stacked layers, and what it
builds on underneath. The edges and the areas each package uses are its
``import pipekit_train`` and ``from pipekit_train… import`` lines on ``main``,
outside its tests; both dependents import it only through an extra, so both
cards are dashed. plumax still imports ``pipekit_train.adapters.equinox``,
which pipekit's ``main`` no longer has, and its card names the old path. kernellib
imports pipekit-train only in its docs notebooks and vardax only in a
docstring, and the workspace's ``pipekit`` package installs it without
importing it, so all three are left out. geostack's
``docs/api/dependency_graph.md`` still says pipekit-train imports
``geopatcher.time``; on ``main`` it does not, so the figure follows ``main``.
``_icon.svg`` is pipekit's ``docs/assets/packages/pipekit-train/icon.svg``,
copied unchanged. Re-run after editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pipekit_train_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the pipekit-train dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

TAGLINE = "training runs as operators, with a record of each"
TIER = "Core · tier 0"
# (layer, caption, [(package, what it is, the areas it imports, its extra)]).
# A name imported from the top level is counted in the area it is defined in.
LAYERS = [
    (
        "GeoStack",
        "geospatial pipelines",
        [("geotoolz-patcher", "patches and tiles", "data", "train")],
    ),
    (
        "GeoModels",
        "physical models",
        [
            (
                "plumax",
                "plume models",
                ("data · losses · loop", "old adapters.equinox path"),
                "emulate",
            )
        ],
    ),
]
BASE = "pipekit-ops"
BASE_EXTRAS = (
    "and, through extras, Equinox, optax, Grain, Orbax, NumPyro, BlackJAX, "
    "Keras, pipekit-cycle, pipekit-experiment, zarr and fsspec"
)

THEMES = {
    "light": {
        "accent": "#15803D",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "grey": "#9CA3AF",
        "uses": "#334155",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#4ADE80",
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
PER_ROW, CARD_GAP, CARD_H, ROW_GAP, LAYER_GAP = 4, 16, 108, 14, 16


def icon(x: float, y: float, theme: str) -> str:
    """pipekit-train's icon at 48 px, with its gradient id made unique per theme."""
    src = (HERE / "_icon.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    body = body.replace("pipekittrain-i", f"pkt-{theme}").strip()
    return (
        f'<svg x="{x}" y="{y}" width="48" height="48" viewBox="0 0 128 128">'
        f"{body}</svg>"
    )


def card(x: float, y: float, w: float, pkg: tuple, t: dict) -> str:
    """One package: its name, what it is, its extra, and the areas it imports."""
    name, what, areas, extra = pkg
    dash = ' stroke-dasharray="6 4"' if extra else ""
    stroke = t["grey"] if extra else t["accent"]
    cx = x + w / 2
    about = [what] + ([f"through its {extra} extra"] if extra else [])
    uses = [areas] if isinstance(areas, str) else list(areas)
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
    ty += 3
    for line in uses:
        out.append(
            f'<text x="{cx:.1f}" y="{ty}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="10.5" fill="{t["uses"]}">{escape(line)}</text>'
        )
        ty += 15
    return "".join(out)


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
        f'font-weight="700" fill="{t["ink"]}">pipekit-train</text>',
        f'<text x="328" y="74" font-family="{SANS}" font-size="13" '
        f'fill="{t["muted"]}">{escape(TAGLINE)}</text>',
        f'<text x="694" y="52" text-anchor="end" font-family="{SANS}" '
        f'font-size="12" fill="{t["muted"]}">{escape(TIER)}</text>',
        # every layer below imports pipekit-train
        f'<path d="M480 {TOP_Y} L480 95" fill="none" stroke="{t["accent"]}" '
        f'stroke-width="2.2" marker-end="url(#ea-{theme})"/>',
        f'<text x="492" y="122" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imported by every package below, through an extra</text>',
    ]
    card_w = (BOX_W - 36 - CARD_GAP * (PER_ROW - 1)) / PER_ROW
    y = TOP_Y
    fill, stroke = t["box"]
    for layer, caption, packages in LAYERS:
        rows = [packages[i : i + PER_ROW] for i in range(0, len(packages), PER_ROW)]
        box_h = 44 + len(rows) * CARD_H + (len(rows) - 1) * ROW_GAP + 18
        out.append(
            f'<rect x="{BOX_X}" y="{y}" width="{BOX_W}" height="{box_h}" rx="14" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{BOX_X + 18}" y="{y + 26}" font-family="{SANS}" '
            f'font-size="15" font-weight="700" fill="{t["ink"]}">{layer}</text>'
            f'<text x="{BOX_X + BOX_W - 18}" y="{y + 26}" text-anchor="end" '
            f'font-family="{SANS}" font-size="12.5" fill="{t["muted"]}">'
            f"{escape(caption)}</text>"
        )
        for r, row in enumerate(rows):
            cy = y + 44 + r * (CARD_H + ROW_GAP)
            for i, pkg in enumerate(row):
                out.append(
                    card(BOX_X + 18 + i * (card_w + CARD_GAP), cy, card_w, pkg, t)
                )
        y += box_h + LAYER_GAP
    # what pipekit-train builds on
    by = y + 6
    bfill, bstroke = t["base"]
    out.append(
        f'<rect x="{BOX_X}" y="{by}" width="{BOX_W}" height="62" rx="10" '
        f'fill="{bfill}" stroke="{bstroke}"/>'
        f'<text x="480" y="{by + 26}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="14" fill="{t["muted"]}">pipekit-train builds on '
        f'<tspan font-weight="700" fill="{t["ink"]}">{escape(BASE)}</tspan></text>'
        f'<text x="480" y="{by + 48}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="13" fill="{t["muted"]}">{escape(BASE_EXTRAS)}</text>'
    )
    # legend: only the marks the figure uses
    ly = by + 92
    items = [
        (
            f'<rect x="{{x}}" y="{ly - 12}" width="26" height="16" rx="4" '
            f'fill="none" stroke="{t["grey"]}" stroke-width="1.6" '
            'stroke-dasharray="6 4"/>',
            "imports pipekit-train through an optional extra (the areas it imports)",
        ),
    ]
    x = 40.0
    for mark, text in items:
        out.append(mark.format(x=x))
        out.append(
            f'<text x="{x + 34}" y="{ly + 1}" font-family="{SANS}" '
            f'font-size="12.5" fill="{t["muted"]}">{escape(text)}</text>'
        )
        x += 34 + 6.6 * len(text) + 40
    height = ly + 22
    names = ", ".join(p[0] for _, _, ps in LAYERS for p in ps)
    label = (
        f"The packages that import pipekit-train on their main branches: {names}; "
        "geotoolz-patcher through its train extra and plumax through its emulate "
        "extra. pipekit-train builds on pipekit-ops, with Equinox, optax, Grain, "
        "Orbax, NumPyro, BlackJAX, Keras, pipekit-cycle, pipekit-experiment, zarr "
        "and fsspec as extras."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>pipekit-train and the packages that import it</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``pipekit_train_ecosystem-light.svg`` and ``-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"pipekit_train_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
