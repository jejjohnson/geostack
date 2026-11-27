"""Write the pipekit-ops dependency figure as light and dark SVGs.

The style follows vardax's dependency figure (``vardax_ecosystem/``): the
package at the top, the packages that import it below in their stack layers,
and what it builds on underneath. pipekit-ops has many dependents, so the
layers are stacked and each wraps its cards into rows. The edges and the
areas each package uses are its ``from pipekit_ops… import`` lines on
``main``, outside its tests; geostack's ``docs/api/dependency_graph.md``
names only the first pipekit package of each repository, so it does not list
them all. ``_icon.svg`` is pipekit's
``docs/assets/packages/pipekit-ops/icon.svg``, copied unchanged. Re-run after
editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pipekit_ops_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the pipekit-ops dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

TAGLINE = "operators and operator graphs for any carrier"
TIER = "Core · tier 0"
# (layer, caption, [(package, what it is, the areas it imports, its extra)])
LAYERS = [
    (
        "pipekit",
        "the rest of the workspace",
        [
            ("pipekit-array", "named-dim arrays", "carriers · core", None),
            ("pipekit-cycle", "time stepping and DA", "core", None),
            ("pipekit-train", "training loops", "core · operators · saving", None),
            ("pipekit-experiment", "model registries", "core · saving", None),
            ("pipekit-evaluate", "scoring and benchmarks", "saving", None),
            ("pipekit-jax", "JAX operators", "core", None),
        ],
    ),
    (
        "GeoStack",
        "geospatial pipelines",
        [
            (
                "geotoolz-ops",
                "geospatial operators",
                ("carriers · core · saving", "operators · observe"),
                None,
            ),
            ("xrtoolz-ops", "xarray operators", "carriers · core", None),
            ("geotoolz-array", "the GeoTensor carrier", "carriers", None),
            ("geotoolz-patcher", "patches and tiles", "core", "pipekit"),
        ],
    ),
    (
        "GeoNumerics",
        "numerical building blocks",
        [
            ("gridax", "grids and fields", "carriers · core", None),
            ("diagnostix", "diagnostics", "carriers", None),
        ],
    ),
    (
        "GeoML",
        "probabilistic ML",
        [("vardax", "variational DA", "core", None)],
    ),
    (
        "GeoModels",
        "physical models",
        [
            ("plumax", "plume models", "carriers · core", None),
            ("radtrax", "radiative transfer", "core", "geo"),
            ("sesmax-core", "Earth-system models", "core", None),
            (
                "sesmax-experiment",
                "runs of those models",
                ("core · saving", "operators · observe"),
                None,
            ),
        ],
    ),
]
BASE = "the Python standard library alone: no third-party dependency"

THEMES = {
    "light": {
        "accent": "#EA580C",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "grey": "#9CA3AF",
        "uses": "#334155",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#FB923C",
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
PER_ROW, CARD_GAP, CARD_H, ROW_GAP, LAYER_GAP = 4, 16, 92, 14, 16


def icon(x: float, y: float, theme: str) -> str:
    """pipekit-ops's icon at 48 px, with its gradient id made unique per theme."""
    src = (HERE / "_icon.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    body = body.replace("pipekitops-i", f"pko-{theme}").strip()
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
        f'font-weight="700" fill="{t["ink"]}">pipekit-ops</text>',
        f'<text x="328" y="74" font-family="{SANS}" font-size="13" '
        f'fill="{t["muted"]}">{escape(TAGLINE)}</text>',
        f'<text x="694" y="52" text-anchor="end" font-family="{SANS}" '
        f'font-size="12" fill="{t["muted"]}">{escape(TIER)}</text>',
        # every layer below imports pipekit-ops
        f'<path d="M480 {TOP_Y} L480 95" fill="none" stroke="{t["accent"]}" '
        f'stroke-width="2.2" marker-end="url(#ea-{theme})"/>',
        f'<text x="492" y="122" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imported by every package below</text>',
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
                out.append(card(BOX_X + 18 + i * (card_w + CARD_GAP), cy, card_w, pkg, t))
        y += box_h + LAYER_GAP
    # what pipekit-ops builds on
    by = y + 6
    bfill, bstroke = t["base"]
    out.append(
        f'<rect x="{BOX_X}" y="{by}" width="{BOX_W}" height="40" rx="10" '
        f'fill="{bfill}" stroke="{bstroke}"/>'
        f'<text x="480" y="{by + 26}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="14" fill="{t["muted"]}">pipekit-ops builds on '
        f'<tspan font-weight="700" fill="{t["ink"]}">{escape(BASE)}</tspan></text>'
    )
    # legend: only the marks the figure uses
    ly = by + 70
    items = [
        (
            f'<rect x="{{x}}" y="{ly - 12}" width="26" height="16" rx="4" '
            f'fill="none" stroke="{t["accent"]}" stroke-width="1.6"/>',
            "imports pipekit-ops (the areas it imports)",
        ),
        (
            f'<rect x="{{x}}" y="{ly - 12}" width="26" height="16" rx="4" '
            f'fill="none" stroke="{t["grey"]}" stroke-width="1.6" '
            'stroke-dasharray="6 4"/>',
            "imports it through an optional extra",
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
        f"The packages that import pipekit-ops on their main branches: {names}; "
        "geotoolz-patcher through its pipekit extra and radtrax through its geo "
        "extra. pipekit-ops builds on the Python standard library alone."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>pipekit-ops and the packages that import it</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``pipekit_ops_ecosystem-light.svg`` and ``-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"pipekit_ops_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
