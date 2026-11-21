"""Write the gridax dependency figure as light and dark SVGs.

The layout follows functax's ecosystem figure (``functax_ecosystem/``):
gridax at the top, the packages that import it below in their stack layers,
and what gridax builds on underneath. geostack's dependency graph
(``docs/api/dependency_graph.md``) still knows the repository as finitevolX,
so the cards and the names each package uses are read from its imports on
``main``: sesmax-core imports ``gridax``, and plumax only the deprecated
``finitevolx`` shim, drawn dashed until it switches (gridax#416).
``_icon.svg`` is the tile of gridax's logo (``docs/assets/logo-light.svg``).
Re-run after editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/gridax_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the gridax dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# (layer, caption, [(package, what it is, [gridax names it uses, one line
#  each], through the finitevolx shim)]). A layer's box is as wide as its cards.
LAYERS = [
    (
        "GeoNumerics",
        "",
        [("gridax-jax", "solvers, pipekit", ["re-exports all", "of gridax"], False)],
    ),
    (
        "GeoModels",
        "the models",
        [
            (
                "sesmax-core",
                "model core",
                ["grids · masks", "Interpolation2D", "stencils"],
                False,
            ),
            ("plumax", "plume models", ["CartesianGrid3D", "Advection3D", "BCs"], True),
        ],
    ),
]
# edges between dependents, drawn grey and routed under the layers:
# (from, to, dashed); both reach gridax-jax's names through finitevolx today
BUILDS_ON = [
    ("sesmax-core", "gridax-jax", True),
    ("plumax", "gridax-jax", True),
]
BASE = ["NumPy · SciPy", "jaxtyping", "geophysix", "JAX, when installed"]

THEMES = {
    "light": {
        "accent": "#1E3A5F",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "grey": "#9CA3AF",
        "uses": "#334155",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#38BDF8",
        "card": "#111827",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "box": ("#0F172A", "#1F2937"),
        "grey": "#6B7280",
        "uses": "#CBD5E1",
        "base": ("#1F2937", "#374151"),
    },
}

BOX_Y, CARD_W, CARD_H, CARD_GAP, PAD, LAYER_GAP = 140, 112, 98, 8, 12, 12
FX_X, FX_W = 250, 460


def icon(x: float, y: float, theme: str) -> str:
    """gridax's icon at 48 px, with its gradient id made unique per theme."""
    src = (HERE / "_icon.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    body = body.replace('id="ic"', f'id="ix-{theme}"').replace(
        "url(#ic)", f"url(#ix-{theme})"
    )
    return (
        f'<svg x="{x}" y="{y}" width="48" height="48" viewBox="0 0 128 128">'
        f"{body.strip()}</svg>"
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    out = [
        f'<defs><marker id="eh-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["grey"]}"/></marker>'
        f'<marker id="ea-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["accent"]}"/></marker></defs>',
        # gridax
        f'<rect x="{FX_X}" y="18" width="{FX_W}" height="74" rx="14" '
        f'fill="{t["card"]}" stroke="{t["accent"]}" stroke-width="2"/>',
        icon(FX_X + 16, 31, theme),
        f'<text x="{FX_X + 78}" y="52" font-family="{SANS}" font-size="22" '
        f'font-weight="700" fill="{t["ink"]}">gridax</text>',
        f'<text x="{FX_X + 78}" y="74" font-family="{SANS}" font-size="13" '
        f'fill="{t["muted"]}">C-grids, masks and the operators on them'
        "</text>",
    ]
    box_h = 44 + CARD_H + 30
    centres = {}
    widths = [2 * PAD + len(p) * CARD_W + (len(p) - 1) * CARD_GAP for *_, p in LAYERS]
    total = sum(widths) + LAYER_GAP * (len(LAYERS) - 1)
    box_x = 20 + (920 - total) / 2
    targets = [FX_X + FX_W * (i + 1) / (len(LAYERS) + 1) for i in range(len(LAYERS))]
    for (layer, caption, packages), box_w, target in zip(
        LAYERS, widths, targets, strict=True
    ):
        fill, stroke = t["box"]
        out.append(
            f'<rect x="{box_x}" y="{BOX_Y}" width="{box_w}" height="{box_h}" rx="14" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{box_x + PAD}" y="{BOX_Y + 26}" font-family="{SANS}" '
            f'font-size="15" font-weight="700" fill="{t["ink"]}">{layer}</text>'
            f'<text x="{box_x + box_w - PAD}" y="{BOX_Y + 26}" text-anchor="end" '
            f'font-family="{SANS}" font-size="12.5" fill="{t["muted"]}">'
            f"{escape(caption)}</text>"
        )
        # the layer's arrow up to gridax
        mid = box_x + box_w / 2
        out.append(
            f'<path d="M{mid} {BOX_Y} L{mid} 116 L{target} 116 L{target} 95" '
            f'fill="none" stroke="{t["accent"]}" stroke-width="2.2" '
            f'marker-end="url(#ea-{theme})"/>'
        )
        x = box_x + PAD
        for name, what, uses, extra in packages:
            y = BOX_Y + 40
            dash = ' stroke-dasharray="6 4"' if extra else ""
            stroke = t["grey"] if extra else t["accent"]
            cx = x + CARD_W / 2
            out.append(
                f'<rect x="{x}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="10" '
                f'fill="{t["card"]}" stroke="{stroke}" stroke-width="1.6"{dash}/>'
                f'<text x="{cx}" y="{y + 22}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="12.5" font-weight="700" '
                f'fill="{t["ink"]}">{name}</text>'
                f'<text x="{cx}" y="{y + 40}" text-anchor="middle" '
                f'font-family="{SANS}" font-size="11.5" fill="{t["muted"]}">'
                f"{escape(what)}</text>"
            )
            for i, line in enumerate(uses):
                out.append(
                    f'<text x="{cx}" y="{y + 60 + 14 * i}" text-anchor="middle" '
                    f'font-family="{MONO}" font-size="10" fill="{t["uses"]}">'
                    f"{escape(line)}</text>"
                )
            centres[name] = (x, y)
            x += CARD_W + CARD_GAP
        box_x += box_w + LAYER_GAP
    # each edge leaves the bottom of its card, runs under the layer boxes at
    # its own depth, and comes up into the bottom of the card it builds on
    box_bottom = BOX_Y + box_h
    for depth, (src, dst, extra) in enumerate(BUILDS_ON, start=1):
        (sx, sy), (dx, _) = centres[src], centres[dst]
        lane = box_bottom + 12 * depth
        start = sx + CARD_W / 2 + 14 * (depth % 2 * 2 - 1)
        end = dx + CARD_W / 2 + 10 * (depth - 2.5)
        dash = ' stroke-dasharray="5 4"' if extra else ""
        out.append(
            f'<path d="M{start} {sy + CARD_H} L{start} {lane} L{end} {lane} '
            f'L{end} {sy + CARD_H + 2}" fill="none" stroke="{t["grey"]}" '
            f'stroke-width="1.5"{dash} marker-end="url(#eh-{theme})"/>'
        )
    # what gridax builds on
    by = box_bottom + 12 * len(BUILDS_ON) + 22
    fill, stroke = t["base"]
    names = " · ".join(
        f'<tspan font-weight="700" fill="{t["ink"]}">{escape(n)}</tspan>' for n in BASE
    )
    out.append(
        f'<rect x="20" y="{by}" width="920" height="40" rx="10" fill="{fill}" '
        f'stroke="{stroke}"/>'
        f'<text x="480" y="{by + 25.5}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="13.5" fill="{t["muted"]}">gridax builds on: {names}</text>'
    )
    # legend
    ly = by + 66
    out.append(
        f'<rect x="40" y="{ly - 12}" width="26" height="16" rx="4" fill="none" '
        f'stroke="{t["accent"]}" stroke-width="1.6"/>'
        f'<text x="74" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imports gridax (the names it uses below)</text>'
        f'<path d="M372 {ly - 4} L406 {ly - 4}" stroke="{t["grey"]}" '
        f'stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#eh-{theme})"/>'
        f'<text x="414" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">through the finitevolx shim (#416)</text>'
        f'<path d="M640 {ly - 4} L674 {ly - 4}" stroke="{t["grey"]}" '
        f'stroke-width="1.6" marker-end="url(#eh-{theme})"/>'
        f'<text x="682" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">builds on</text>'
    )
    height = ly + 22
    label = (
        "Packages that import gridax: gridax-jax in GeoNumerics, in the same "
        "repository, which re-exports all of it; and sesmax-core in GeoModels, "
        "which imports the grids, masks, Interpolation2D and the raw stencils. "
        "plumax in GeoModels still imports the deprecated finitevolx shim, drawn "
        "dashed, and both models reach gridax-jax's names through it. gridax "
        "builds on NumPy, SciPy, jaxtyping and geophysix, and on JAX when it is "
        "installed."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>gridax in the GeoNumerics and GeoModels layers</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``gridax_ecosystem-light.svg`` and ``gridax_ecosystem-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"gridax_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
