"""Write the functax dependency figure as light and dark SVGs.

The layout follows argminax's ecosystem figure (``argminax_ecosystem/``):
functax at the top, the packages that import it below in their stack layers,
and what functax builds on underneath. The edges are geostack's dependency
graph (``docs/api/dependency_graph.md``) and, for geonnax, which the graph does
not list yet, geonnax's own pin; the modules each package uses are its
``from functax… import`` lines on ``main``. ``_icon.svg`` is functax's
``docs/assets/icon.svg``, unchanged. Re-run after editing the data below, from
the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/functax_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the functax dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# (layer, caption, [(package, what it is, [functax modules it uses, one line
#  each], extra)]). A layer's box is as wide as its cards.
LAYERS = [
    (
        "Core",
        "",
        [("integrix", "integration", ["quad"], False)],
    ),
    (
        "GeoML",
        "layers and probabilistic models",
        [
            (
                "geonnax",
                "layers and fields",
                ["basis · transform", "diffops · quad", "geometry"],
                False,
            ),
            ("pyrox-nn", "Bayesian nets", ["basis · geometry"], False),
            ("pyrox-gp", "GP models", ["basis"], False),
            ("pyrox-lgm", "latent Gaussians", ["geometry · interp"], False),
        ],
    ),
    (
        "GeoStack",
        "arrays and operators",
        [
            ("geotoolz-array", "geoarray carrier", ["geometry"], False),
            ("geotoolz-ops", "geoops operators", ["geometry · interp"], True),
        ],
    ),
]
# edges between dependents, drawn grey: (from, to)
BUILDS_ON = [("pyrox-nn", "geonnax")]
NOTES = {"geotoolz-array": "NumPy-only install", "geotoolz-ops": "interp extra"}
BASE = [
    "NumPy",
    "JAX · Equinox · lineax · einx · jaxtyping (jax extra)",
    "SciPy (scipy extra)",
]

THEMES = {
    "light": {
        "accent": "#1D4ED8",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "grey": "#9CA3AF",
        "uses": "#334155",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#60A5FA",
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
    """functax's icon at 48 px, with its gradient id made unique per theme."""
    src = (HERE / "_icon.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    body = body.replace('id="ic"', f'id="fx-{theme}"').replace(
        "url(#ic)", f"url(#fx-{theme})"
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
        # functax
        f'<rect x="{FX_X}" y="18" width="{FX_W}" height="74" rx="14" '
        f'fill="{t["card"]}" stroke="{t["accent"]}" stroke-width="2"/>',
        icon(FX_X + 16, 31, theme),
        f'<text x="{FX_X + 78}" y="52" font-family="{SANS}" font-size="22" '
        f'font-weight="700" fill="{t["ink"]}">functax</text>',
        f'<text x="{FX_X + 78}" y="74" font-family="{SANS}" font-size="13" '
        f'fill="{t["muted"]}">bases, plans with adjoints, rules and transforms'
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
        # the layer's arrow up to functax
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
            if name in NOTES:
                out.append(
                    f'<text x="{cx}" y="{y + CARD_H + 18}" text-anchor="middle" '
                    f'font-family="{SANS}" font-size="11.5" fill="{t["muted"]}">'
                    f"{escape(NOTES[name])}</text>"
                )
            centres[name] = (x, y)
            x += CARD_W + CARD_GAP
        box_x += box_w + LAYER_GAP
    for src, dst in BUILDS_ON:
        (sx, sy), (dx, _) = centres[src], centres[dst]
        ymid = sy + CARD_H / 2
        out.append(
            f'<path d="M{sx - 1} {ymid} L{dx + CARD_W + 2} {ymid}" fill="none" '
            f'stroke="{t["grey"]}" stroke-width="1.6" marker-end="url(#eh-{theme})"/>'
        )
    # what functax builds on
    by = BOX_Y + box_h + 22
    fill, stroke = t["base"]
    names = " · ".join(
        f'<tspan font-weight="700" fill="{t["ink"]}">{escape(n)}</tspan>' for n in BASE
    )
    out.append(
        f'<rect x="20" y="{by}" width="920" height="40" rx="10" fill="{fill}" '
        f'stroke="{stroke}"/>'
        f'<text x="480" y="{by + 25.5}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="13.5" fill="{t["muted"]}">functax builds on: {names}</text>'
    )
    # legend
    ly = by + 66
    out.append(
        f'<rect x="40" y="{ly - 12}" width="26" height="16" rx="4" fill="none" '
        f'stroke="{t["accent"]}" stroke-width="1.6"/>'
        f'<text x="74" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imports functax (the modules it uses below)</text>'
        f'<rect x="380" y="{ly - 12}" width="26" height="16" rx="4" fill="none" '
        f'stroke="{t["grey"]}" stroke-width="1.6" stroke-dasharray="6 4"/>'
        f'<text x="414" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">through an optional extra</text>'
        f'<path d="M640 {ly - 4} L674 {ly - 4}" stroke="{t["grey"]}" '
        f'stroke-width="1.6" marker-end="url(#eh-{theme})"/>'
        f'<text x="682" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">builds on</text>'
    )
    height = ly + 22
    label = (
        "Packages that import functax: integrix in Core; geonnax, pyrox-nn, "
        "pyrox-gp and pyrox-lgm in GeoML; geotoolz-array, and geotoolz-ops through "
        "its interp extra, in GeoStack. functax builds on NumPy, on JAX, Equinox, "
        "lineax, einx and jaxtyping through its jax extra, and on SciPy through its "
        "scipy extra."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>functax in the Core, GeoML and GeoStack layers</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``functax_ecosystem-light.svg`` and ``functax_ecosystem-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"functax_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
