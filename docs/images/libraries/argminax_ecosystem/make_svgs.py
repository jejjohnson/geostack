"""Write the argminax dependency figure as light and dark SVGs.

The layout follows gaussx's ecosystem figure: argminax at the top, the
packages that import it below in their stack layers, and what argminax
builds on underneath. The edges are geostack's dependency graph
(``docs/api/dependency_graph.md``); the modules each package uses are its
``from argminax… import`` lines at the submodule pins. Re-run after editing
the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/argminax_ecosystem/make_svgs.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the argminax dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# (layer, caption, box width, card width, card gap,
#  [(package, what it is, argminax modules it uses, extra)])
LAYERS = [
    (
        "GeoML",
        "probabilistic ML",
        564,
        168,
        12,
        [
            ("vardax", "3D / 4DVar", "solve · adjoint · curvature", False),
            ("pyrox-lgm", "latent Gaussians", "solve", False),
            ("pyrox-nn", "Bayesian networks", "curvature · diagnose", False),
        ],
    ),
    (
        "GeoModels",
        "physical models",
        336,
        135,
        30,
        [
            ("radtrax", "radiative transfer", "curvature · diagnose", False),
            ("plumax", "plume models", "curvature", True),
        ],
    ),
]
# edges between dependents, drawn grey: (from, to)
BUILDS_ON = [("plumax", "radtrax")]
NOTES = {"plumax": "assim extra, with vardax"}
BASE = ["gaussx", "optimistix", "lineax", "JAX · Equinox", "optax (extra)"]

THEMES = {
    "light": {
        "accent": "#475569",
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

ICON = (  # argminax's docs/assets/icon.svg, drawn at 48 px
    '<svg x="{x}" y="{y}" width="48" height="48" viewBox="0 0 128 128">'
    '<defs><linearGradient id="amx-{theme}" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#64748B"/><stop offset="1" stop-color="#1E293B"/>'
    "</linearGradient></defs>"
    '<rect x="4" y="4" width="120" height="120" rx="26" fill="url(#amx-{theme})"/>'
    '<ellipse cx="70" cy="70" rx="44" ry="26" transform="rotate(-30 70 70)" '
    'fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-opacity="0.35"/>'
    '<ellipse cx="70" cy="70" rx="28" ry="16" transform="rotate(-30 70 70)" '
    'fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-opacity="0.5"/>'
    '<ellipse cx="70" cy="70" rx="13" ry="7.5" transform="rotate(-30 70 70)" '
    'fill="#FFFFFF" fill-opacity="0.3" stroke="#FFFFFF" stroke-width="2.8" '
    'stroke-opacity="0.95"/>'
    '<path d="M24 30 L46 34 L52 56 L64 60 L67 67 L70 70" fill="none" '
    'stroke="#FFFFFF" stroke-width="3.4" stroke-linecap="round" '
    'stroke-linejoin="round"/>'
    + "".join(
        f'<circle cx="{cx}" cy="{cy}" r="3.6" fill="#FFFFFF"/>'
        for cx, cy in [(24, 30), (46, 34), (52, 56), (64, 60), (67, 67)]
    )
    + '<circle cx="70" cy="70" r="5.5" fill="#FFFFFF"/></svg>'
)

BOX_Y, BOX_H, CARD_H = 140, 150, 74


def build(theme: str) -> str:
    t = THEMES[theme]
    out = [
        f'<defs><marker id="eh-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["grey"]}"/></marker>'
        f'<marker id="ea-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["accent"]}"/></marker></defs>',
        # argminax
        f'<rect x="260" y="18" width="440" height="74" rx="14" fill="{t["card"]}" '
        f'stroke="{t["accent"]}" stroke-width="2"/>',
        ICON.format(x=276, y=31, theme=theme),
        f'<text x="338" y="52" font-family="{SANS}" font-size="22" '
        f'font-weight="700" fill="{t["ink"]}">argminax</text>',
        f'<text x="338" y="74" font-family="{SANS}" font-size="13" '
        f'fill="{t["muted"]}">optimisation with uncertainty at the minimum</text>',
    ]
    centres = {}
    box_x = 20
    for layer, caption, box_w, card_w, gap, packages in LAYERS:
        fill, stroke = t["box"]
        out.append(
            f'<rect x="{box_x}" y="{BOX_Y}" width="{box_w}" height="{BOX_H}" rx="14" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{box_x + 18}" y="{BOX_Y + 26}" font-family="{SANS}" '
            f'font-size="15" font-weight="700" fill="{t["ink"]}">{layer}</text>'
            f'<text x="{box_x + box_w - 18}" y="{BOX_Y + 26}" text-anchor="end" '
            f'font-family="{SANS}" font-size="12.5" fill="{t["muted"]}">'
            f"{caption}</text>"
        )
        # the layer's arrow up to argminax
        mid = box_x + box_w / 2
        target = 420 if box_x < 480 else 540
        out.append(
            f'<path d="M{mid} {BOX_Y} L{mid} 116 L{target} 116 L{target} 95" '
            f'fill="none" stroke="{t["accent"]}" stroke-width="2.2" '
            f'marker-end="url(#ea-{theme})"/>'
        )
        x = box_x + 18
        for name, what, uses, extra in packages:
            y = BOX_Y + 44
            dash = ' stroke-dasharray="6 4"' if extra else ""
            stroke = t["grey"] if extra else t["accent"]
            out.append(
                f'<rect x="{x}" y="{y}" width="{card_w}" height="{CARD_H}" rx="10" '
                f'fill="{t["card"]}" stroke="{stroke}" stroke-width="1.6"{dash}/>'
                f'<text x="{x + card_w / 2}" y="{y + 23}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="13.5" font-weight="700" '
                f'fill="{t["ink"]}">{name}</text>'
                f'<text x="{x + card_w / 2}" y="{y + 42}" text-anchor="middle" '
                f'font-family="{SANS}" font-size="12" fill="{t["muted"]}">'
                f"{escape(what)}</text>"
                f'<text x="{x + card_w / 2}" y="{y + 61}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="10" fill="{t["uses"]}">'
                f"{escape(uses)}</text>"
            )
            if name in NOTES:
                out.append(
                    f'<text x="{x + card_w / 2}" y="{y + CARD_H + 17}" '
                    f'text-anchor="middle" font-family="{SANS}" font-size="11.5" '
                    f'fill="{t["muted"]}">{escape(NOTES[name])}</text>'
                )
            centres[name] = (x, y, card_w)
            x += card_w + gap
        box_x += box_w + 20
    for src, dst in BUILDS_ON:
        (sx, sy, _), (dx, _, dw) = centres[src], centres[dst]
        ymid = sy + CARD_H / 2
        out.append(
            f'<path d="M{sx - 2} {ymid} L{dx + dw + 3} {ymid}" fill="none" '
            f'stroke="{t["grey"]}" stroke-width="1.6" marker-end="url(#eh-{theme})"/>'
        )
    # what argminax builds on
    by = BOX_Y + BOX_H + 22
    fill, stroke = t["base"]
    names = " · ".join(
        f'<tspan font-weight="700" fill="{t["ink"]}">{escape(n)}</tspan>' for n in BASE
    )
    out.append(
        f'<rect x="20" y="{by}" width="920" height="40" rx="10" fill="{fill}" '
        f'stroke="{stroke}"/>'
        f'<text x="480" y="{by + 25.5}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="14" fill="{t["muted"]}">argminax builds on: {names}</text>'
    )
    # legend
    ly = by + 66
    out.append(
        f'<rect x="40" y="{ly - 12}" width="26" height="16" rx="4" fill="none" '
        f'stroke="{t["accent"]}" stroke-width="1.6"/>'
        f'<text x="74" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imports argminax (the modules it uses below)</text>'
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
        "Packages that import argminax: vardax, pyrox-lgm and pyrox-nn in GeoML; "
        "radtrax, and plumax through an extra, in GeoModels. argminax builds on "
        "gaussx, optimistix, lineax, JAX and Equinox, and optax through an extra."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>argminax in the GeoML and GeoModels layers</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``argminax_ecosystem-light.svg`` and ``argminax_ecosystem-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"argminax_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
