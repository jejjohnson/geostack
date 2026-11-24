"""Write the pyrox-nn dependency figure as light and dark SVGs.

The layout follows pyrox's ecosystem figure (``pyrox_ecosystem/``): pyrox-nn at
the top, what imports it below, and what pyrox-nn builds on underneath.
pyrox-nn is ``packages/pyrox-nn`` in the pyrox workspace. No package imports it
on pyrox's ``main``; the one edge is the core's ``pyrox.inference``, which looks
pyrox-nn up lazily, through ``importlib``, to serve two names that moved there,
with a ``DeprecationWarning``. geostack counts a lazy import as an edge, so it
is drawn, dashed. ``_icon.svg`` is pyrox's ``docs/assets/icon-nn.svg``, copied
unchanged. Re-run after editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pyrox_nn_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the pyrox-nn dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

LAYER = ("GeoML", "the pyrox workspace; no other package imports pyrox-nn")
# (package, what it is, the pyrox-nn names it uses, dashed)
PACKAGES = [
    (
        "pyrox.inference",
        "a deprecated lazy lookup",
        "LastLayerLaplace · last_layer_laplace",
        True,
    ),
]
BASE = [
    "pyrox",
    "geonnax",
    "functax",
    "kernellib",
    "argminax",
    "gaussx",
    "lineax",
    "JAX",
    "Equinox",
    "NumPyro",
    "einx",
    "optax, pandas, xarray (extras)",
]

THEMES = {
    "light": {
        "accent": "#475569",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "uses": "#334155",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#94A3B8",
        "card": "#111827",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "box": ("#0F172A", "#1F2937"),
        "uses": "#CBD5E1",
        "base": ("#1F2937", "#374151"),
    },
}

BOX_X, BOX_W, BOX_Y, BOX_H = 20, 920, 140, 140
CARD_W, CARD_H, CARD_GAP = 320, 74, 12


def icon(x: float, y: float, theme: str) -> str:
    """pyrox-nn's icon at 48 px, with its gradient id made unique per theme."""
    src = (HERE / "_icon.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    body = body.replace("icon-nn-g", f"prxnn-{theme}").strip()
    return (
        f'<svg x="{x}" y="{y}" width="48" height="48" viewBox="0 0 128 128">'
        f"{body}</svg>"
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    out = [
        f'<defs><marker id="ea-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["accent"]}"/></marker></defs>',
        # pyrox-nn
        f'<rect x="250" y="18" width="460" height="74" rx="14" fill="{t["card"]}" '
        f'stroke="{t["accent"]}" stroke-width="2"/>',
        icon(266, 31, theme),
        f'<text x="328" y="52" font-family="{SANS}" font-size="22" '
        f'font-weight="700" fill="{t["ink"]}">pyrox-nn</text>',
        f'<text x="328" y="74" font-family="{SANS}" font-size="13" '
        f'fill="{t["muted"]}">uncertainty-aware layers and neural fields</text>',
    ]
    # the layer box, with its dashed arrow up to pyrox-nn
    layer, caption = LAYER
    fill, stroke = t["box"]
    out.append(
        f'<rect x="{BOX_X}" y="{BOX_Y}" width="{BOX_W}" height="{BOX_H}" rx="14" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        f'<text x="{BOX_X + 18}" y="{BOX_Y + 26}" font-family="{SANS}" '
        f'font-size="15" font-weight="700" fill="{t["ink"]}">{layer}</text>'
        f'<text x="{BOX_X + BOX_W - 18}" y="{BOX_Y + 26}" text-anchor="end" '
        f'font-family="{SANS}" font-size="12.5" fill="{t["muted"]}">'
        f"{escape(caption)}</text>"
        f'<path d="M480 {BOX_Y} L480 95" fill="none" stroke="{t["accent"]}" '
        f'stroke-width="2.2" stroke-dasharray="6 5" marker-end="url(#ea-{theme})"/>'
    )
    row_w = len(PACKAGES) * CARD_W + (len(PACKAGES) - 1) * CARD_GAP
    x = BOX_X + (BOX_W - row_w) / 2
    y = BOX_Y + 48
    for name, what, uses, dashed in PACKAGES:
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        out.append(
            f'<rect x="{x}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="10" '
            f'fill="{t["card"]}" stroke="{t["accent"]}" stroke-width="1.6"{dash}/>'
            f'<text x="{x + CARD_W / 2}" y="{y + 23}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="13.5" font-weight="700" '
            f'fill="{t["ink"]}">{name}</text>'
            f'<text x="{x + CARD_W / 2}" y="{y + 42}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="12" fill="{t["muted"]}">'
            f"{escape(what)}</text>"
            f'<text x="{x + CARD_W / 2}" y="{y + 61}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="11" fill="{t["uses"]}">'
            f"{escape(uses)}</text>"
        )
        x += CARD_W + CARD_GAP
    # what pyrox-nn builds on, on two lines: the GeoStack libraries, then the rest
    by = BOX_Y + BOX_H + 22
    fill, stroke = t["base"]
    lines = [BASE[:7], BASE[7:]]
    out.append(
        f'<rect x="20" y="{by}" width="920" height="62" rx="10" fill="{fill}" '
        f'stroke="{stroke}"/>'
    )
    for i, line in enumerate(lines):
        names = " · ".join(
            f'<tspan font-weight="700" fill="{t["ink"]}">{escape(n)}</tspan>'
            for n in line
        )
        lead = "pyrox-nn builds on: " if i == 0 else ""
        out.append(
            f'<text x="480" y="{by + 25.5 + 22 * i}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="14" fill="{t["muted"]}">'
            f"{lead}{names}</text>"
        )
    # legend
    ly = by + 88
    out.append(
        f'<rect x="40" y="{ly - 12}" width="26" height="16" rx="4" fill="none" '
        f'stroke="{t["accent"]}" stroke-width="1.6" stroke-dasharray="5 4"/>'
        f'<text x="74" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">looks pyrox-nn up lazily (the names it serves below), '
        "and does not depend on it</text>"
    )
    height = ly + 22
    label = (
        "No package imports pyrox-nn. The pyrox core's pyrox.inference looks it up "
        "lazily to serve LastLayerLaplace and last_layer_laplace with a "
        "deprecation warning, drawn dashed, inside the pyrox workspace in GeoML. "
        "pyrox-nn builds on pyrox, geonnax, functax, kernellib, argminax, gaussx, "
        "lineax, JAX, Equinox, NumPyro and einx, and on optax, pandas and xarray "
        "through extras."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>pyrox-nn in the GeoML layer</title>\n" + "\n".join(out) + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``pyrox_nn_ecosystem-light.svg`` and ``pyrox_nn_ecosystem-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"pyrox_nn_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
