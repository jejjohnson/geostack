"""Write the vardax dependency figure as light and dark SVGs.

The layout follows argminax's dependency figure (``argminax_ecosystem/``):
vardax at the top, the packages that import it below in their stack layer, and
what vardax builds on underneath. The edges are geostack's dependency graph
(``docs/api/dependency_graph.md``); the names each package uses are its
``vardax.`` attribute uses on its ``main``, read through plumax's
``require_vardax()`` alias. ``_icon.svg`` is vardax's ``docs/assets/icon.svg``,
copied unchanged. Re-run after editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/vardax_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the vardax dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

TAGLINE = "variational data assimilation in JAX"
TIER = "GeoML · tier 3"
# (layer, caption, [(package, what it is, lines of the areas it uses, the methods
#  it calls, the extra it comes through)])
LAYERS = [
    (
        "GeoModels",
        "physical models",
        [
            (
                "plumax",
                "plume transport and source inversion",
                (
                    "models · observation operators · controls · minimisers",
                    "posterior · validation · data",
                ),
                "OptimalInterpolation · ThreeDVar · IncrementalFourDVar",
                "assim",
            ),
            (
                "sesmax-core",
                "simple Earth-system models",
                ("models · observation operators",),
                "OptimalInterpolation · ThreeDVar",
                "da",
            ),
        ],
    ),
]
# edges between dependents, drawn grey: (from, to); none at the pins
BUILDS_ON: list[tuple[str, str]] = []
NOTES = {
    "plumax": "the assim extra, with argminax",
    "sesmax-core": "the da extra, with filterax",
}
BASE = [
    "JAX",
    "Equinox",
    "lineax",
    "gaussx",
    "optimistix",
    "optax",
    "argminax",
    "diffrax",
    "integrix",
    "geonnax",
    "pipekit-cycle",
]

THEMES = {
    "light": {
        "accent": "#BE123C",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "grey": "#9CA3AF",
        "uses": "#334155",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#FB7185",
        "card": "#111827",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "box": ("#0F172A", "#1F2937"),
        "grey": "#6B7280",
        "uses": "#CBD5E1",
        "base": ("#1F2937", "#374151"),
    },
}

BOX_X, BOX_W, BOX_Y, CARD_GAP = 20, 920, 140, 20


def icon(x: float, y: float, theme: str) -> str:
    """vardax's icon at 48 px, with its gradient id made unique per theme."""
    src = (HERE / "_icon.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    body = body.replace("vardax-icon", f"vdx-{theme}").strip()
    return (
        f'<svg x="{x}" y="{y}" width="48" height="48" viewBox="0 0 128 128">'
        f"{body}</svg>"
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
        # vardax
        f'<rect x="250" y="18" width="460" height="74" rx="14" fill="{t["card"]}" '
        f'stroke="{t["accent"]}" stroke-width="2"/>',
        icon(266, 31, theme),
        f'<text x="328" y="52" font-family="{SANS}" font-size="22" '
        f'font-weight="700" fill="{t["ink"]}">vardax</text>',
        f'<text x="328" y="74" font-family="{SANS}" font-size="13" '
        f'fill="{t["muted"]}">{escape(TAGLINE)}</text>',
        f'<text x="694" y="52" text-anchor="end" font-family="{SANS}" '
        f'font-size="12" fill="{t["muted"]}">{escape(TIER)}</text>',
    ]
    centres = {}
    box_h = 0
    card_h = 0
    for layer, caption, packages in LAYERS:
        card_h = 74 + 17 * max(len(p[2]) for p in packages)
        box_h = 44 + card_h + 36
        fill, stroke = t["box"]
        out.append(
            f'<rect x="{BOX_X}" y="{BOX_Y}" width="{BOX_W}" height="{box_h}" '
            f'rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{BOX_X + 18}" y="{BOX_Y + 26}" font-family="{SANS}" '
            f'font-size="15" font-weight="700" fill="{t["ink"]}">{layer}</text>'
            f'<text x="{BOX_X + BOX_W - 18}" y="{BOX_Y + 26}" text-anchor="end" '
            f'font-family="{SANS}" font-size="12.5" fill="{t["muted"]}">'
            f"{caption}</text>"
        )
        # the layer's arrow up to vardax, dashed: every import is through an extra
        out.append(
            f'<path d="M480 {BOX_Y} L480 95" fill="none" stroke="{t["accent"]}" '
            f'stroke-width="2.2" stroke-dasharray="7 5" '
            f'marker-end="url(#ea-{theme})"/>'
        )
        card_w = (BOX_W - 36 - CARD_GAP * (len(packages) - 1)) / len(packages)
        x = BOX_X + 18
        for name, what, areas, methods, extra in packages:
            y = BOX_Y + 44
            dash = ' stroke-dasharray="6 4"' if extra else ""
            stroke = t["grey"] if extra else t["accent"]
            cx = x + card_w / 2
            out.append(
                f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="10" '
                f'fill="{t["card"]}" stroke="{stroke}" stroke-width="1.6"{dash}/>'
                f'<text x="{cx}" y="{y + 23}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="13.5" font-weight="700" '
                f'fill="{t["ink"]}">{name}</text>'
                f'<text x="{cx}" y="{y + 42}" text-anchor="middle" '
                f'font-family="{SANS}" font-size="12" fill="{t["muted"]}">'
                f"{escape(what)}</text>"
                + "".join(
                    f'<text x="{cx}" y="{y + 61 + 17 * i}" text-anchor="middle" '
                    f'font-family="{SANS}" font-size="11" fill="{t["uses"]}">'
                    f"{escape(line)}</text>"
                    for i, line in enumerate(areas)
                )
                + f'<text x="{cx}" y="{y + card_h - 13}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="10.5" fill="{t["uses"]}">'
                f"{escape(methods)}</text>"
            )
            if name in NOTES:
                out.append(
                    f'<text x="{cx}" y="{y + card_h + 20}" '
                    f'text-anchor="middle" font-family="{SANS}" font-size="11.5" '
                    f'fill="{t["muted"]}">{escape(NOTES[name])}</text>'
                )
            centres[name] = (x, y, card_w)
            x += card_w + CARD_GAP
    for src, dst in BUILDS_ON:
        (sx, sy, _), (dx, _, dw) = centres[src], centres[dst]
        ymid = sy + card_h / 2
        out.append(
            f'<path d="M{sx - 2} {ymid} L{dx + dw + 3} {ymid}" fill="none" '
            f'stroke="{t["grey"]}" stroke-width="1.6" marker-end="url(#eh-{theme})"/>'
        )
    # what vardax builds on, on two lines
    by = BOX_Y + box_h + 22
    fill, stroke = t["base"]
    half = (len(BASE) + 1) // 2
    out.append(
        f'<rect x="20" y="{by}" width="920" height="62" rx="10" fill="{fill}" '
        f'stroke="{stroke}"/>'
    )
    for i, names in enumerate([BASE[:half], BASE[half:]]):
        bold = " · ".join(
            f'<tspan font-weight="700" fill="{t["ink"]}">{escape(n)}</tspan>'
            for n in names
        )
        lead = "vardax builds on: " if i == 0 else ""
        out.append(
            f'<text x="480" y="{by + 26 + 22 * i}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="14" fill="{t["muted"]}">'
            f"{lead}{bold}</text>"
        )
    # legend: only the marks the figure uses
    ly = by + 88
    items = []
    extras = [p[4] for _, _, ps in LAYERS for p in ps]
    if not all(extras):
        items.append(
            (
                f'<rect x="{{x}}" y="{ly - 12}" width="26" height="16" rx="4" '
                f'fill="none" stroke="{t["accent"]}" stroke-width="1.6"/>',
                "imports vardax (the areas and methods it uses)",
            )
        )
    if any(extras):
        items.append(
            (
                f'<rect x="{{x}}" y="{ly - 12}" width="26" height="16" rx="4" '
                f'fill="none" stroke="{t["grey"]}" stroke-width="1.6" '
                'stroke-dasharray="6 4"/>',
                "imports vardax through an optional extra "
                "(the areas and methods it uses)",
            )
        )
    if BUILDS_ON:
        items.append(
            (
                f'<path d="M{{x}} {ly - 4} L{{x34}} {ly - 4}" stroke="{t["grey"]}" '
                f'stroke-width="1.6" marker-end="url(#eh-{theme})"/>',
                "builds on",
            )
        )
    x = 40.0
    for mark, text in items:
        out.append(mark.format(x=x, x34=x + 34))
        out.append(
            f'<text x="{x + 34}" y="{ly + 1}" font-family="{SANS}" '
            f'font-size="12.5" fill="{t["muted"]}">{escape(text)}</text>'
        )
        x += 34 + 6.6 * len(text) + 40
    height = ly + 22
    label = (
        "Packages that import vardax: plumax through its assim extra and "
        "sesmax-core through its da extra, both in GeoModels. vardax builds on "
        "JAX, Equinox, lineax, gaussx, optimistix, optax, argminax, diffrax, "
        "integrix, geonnax and pipekit-cycle."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>vardax in the GeoModels layer</title>\n" + "\n".join(out) + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``vardax_ecosystem-light.svg`` and ``vardax_ecosystem-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"vardax_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
