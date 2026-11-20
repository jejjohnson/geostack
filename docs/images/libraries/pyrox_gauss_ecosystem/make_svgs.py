"""Write the pyrox-gauss dependency figure as light and dark SVGs.

The layout follows pyrox's ecosystem figure (``pyrox_ecosystem/``): pyrox-gauss
at the top, the packages that import it below, and what pyrox-gauss builds on
underneath. pyrox-gauss is ``packages/pyrox-gauss`` in the pyrox workspace;
every package that imports it is a sibling in the same workspace, so one GeoML
box holds them all. The names each package uses are its ``from pyrox_gauss
import`` lines on pyrox's ``main``, read from the code and not from its
docstrings; pyrox-ssm's state-space names come through its ``flows`` extra.
``_icon.svg`` is pyrox's ``docs/assets/icon-gauss.svg``, copied unchanged.
Re-run after editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pyrox_gauss_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the pyrox-gauss dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

LAYER = ("GeoML", "the pyrox workspace; nothing outside it imports pyrox-gauss")
# (package, what it is, the lines of pyrox_gauss names it uses)
PACKAGES = [
    (
        "pyrox-gp",
        "Gaussian processes",
        ["MultivariateNormal", "MultivariateNormalPrecision"],
    ),
    (
        "pyrox-ssm",
        "state-space models",
        ["MultivariateNormal", "flows: LGSSM · MaskedLGSSM · LGSSMFactory"],
    ),
    ("pyrox-lgm", "latent Gaussian models", ["GMRFDistribution"]),
]
BASE = ["gaussx", "lineax", "NumPyro", "JAX", "Equinox", "einx", "jaxtyping"]

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

BOX_X, BOX_W, BOX_Y, BOX_H = 20, 920, 140, 158
CARD_W, CARD_H, CARD_GAP = 290, 92, 12


def icon(x: float, y: float, theme: str) -> str:
    """pyrox-gauss's icon at 48 px, with its gradient id made unique per theme."""
    src = (HERE / "_icon.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    body = body.replace("icon-gauss-g", f"prxgauss-{theme}").strip()
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
        # pyrox-gauss
        f'<rect x="250" y="18" width="460" height="74" rx="14" fill="{t["card"]}" '
        f'stroke="{t["accent"]}" stroke-width="2"/>',
        icon(266, 31, theme),
        f'<text x="328" y="52" font-family="{SANS}" font-size="22" '
        f'font-weight="700" fill="{t["ink"]}">pyrox-gauss</text>',
        f'<text x="328" y="74" font-family="{SANS}" font-size="13" '
        f'fill="{t["muted"]}">Gaussian NumPyro distributions on gaussx operators'
        "</text>",
    ]
    # the layer box, with its arrow up to pyrox-gauss
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
        f'stroke-width="2.2" marker-end="url(#ea-{theme})"/>'
    )
    row_w = len(PACKAGES) * CARD_W + (len(PACKAGES) - 1) * CARD_GAP
    x = BOX_X + (BOX_W - row_w) / 2
    y = BOX_Y + 48
    for name, what, uses in PACKAGES:
        out.append(
            f'<rect x="{x}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="10" '
            f'fill="{t["card"]}" stroke="{t["accent"]}" stroke-width="1.6"/>'
            f'<text x="{x + CARD_W / 2}" y="{y + 23}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="13.5" font-weight="700" '
            f'fill="{t["ink"]}">{name}</text>'
            f'<text x="{x + CARD_W / 2}" y="{y + 42}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="12" fill="{t["muted"]}">'
            f"{escape(what)}</text>"
        )
        for i, line in enumerate(uses):
            out.append(
                f'<text x="{x + CARD_W / 2}" y="{y + 61 + 16 * i}" '
                'text-anchor="middle" '
                f'font-family="{MONO}" font-size="11" fill="{t["uses"]}">'
                f"{escape(line)}</text>"
            )
        x += CARD_W + CARD_GAP
    # what pyrox-gauss builds on
    by = BOX_Y + BOX_H + 22
    fill, stroke = t["base"]
    names = " · ".join(
        f'<tspan font-weight="700" fill="{t["ink"]}">{escape(n)}</tspan>' for n in BASE
    )
    out.append(
        f'<rect x="20" y="{by}" width="920" height="40" rx="10" fill="{fill}" '
        f'stroke="{stroke}"/>'
        f'<text x="480" y="{by + 25.5}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="14" fill="{t["muted"]}">pyrox-gauss builds on: {names}</text>'
    )
    # legend
    ly = by + 66
    out.append(
        f'<rect x="40" y="{ly - 12}" width="26" height="16" rx="4" fill="none" '
        f'stroke="{t["accent"]}" stroke-width="1.6"/>'
        f'<text x="74" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imports pyrox-gauss (the names it uses below; '
        "flows: through that extra)</text>"
    )
    height = ly + 22
    label = (
        "Packages that import pyrox-gauss: pyrox-gp (MultivariateNormal and "
        "MultivariateNormalPrecision), pyrox-ssm (MultivariateNormal, and LGSSM, "
        "MaskedLGSSM and LGSSMFactory through its flows extra) and pyrox-lgm "
        "(GMRFDistribution), all in the pyrox workspace in GeoML. pyrox-gauss "
        "builds on gaussx, lineax, NumPyro, JAX, Equinox, einx and jaxtyping."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>pyrox-gauss in the GeoML layer</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write the light and dark ``pyrox_gauss_ecosystem`` SVGs."""
    for theme in THEMES:
        path = HERE / f"pyrox_gauss_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
