"""Write the pyrox-lgm dependency figure as light and dark SVGs.

The layout follows pyrox-ssm's ecosystem figure (``pyrox_ssm_ecosystem/``):
pyrox-lgm at the top, what imports it below, and what pyrox-lgm builds on
underneath. pyrox-lgm is ``packages/pyrox-lgm`` in the pyrox workspace. No
package imports it on pyrox's ``main``, lazily or otherwise, so the layer box
holds no card and no arrow. It names instead the three sibling packages whose
import-guard tests forbid importing pyrox-lgm. ``_icon.svg`` is pyrox's
``docs/assets/icon-lgm.svg``, copied unchanged. Re-run after editing the data
below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pyrox_lgm_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the pyrox-lgm dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

LAYER = ("GeoML", "the pyrox workspace; no package imports pyrox-lgm")
# The siblings whose import-guard tests forbid importing pyrox_lgm.
GUARDED = ["pyrox-gauss", "pyrox-ssm", "pyrox-xtreme"]
BASE = [
    "pyrox",
    "pyrox-gauss",
    "gaussx",
    "lineax",
    "kernellib",
    "functax",
    "argminax",
    "optimistix",
    "matfree",
    "JAX",
    "Equinox",
    "NumPyro",
    "einx",
    "xarray (extra)",
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

BOX_X, BOX_W, BOX_Y, BOX_H = 20, 920, 140, 126
CHIP_H, CHIP_GAP = 26, 12


def chip_width(label: str) -> float:
    """Width of a monospace chip at 13 px."""
    return 7.81 * len(label) + 24


def icon(x: float, y: float, theme: str) -> str:
    """pyrox-lgm's icon at 48 px, with its gradient id made unique per theme."""
    src = (HERE / "_icon.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    body = body.replace("icon-lgm-g", f"prxlgm-{theme}").strip()
    return (
        f'<svg x="{x}" y="{y}" width="48" height="48" viewBox="0 0 128 128">'
        f"{body}</svg>"
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    out = [
        # pyrox-lgm
        f'<rect x="250" y="18" width="460" height="74" rx="14" fill="{t["card"]}" '
        f'stroke="{t["accent"]}" stroke-width="2"/>',
        icon(266, 31, theme),
        f'<text x="328" y="52" font-family="{SANS}" font-size="22" '
        f'font-weight="700" fill="{t["ink"]}">pyrox-lgm</text>',
        f'<text x="328" y="74" font-family="{SANS}" font-size="13" '
        f'fill="{t["muted"]}">latent Gaussian models for NumPyro</text>',
    ]
    # the layer box: no card and no arrow, since nothing imports pyrox-lgm
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
        f'<text x="480" y="{BOX_Y + 62}" text-anchor="middle" '
        f'font-family="{SANS}" font-size="13" fill="{t["muted"]}">'
        "these siblings forbid importing pyrox-lgm, each in an import-guard test"
        "</text>"
    )
    widths = [chip_width(name) for name in GUARDED]
    row_w = sum(widths) + CHIP_GAP * (len(GUARDED) - 1)
    x = BOX_X + (BOX_W - row_w) / 2
    y = BOX_Y + 80
    for name, w in zip(GUARDED, widths, strict=True):
        out.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{CHIP_H}" rx="7" '
            f'fill="{t["card"]}" stroke="{t["accent"]}" stroke-width="1.2" '
            f'stroke-dasharray="5 4"/>'
            f'<text x="{x + w / 2:.1f}" y="{y + 17.5}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="13" fill="{t["uses"]}">'
            f"{escape(name)}</text>"
        )
        x += w + CHIP_GAP
    # what pyrox-lgm builds on, on two lines: the GeoStack libraries, then the rest
    by = BOX_Y + BOX_H + 22
    fill, stroke = t["base"]
    lines = [BASE[:9], BASE[9:]]
    out.append(
        f'<rect x="20" y="{by}" width="920" height="62" rx="10" fill="{fill}" '
        f'stroke="{stroke}"/>'
    )
    for i, line in enumerate(lines):
        names = " · ".join(
            f'<tspan font-weight="700" fill="{t["ink"]}">{escape(n)}</tspan>'
            for n in line
        )
        lead = "pyrox-lgm builds on: " if i == 0 else ""
        out.append(
            f'<text x="480" y="{by + 25.5 + 22 * i}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="14" fill="{t["muted"]}">'
            f"{lead}{names}</text>"
        )
    # note
    ly = by + 88
    out.append(
        f'<text x="40" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">no arrow: nothing imports pyrox-lgm, and pyrox-lgm '
        "never imports pyrox-gp, its covariance-form sibling.</text>"
    )
    height = ly + 22
    label = (
        "No package imports pyrox-lgm. In the pyrox workspace in GeoML, "
        "pyrox-gauss, pyrox-ssm and pyrox-xtreme forbid importing it in their "
        "import-guard tests. pyrox-lgm builds on pyrox, pyrox-gauss, gaussx, "
        "lineax, kernellib, functax, argminax, optimistix, matfree, JAX, "
        "Equinox, NumPyro and einx, and on xarray through an extra."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>pyrox-lgm in the GeoML layer</title>\n" + "\n".join(out) + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``pyrox_lgm_ecosystem-light.svg`` and ``pyrox_lgm_ecosystem-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"pyrox_lgm_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
