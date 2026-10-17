"""Render the GeoStack README assets: the icon and the two logos.

Every output is a plain SVG written next to this file. The icon shares the
tile of the stack's libraries (gaussx, kernellib, geonnax, pyrox: a 120-unit
rounded square, r = 26, white glyph) in the GeoStack blue-to-teal gradient,
the observation and state hues of the overview figures. Its glyph is the
stack itself, as the four-layer figure draws it: GeoModels on top, GeoML
below, GeoNumerics and GeoStack side by side at the bottom.

The logos come in a light and a dark variant for GitHub's ``<picture>`` /
``prefers-color-scheme`` switch.

Run from the repo root (any Python 3.12+, no dependencies)::

    python docs/assets/render.py
"""

from __future__ import annotations

from pathlib import Path


OUT = Path(__file__).resolve().parent

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# Blue to teal: observations (GeoTensor) to states (xarray).
C1, C2 = "#1D4ED8", "#0D9488"
ACCENT = {"light": "#0F766E", "dark": "#5EEAD4"}
TEXT = {"light": "#1F2937", "dark": "#E5E7EB"}
W = "#FFFFFF"


def glyph_stack() -> str:
    """Four layers: two full-width slabs over two half-width slabs."""
    x0, x1, h, gap = 26, 102, 16, 8
    half = (x1 - x0 - gap) / 2
    rows = [
        [(x0, x1 - x0, 1.0)],  # GeoModels
        [(x0, x1 - x0, 0.8)],  # GeoML
        [(x0, half, 0.6), (x0 + half + gap, half, 0.6)],  # GeoNumerics, GeoStack
    ]
    y0 = 64 - (len(rows) * h + (len(rows) - 1) * gap) / 2
    return "".join(
        f'<rect x="{x}" y="{y0 + i * (h + gap)}" width="{w}" height="{h}" rx="4"'
        f' fill="{W}" fill-opacity="{a}"/>'
        for i, row in enumerate(rows)
        for x, w, a in row
    )


def tile(gid: str, x: float = 0, y: float = 0, size: float = 128) -> str:
    """The gradient tile with the stack glyph, placed at (x, y), ``size`` wide."""
    s = size / 128
    return (
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{C1}"/><stop offset="1" stop-color="{C2}"/>'
        "</linearGradient></defs>"
        f'<g transform="translate({x} {y}) scale({s:g})">'
        f'<rect x="4" y="4" width="120" height="120" rx="26" fill="url(#{gid})"/>'
        f"{glyph_stack()}</g>"
    )


def svg(width: int, height: int, label: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}" role="img" aria-label="{label}">\n'
        f"<title>{label}</title>\n{body}\n</svg>\n"
    )


def write_icon() -> None:
    (OUT / "icon.svg").write_text(svg(128, 128, "GeoStack", tile("ic")))


def write_logos() -> None:
    for mode in ("light", "dark"):
        body = tile(f"logo-{mode}", x=6, y=6) + (
            f'<text x="154" y="92" font-family="{SANS}" font-size="72"'
            f' font-weight="700" letter-spacing="-1.5" fill="{TEXT[mode]}">'
            f'Geo<tspan fill="{ACCENT[mode]}">Stack</tspan></text>'
        )
        (OUT / f"logo-{mode}.svg").write_text(svg(520, 140, "GeoStack", body))


if __name__ == "__main__":
    write_icon()
    write_logos()
