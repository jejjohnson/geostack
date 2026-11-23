"""Write the sesmax-core dependency figure as light and dark SVGs.

The layout follows pyrox-gauss's figure (``pyrox_gauss_ecosystem/``):
sesmax-core at the top, its dependents below it, and what it builds on
underneath. Nothing outside the sesmax workspace imports sesmax-core, so the
dependents are the components the design plans (geostack's
``docs/design/esmax.md``), drawn dashed; a card turns solid, with the names it
uses, when its package imports the core. The core's own modules under each
foundation are read from its ``import`` lines on sesmax's ``main``.
``_icon.svg`` is sesmax's ``docs/assets/packages/sesmax-core/icon.svg``, copied
unchanged, and ``_icon-<part>.svg`` are the icons drawn for the planned
packages. Re-run after editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/sesmax_core_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the sesmax-core dependency figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

LAYER = ("GeoModels", "the sesmax workspace: planned, nothing imports the core yet")
# (part, what it is), tier 1 and then the coupler on top of them
COMPONENTS = [
    ("ocean", "shallow water, QG"),
    ("atmos", "simple atmosphere"),
    ("land", "buckets, energy balance"),
    ("ice", "slab sea ice"),
]
COUPLER = ("couple", "fluxes, regridding, coupled runs")
# (package, what the core takes from it, the sesmax_core modules, extra or None)
FOUNDATIONS = [
    ("gridax", "grids, masks, operators", ["forcing · gfd", "vertical · testing"], ""),
    ("gaussx", "conjugate gradients", ["gfd"], ""),
    ("spectraldiffx", "spectral solves", ["testing"], ""),
    ("geonnax", "spatial bases", ["forcing"], ""),
    ("integrix", "IMEX stepping", ["solvers"], ""),
    ("geophysix", "physical constants", ["forcing · testing"], ""),
    ("pipekit", "operators, rollouts", ["operators · train", "terms"], ""),
    ("filterax", "ensemble filters", ["da"], "da"),
    ("vardax", "variational methods", ["da"], "da"),
    ("flowjax", "normalising flows", ["flows"], "flows"),
]
BASE = ["JAX", "Equinox", "diffrax", "paramax", "einops", "jaxtyping"]

THEMES = {
    "light": {
        "accent": "#475569",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "uses": "#334155",
        "base": ("#F3F4F6", "#D1D5DB"),
        "arrow": "#9CA3AF",
    },
    "dark": {
        "accent": "#94A3B8",
        "card": "#111827",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "box": ("#0F172A", "#1F2937"),
        "uses": "#CBD5E1",
        "base": ("#1F2937", "#374151"),
        "arrow": "#6B7280",
    },
}

BOX_X, BOX_W = 20, 920
PLAN_Y, PLAN_H = 140, 236
CARD_W, CARD_H, CARD_GAP = 212, 70, 14
FOUND_W, FOUND_H, FOUND_GAP = 168, 92, 14
DASH = ' stroke-dasharray="6 4"'


def icon(part: str, x: float, y: float, size: int, theme: str) -> str:
    """One package's icon, with its ids made unique per theme."""
    name = f"_icon-{part}.svg" if part else "_icon.svg"
    src = (HERE / name).read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    body = body.replace(f"sesmax-{part or 'core'}-i", f"sxe-{part}-{theme}-").strip()
    return (
        f'<svg x="{x}" y="{y}" width="{size}" height="{size}" '
        f'viewBox="0 0 128 128">{body}</svg>'
    )


def text(
    x: float, y: float, body: str, size: float, fill: str, *, anchor: str = "start"
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{SANS}" '
        f'font-size="{size}" fill="{fill}">{escape(body)}</text>'
    )


def mono(
    x: float, y: float, body: str, size: float, fill: str, *, bold: bool, anchor: str
) -> str:
    weight = ' font-weight="700"' if bold else ""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{MONO}" '
        f'font-size="{size}"{weight} fill="{fill}">{escape(body)}</text>'
    )


def planned(
    part: str, what: str, x: float, y: float, w: float, t: dict, th: str
) -> str:
    """A dashed card for a planned package, with its icon."""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{CARD_H}" rx="10" '
        f'fill="{t["card"]}" stroke="{t["arrow"]}" stroke-width="1.6"{DASH}/>'
        + icon(part, x + 10, y + 15, 40, th)
        + mono(
            x + 58, y + 31, f"sesmax-{part}", 13, t["ink"], bold=True, anchor="start"
        )
        + text(x + 58, y + 50, what, 11.5, t["muted"])
    )


def heading(y: float, title: str, caption: str, t: dict) -> str:
    return (
        f'<text x="{BOX_X + 18}" y="{y + 26}" font-family="{SANS}" font-size="15" '
        f'font-weight="700" fill="{t["ink"]}">{escape(title)}</text>'
        + text(BOX_X + BOX_W - 18, y + 26, caption, 12.5, t["muted"], anchor="end")
    )


def box(y: float, h: float, t: dict) -> str:
    fill, stroke = t["box"]
    return (
        f'<rect x="{BOX_X}" y="{y}" width="{BOX_W}" height="{h}" rx="14" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    out = []
    for key in ("accent", "arrow"):
        out.append(
            f'<defs><marker id="e-{key}-{theme}" viewBox="0 0 10 10" refX="9" '
            'refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0 0 L10 5 L0 10 z" fill="{t[key]}"/></marker></defs>'
        )
    # sesmax-core
    out.append(
        f'<rect x="230" y="18" width="500" height="74" rx="14" fill="{t["card"]}" '
        f'stroke="{t["accent"]}" stroke-width="2"/>'
        + icon("", 246, 31, 48, theme)
        + f'<text x="308" y="52" font-family="{SANS}" font-size="22" '
        f'font-weight="700" fill="{t["ink"]}">sesmax-core</text>'
        + text(308, 74, "what every simple Earth-system model shares", 13, t["muted"])
    )
    # the planned components, and the coupler on top of them
    out.append(box(PLAN_Y, PLAN_H, t) + heading(PLAN_Y, *LAYER, t))
    out.append(
        f'<path d="M480 {PLAN_Y} L480 95" fill="none" stroke="{t["arrow"]}" '
        f'stroke-width="2.2"{DASH} marker-end="url(#e-arrow-{theme})"/>'
    )
    row_w = len(COMPONENTS) * CARD_W + (len(COMPONENTS) - 1) * CARD_GAP
    x0 = BOX_X + (BOX_W - row_w) / 2
    y = PLAN_Y + 44
    cy = y + CARD_H + 34
    bus = y + CARD_H + 18
    cw = 290
    for i, (part, what) in enumerate(COMPONENTS):
        x = x0 + i * (CARD_W + CARD_GAP)
        out.append(planned(part, what, x, y, CARD_W, t, theme))
        mid = x + CARD_W / 2
        out.append(
            f'<path d="M{mid} {bus} L{mid} {y + CARD_H + 3}" fill="none" '
            f'stroke="{t["arrow"]}" stroke-width="1.6"{DASH} '
            f'marker-end="url(#e-arrow-{theme})"/>'
        )
    first = x0 + CARD_W / 2
    last = x0 + row_w - CARD_W / 2
    out.append(
        f'<path d="M{first} {bus} L{last} {bus} M480 {cy} L480 {bus}" fill="none" '
        f'stroke="{t["arrow"]}" stroke-width="1.6"{DASH}/>'
        + planned(*COUPLER, 480 - cw / 2, cy, cw, t, theme)
    )
    # what sesmax-core builds on
    fy = PLAN_Y + PLAN_H + 22
    rows = [FOUNDATIONS[i : i + 5] for i in range(0, len(FOUNDATIONS), 5)]
    fh = 44 + len(rows) * (FOUND_H + FOUND_GAP) + 4
    out.append(
        box(fy, fh, t)
        + heading(fy, "sesmax-core builds on", "its modules that import each one", t)
    )
    for r, row in enumerate(rows):
        row_w = len(row) * FOUND_W + (len(row) - 1) * FOUND_GAP
        x = BOX_X + (BOX_W - row_w) / 2
        y = fy + 44 + r * (FOUND_H + FOUND_GAP)
        for name, what, uses, extra in row:
            mid = x + FOUND_W / 2
            stroke, dash = (t["arrow"], DASH) if extra else (t["accent"], "")
            out.append(
                f'<rect x="{x}" y="{y}" width="{FOUND_W}" height="{FOUND_H}" rx="10" '
                f'fill="{t["card"]}" stroke="{stroke}" stroke-width="1.6"{dash}/>'
                + mono(mid, y + 23, name, 13.5, t["ink"], bold=True, anchor="middle")
                + text(mid, y + 42, what, 12, t["muted"], anchor="middle")
            )
            lines = [*uses, f"extra: {extra}"] if extra else uses
            for i, ln in enumerate(lines):
                out.append(
                    mono(
                        mid,
                        y + 61 + 16 * i,
                        ln,
                        11,
                        t["uses"],
                        bold=False,
                        anchor="middle",
                    )
                )
            x += FOUND_W + FOUND_GAP
    # the third-party strip
    by = fy + fh + 16
    fill, stroke = t["base"]
    names = " · ".join(
        f'<tspan font-weight="700" fill="{t["ink"]}">{escape(n)}</tspan>' for n in BASE
    )
    out.append(
        f'<rect x="{BOX_X}" y="{by}" width="{BOX_W}" height="40" rx="10" '
        f'fill="{fill}" stroke="{stroke}"/>'
        f'<text x="480" y="{by + 25.5}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="14" fill="{t["muted"]}">and on: {names}'
        "; extras: xarray · zarr · orbax</text>"
    )
    # legend
    ly = by + 66
    out.append(
        f'<rect x="40" y="{ly - 12}" width="26" height="16" rx="4" fill="none" '
        f'stroke="{t["accent"]}" stroke-width="1.6"/>'
        + text(74, ly + 1, "imported by the base install", 12.5, t["muted"])
        + f'<rect x="300" y="{ly - 12}" width="26" height="16" rx="4" fill="none" '
        f'stroke="{t["arrow"]}" stroke-width="1.6"{DASH}/>'
        + text(334, ly + 1, "planned, or through the extra it names", 12.5, t["muted"])
    )
    height = ly + 22
    label = (
        "sesmax-core in the GeoModels layer. Planned on top of it, with no import "
        "yet: sesmax-ocean, sesmax-atmos, sesmax-land and sesmax-ice, and "
        "sesmax-couple on top of those. It builds on gridax, gaussx, "
        "spectraldiffx, geonnax, integrix, geophysix and pipekit, on filterax and "
        "vardax through its da extra and flowjax through its flows extra, each "
        "with the modules that import it, and on JAX, Equinox, diffrax, paramax, "
        "einops and jaxtyping."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>sesmax-core in the GeoModels layer</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write the light and dark ``sesmax_core_ecosystem`` SVGs."""
    for theme in THEMES:
        path = HERE / f"sesmax_core_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
