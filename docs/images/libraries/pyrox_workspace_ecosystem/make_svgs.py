"""Write the pyrox workspace's dependency figure as light and dark SVGs.

The seven packages of the pyrox workspace and the imports between them: the
four that write a prior on a function on top, the core, pyrox-gauss and
pyrox-xtreme below, and what they all build on underneath. The edges and the
libraries on each card are its ``dependencies`` and its ``import`` lines on
pyrox's ``main``. JAX, Equinox, NumPyro, lineax and einx are in the strip and not
on the cards, because most of the packages import them. ``_icon*.svg`` are pyrox's
``docs/assets/icon*.svg``, copied unchanged. Re-run after editing the data
below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pyrox_workspace_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the pyrox workspace's dependency figure as two SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"


@dataclass(frozen=True)
class Package:
    """One card: the package, what it holds and the libraries it imports."""

    name: str
    icon: str
    what: str
    libs: tuple[str, ...]
    extras: str = ""


TOP = [
    Package(
        "pyrox-nn",
        "-nn",
        "Bayesian networks",
        ("geonnax · kernellib", "functax · gaussx · argminax"),
        "extras: optax, pandas, xarray",
    ),
    Package(
        "pyrox-gp",
        "-gp",
        "Gaussian processes",
        ("kernellib · functax · flowjax", "gaussx · argminax · optimistix"),
        "extra: gauss_flows",
    ),
    Package(
        "pyrox-lgm",
        "-lgm",
        "latent Gaussian models",
        ("kernellib · functax · matfree", "gaussx · argminax · optimistix"),
        "extra: xarray",
    ),
    Package(
        "pyrox-ssm",
        "-ssm",
        "state-space models",
        ("gaussx", "argminax · optimistix"),
        "extras: filterax, gauss_flows",
    ),
]
CORE = Package("pyrox", "", "the Equinox-to-NumPyro core", ("extra: optax",))
GAUSS = Package("pyrox-gauss", "-gauss", "Gaussian distributions", ("gaussx",))
XTREME = Package(
    "pyrox-xtreme",
    "-xtreme",
    "extremes, point processes",
    ("xarray · used by plumax (extra)",),
)
USES_CORE = ("pyrox-nn", "pyrox-gp", "pyrox-lgm", "pyrox-ssm")
USES_GAUSS = ("pyrox-gp", "pyrox-lgm", "pyrox-ssm")
BASE = [
    ("GeoML and GeoNumerics", "kernellib · geonnax · gaussx · functax · argminax"),
    ("Foundations", "JAX · Equinox · NumPyro · lineax · einx"),
]

THEMES = {
    "light": {
        "core": "#EA580C",
        "gauss": "#475569",
        "core_fill": "#FFF7ED",
        "card": "#FFFFFF",
        "line": "#CBD5E1",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "uses": "#334155",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "core": "#FB923C",
        "gauss": "#94A3B8",
        "core_fill": "#2A1408",
        "card": "#111827",
        "line": "#374151",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "box": ("#0F172A", "#1F2937"),
        "uses": "#CBD5E1",
        "base": ("#1F2937", "#374151"),
    },
}

WIDTH = 1000
BOX_X, BOX_Y, BOX_W, BOX_H = 16, 16, 968, 388
TOP_Y, TOP_H, TOP_W, GAP = 66, 136, 224, 14
LOW_Y, LOW_H, LOW_W, LOW_GAP = 290, 92, 300, 19
GAUSS_BUS, CORE_BUS = 228, 256
LEFT = BOX_X + (BOX_W - 4 * TOP_W - 3 * GAP) / 2


def text(
    x: float,
    y: float,
    body: str,
    size: float,
    fill: str,
    *,
    font: str = SANS,
    bold: bool = False,
    anchor: str = "start",
) -> str:
    weight = ' font-weight="700"' if bold else ""
    return (
        f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" font-family="{font}" '
        f'font-size="{size}"{weight} fill="{fill}">{escape(body)}</text>'
    )


def icon(suffix: str, x: float, y: float, theme: str) -> str:
    """One package's icon at 44 px, with its gradient id made unique."""
    src = (HERE / f"_icon{suffix}.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    (gid,) = re.findall(r'linearGradient id="([^"]+)"', body)
    body = body.replace(gid, f"prw{suffix}-{theme}").strip()
    return (
        f'<svg x="{x:g}" y="{y:g}" width="44" height="44" viewBox="0 0 128 128">'
        f"{body}</svg>"
    )


def card(p: Package, x: float, y: float, w: float, h: float, theme: str) -> str:
    t = THEMES[theme]
    fill, stroke, width = t["card"], t["line"], 1.5
    if p is CORE:
        fill, stroke, width = t["core_fill"], t["core"], 2
    elif p is GAUSS:
        stroke, width = t["gauss"], 2
    out = [
        f'<rect x="{x:g}" y="{y:g}" width="{w}" height="{h}" rx="12" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>',
        icon(p.icon, x + 12, y + 12, theme),
        text(x + 66, y + 31, p.name, 15, t["ink"], font=MONO, bold=True),
        text(x + 66, y + 50, p.what, 12.5, t["muted"]),
    ]
    ly = y + 79
    for line in p.libs:
        out.append(text(x + 14, ly, line, 12, t["uses"]))
        ly += 18
    if p.extras:
        out.append(text(x + 14, ly + 4, p.extras, 11.5, t["muted"]))
    return "".join(out)


def bus(sources: list[float], target: float, y: float, kind: str, theme: str) -> str:
    """Drops from the cards to a horizontal line, and one arrow off it."""
    colour = THEMES[theme][kind]
    ends = [*sources, target]
    d = "".join(f"M{x:g} {TOP_Y + TOP_H} L{x:g} {y} " for x in sources)
    d += f"M{min(ends):g} {y} L{max(ends):g} {y}"
    return (
        f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" '
        'stroke-linecap="round"/>'
        f'<path d="M{target:g} {y} L{target:g} {LOW_Y - 3}" fill="none" '
        f'stroke="{colour}" stroke-width="2" marker-end="url(#w-{kind}-{theme})"/>'
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    fill, stroke = t["box"]
    markers = "".join(
        f'<marker id="w-{k}-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t[k]}"/></marker>'
        for k in ("core", "gauss")
    )
    out = [
        f"<defs>{markers}</defs>",
        f'<rect x="{BOX_X}" y="{BOX_Y}" width="{BOX_W}" height="{BOX_H}" rx="16" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        text(BOX_X + 20, BOX_Y + 31, "The pyrox workspace", 16, t["ink"], bold=True),
        text(
            BOX_X + BOX_W - 20,
            BOX_Y + 31,
            "an arrow points at the package that is imported",
            12.5,
            t["muted"],
            anchor="end",
        ),
    ]
    centre = {p.name: LEFT + i * (TOP_W + GAP) + TOP_W / 2 for i, p in enumerate(TOP)}
    low_x = [LEFT + i * (LOW_W + LOW_GAP) for i in range(3)]
    core_to, gauss_to = low_x[0] + LOW_W / 2, low_x[1] + LOW_W / 2
    core_from = [centre[n] - 30 for n in USES_CORE]
    gauss_from = [centre[n] + 30 for n in USES_GAUSS]
    out.append(bus(gauss_from, gauss_to, GAUSS_BUS, "gauss", theme))
    out.append(bus(core_from, core_to, CORE_BUS, "core", theme))
    for p in TOP:
        out.append(card(p, centre[p.name] - TOP_W / 2, TOP_Y, TOP_W, TOP_H, theme))
    for p, x in zip((CORE, GAUSS, XTREME), low_x, strict=True):
        out.append(card(p, x, LOW_Y, LOW_W, LOW_H, theme))
    # what the workspace builds on
    by = BOX_Y + BOX_H + 16
    fill, stroke = t["base"]
    w = (BOX_W - 14) / 2
    for i, (head, names) in enumerate(BASE):
        x = BOX_X + i * (w + 14)
        out.append(
            f'<rect x="{x:g}" y="{by}" width="{w:g}" height="58" rx="12" '
            f'fill="{fill}" stroke="{stroke}"/>'
            + text(x + 18, by + 24, head, 13.5, t["ink"], bold=True)
            + text(x + 18, by + 44, names, 13, t["muted"])
        )
    height = by + 58 + 16
    label = (
        "The seven packages of the pyrox workspace. pyrox-nn, pyrox-gp, pyrox-lgm "
        "and pyrox-ssm import the core pyrox; pyrox-gp, pyrox-lgm and pyrox-ssm "
        "also import pyrox-gauss; pyrox-gauss and pyrox-xtreme import no sibling."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>The packages of the pyrox workspace</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``pyrox_workspace_ecosystem-light.svg`` and ``-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"pyrox_workspace_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
