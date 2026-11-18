"""Write vardax's module-import figure as light and dark SVGs.

vardax keeps its code in 22 areas under ``src/vardax/_src/`` and exports almost
all of it from the package root, so the cards group the areas by job: each card
names its areas and the libraries they import (dashed when through an extra).
Arrows run only where an import crosses between cards and read "imports";
every one is at module level on ``main``, though some areas add imports inside
functions too. ``_types``, ``errors``,
``protocols`` and ``utils`` are imported across the package and are named in
the strips below. The edges are the ``import`` lines on vardax's ``main``.
Re-run after editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/vardax_imports/make_svgs.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write vardax's module-import figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

COLS = (40, 340, 640)
ROWS = (72, 210, 426, 600)
W, H = 280, 112
H2 = 129  # the second row, whose 4DVarNet card lists its areas on two lines
# key: (title, areas, what it does, [(import, through an extra)], x, y, width)
CARDS = {
    "cycle": (
        "cycle",
        "cycle",
        "builds pipekit-cycle cycles from any model",
        [("pipekit-cycle", False)],
        COLS[0],
        ROWS[0],
        W,
    ),
    "adapters": (
        "adapters",
        "adapters",
        "a pipekit-train adapter",
        [("pipekit-ops", False)],
        COLS[1],
        ROWS[0],
        W,
    ),
    "examples": (
        "examples",
        "examples",
        "notebook helpers, not library API",
        [("optax", False)],
        COLS[2],
        ROWS[0],
        W,
    ),
    "methods": (
        "the variational methods",
        "models/ · optimal_estimation · costs · minimisers",
        "OI, 3DVar, the 4DVars, neural-field 4DVar, Rodgers OE",
        [
            ("optimistix", False),
            ("lineax", False),
            ("gaussx", False),
            ("argminax", False),
            ("integrix", False),
            ("optax", False),
        ],
        COLS[0],
        ROWS[1],
        COLS[1] + W - COLS[0],
    ),
    "net": (
        "4DVarNet",
        "model · solver · grad_mod\npriors · training",
        "the learned solver and its priors",
        [("argminax", False), ("optimistix", False), ("optax", False)],
        COLS[2],
        ROWS[1],
        W,
    ),
    "obs": (
        "observation operators",
        "obs_operators/",
        "h, its tangent and adjoint",
        [("lineax", False), ("gaussx", False), ("pipekit-cycle", False)],
        COLS[0],
        ROWS[2],
        W,
    ),
    "posterior": (
        "posterior",
        "posterior/",
        "uncertainty at the analysis",
        [("lineax", False), ("gaussx", False), ("argminax", False)],
        COLS[1],
        ROWS[2],
        W,
    ),
    "controls": (
        "controls",
        "basis/ · parameterization",
        "reduced bases and control maps",
        [("geonnax", False)],
        COLS[2],
        ROWS[2],
        W,
    ),
    "amortized": (
        "amortized",
        "amortized/",
        "q(x | y) in one forward pass",
        [],
        COLS[0],
        ROWS[3],
        W,
    ),
    "dynamics": (
        "dynamical priors",
        "priors_dynamical",
        "ODE priors and forward models",
        [("diffrax", False)],
        COLS[1],
        ROWS[3],
        W,
    ),
    "adjoints": (
        "adjoints",
        "adjoints/",
        "optimistix and argminax adjoints",
        [("optimistix", False), ("argminax", False)],
        COLS[2],
        ROWS[3],
        W,
    ),
}
# (from, to, inside a function, route): "down" and "up" are vertical at the
# given x; "under" leaves the source's bottom at one x and enters the target's
# bottom at another, below the row; "elbow" goes down, across and down.
ARROWS = [
    ("examples", "net", False, ("down", COLS[2] + W / 2)),
    ("methods", "obs", False, ("down", COLS[0] + 90)),
    ("obs", "methods", False, ("up", COLS[0] + 190)),
    ("methods", "posterior", False, ("down", COLS[1] + 90)),
    ("posterior", "methods", False, ("up", COLS[1] + 190)),
    ("posterior", "obs", False, ("under", (COLS[1] + 60, COLS[0] + 210))),
    ("posterior", "controls", False, ("under", (COLS[1] + 220, COLS[2] + 70))),
    ("methods", "controls", False, ("elbow", COLS[1] + W - 20)),
]
NOTES = {
    "cycle": "imports nothing else from vardax",
    "adapters": "imports only the shared areas below",
    "amortized": "imports only the shared areas below",
    "dynamics": "imports only the shared areas below",
    "adjoints": "imports only the shared areas below",
}
SHARED = [
    "_types · errors · protocols: imported across the package; "
    "protocols re-exports pipekit-cycle's",
    "utils: validation gates, Lorenz systems, patches, plots; "
    "imports posterior, and the root for deprecations",
]
THROUGHOUT = ["JAX", "Equinox", "jaxtyping", "einx"]
UTILS_EXTRAS = [("xarray", "data"), ("matplotlib", "viz")]

THEMES = {
    "light": {
        "accent": "#BE123C",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "chip": ("#FFFFFF", "#D1D5DB", "#1F2937"),
        "extra": "#9CA3AF",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#FB7185",
        "card": "#111827",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "box": ("#0F172A", "#1F2937"),
        "chip": ("#111827", "#374151", "#E5E7EB"),
        "extra": "#6B7280",
        "base": ("#1F2937", "#374151"),
    },
}


def height(y: float) -> float:
    """Height of the cards in the row at ``y``."""
    return H2 if y == ROWS[1] else H


def chip_width(label: str) -> float:
    return 7.51 * len(label) + 19.5


def chip(x: float, y: float, label: str, extra: bool, t: dict) -> str:
    fill, stroke, ink = t["chip"]
    w = chip_width(label)
    edge = (
        f' stroke="{t["extra"]}" stroke-dasharray="5 4"'
        if extra
        else f' stroke="{stroke}"'
    )
    return (
        f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" rx="6" '
        f'fill="{fill}"{edge}/>'
        f'<text x="{x + w / 2:.1f}" y="{y + 15}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="12.5" fill="{ink}">{escape(label)}</text>'
    )


def card(key: str, t: dict) -> list[str]:
    title, areas, what, items, x, y, w = CARDS[key]
    h = height(y)
    lines = areas.split("\n")
    shift = 17 * (len(lines) - 1)
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{t["card"]}" '
        f'stroke="{t["accent"]}" stroke-width="1.6"/>'
        f'<text x="{x + 14}" y="{y + 23}" font-family="{SANS}" font-size="14.5" '
        f'font-weight="700" fill="{t["ink"]}">{escape(title)}</text>'
    ]
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x + 14}" y="{y + 42 + 17 * i}" font-family="{MONO}" '
            f'font-size="11.5" fill="{t["accent"]}">{escape(line)}</text>'
        )
    out.append(
        f'<text x="{x + 14}" y="{y + 60 + shift}" font-family="{SANS}" '
        f'font-size="12" fill="{t["muted"]}">{escape(what)}</text>'
    )
    cx = x + 14
    if not items:
        out.append(
            f'<text x="{cx}" y="{y + 89 + shift}" font-family="{SANS}" '
            f'font-size="12" font-style="italic" fill="{t["muted"]}">only the '
            "libraries used throughout</text>"
        )
    for label, extra in items:
        out.append(chip(cx, y + 74 + shift, label, extra, t))
        cx += chip_width(label) + 8
    if key in NOTES:
        out.append(
            f'<text x="{x + w / 2}" y="{y + h + 17}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="11.5" fill="{t["muted"]}">'
            f"{escape(NOTES[key])}</text>"
        )
    return out


def arrow(src: str, dst: str, lazy: bool, route: tuple, t: dict, theme: str) -> str:
    sy = CARDS[src][5]
    _, _, _, _, dx, dy, dw = CARDS[dst]
    sh, dh = height(sy), height(dy)
    kind, at = route
    if kind == "down":
        d = f"M{at} {sy + sh} L{at} {dy - 3}"
    elif kind == "up":
        d = f"M{at} {sy} L{at} {dy + dh + 3}"
    elif kind == "under":
        x0, x1 = at
        low = sy + sh + 24
        d = f"M{x0} {sy + sh} L{x0} {low} L{x1} {low} L{x1} {dy + dh + 3}"
    else:  # elbow: down from the source at x, then across to the target's centre
        mid = (sy + sh + dy) / 2
        d = f"M{at} {sy + sh} L{at} {mid} L{dx + dw / 2} {mid} L{dx + dw / 2} {dy - 3}"
    dash = ' stroke-dasharray="6 4"' if lazy else ""
    return (
        f'<path d="{d}" fill="none" stroke="{t["accent"]}" stroke-width="2"{dash} '
        f'marker-end="url(#im-{theme})"/>'
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    fill, stroke = t["box"]
    box_h = ROWS[3] + H + 40 - 20
    out = [
        f'<defs><marker id="im-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["accent"]}"/></marker></defs>',
        f'<rect x="20" y="20" width="920" height="{box_h}" rx="14" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="40" y="48" font-family="{SANS}" font-size="15" font-weight="700" '
        f'fill="{t["ink"]}">vardax</text>',
        f'<text x="920" y="48" text-anchor="end" font-family="{SANS}" '
        f'font-size="12.5" fill="{t["muted"]}">22 areas under _src/, grouped by '
        "job; arrows read “imports”</text>",
    ]
    for src, dst, lazy, route in ARROWS:
        out.append(arrow(src, dst, lazy, route, t, theme))
    for key in CARDS:
        out += card(key, t)
    # the strips: shared areas, then the libraries used throughout
    by = 20 + box_h + 20
    bf, bs = t["base"]
    out.append(
        f'<rect x="20" y="{by}" width="920" height="62" rx="10" fill="{bf}" '
        f'stroke="{bs}"/>'
    )
    for i, line in enumerate(SHARED):
        head, rest = line.split(": ", 1)
        out.append(
            f'<text x="40" y="{by + 25 + 22 * i}" font-family="{SANS}" '
            f'font-size="13" fill="{t["muted"]}">'
            f'<tspan font-family="{MONO}" font-weight="700" fill="{t["ink"]}">'
            f"{escape(head)}</tspan>: {escape(rest)}</text>"
        )
    ty = by + 62 + 14
    names = " · ".join(
        f'<tspan font-weight="700" fill="{t["ink"]}">{n}</tspan>' for n in THROUGHOUT
    )
    extras = ", ".join(
        f'<tspan font-weight="700" fill="{t["ink"]}">{lib}</tspan> ({extra} extra)'
        for lib, extra in UTILS_EXTRAS
    )
    out.append(
        f'<rect x="20" y="{ty}" width="920" height="40" rx="10" fill="{bf}" '
        f'stroke="{bs}"/>'
        f'<text x="480" y="{ty + 25.5}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="13.5" fill="{t["muted"]}">throughout: {names}; utils imports '
        f"{extras} inside its functions</text>"
    )
    ly = ty + 66
    fill_c, stroke_c, _ = t["chip"]
    out.append(
        f'<rect x="40" y="{ly - 13}" width="30" height="18" rx="5" fill="{fill_c}" '
        f'stroke="{stroke_c}"/>'
        f'<text x="78" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">a base dependency</text>'
        f'<path d="M260 {ly - 4} L294 {ly - 4}" stroke="{t["accent"]}" '
        f'stroke-width="2" marker-end="url(#im-{theme})"/>'
        f'<text x="302" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imports</text>'
    )
    height = ly + 22
    label = (
        "vardax's areas grouped by job and their imports: the variational methods "
        "import the observation operators, the controls and the posterior; the "
        "observation operators and the posterior import the methods back; "
        "examples imports the 4DVarNet training; cycle imports nothing else from "
        "vardax, and adapters, amortized, the dynamical priors and adjoints import "
        "only the shared areas."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>vardax's modules and their imports</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``vardax_imports-light.svg`` and ``vardax_imports-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"vardax_imports-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
