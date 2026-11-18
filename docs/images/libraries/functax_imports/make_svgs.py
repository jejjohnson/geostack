"""Write functax's module-import figure as light and dark SVGs.

Each public area of functax is a card listing the libraries it imports (dashed
when through an extra), and arrows between cards read "imports". The edges are
the ``import`` lines in ``src/functax/_src/<area>/`` on functax's ``main``,
found by an AST pass and checked by hand; ``TYPE_CHECKING`` imports are not
edges. geometry and quad import each other, so they share the floor. The
layout follows argminax's code map (``argminax_imports/``). Re-run after
editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/functax_imports/make_svgs.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write functax's module-import figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# (area, what it does, [(import, through an extra)]), left to right
TOP = [
    ("project", "fits coefficients to data", [("lineax", True)]),
    (
        "interp",
        "plans: a read-out, its adjoint and a fit",
        [("lineax", True), ("numpy", False), ("scipy", True)],
    ),
    ("transform", "values to coefficients and back", [("numpy", False)]),
    ("basis", "the basis functions", [("numpy", False)]),
    ("diffops", "derivatives by autodiff", []),
]
FLOOR = [
    ("quad", "rules as data: nodes and weights", [("numpy", False)]),
    (
        "geometry",
        "where s lives; the affine kernels need NumPy alone",
        [("numpy", False)],
    ),
]
# imports between top cards, drawn straight between neighbours or as an arc
SIDEWAYS = [("project", "interp"), ("interp", "transform"), ("transform", "basis")]
ARCS = [("interp", "basis")]
# imports from a top card down to the floor
DOWN = {
    "interp": ["quad", "geometry"],
    "transform": ["quad", "geometry"],
    "basis": ["quad", "geometry"],
    "diffops": ["geometry"],
}
NOTES: dict[str, str] = {}
THROUGHOUT = ["JAX", "einx", "jaxtyping"]

THEMES = {
    "light": {
        "accent": "#1D4ED8",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "floor": ("#EFF6FF", "#BFDBFE"),
        "chip": ("#FFFFFF", "#D1D5DB", "#1F2937"),
        "extra": "#9CA3AF",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#60A5FA",
        "card": "#111827",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "box": ("#0F172A", "#1F2937"),
        "floor": ("#172554", "#1E3A8A"),
        "chip": ("#111827", "#374151", "#E5E7EB"),
        "extra": "#6B7280",
        "base": ("#1F2937", "#374151"),
    },
}

X0, X1 = 40, 920
CARD_GAP = 16
TOP_W = (X1 - X0 - CARD_GAP * (len(TOP) - 1)) / len(TOP)
TOP_Y, TOP_H = 96, 138
FLOOR_Y, FLOOR_H = 320, 90
FLOOR_W = (X1 - X0 - 3 * CARD_GAP) / 2


def chip_width(label: str) -> float:
    return 7.51 * len(label) + 19.5


def chips(x0: float, x1: float, y: float, items, t: dict) -> list[str]:
    """Lay chips out from x0, wrapping at x1."""
    out, x = [], x0
    for label, extra in items:
        w = chip_width(label)
        if x + w > x1:
            x, y = x0, y + 28
        fill, stroke, ink = t["chip"]
        dash = (
            f' stroke-dasharray="5 4" stroke="{t["extra"]}"'
            if extra
            else f' stroke="{stroke}"'
        )
        out.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" rx="6" '
            f'fill="{fill}"{dash}/>'
            f'<text x="{x + w / 2:.1f}" y="{y + 15}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="12.5" fill="{ink}">{escape(label)}</text>'
        )
        x += w + 8
    return out


def wrap(text: str, width: int) -> list[str]:
    line, lines = "", []
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line]


def card(x, y, w, h, name, what, items, t, width_chars) -> list[str]:
    out = [
        f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="10" '
        f'fill="{t["card"]}" stroke="{t["accent"]}" stroke-width="1.6"/>'
        f'<text x="{x + 14:.1f}" y="{y + 24}" font-family="{MONO}" font-size="15" '
        f'font-weight="700" fill="{t["ink"]}">{name}</text>'
    ]
    lines = wrap(what, width_chars)
    for i, ln in enumerate(lines):
        out.append(
            f'<text x="{x + 14:.1f}" y="{y + 43 + 15 * i}" font-family="{SANS}" '
            f'font-size="12" fill="{t["muted"]}">{escape(ln)}</text>'
        )
    return out + chips(x + 14, x + w - 10, y + 50 + 15 * len(lines), items, t)


def arrow(d: str, t: dict, theme: str) -> str:
    return (
        f'<path d="{d}" fill="none" stroke="{t["accent"]}" stroke-width="1.8" '
        f'marker-end="url(#im-{theme})"/>'
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    fill, stroke = t["box"]
    box_h = FLOOR_Y + FLOOR_H + 24 - 20
    out = [
        f'<defs><marker id="im-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["accent"]}"/></marker></defs>',
        f'<rect x="20" y="20" width="920" height="{box_h}" rx="14" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="40" y="48" font-family="{SANS}" font-size="15" font-weight="700" '
        f'fill="{t["ink"]}">functax</text>',
        f'<text x="920" y="48" text-anchor="end" font-family="{SANS}" '
        f'font-size="12.5" fill="{t["muted"]}">one public module per area; '
        "arrows read “imports”</text>",
    ]
    xs = {}
    for i, (name, what, items) in enumerate(TOP):
        x = X0 + i * (TOP_W + CARD_GAP)
        xs[name] = x
        out += card(x, TOP_Y, TOP_W, TOP_H, name, what, items, t, 24)
        if name in NOTES:
            for j, ln in enumerate(wrap(NOTES[name], 28)):
                out.append(
                    f'<text x="{x + TOP_W / 2:.1f}" y="{TOP_Y + TOP_H + 17 + 14 * j}" '
                    f'text-anchor="middle" font-family="{SANS}" font-size="11.5" '
                    f'fill="{t["muted"]}">{escape(ln)}</text>'
                )
    # the floor: geometry and quad, which import each other
    ff, fs = t["floor"]
    out.append(
        f'<rect x="{X0 - 8}" y="{FLOOR_Y - 10}" width="{X1 - X0 + 16}" '
        f'height="{FLOOR_H + 20}" rx="12" fill="{ff}" stroke="{fs}" '
        'stroke-width="1.2"/>'
    )
    fx = {}
    for i, (name, what, items) in enumerate(FLOOR):
        x = X0 + i * (FLOOR_W + 3 * CARD_GAP)
        fx[name] = x
        out += card(x, FLOOR_Y, FLOOR_W, FLOOR_H, name, what, items, t, 60)
    gx0, gx1 = X0 + FLOOR_W, X0 + FLOOR_W + 3 * CARD_GAP
    mid = FLOOR_Y + FLOOR_H / 2
    out.append(arrow(f"M{gx0 + 2:.1f} {mid - 9} L{gx1 - 3:.1f} {mid - 9}", t, theme))
    out.append(arrow(f"M{gx1 - 2:.1f} {mid + 9} L{gx0 + 3:.1f} {mid + 9}", t, theme))
    # sideways imports between neighbours, at mid height
    ym = TOP_Y + 30
    for src, dst in SIDEWAYS:
        a, b = xs[src] + TOP_W, xs[dst]
        out.append(arrow(f"M{a + 1:.1f} {ym} L{b - 3:.1f} {ym}", t, theme))
    # an import that skips a card, as an arc above the row
    for src, dst in ARCS:
        a, b = xs[src] + TOP_W / 2, xs[dst] + TOP_W / 2
        out.append(
            arrow(
                f"M{a:.1f} {TOP_Y} L{a:.1f} {TOP_Y - 16} L{b:.1f} {TOP_Y - 16} "
                f"L{b:.1f} {TOP_Y - 3}",
                t,
                theme,
            )
        )
    # imports down to the floor, fanned so each lands on its own card
    landings = {name: [] for name, *_ in FLOOR}
    for src, dsts in DOWN.items():
        for d in dsts:
            landings[d].append(src)
    for d, srcs in landings.items():
        n = len(srcs)
        for k, src in enumerate(srcs):
            land = fx[d] + FLOOR_W * (k + 1) / (n + 1)
            off = -14 if d == "quad" and len(DOWN[src]) > 1 else 14
            if len(DOWN[src]) == 1:
                off = 0
            start = xs[src] + TOP_W / 2 + off
            out.append(
                arrow(
                    f"M{start:.1f} {TOP_Y + TOP_H} L{land:.1f} {FLOOR_Y - 13}", t, theme
                )
            )
    by = 20 + box_h + 20
    bf, bs = t["base"]
    names = " · ".join(
        f'<tspan font-weight="700" fill="{t["ink"]}">{n}</tspan>' for n in THROUGHOUT
    )
    out.append(
        f'<rect x="20" y="{by}" width="920" height="58" rx="10" fill="{bf}" '
        f'stroke="{bs}"/>'
        f'<text x="480" y="{by + 24}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="13.5" fill="{t["muted"]}">every area imports {names} '
        f'through the jax extra, and all but diffops <tspan font-weight="700" '
        f'fill="{t["ink"]}">Equinox</tspan></text>'
        f'<text x="480" y="{by + 44}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="13.5" fill="{t["muted"]}">every area imports functax\'s '
        "private _backend and exceptions</text>"
    )
    ly = by + 84
    fill_c, stroke_c, _ = t["chip"]
    out.append(
        f'<rect x="40" y="{ly - 13}" width="30" height="18" rx="5" fill="{fill_c}" '
        f'stroke="{stroke_c}"/>'
        f'<text x="78" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">the base install</text>'
        f'<rect x="230" y="{ly - 13}" width="30" height="18" rx="5" fill="{fill_c}" '
        f'stroke="{t["extra"]}" stroke-dasharray="5 4"/>'
        f'<text x="268" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">through an extra: jax, or scipy when a scattered plan '
        "is built</text>"
        f'<path d="M720 {ly - 4} L754 {ly - 4}" stroke="{t["accent"]}" '
        f'stroke-width="2" marker-end="url(#im-{theme})"/>'
        f'<text x="762" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imports</text>'
    )
    height = ly + 22
    label = (
        "functax's modules and their imports: project imports interp; interp "
        "imports transform, basis, quad and geometry; transform and basis import "
        "quad and geometry; diffops imports geometry; geometry and quad import each "
        "other. scipy comes through an extra when a scattered plan is built."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>functax's modules and their imports</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``functax_imports-light.svg`` and ``functax_imports-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"functax_imports-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
