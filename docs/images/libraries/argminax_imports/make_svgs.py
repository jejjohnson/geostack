"""Write argminax's module-import figure as light and dark SVGs.

Each public area of argminax is a card listing the libraries it imports
(dashed when through an extra), and arrows between cards read "imports".
The edges are the ``import`` lines in ``src/argminax/_src/<area>/`` on
argminax's ``main``. Re-run after editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/argminax_imports/make_svgs.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write argminax's module-import figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# (area, what it does, [(import, through an extra)], x)
TOP = [
    (
        "diagnose",
        "reads a Laplace",
        [("gaussx", False), ("lineax", False), ("matfree", True)],
        40,
    ),
    (
        "solve",
        "builds a Laplace by solving",
        [
            ("optimistix", False),
            ("gaussx", False),
            ("lineax", False),
            ("numpyro", True),
        ],
        262,
    ),
    (
        "blr",
        "builds a Laplace by training",
        [("gaussx", False), ("lineax", False), ("optax", True)],
        484,
    ),
    ("adjoint", "differentiates a solve", [("optimistix", False)], 718),
]
BOTTOM = (
    "curvature",
    "the Laplace object: a mode and a precision operator, "
    "which every other area builds or reads",
    [("gaussx", False), ("lineax", False)],
)
ARROWS = ["diagnose", "solve", "blr"]  # each imports curvature
THROUGHOUT = ["JAX", "Equinox"]

THEMES = {
    "light": {
        "accent": "#475569",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "chip": ("#FFFFFF", "#D1D5DB", "#1F2937"),
        "extra": "#9CA3AF",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#94A3B8",
        "card": "#111827",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "box": ("#0F172A", "#1F2937"),
        "chip": ("#111827", "#374151", "#E5E7EB"),
        "extra": "#6B7280",
        "base": ("#1F2937", "#374151"),
    },
}

CARD_W, TOP_Y, BOT_Y = 202, 72, 262


def chip_width(label: str) -> float:
    return 7.51 * len(label) + 19.5


def chips(x0: float, x1: float, y: float, items, t: dict) -> tuple[list[str], float]:
    """Lay chips out from x0, wrapping at x1; return the SVG and the last row's y."""
    out, x = [], x0
    for label, extra in items:
        w = chip_width(label)
        if x + w > x1:
            x, y = x0, y + 28
        fill, stroke, ink = t["chip"]
        dash = (
            f' stroke-dasharray="5 4" stroke="{t["extra"]}"'
            if extra
            else (f' stroke="{stroke}"')
        )
        out.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" rx="6" '
            f'fill="{fill}"{dash}/>'
            f'<text x="{x + w / 2:.1f}" y="{y + 15}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="12.5" fill="{ink}">{escape(label)}</text>'
        )
        x += w + 8
    return out, y


def card(x, y, w, h, name, what, items, t) -> list[str]:
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{t["card"]}" '
        f'stroke="{t["accent"]}" stroke-width="1.6"/>'
        f'<text x="{x + 14}" y="{y + 24}" font-family="{MONO}" font-size="15" '
        f'font-weight="700" fill="{t["ink"]}">{name}</text>'
        f'<text x="{x + 14}" y="{y + 43}" font-family="{SANS}" font-size="12" '
        f'fill="{t["muted"]}">{escape(what)}</text>'
    ]
    body, _ = chips(x + 14, x + w - 10, y + 54, items, t)
    return out + body


def build(theme: str) -> str:
    t = THEMES[theme]
    fill, stroke = t["box"]
    out = [
        f'<defs><marker id="im-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["accent"]}"/></marker></defs>',
        f'<rect x="20" y="20" width="920" height="352" rx="14" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="40" y="48" font-family="{SANS}" font-size="15" font-weight="700" '
        f'fill="{t["ink"]}">argminax</text>',
        f'<text x="920" y="48" text-anchor="end" font-family="{SANS}" '
        f'font-size="12.5" fill="{t["muted"]}">one public module per area; '
        "arrows read “imports”</text>",
    ]
    top_h = 116
    for name, what, items, x in TOP:
        out += card(x, TOP_Y, CARD_W, top_h, name, what, items, t)
    adj_x = TOP[-1][3]
    out.append(
        f'<text x="{adj_x + CARD_W / 2}" y="{TOP_Y + top_h + 20}" text-anchor="middle" '
        f'font-family="{SANS}" font-size="11.5" fill="{t["muted"]}">'
        "imports nothing else from argminax</text>"
    )
    name, what, items = BOTTOM
    bot_w = TOP[2][3] + CARD_W - 40
    out += card(40, BOT_Y, bot_w, 90, name, what, items, t)
    for x_name in ARROWS:
        x = next(c[3] for c in TOP if c[0] == x_name) + CARD_W / 2
        out.append(
            f'<path d="M{x} {TOP_Y + top_h} L{x} {BOT_Y - 3}" stroke="{t["accent"]}" '
            f'stroke-width="2" marker-end="url(#im-{theme})"/>'
        )
    by = 392
    bf, bs = t["base"]
    names = " · ".join(
        f'<tspan font-weight="700" fill="{t["ink"]}">{n}</tspan>' for n in THROUGHOUT
    )
    out.append(
        f'<rect x="20" y="{by}" width="920" height="40" rx="10" fill="{bf}" '
        f'stroke="{bs}"/>'
        f'<text x="480" y="{by + 25.5}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="14" fill="{t["muted"]}">throughout: {names}, and '
        f'<tspan font-weight="700" fill="{t["ink"]}">einx</tspan> '
        "in all but adjoint</text>"
    )
    ly = by + 66
    fill_c, stroke_c, _ = t["chip"]
    out.append(
        f'<rect x="40" y="{ly - 13}" width="30" height="18" rx="5" fill="{fill_c}" '
        f'stroke="{stroke_c}"/>'
        f'<text x="78" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">a base dependency</text>'
        f'<rect x="250" y="{ly - 13}" width="30" height="18" rx="5" fill="{fill_c}" '
        f'stroke="{t["extra"]}" stroke-dasharray="5 4"/>'
        f'<text x="288" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">through an extra, imported when first used</text>'
        f'<path d="M620 {ly - 4} L654 {ly - 4}" stroke="{t["accent"]}" '
        f'stroke-width="2" marker-end="url(#im-{theme})"/>'
        f'<text x="662" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imports</text>'
    )
    height = ly + 22
    label = (
        "argminax's modules and their imports: diagnose, solve and blr import "
        "curvature; adjoint imports only optimistix; curvature imports gaussx and "
        "lineax; matfree, numpyro and optax come through extras."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>argminax's modules and their imports</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``argminax_imports-light.svg`` and ``argminax_imports-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"argminax_imports-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
