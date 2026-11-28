"""Write pipekit-array's area-import figure as light and dark SVGs.

Each area of pipekit-array is a card listing the libraries it imports (solid
when the area's module imports it, dashed when a function imports it the first
time it runs), and arrows between cards read "imports". The test kit sits
beside the package, in a dashed box. The edges are the ``import`` lines in
``packages/pipekit-array/src/pipekit_array/<area>/`` on pipekit's ``main``;
``_recorded``, the private module a saved operator names, is left out. Re-run
after editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pipekit_array_imports/make_svgs.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write pipekit-array's area-import figure as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# area -> (what it does, [(import, when first used)], x, y, width)
CARDS = {
    "bridges": (
        "xarray, einx and scikit-learn on named carriers, one module each",
        [
            ("xarray", False),
            ("numpy", False),
            ("einx", True),
            ("scikit-learn", True),
            ("jax", True),
        ],
        40,
        72,
        640,
    ),
    "operators": (
        "the named functions as steps",
        [("numpy", True)],
        40,
        236,
        200,
    ),
    "kernels": (
        "one array function on every carrier",
        [("numpy", True), ("jax", True)],
        300,
        236,
        380,
    ),
    "named": (
        "functions of carriers with dim names",
        [
            ("numpy", True),
            ("jax", True),
            ("dask", True),
            ("xarray", True),
            ("array_api_compat", True),
        ],
        40,
        400,
        360,
    ),
    "arrays": (
        "formulas on plain arrays",
        [("numpy", False), ("array_api_compat", False), ("jax", True)],
        420,
        400,
        260,
    ),
}
CARD_H = 116
# (from, to, x at the start, x at the end); every arrow runs straight down
ARROWS = [
    ("bridges", "kernels", 490, 490),
    ("bridges", "named", 270, 270),
    ("operators", "named", 140, 140),
    ("kernels", "named", 380, 380),
    ("kernels", "arrays", 580, 580),
]
SIDE = (
    "testing",
    "the checks for the suites built on pipekit-array; imports named and kernels",
    [
        ("numpy", False),
        ("jax", True),
        ("dask", True),
        ("xarray", True),
        ("array_api_strict", True),
    ],
)

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


def chip_width(label: str) -> float:
    return 7.51 * len(label) + 19.5


def chips(x0: float, x1: float, y: float, items, t: dict) -> tuple[list[str], float]:
    """Lay chips out from x0, wrapping at x1; return the SVG and the last row's y."""
    out, x = [], x0
    for label, lazy in items:
        w = chip_width(label)
        if x + w > x1:
            x, y = x0, y + 28
        fill, stroke, ink = t["chip"]
        dash = (
            f' stroke-dasharray="5 4" stroke="{t["extra"]}"'
            if lazy
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


def wrap(text: str, width: float) -> list[str]:
    """Split ``text`` into lines of at most ``width`` px at 12 px sans."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if len(trial) * 6.3 > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line]


def card(x, y, w, h, name, what, items, t, *, dashed=False) -> list[str]:
    border = ' stroke-dasharray="7 5"' if dashed else ""
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{t["card"]}" '
        f'stroke="{t["accent"]}" stroke-width="1.6"{border}/>'
        f'<text x="{x + 14}" y="{y + 24}" font-family="{MONO}" font-size="15" '
        f'font-weight="700" fill="{t["ink"]}">{name}</text>'
    ]
    lines = wrap(what, w - 28)
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x + 14}" y="{y + 43 + 16 * i}" font-family="{SANS}" '
            f'font-size="12" fill="{t["muted"]}">{escape(line)}</text>'
        )
    body, _ = chips(x + 14, x + w - 10, y + 38 + 16 * len(lines), items, t)
    return out + body


def build(theme: str) -> str:
    t = THEMES[theme]
    fill, stroke = t["box"]
    out = [
        f'<defs><marker id="im-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["accent"]}"/></marker></defs>',
        f'<rect x="20" y="20" width="680" height="516" rx="14" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="40" y="48" font-family="{SANS}" font-size="15" font-weight="700" '
        f'fill="{t["ink"]}">pipekit-array</text>',
        f'<text x="680" y="48" text-anchor="end" font-family="{SANS}" '
        f'font-size="12.5" fill="{t["muted"]}">one public module per area; '
        "arrows read “imports”</text>",
    ]
    for name, (what, items, x, y, w) in CARDS.items():
        out += card(x, y, w, CARD_H, name, what, items, t)
    for start, end, x0, x1 in ARROWS:
        y0 = CARDS[start][3] + CARD_H
        y1 = CARDS[end][3] - 3
        out.append(
            f'<path d="M{x0} {y0} L{x1} {y1}" stroke="{t["accent"]}" '
            f'stroke-width="2" marker-end="url(#im-{theme})"/>'
        )
    name, what, items = SIDE
    out.append(
        f'<text x="830" y="48" text-anchor="middle" font-family="{SANS}" '
        f'font-size="12.5" fill="{t["muted"]}">beside the package</text>'
    )
    out += card(720, 72, 220, 200, name, what, items, t, dashed=True)
    by = 556
    bf, bs = t["base"]
    out.append(
        f'<rect x="20" y="{by}" width="920" height="40" rx="10" fill="{bf}" '
        f'stroke="{bs}"/>'
        f'<text x="480" y="{by + 25.5}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="14" fill="{t["muted"]}">throughout: '
        f'<tspan font-weight="700" fill="{t["ink"]}">pipekit-ops</tspan>, the one '
        "base dependency; the others come through extras, but for "
        "array_api_strict, a test reference</text>"
    )
    ly = by + 66
    fill_c, stroke_c, _ = t["chip"]
    out.append(
        f'<rect x="40" y="{ly - 13}" width="30" height="18" rx="5" fill="{fill_c}" '
        f'stroke="{stroke_c}"/>'
        f'<text x="78" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imported with the module</text>'
        f'<rect x="270" y="{ly - 13}" width="30" height="18" rx="5" fill="{fill_c}" '
        f'stroke="{t["extra"]}" stroke-dasharray="5 4"/>'
        f'<text x="308" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imported when first used</text>'
        f'<path d="M520 {ly - 4} L554 {ly - 4}" stroke="{t["accent"]}" '
        f'stroke-width="2" marker-end="url(#im-{theme})"/>'
        f'<text x="562" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">imports</text>'
    )
    height = ly + 22
    label = (
        "pipekit-array's areas and their imports: bridges imports kernels and "
        "named, operators imports named, kernels imports named and arrays, and "
        "named and arrays import no other area; testing, beside the package, "
        "imports named and kernels. Every area imports pipekit-ops; the other "
        "libraries come through extras, but for array_api_strict, a test reference."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>pipekit-array's areas and their imports</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``pipekit_array_imports-light.svg`` and ``-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"pipekit_array_imports-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
