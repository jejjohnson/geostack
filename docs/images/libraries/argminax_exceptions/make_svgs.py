"""Write argminax's exception hierarchy as light and dark SVGs.

Each argminax error inherits from the package base and from the closest
standard-library exception; the cards say when each is raised. The classes
are ``argminax.exceptions`` on argminax's ``main`` and the raise sites are
its ``_src/<area>/`` modules. Re-run after editing the data below, from the
repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/argminax_exceptions/make_svgs.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write argminax's exception hierarchy as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# standard-library parents: (name, centre x)
STDLIB = [("ValueError", 200), ("Exception", 480), ("ImportError", 760)]
BASE = (
    "ArgminaxError",
    "every error the package raises: one except clause traps them all",
)
# (name, centre x, when it is raised, chips)
LEAVES = [
    (
        "ValidationError",
        200,
        "a static check failed: a shape, a dtype or an option name, "
        "never a traced value",
        ["solve", "curvature", "diagnose", "adjoint", "blr"],
        "raised in",
    ),
    (
        "MissingExtraError",
        760,
        "a function needs an extra that is not installed; the message names it",
        ["optax · blr", "matfree · diagnose", "numpyro · solve"],
        "extras",
    ),
]

THEMES = {
    "light": {
        "accent": "#475569",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "std": ("#F9FAFB", "#9CA3AF"),
        "chip": ("#FFFFFF", "#D1D5DB", "#1F2937"),
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#94A3B8",
        "card": "#111827",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "std": ("#0F172A", "#4B5563"),
        "chip": ("#111827", "#374151", "#E5E7EB"),
        "base": ("#1F2937", "#374151"),
    },
}

STD_Y, STD_H = 20, 44
BASE_Y, BASE_H, BASE_W = 108, 84, 260
LEAF_Y, LEAF_W = 250, 300


def chip_width(label: str) -> float:
    return 7.51 * len(label) + 19.5


def wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line]


def arrow(points: list[tuple[float, float]], t: dict, theme: str) -> str:
    d = " ".join(f"{'M' if i == 0 else 'L'}{x} {y}" for i, (x, y) in enumerate(points))
    return (
        f'<path d="{d}" fill="none" stroke="{t["accent"]}" stroke-width="2" '
        f'marker-end="url(#ex-{theme})"/>'
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    out = [
        f'<defs><marker id="ex-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["accent"]}"/></marker></defs>'
    ]
    sf, ss = t["std"]
    for name, cx in STDLIB:
        w = 200
        out.append(
            f'<rect x="{cx - w / 2}" y="{STD_Y}" width="{w}" height="{STD_H}" rx="10" '
            f'fill="{sf}" stroke="{ss}" stroke-width="1.5" stroke-dasharray="6 5"/>'
            f'<text x="{cx}" y="{STD_Y + 27}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="14" fill="{t["muted"]}">{name}</text>'
        )
    name, what = BASE
    bx = 480 - BASE_W / 2
    out.append(
        f'<rect x="{bx}" y="{BASE_Y}" width="{BASE_W}" height="{BASE_H}" rx="12" '
        f'fill="{t["card"]}" stroke="{t["accent"]}" stroke-width="2"/>'
        f'<text x="480" y="{BASE_Y + 28}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="16" font-weight="700" fill="{t["ink"]}">{name}</text>'
    )
    for i, line in enumerate(wrap(what, 36)):
        out.append(
            f'<text x="480" y="{BASE_Y + 50 + 16 * i}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="12" fill="{t["muted"]}">'
            f"{escape(line)}</text>"
        )
    out.append(arrow([(480, BASE_Y), (480, STD_Y + STD_H + 3)], t, theme))
    base_mid = BASE_Y + BASE_H / 2
    fill, stroke, ink = t["chip"]
    layouts = []
    for _name, cx, _when, chips, _key in LEAVES:
        x0 = cx - LEAF_W / 2
        placed, x, cy = [], x0 + 16, LEAF_Y + 94
        for label in chips:
            w = 7.51 * len(label) + 16
            if x + w > x0 + LEAF_W - 12:
                x, cy = x0 + 16, cy + 28
            placed.append((x, cy, w, label))
            x += w + 6
        layouts.append(placed)
    leaf_h = max(p[-1][1] for p in layouts) - LEAF_Y + 36
    for (name, cx, when, _chips, key), placed in zip(LEAVES, layouts, strict=True):
        x0 = cx - LEAF_W / 2
        out.append(
            f'<rect x="{x0}" y="{LEAF_Y}" width="{LEAF_W}" height="{leaf_h}" rx="12" '
            f'fill="{t["card"]}" stroke="{t["accent"]}" stroke-width="1.6"/>'
            f'<text x="{x0 + 16}" y="{LEAF_Y + 26}" font-family="{MONO}" '
            f'font-size="15" font-weight="700" fill="{t["ink"]}">{name}</text>'
        )
        for i, line in enumerate(wrap(when, 44)):
            out.append(
                f'<text x="{x0 + 16}" y="{LEAF_Y + 47 + 16 * i}" font-family="{SANS}" '
                f'font-size="12" fill="{t["muted"]}">{escape(line)}</text>'
            )
        out.append(
            f'<text x="{x0 + 16}" y="{LEAF_Y + 86}" font-family="{SANS}" '
            f'font-size="10.5" font-weight="700" letter-spacing="0.06em" '
            f'fill="{t["muted"]}">'
            f"{key.upper()}</text>"
        )
        for x, cy, w, label in placed:
            out.append(
                f'<rect x="{x:.1f}" y="{cy}" width="{w:.1f}" height="22" rx="6" '
                f'fill="{fill}" stroke="{stroke}"/>'
                f'<text x="{x + w / 2:.1f}" y="{cy + 15}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="12" fill="{ink}">'
                f"{escape(label)}</text>"
            )
        # to its standard-library parent, straight up
        out.append(arrow([(cx, LEAF_Y), (cx, STD_Y + STD_H + 3)], t, theme))
        # to ArgminaxError, by an elbow on the side facing the base
        side = 1 if cx < 480 else -1
        ex = cx + side * 100
        tx = 480 - side * BASE_W / 2 - side * 3
        out.append(arrow([(ex, LEAF_Y), (ex, base_mid), (tx, base_mid)], t, theme))
    by = LEAF_Y + leaf_h + 24
    bf, bs = t["base"]
    out.append(
        f'<rect x="20" y="{by}" width="920" height="40" rx="10" fill="{bf}" '
        f'stroke="{bs}"/>'
        f'<text x="480" y="{by + 25.5}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="13" fill="{t["muted"]}">not an exception: a failure that depends '
        f"on traced values, such as a solve that did not converge, is in "
        f'<tspan font-family="{MONO}" fill="{t["ink"]}">solution.result</tspan>'
        "</text>"
    )
    ly = by + 66
    out.append(
        f'<rect x="40" y="{ly - 13}" width="30" height="18" rx="5" fill="{sf}" '
        f'stroke="{ss}" stroke-dasharray="5 4"/>'
        f'<text x="78" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">the standard library</text>'
        f'<path d="M260 {ly - 4} L294 {ly - 4}" stroke="{t["accent"]}" '
        f'stroke-width="2" marker-end="url(#ex-{theme})"/>'
        f'<text x="302" y="{ly + 1}" font-family="{SANS}" font-size="12.5" '
        f'fill="{t["muted"]}">inherits from, so either except clause catches it</text>'
    )
    height = ly + 22
    label = (
        "argminax's exceptions: ArgminaxError inherits from Exception; "
        "ValidationError inherits from ArgminaxError and ValueError; "
        "MissingExtraError inherits from ArgminaxError and ImportError."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
        f'viewBox="0 0 960 {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>argminax's exceptions</title>\n" + "\n".join(out) + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``argminax_exceptions-light.svg`` and ``argminax_exceptions-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"argminax_exceptions-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
