"""Write functax's exception hierarchy as light and dark SVGs.

Each functax error inherits from the package base and, below it, from the
closest standard-library exception; three of them refine ``ValidationError``.
The cards say when each is raised and in which areas. The classes are
``functax.exceptions`` on functax's ``main`` and the raise sites are its
``_src/<area>/`` modules, ordered by how often each area raises it. The
layout follows argminax's (``argminax_exceptions/``), with a third row for
the validation errors. Re-run after editing the data below, from the repo
root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/functax_exceptions/make_svgs.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write functax's exception hierarchy as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

BASE = ("FunctaxError", "every error functax raises: one except clause traps them all")
# the errors directly under FunctaxError:
# (name, centre x, standard-library parent, its centre x, when, chips, chip key)
MIDDLE = [
    (
        "ValidationError",
        170,
        "ValueError",
        110,
        "an argument failed a check: a shape, a dtype, a value or an option",
        ["interp", "basis", "quad", "project", "transform", "diffops"],
        "raised in",
    ),
    (
        "NonlinearSchemeError",
        480,
        "TypeError",
        420,
        "a linear-only operation met a plan whose fit depends on y "
        "(Pchip, Akima, Steffen); use plan.linearize(y)",
        ["interp"],
        "raised in",
    ),
    (
        "MissingExtraError",
        790,
        "ImportError",
        850,
        "a module or a plan needs an extra that is not installed; the message names it",
        ["functax[jax]", "functax[scipy]"],
        "extras",
    ),
]
# the refinements of ValidationError: (name, centre x, when, chips)
LEAVES = [
    (
        "GeometryError",
        170,
        "an axis, grid or point set is malformed; the message names the field",
        ["geometry", "interp", "quad", "basis"],
    ),
    (
        "PolicyError",
        480,
        "a policy names an option that does not exist, or cannot act there, "
        'such as outside="raise" on traced targets',
        ["interp"],
    ),
    (
        "OutsideDomainError",
        790,
        'under outside="raise", a target lies off the domain when the plan is '
        "built on the host",
        ["interp"],
    ),
]

THEMES = {
    "light": {
        "accent": "#1D4ED8",
        "card": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "std": ("#F9FAFB", "#9CA3AF"),
        "chip": ("#FFFFFF", "#D1D5DB", "#1F2937"),
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "accent": "#60A5FA",
        "card": "#111827",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "std": ("#0F172A", "#4B5563"),
        "chip": ("#111827", "#374151", "#E5E7EB"),
        "base": ("#1F2937", "#374151"),
    },
}

STD_Y, STD_H, STD_W = 20, 40, 170
BASE_Y, BASE_H, BASE_W = 96, 76, 300
TAG_Y, TAG_H, TAG_W = 196, 34, 150
MID_Y, CARD_W = 262, 270
BUS_GAP = 30


def wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line]


def arrow(points: list[tuple[float, float]], t: dict, theme: str, head=True) -> str:
    d = " ".join(f"{'M' if i == 0 else 'L'}{x} {y}" for i, (x, y) in enumerate(points))
    marker = f' marker-end="url(#ex-{theme})"' if head else ""
    return (
        f'<path d="{d}" fill="none" stroke="{t["accent"]}" stroke-width="2"{marker}/>'
    )


def std_box(cx: float, y: float, w: float, h: float, name: str, t: dict) -> str:
    sf, ss = t["std"]
    return (
        f'<rect x="{cx - w / 2}" y="{y}" width="{w}" height="{h}" rx="9" '
        f'fill="{sf}" stroke="{ss}" stroke-width="1.5" stroke-dasharray="6 5"/>'
        f'<text x="{cx}" y="{y + h / 2 + 5}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="13.5" fill="{t["muted"]}">{name}</text>'
    )


def card_height(when: str, chips: list[str]) -> float:
    rows, x = 1, 16
    for label in chips:
        w = 7.51 * len(label) + 16
        if x + w > CARD_W - 12:
            rows, x = rows + 1, 16
        x += w + 6
    return 74 + 16 * len(wrap(when, 40)) + 28 * rows


def card(cx, y, h, name, when, chips, key, t) -> list[str]:
    x0 = cx - CARD_W / 2
    fill, stroke, ink = t["chip"]
    out = [
        f'<rect x="{x0}" y="{y}" width="{CARD_W}" height="{h}" rx="12" '
        f'fill="{t["card"]}" stroke="{t["accent"]}" stroke-width="1.6"/>'
        f'<text x="{x0 + 16}" y="{y + 26}" font-family="{MONO}" '
        f'font-size="15" font-weight="700" fill="{t["ink"]}">{name}</text>'
    ]
    lines = wrap(when, 40)
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x0 + 16}" y="{y + 47 + 16 * i}" font-family="{SANS}" '
            f'font-size="12" fill="{t["muted"]}">{escape(line)}</text>'
        )
    ky = y + 47 + 16 * len(lines) + 8
    out.append(
        f'<text x="{x0 + 16}" y="{ky}" font-family="{SANS}" font-size="10.5" '
        f'font-weight="700" letter-spacing="0.06em" fill="{t["muted"]}">'
        f"{key.upper()}</text>"
    )
    x, cy = x0 + 16, ky + 8
    for label in chips:
        w = 7.51 * len(label) + 16
        if x + w > x0 + CARD_W - 12:
            x, cy = x0 + 16, cy + 28
        out.append(
            f'<rect x="{x:.1f}" y="{cy}" width="{w:.1f}" height="22" rx="6" '
            f'fill="{fill}" stroke="{stroke}"/>'
            f'<text x="{x + w / 2:.1f}" y="{cy + 15}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="12" fill="{ink}">{escape(label)}</text>'
        )
        x += w + 6
    return out


def build(theme: str) -> str:
    t = THEMES[theme]
    out = [
        f'<defs><marker id="ex-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["accent"]}"/></marker></defs>',
        std_box(480, STD_Y, STD_W, STD_H, "Exception", t),
    ]
    name, what = BASE
    bx = 480 - BASE_W / 2
    out.append(
        f'<rect x="{bx}" y="{BASE_Y}" width="{BASE_W}" height="{BASE_H}" rx="12" '
        f'fill="{t["card"]}" stroke="{t["accent"]}" stroke-width="2"/>'
        f'<text x="480" y="{BASE_Y + 28}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="16" font-weight="700" fill="{t["ink"]}">{name}</text>'
    )
    for i, line in enumerate(wrap(what, 40)):
        out.append(
            f'<text x="480" y="{BASE_Y + 50 + 16 * i}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="12" fill="{t["muted"]}">'
            f"{escape(line)}</text>"
        )
    out.append(arrow([(480, BASE_Y), (480, STD_Y + STD_H + 3)], t, theme))
    base_mid = BASE_Y + BASE_H / 2
    mid_h = max(card_height(m[4], m[5]) for m in MIDDLE)
    for name, cx, std, scx, when, chips, key in MIDDLE:
        out += card(cx, MID_Y, mid_h, name, when, chips, key, t)
        # its standard-library parent, as a dashed tag just above it
        out.append(std_box(scx, TAG_Y, TAG_W, TAG_H, std, t))
        out.append(arrow([(scx, MID_Y), (scx, TAG_Y + TAG_H + 3)], t, theme))
        # up to FunctaxError: straight from below, or by an elbow from the side
        if abs(cx - 480) < 1:
            out.append(
                arrow([(cx + 60, MID_Y), (cx + 60, BASE_Y + BASE_H + 3)], t, theme)
            )
        else:
            side = 1 if cx < 480 else -1
            ex = cx + side * 60
            tx = 480 - side * BASE_W / 2 - side * 3
            out.append(arrow([(ex, MID_Y), (ex, base_mid), (tx, base_mid)], t, theme))
    # the refinements of ValidationError, joined by a bus to its card
    leaf_y = MID_Y + mid_h + 2 * BUS_GAP
    bus_y = MID_Y + mid_h + BUS_GAP
    leaf_h = max(card_height(lf[2], lf[3]) for lf in LEAVES)
    val_cx = MIDDLE[0][1]
    for name, cx, when, chips in LEAVES:
        out += card(cx, leaf_y, leaf_h, name, when, chips, "raised in", t)
        out.append(arrow([(cx, leaf_y), (cx, bus_y), (val_cx, bus_y)], t, theme, False))
    out.append(arrow([(val_cx, bus_y), (val_cx, MID_Y + mid_h + 3)], t, theme))
    by = leaf_y + leaf_h + 24
    bf, bs = t["base"]
    out.append(
        f'<rect x="20" y="{by}" width="920" height="40" rx="10" fill="{bf}" '
        f'stroke="{bs}"/>'
        f'<text x="480" y="{by + 25.5}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="13" fill="{t["muted"]}">not an exception: by default a target '
        "off the domain gets the fill value (NaN) and a NaN sample propagates, as "
        f'<tspan font-family="{MONO}" fill="{t["ink"]}">Policies()</tspan> sets</text>'
    )
    ly = by + 66
    sf, ss = t["std"]
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
        "functax's exceptions: FunctaxError inherits from Exception; "
        "ValidationError from FunctaxError and ValueError; NonlinearSchemeError "
        "from FunctaxError and TypeError; MissingExtraError from FunctaxError and "
        "ImportError; GeometryError, PolicyError and OutsideDomainError from "
        "ValidationError."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height:.0f}" '
        f'viewBox="0 0 960 {height:.0f}" role="img" aria-label="{escape(label)}">\n'
        "<title>functax's exceptions</title>\n" + "\n".join(out) + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``functax_exceptions-light.svg`` and ``functax_exceptions-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"functax_exceptions-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
