"""Write argminax's role ladder as light and dark SVGs.

The layout and palettes follow gaussx's architecture figure, so the two
libraries' ladders read alike. Re-run after editing the rows below, from the
repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/argminax_roles/make_svgs.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the argminax layer diagram as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# The rungs of the GeoStack role ladder (the role-ladder skill), top to bottom,
# with argminax's names on each as placed by geostack's
# docs/design/skill_tree_argminax.md. Each row is
# (badge, role, grammar, caption, chips, palette key, borrowed).
ROWS = [
    (
        "4",
        "algorithm",
        "the sentence",
        "a method over models: do this, to that, this way",
        [
            "laplace_mode",
            "dual_solve",
            "incremental_dual_solve",
            "averaging_kernel",
            "degrees_of_freedom",
            "information_content",
            "evidence",
            "optimize_prior_precision",
            "blr_posterior",
        ],
        "pink",
        False,
    ),
    (
        "3",
        "model",
        "the noun phrase",
        "the Laplace approximation and what it reads out",
        ["Laplace", "marginal_variance", "push_forward", "sample_by_perturbation"],
        "violet",
        False,
    ),
    (
        "2",
        "strategy",
        "the adverb",
        "how to solve, step or differentiate: settings only",
        [
            "GaussNewtonCG",
            "Reparametrised",
            "KStep",
            "OneStep",
            "blr_diagonal",
            "ivon",
            "blr_low_rank",
            "blr_full_rank",
        ],
        "blue",
        False,
    ),
    (
        "1",
        "primitive",
        "the verb",
        "none of argminax's own: one step is gaussx's",
        ["gaussx.newton_update", "gaussx.solve", "gaussx.logdet"],
        "grey",
        True,
    ),
    (
        "0",
        "structure",
        "the noun",
        "the Hessian at the minimum, as a gaussx operator",
        ["gauss_newton", "low_rank"],
        "cyan",
        False,
    ),
]
SIDE = (
    "optimistix",
    "the floor argminax stands on: its solvers and adjoints are optimistix's",
    ["optx.least_squares", "optx.minimise", "optx.ImplicitAdjoint"],
)
BASE = ["JAX", "Equinox", "lineax", "gaussx", "optimistix", "optax (extra)"]

THEMES = {
    "light": {
        "rows": {
            "pink": ("#FDF2F8", "#EC4899", "#BE185D"),
            "violet": ("#F5F3FF", "#8B5CF6", "#6D28D9"),
            "blue": ("#EFF6FF", "#3B82F6", "#1D4ED8"),
            "cyan": ("#ECFEFF", "#06B6D4", "#0E7490"),
            "grey": ("#F9FAFB", "#9CA3AF", "#4B5563"),
        },
        "chip": ("#FFFFFF", "#D1D5DB", "#1F2937"),
        "muted": "#6B7280",
        "ink": "#1F2937",
        "base": ("#F3F4F6", "#D1D5DB"),
        "side": ("#F9FAFB", "#9CA3AF"),
        "arrow": "#9CA3AF",
    },
    "dark": {
        "rows": {
            "pink": ("#2A1020", "#DB2777", "#F9A8D4"),
            "violet": ("#1E1636", "#7C3AED", "#C4B5FD"),
            "blue": ("#0F1D33", "#3B82F6", "#93C5FD"),
            "cyan": ("#0B2429", "#0891B2", "#67E8F9"),
            "grey": ("#111827", "#4B5563", "#9CA3AF"),
        },
        "chip": ("#111827", "#374151", "#E5E7EB"),
        "muted": "#9CA3AF",
        "ink": "#E5E7EB",
        "base": ("#1F2937", "#374151"),
        "side": ("#111827", "#4B5563"),
        "arrow": "#6B7280",
    },
}

ROW_X, ROW_W, GAP, TOP = 20, 670, 10, 20
CHIP_X0, CHIP_X1 = 72.0, 674.0
SIDE_X, SIDE_W = 730, 210


def chip_width(label: str) -> float:
    """Width of a monospace chip at 12.5 px, as in gaussx's figure."""
    return 7.51 * len(label) + 19.5


def chip(x: float, y: float, label: str, t: dict) -> str:
    fill, stroke, ink = t["chip"]
    w = chip_width(label)
    return (
        f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" rx="6" '
        f'fill="{fill}" stroke="{stroke}"/>'
        f'<text x="{x + w / 2:.1f}" y="{y + 15}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="12.5" fill="{ink}">{escape(label)}</text>'
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    out = []
    y = TOP
    rows_y = {}
    for badge, role, grammar, caption, chips, key, borrowed in ROWS:
        bg, stroke, ink = t["rows"][key]
        # lay the chips out first, wrapping onto a new line at the right edge
        placed, x, cy = [], CHIP_X0, 0
        for label in chips:
            if x + chip_width(label) > CHIP_X1:
                x, cy = CHIP_X0, cy + 28
            placed.append((x, cy, label))
            x += chip_width(label) + 8
        height = 78 + cy
        rows_y[role] = (y, height)
        dash = ' stroke-dasharray="6 5"' if borrowed else ""
        out.append(
            f'<rect x="{ROW_X}" y="{y}" width="{ROW_W}" height="{height}" rx="12" '
            f'fill="{bg}" stroke="{stroke}" stroke-width="1.5"{dash}/>'
            f'<circle cx="49" cy="{y + 25}" r="13" fill="{stroke}"/>'
            f'<text x="49" y="{y + 30.5}" text-anchor="middle" font-family="{SANS}" '
            f'font-size="16" font-weight="700" fill="#FFFFFF">{badge}</text>'
            f'<text x="72" y="{y + 31}" font-family="{SANS}" font-size="16" '
            f'font-weight="700" fill="{ink}">{role}'
            f'<tspan font-size="13" font-weight="400" fill="{t["muted"]}"> · {grammar}'
            "</tspan></text>"
            f'<text x="674" y="{y + 31}" text-anchor="end" font-family="{SANS}" '
            f'font-size="13" fill="{t["muted"]}">{escape(caption)}</text>'
        )
        for cx, dy, label in placed:
            out.append(chip(cx, y + 44 + dy, label, t))
        y += height + GAP
    base_y = y + 4
    fill, stroke = t["base"]
    out.append(
        f'<rect x="{ROW_X}" y="{base_y}" width="{ROW_W}" height="40" rx="10" '
        f'fill="{fill}" stroke="{stroke}"/>'
    )
    names = " · ".join(
        f'<tspan font-weight="700" fill="{t["ink"]}">{escape(n)}</tspan>' for n in BASE
    )
    out.append(
        f'<text x="{ROW_X + ROW_W / 2}" y="{base_y + 25.5}" text-anchor="middle" '
        f'font-family="{SANS}" font-size="14" fill="{t["muted"]}">'
        f"built on: {names}</text>"
    )
    # the adjoint box, beside the solve row
    title, caption, chips = SIDE
    words, line, lines = caption.split(), "", []
    for w in words:  # wrap the caption at about 28 characters
        if len(line) + len(w) + 1 > 28:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    side_h = 60 + 16 * len(lines) + 28 * len(chips)
    row_y, row_h = rows_y["strategy"]
    solve_mid = row_y + row_h / 2
    sy = solve_mid - side_h / 2
    fill, stroke = t["side"]
    out.append(
        f'<rect x="{SIDE_X}" y="{sy}" width="{SIDE_W}" height="{side_h}" rx="12" '
        f'fill="{fill}" stroke="{stroke}" stroke-dasharray="6 5" stroke-width="1.5"/>'
        f'<text x="{SIDE_X + 16}" y="{sy + 28}" font-family="{MONO}" font-size="15" '
        f'font-weight="700" fill="{t["ink"]}">{title}</text>'
    )
    for i, ln in enumerate(lines):
        out.append(
            f'<text x="{SIDE_X + 16}" y="{sy + 48 + 16 * i}" font-family="{SANS}" '
            f'font-size="12.5" fill="{t["muted"]}">{escape(ln)}</text>'
        )
    cy = sy + 52 + 16 * len(lines)
    for label in chips:
        out.append(chip(SIDE_X + 16, cy, label, t))
        cy += 28
    out.append(
        f'<defs><marker id="ah-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["arrow"]}"/></marker></defs>'
        f'<path d="M{SIDE_X - 4} {solve_mid} L{ROW_X + ROW_W + 6} {solve_mid}" '
        f'stroke="{t["arrow"]}" stroke-width="2" marker-end="url(#ah-{theme})"/>'
    )
    height = max(base_y + 40, sy + side_h) + 20
    label = (
        "argminax on the GeoStack role ladder: algorithms, the Laplace model, "
        "strategies for solving, stepping and differentiating, no primitives of "
        "its own (gaussx's), and the Hessian as a gaussx structure; optimistix "
        "is the floor its solvers and adjoints extend."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height:.0f}" '
        f'viewBox="0 0 960 {height:.0f}" role="img" aria-label="{escape(label)}">\n'
        "<title>argminax on the role ladder</title>\n" + "\n".join(out) + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``argminax_roles-light.svg`` and ``argminax_roles-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"argminax_roles-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
