"""Write pyrox-gp's role ladder as light and dark SVGs.

The layout and palettes follow pyrox's ladder (``pyrox_roles/``), so the
workspace's ladders read alike. pyrox-gp (``packages/pyrox-gp`` in the pyrox
workspace) holds kernels with priors, GP priors and their posteriors, guides,
likelihoods, inference strategies and pathwise samplers. Each rung groups its
chips under a short label, and two side boxes hold what sits beside the ladder:
the setup functions that build a model's starting state, and the gaussx solver
strategies every model takes as ``solver=``. Re-run after editing the rows
below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pyrox_gp_roles/make_svgs.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the pyrox-gp role ladder as light and dark SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# The rungs of the GeoStack role ladder (the role-ladder skill), top to bottom,
# with the public names of pyrox_gp on each. geostack has no
# skill_tree_pyrox_gp.md yet, so every name is placed by the role-ladder skill's
# §2 test. A chip that starts with a dot is a public method; a "+N more" chip
# stands for the rest of a group, all of which the README's module table lists.
# Each row is (badge, role, grammar, caption, groups, palette key, borrowed), and
# each group is (label, chips): the label starts a new line and the chips wrap
# after it.
ROWS = [
    (
        "4",
        "algorithm",
        "the sentence",
        "condition a prior on data, score it in a model, and draw from it",
        [
            ("condition", [".condition()", ".condition_nongauss()", ".fit()"]),
            ("evidence", ["gp_factor", "mo_gp_factor", "lfr_factor", "lfr_model"]),
            (
                "variational",
                [
                    "svgp_elbo",
                    "svgp_factor",
                    "mo_svgp_elbo",
                    "mo_svgp_factor",
                    ".step()",
                ],
            ),
            ("draws", ["gp_sample", ".sample_paths()"]),
        ],
        "pink",
        False,
    ),
    (
        "3",
        "model",
        "the noun phrase",
        "what a GP model is made of, and what conditioning leaves",
        [
            ("kernels", ["RBF", "Matern", "Periodic", "+6 more"]),
            ("mixing", ["LMCKernel", "ICMKernel", "OILMMKernel"]),
            ("priors", ["GPPrior", "SparseGPPrior", "+4 more"]),
            (
                "likelihoods",
                [
                    "GaussianLikelihood",
                    "PoissonLikelihood",
                    "WarpedGaussianLikelihood",
                    "+5 more",
                ],
            ),
            ("guides", ["WhitenedGuide", "NaturalGuide", "+3 more"]),
            (
                "posteriors",
                [
                    "ConditionedGP",
                    "NonGaussConditionedGP",
                    "GradientPosterior",
                    "+3 more",
                ],
            ),
            (
                "draws",
                ["PathwiseSampler", "DecoupledPathwiseSampler", "PathwiseFunction"],
            ),
            ("protocols", ["Kernel", "Guide", "Likelihood"]),
        ],
        "violet",
        False,
    ),
    (
        "2",
        "strategy",
        "the adverb",
        "how a posterior is computed: settings only",
        [
            (
                "non-Gaussian",
                [
                    "LaplaceInference",
                    "GaussNewtonInference",
                    "PosteriorLinearization",
                    "ExpectationPropagation",
                    "QuasiNewtonInference",
                ],
            ),
            ("variational", ["ConjugateVI"]),
            ("solves", ["preconditioned_cg_solver"]),
        ],
        "blue",
        False,
    ),
    (
        "1",
        "primitive",
        "the verb",
        "one call, with no state of its own",
        [
            (
                "predict",
                [
                    ".predict()",
                    ".predict_mean()",
                    ".predict_var()",
                    ".predict_gradient()",
                    ".predictive_blocks()",
                ],
            ),
            ("densities", [".log_prob()", ".kl_divergence()", ".sample()"]),
            ("kernel", [".frozen()", "funk_hecke_coefficients"]),
            (
                "on arrays",
                [
                    "collapsed_lfr_log_prob",
                    "decoder_posterior",
                    "lfr_predictive_moments",
                    "latent_total_correlation",
                    "+4 more",
                ],
            ),
        ],
        "teal",
        False,
    ),
    (
        "0",
        "structure",
        "the noun",
        "the inducing variables a sparse GP is built on",
        [
            (
                "inducing",
                [
                    "FourierInducingFeatures",
                    "SphericalHarmonicInducingFeatures",
                    "SlepianInducingFeatures",
                    "+5 more",
                ],
            )
        ],
        "cyan",
        False,
    ),
]
# Boxes beside the ladder: (title, caption, chips, rung the arrow points at).
SIDES = [
    (
        "setup",
        "builds a model's starting state from the data, once",
        ["init_inducing", "latent_init"],
        "model",
    ),
    (
        "gaussx",
        "the solver strategies every model takes as solver=",
        ["DenseSolver", "CGSolver", "BBMMSolver", "ComposedSolver"],
        "strategy",
    ),
]
BASE = [
    "JAX",
    "Equinox",
    "NumPyro",
    "einx",
    "pyrox",
    "pyrox-gauss",
    "kernellib",
    "gaussx",
    "lineax",
    "functax",
    "flowjax",
    "optax, gauss-flows (extras)",
]

THEMES = {
    "light": {
        "rows": {
            "amber": ("#FFFBEB", "#F59E0B", "#B45309"),
            "green": ("#F0FDF4", "#22C55E", "#15803D"),
            "pink": ("#FDF2F8", "#EC4899", "#BE185D"),
            "violet": ("#F5F3FF", "#8B5CF6", "#6D28D9"),
            "blue": ("#EFF6FF", "#3B82F6", "#1D4ED8"),
            "teal": ("#F0FDFA", "#14B8A6", "#0F766E"),
            "cyan": ("#ECFEFF", "#06B6D4", "#0E7490"),
            "grey": ("#F9FAFB", "#9CA3AF", "#4B5563"),
        },
        "chip": ("#FFFFFF", "#D1D5DB", "#1F2937"),
        "muted": "#6B7280",
        "ink": "#1F2937",
        "base": ("#F3F4F6", "#D1D5DB"),
        "side": ("#F9FAFB", "#9CA3AF"),
        "arrow": "#9CA3AF",
        "divider": 0.45,
    },
    "dark": {
        "rows": {
            "amber": ("#2A1D08", "#D97706", "#FCD34D"),
            "green": ("#0B2416", "#16A34A", "#86EFAC"),
            "pink": ("#2A1020", "#DB2777", "#F9A8D4"),
            "violet": ("#1E1636", "#7C3AED", "#C4B5FD"),
            "blue": ("#0F1D33", "#3B82F6", "#93C5FD"),
            "teal": ("#0A2421", "#0D9488", "#5EEAD4"),
            "cyan": ("#0B2429", "#0891B2", "#67E8F9"),
            "grey": ("#111827", "#4B5563", "#9CA3AF"),
        },
        "chip": ("#111827", "#374151", "#E5E7EB"),
        "muted": "#9CA3AF",
        "ink": "#E5E7EB",
        "base": ("#1F2937", "#374151"),
        "side": ("#111827", "#4B5563"),
        "arrow": "#6B7280",
        "divider": 0.75,
    },
}

ROW_X, ROW_W, GAP, TOP = 20, 690, 10, 20
LABEL_X, CHIP_X0, CHIP_X1 = 72.0, 168.0, 694.0
SIDE_X, SIDE_W = 734, 212
LINE = 28
GROUP_GAP = 14


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


def wrap(text: str, width: int) -> list[str]:
    """Split ``text`` into lines of at most about ``width`` characters."""
    line, lines = "", []
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line]


def lay_out(groups: list) -> tuple[list, list, list, float]:
    """Place each group's label and chips, with a divider between groups.

    Returns the labels and chips with their offsets below the first chip line,
    the offsets of the dotted dividers, and the height the groups take.
    """
    labels, placed, dividers, dy = [], [], [], 0.0
    for i, (label, chips) in enumerate(groups):
        if i:
            dividers.append(dy + GROUP_GAP / 2 - 4)
            dy += GROUP_GAP
        x0 = CHIP_X0 if label else LABEL_X
        if label:
            labels.append((dy, label))
        x = x0
        for name in chips:
            if x > x0 and x + chip_width(name) > CHIP_X1:
                x, dy = x0, dy + LINE
            placed.append((x, dy, name))
            x += chip_width(name) + 8
        dy += LINE
    return labels, placed, dividers, dy


def side_box(side: tuple, mid: float, t: dict, theme: str) -> tuple[list, float, float]:
    """Draw one dashed side box centred on ``mid``, with an arrow to the ladder."""
    title, caption, chips, _ = side
    lines = wrap(caption, 28)
    height = 60 + 16 * len(lines) + LINE * len(chips)
    top = mid - height / 2
    fill, stroke = t["side"]
    out = [
        f'<rect x="{SIDE_X}" y="{top}" width="{SIDE_W}" height="{height}" rx="12" '
        f'fill="{fill}" stroke="{stroke}" stroke-dasharray="6 5" stroke-width="1.5"/>'
        f'<text x="{SIDE_X + 16}" y="{top + 28}" font-family="{MONO}" font-size="13.5" '
        f'font-weight="700" fill="{t["ink"]}">{escape(title)}</text>'
    ]
    for i, ln in enumerate(lines):
        out.append(
            f'<text x="{SIDE_X + 16}" y="{top + 48 + 16 * i}" font-family="{SANS}" '
            f'font-size="12.5" fill="{t["muted"]}">{escape(ln)}</text>'
        )
    cy = top + 52 + 16 * len(lines)
    for name in chips:
        out.append(chip(SIDE_X + 16, cy, name, t))
        cy += LINE
    out.append(
        f'<path d="M{SIDE_X - 4} {mid} L{ROW_X + ROW_W + 6} {mid}" '
        f'stroke="{t["arrow"]}" stroke-width="2" marker-end="url(#ah-{theme})"/>'
    )
    return out, top, top + height


def build(theme: str) -> str:
    t = THEMES[theme]
    out = [
        f'<defs><marker id="ah-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["arrow"]}"/></marker></defs>'
    ]
    y = TOP
    rows_y = {}
    for badge, role, grammar, caption, groups, key, borrowed in ROWS:
        bg, stroke, ink = t["rows"][key]
        labels, placed, dividers, used = lay_out(groups)
        height = 50 + used
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
            f'<text x="{CHIP_X1}" y="{y + 31}" text-anchor="end" font-family="{SANS}" '
            f'font-size="13" fill="{t["muted"]}">{escape(caption)}</text>'
        )
        for dy in dividers:
            out.append(
                f'<line x1="{LABEL_X}" y1="{y + 44 + dy}" x2="{CHIP_X1}" '
                f'y2="{y + 44 + dy}" stroke="{stroke}" stroke-opacity="{t["divider"]}" '
                'stroke-width="1" stroke-dasharray="2 4"/>'
            )
        for dy, label in labels:
            out.append(
                f'<text x="{LABEL_X}" y="{y + 44 + dy + 15}" '
                f'font-family="{SANS}" font-size="12.5" font-style="italic" '
                f'fill="{t["muted"]}">{escape(label)}</text>'
            )
        for cx, dy, name in placed:
            out.append(chip(cx, y + 44 + dy, name, t))
        y += height + GAP
    # the boxes beside the ladder, each centred on the rung it plugs into
    bottom = y
    for side in SIDES:
        row_y, row_h = rows_y[side[3]]
        drawn, _, end = side_box(side, row_y + row_h / 2, t, theme)
        out.extend(drawn)
        bottom = max(bottom, end)
    # the "built on" strip, across the full width, wrapping its names
    base_y = bottom + 14
    base_w = SIDE_X + SIDE_W - ROW_X
    fill, stroke = t["base"]
    half = (len(BASE) + 1) // 2
    base_lines = [BASE] if len(BASE) <= 6 else [BASE[:half], BASE[half:]]
    base_h = 22 + 22 * len(base_lines)
    out.append(
        f'<rect x="{ROW_X}" y="{base_y}" width="{base_w}" height="{base_h}" rx="10" '
        f'fill="{fill}" stroke="{stroke}"/>'
    )
    for i, names in enumerate(base_lines):
        bold = " · ".join(
            f'<tspan font-weight="700" fill="{t["ink"]}">{escape(n)}</tspan>'
            for n in names
        )
        lead = "built on: " if i == 0 else ""
        out.append(
            f'<text x="{ROW_X + base_w / 2}" y="{base_y + 27 + 22 * i}" '
            f'text-anchor="middle" font-family="{SANS}" font-size="14" '
            f'fill="{t["muted"]}">{lead}{bold}</text>'
        )
    height = base_y + base_h + 20
    label = (
        "pyrox-gp on the GeoStack role ladder: conditioning, the evidence "
        "factors, the variational objectives and the samplers as algorithms; the "
        "kernels, GP priors, likelihoods, guides, posteriors and pathwise "
        "samplers as models; the inference strategies and the CG solver as "
        "strategies; the predict and density methods and the latent-factor "
        "functions as primitives; the inducing features as structure. Beside the "
        "ladder, the setup functions and gaussx's solver strategies."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height:.0f}" '
        f'viewBox="0 0 960 {height:.0f}" role="img" aria-label="{escape(label)}">\n'
        "<title>pyrox-gp on the role ladder</title>\n" + "\n".join(out) + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``pyrox_gp_roles-light.svg`` and ``pyrox_gp_roles-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"pyrox_gp_roles-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
