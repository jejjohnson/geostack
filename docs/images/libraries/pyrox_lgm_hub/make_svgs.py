"""Write pyrox-lgm's prior hub as light and dark SVGs.

One `LGMPrior` fans out to the seven things a model does with it: the exact
NumPyro factor, the log marginal, conditioning and prediction, Laplace
conditioning, the Laplace evidence as a factor, `inla()` on an `LGM` that
holds the prior, and the moments of the integrated fit. The layout is the
pyrox-gp hub's (``pyrox_gp_hub/``): one object card, a trunk, and one card per
use. jejjohnson/pyrox#488 is the contract it draws. Re-run after editing the
data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pyrox_lgm_hub/make_svgs.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the pyrox-lgm prior hub as light and dark SVGs.")

DOT = " \N{MIDDLE DOT} "
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# The object card: its type, the code that builds it, and what it is.
TITLE = "prior: LGMPrior"
CODE = [
    'trend = RW2(104, name="week")',
    'prior = LGMPrior((trend,), FixedEffects(("z",)))',
]
FOOTER = "sparse Q(θ)" + DOT + "gaussx Cholesky" + DOT + "a pytree"
# One card per use: (the call, what it gives). Every name is public on
# pyrox's main.
USES = [
    ('lgm_factor("y", prior, θ, data, σ²)', "a NumPyro factor, for NUTS" + DOT + "SVI"),
    (
        "prior.log_marginal(θ, data, σ²)",
        "the collapsed evidence, under jit" + DOT + "grad",
    ),
    (
        "prior.condition(θ, data, σ²).predict(new)",
        "mean and variance, include_noise for a band",
    ),
    (
        "prior.condition_nongauss(θ, lik, data)",
        "Laplace: predict" + DOT + "joint draws by sample",
    ),
    (
        'lgm_laplace_factor("y", prior, θ, lik, data)',
        "the Laplace evidence, for NUTS",
    ),
    (
        "result = inla(LGM(prior, likelihood=lik), data)",
        "θ integrated out, no sampling",
    ),
    ("result.predict_moments(new)", "mean and variance over the design of θ"),
]

THEME = {
    "light": {
        "text": "#1F2937",
        "muted": "#6B7280",
        "card": "#F9FAFB",
        "line": "#D1D5DB",
        "arrow": "#9CA3AF",
        "hi_fill": "#EEF2FF",
        "hi_line": "#6366F1",
        "accent": "#4338CA",
    },
    "dark": {
        "text": "#E5E7EB",
        "muted": "#9CA3AF",
        "card": "#111827",
        "line": "#374151",
        "arrow": "#6B7280",
        "hi_fill": "#151838",
        "hi_line": "#818CF8",
        "accent": "#A5B4FC",
    },
}

WIDTH, HEIGHT = 1240, 488
# The object card, the trunk's x, and the use cards' x, width, height and gap.
MX, MY, MW, MH = 20, 174, 430, 140
TRUNK_X, EX, EW, EH, GAP = 495, 550, 670, 52, 14


def text(
    x: float,
    y: float,
    s: str,
    *,
    size: float = 14,
    weight: int = 400,
    fill: str,
    family: str = SANS,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'xml:space="preserve">{escape(s)}</text>'
    )


def line(points: list[tuple[float, float]], colour: str, head: str | None) -> str:
    d = " ".join(f"{'M' if i == 0 else 'L'}{x} {y}" for i, (x, y) in enumerate(points))
    end = f' marker-end="url(#{head})"' if head else ""
    return (
        f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="1.6" '
        f'stroke-linejoin="round"{end}/>'
    )


def card(
    x: float, y: float, w: float, h: float, t: dict[str, str], *, hi: bool = False
) -> str:
    fill, stroke, sw = (
        (t["hi_fill"], t["hi_line"], 2) if hi else (t["card"], t["line"], 1)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>'
    )


def build(mode: str) -> str:
    t = THEME[mode]
    head = f"lgm-hub-{mode}-head"
    parts = [
        f'<defs><marker id="{head}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t["arrow"]}"/></marker></defs>',
        card(MX, MY, MW, MH, t, hi=True),
        text(MX + 18, MY + 34, TITLE, size=15, weight=700, fill=t["text"], family=MONO),
    ]
    for i, s in enumerate(CODE):
        parts.append(
            text(MX + 18, MY + 68 + 22 * i, s, size=12, fill=t["text"], family=MONO)
        )
    parts.append(
        text(
            MX + 18,
            MY + 122,
            FOOTER,
            size=12,
            weight=600,
            fill=t["accent"],
            family=MONO,
        )
    )
    cy = MY + MH / 2
    centres = [20 + EH / 2 + i * (EH + GAP) for i in range(len(USES))]
    parts.append(line([(MX + MW, cy), (TRUNK_X, cy)], t["arrow"], None))
    parts.append(
        line([(TRUNK_X, centres[0]), (TRUNK_X, centres[-1])], t["arrow"], None)
    )
    for c in centres:
        parts.append(line([(TRUNK_X, c), (EX - 3, c)], t["arrow"], head))
    for (name, what), c in zip(USES, centres, strict=True):
        parts.append(card(EX, c - EH / 2, EW, EH, t))
        parts.append(
            text(EX + 18, c + 5, name, size=13, weight=700, fill=t["text"], family=MONO)
        )
        parts.append(
            text(EX + EW - 18, c + 5, what, size=12.5, fill=t["muted"], anchor="end")
        )
    label = (
        "One LGM prior is scored as a NumPyro factor, gives its log marginal, "
        "conditions and predicts at new rows, conditions on a non-Gaussian "
        "likelihood by Laplace, scores its Laplace evidence as a factor, and is "
        "fitted with its hyperparameters integrated out by inla, whose result "
        "predicts the mean and variance."
    )
    body = "\n".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(label)}">\n'
        f"<title>One prior, seven uses</title>\n{body}\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``pyrox_lgm_hub-light.svg`` and ``pyrox_lgm_hub-dark.svg``."""
    for mode in THEME:
        path = HERE / f"pyrox_lgm_hub-{mode}.svg"
        path.write_text(build(mode))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
