# /// script
# dependencies = ["cyclopts", "loguru", "pyyaml"]
# ///
"""Check Markdown maths against the style-math standard and the target renderer.

Errors make the exit code 1; warnings are printed but do not fail the check.
Shapes, roles and one-meaning-per-page cannot be checked mechanically: review those
by hand.
"""

import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cyclopts
from _macros import MACROS_FILE, Span, load_macros, math_spans
from loguru import logger


app = cyclopts.App(help=__doc__)

GREEK = (
    "alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|"
    "rho|varrho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|"
    "Upsilon|Phi|Psi|Omega"
)
LOWER_GREEK = (
    r"\\(?:alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|"
    r"mu|nu|xi|pi|rho|varrho|sigma|tau|upsilon|phi|varphi|chi|psi|omega)"
)

# Spacing and sizing macros GitHub (mobile) renders badly; the standard bans them
# everywhere.
BANNED = [
    r"\!",
    r"\,",
    r"\;",
    r"\:",
    r"\quad",
    r"\qquad",
    r"\boxed",
    r"\big",
    r"\Big",
    r"\bigg",
    r"\Bigg",
    r"\bigl",
    r"\bigr",
    r"\Bigl",
    r"\Bigr",
    r"\tfrac",
    r"\dfrac",
    r"\cdots",
    r"\operatorname",
    r"\mathrlap",
]


class Target(StrEnum):
    github = "github"
    myst = "myst"
    mkdocs = "mkdocs"


@dataclass(frozen=True)
class Finding:
    line: int
    rule: str
    message: str
    error: bool


def _strip_words(text: str) -> str:
    """Drop \\text{…}, \\mathrm{…} and \\op{…} contents: words, not symbols."""
    return re.sub(r"\\(?:text|mathrm|op|operatorname)\{[^{}]*\}", " ", text)


def check_span(span: Span, target: Target, macro_names: list[str]) -> list[Finding]:
    found: list[Finding] = []
    t = span.text

    def add(rule: str, message: str, error: bool = True) -> None:
        found.append(Finding(span.line, rule, message, error))

    for macro in BANNED:
        pattern = re.escape(macro) + (r"(?![A-Za-z])" if macro[-1].isalpha() else "")
        if re.search(pattern, t):
            add(
                "banned",
                f"{macro} is banned (GitHub renders it badly); "
                "use plain spacing or \\frac/\\ldots/\\mathrm",
            )
    if re.search(r"\\mathbf\{\\(?:" + GREEK + r")\}", t):
        add(
            "bold-greek",
            "\\mathbf on a Greek letter renders unbolded; "
            "use \\boldsymbol (or \\gvect/\\gmat)",
        )
    if re.search(r"\\(?:vect|mat)\{\\(?:" + GREEK + r")\}", t):
        add(
            "latin-macro-on-greek",
            "\\vect/\\mat are for Latin letters; use \\gvect/\\gmat for Greek",
        )
    if re.search(r"\\(?:gvect|gmat)\{[A-Za-z]\}", t):
        add(
            "greek-macro-on-latin",
            "\\gvect/\\gmat on a Latin letter gives bold italic; use \\vect/\\mat",
            False,
        )
    if re.search(r"\\bm(?![A-Za-z])", t):
        add(
            "bm",
            "\\bm is not available in MathJax (GitHub); use \\mathbf or \\boldsymbol",
        )
    if re.search(
        r"\^\{(?:T|\\mathsf\{T\}|\\mathrm\{T\}|\\intercal)\}|\^(?:T|\\intercal)(?![A-Za-z_{])",
        t,
    ):
        add("transpose", "transpose is ^\\top only; T is the number of time steps")
    if "\\|" in t:
        add(
            "norm-bar",
            "use \\Vert for norms and KL separators; \\| breaks GitHub tables",
        )
    if re.search(r"(?:\b[pq]|\\mathbb\{P\})\([^()]*(?<!\\)\|", t):
        add("conditioning-bar", "condition with \\mid, never a bare |")
    if re.search(r"\^\{?\*\}?", t):
        add(
            "star",
            "^* reads as a conjugate or adjoint; "
            "use ^\\star for an optimum, _* for a test point",
        )
    if "\\mathbb{1}" in t:
        add(
            "indicator", "\\mathbb{1} renders as a plain 1 in KaTeX; use \\mathbf{1}[…]"
        )
    if "\\int" in t and re.search(
        r"(?<![\\A-Za-z{])d(?:[a-zA-Z]|\\(?:mathbf|boldsymbol|" + GREEK + r"))",
        _strip_words(t),
    ):
        add(
            "differential",
            "the differential is upright: \\mathrm{d}t, \\mathrm{d}\\mathbf{s}",
            False,
        )
    if (
        re.search(r"\\(?:mathbf|boldsymbol)\{(?:\\Sigma|\\Lambda)\}[^$]*", t)
        and "\\Lambda" in t
        and re.search(r"\\Sigma\}?_?\{?[^}]*\}?\^\{-1\}", t)
        and not re.search(r"\\Lambda\}_?\{?[^}=]*\}?\s*=\s*\\boldsymbol\{\\Sigma\}", t)
    ):
        add(
            "parameterisation",
            "covariance inverses and precisions mixed in one expression; "
            "pick one form per page",
            False,
        )
    if re.search(
        r"(?:\||\\vert)\s*\\(?:mathbf|boldsymbol)\{\\?[A-Z][a-z]*\}\s*(?:\||\\vert)", t
    ):
        add("determinant", "|A| is set size; write \\det A", False)
    if "\\bar{F}" in t:
        add("survival", "\\bar{F} reads as a sample average; write 1 - F(y)", False)
    if re.search(r"\\partial_\{?x(?![A-Za-z])", t):
        add(
            "spatial-partial",
            "\\partial_x reads as a derivative in the state; "
            "spatial directions are s_d",
            False,
        )
    if target is Target.github:
        used = [m for m in macro_names if re.search(r"\\" + m + r"(?![A-Za-z])", t)]
        if used:
            add(
                "macro-on-github",
                f"GitHub has no custom macros ({', '.join(used)}); "
                "run expand_macros.py first",
            )
        if not span.display and t != t.strip():
            add("inline-space", "no space just inside $…$")
        if not span.display and span.in_table and re.search(r"(?<!\\)\|", t):
            add(
                "table-bar",
                "a bare | inside a table cell splits the cell; use \\vert or \\mid",
            )
    return found


def check_file(path: Path, target: Target, macro_names: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    text = path.read_text(encoding="utf-8")
    for span in math_spans(text):
        findings.extend(check_span(span, target, macro_names))
    if target is Target.github:
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if (
                stripped.startswith("$$")
                and stripped != "$$"
                and not stripped.endswith("$$")
            ):
                findings.append(
                    Finding(
                        number,
                        "display",
                        "$$ opened mid-line; use a ```math block",
                        True,
                    )
                )
    return findings


@app.default
def main(
    *paths: Path, target: Target = Target.github, macros: Path = MACROS_FILE
) -> None:
    """Check each Markdown file in PATHS.

    Parameters
    ----------
    paths
        Markdown files to check.
    target
        github also enforces GitHub's limits (no custom macros, table bars,
        inline spacing).
    macros
        The macro definitions, used to spot macros left in GitHub-bound files.
    """
    names = [m.name for m in load_macros(macros)]
    n_errors = 0
    for path in paths:
        for f in check_file(path, target, names):
            log = logger.error if f.error else logger.warning
            log("{}:{}: [{}] {}", path, f.line, f.rule, f.message)
            n_errors += f.error
    if n_errors:
        logger.error("{} error(s)", n_errors)
        sys.exit(1)
    logger.info("clean: {} file(s), target {}", len(paths), target.value)


if __name__ == "__main__":
    app()
