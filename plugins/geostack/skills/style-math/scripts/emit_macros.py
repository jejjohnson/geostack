# /// script
# dependencies = ["cyclopts", "loguru", "pyyaml"]
# ///
"""Write the standard's maths macros in the form each renderer expects."""

import json
from enum import StrEnum
from pathlib import Path

import cyclopts
from _macros import MACROS_FILE, load_macros
from loguru import logger


app = cyclopts.App(help=__doc__)


class Target(StrEnum):
    myst = "myst"
    mkdocs_mathjax = "mkdocs-mathjax"
    mkdocs_katex = "mkdocs-katex"
    json = "json"


def _yaml_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def render(target: Target, macros_path: Path = MACROS_FILE) -> str:
    macros = load_macros(macros_path)
    if target is Target.myst:
        lines = ["math:"] + [
            f"  {_yaml_quote(chr(92) + m.name)}: {_yaml_quote(m.body)}" for m in macros
        ]
        return "\n".join(lines) + "\n"
    if target is Target.mkdocs_mathjax:
        tex_macros = {
            m.name: [m.body, m.n_args] if m.n_args else m.body for m in macros
        }
        config = {
            "tex": {
                "inlineMath": [["\\(", "\\)"]],
                "displayMath": [["\\[", "\\]"]],
                "processEscapes": True,
                "processEnvironments": True,
                "macros": tex_macros,
            },
            "options": {"ignoreHtmlClass": ".*|", "processHtmlClass": "arithmatex"},
        }
        return "window.MathJax = " + json.dumps(config, indent=2) + ";\n"
    katex_macros = {"\\" + m.name: m.body for m in macros}
    if target is Target.mkdocs_katex:
        # pymdownx.arithmatex (generic: true) emits \( \) and \[ \]; works with or
        # without mkdocs-material.
        return (
            "function renderMath() {\n"
            '  for (const el of document.querySelectorAll(".arithmatex")) {\n'
            "    renderMathInElement(el, {\n"
            "      delimiters: [\n"
            '        { left: "\\\\(", right: "\\\\)", display: false },\n'
            '        { left: "\\\\[", right: "\\\\]", display: true },\n'
            "      ],\n"
            f"      macros: {json.dumps(katex_macros)},\n"
            "    });\n"
            "  }\n"
            "}\n"
            "if (window.document$) { document$.subscribe(renderMath); }\n"
            'else { document.addEventListener("DOMContentLoaded", renderMath); }\n'
        )
    return json.dumps(katex_macros, indent=2) + "\n"


@app.default
def main(
    target: Target, output: Path | None = None, macros: Path = MACROS_FILE
) -> None:
    """Emit the macros for TARGET.

    Parameters
    ----------
    target
        myst prints a `math:` block to paste under `project:` in myst.yml;
        mkdocs-mathjax and mkdocs-katex write the JavaScript config that mkdocs loads
        through `extra_javascript`.
    output
        File to write; prints to stdout when omitted.
    macros
        The macro definitions.
    """
    text = render(target, macros)
    if output is None:
        print(text, end="")
        return
    output.write_text(text, encoding="utf-8")
    logger.info(
        "wrote {} macros for {} to {}", len(load_macros(macros)), target.value, output
    )


if __name__ == "__main__":
    app()
