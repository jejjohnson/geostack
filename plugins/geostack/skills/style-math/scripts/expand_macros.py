# /// script
# dependencies = ["cyclopts", "loguru", "pyyaml"]
# ///
"""Expand the standard's maths macros into plain LaTeX, for GitHub."""

from pathlib import Path

import cyclopts
from _macros import MACROS_FILE, expand_markdown, load_macros
from loguru import logger


app = cyclopts.App(help=__doc__)


@app.default
def main(source: Path, output: Path | None = None, macros: Path = MACROS_FILE) -> None:
    """Expand every macro in SOURCE.

    Parameters
    ----------
    source
        Markdown file written with the macros.
    output
        Where to write the expanded file; defaults to overwriting SOURCE.
    macros
        The macro definitions.
    """
    text = source.read_text(encoding="utf-8")
    expanded = expand_markdown(text, load_macros(macros))
    target = output or source
    target.write_text(expanded, encoding="utf-8")
    logger.info("expanded {} -> {}", source, target)


if __name__ == "__main__":
    app()
