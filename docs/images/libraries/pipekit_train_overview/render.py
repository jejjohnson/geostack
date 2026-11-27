"""Render the pipekit-train overview schematic from its HTML source to a 2x PNG.

``pipekit_train_overview.html`` is the editable source; ``pipekit_train_overview.png``
next to it is what the pipekit-train README and docs embed, so the schematic
is kept up to date in one place. Re-run after
editing the HTML, from the repo root::

    uv run --no-project --with playwright --with loguru --with cyclopts \\
        python docs/images/libraries/pipekit_train_overview/render.py

Playwright uses its own Chromium (``playwright install chromium``); set
``CHROMIUM_PATH`` to point it at another build. Where Chromium cannot find
``libgbm.so.1``, take it from conda-forge without touching the system::

    export LD_LIBRARY_PATH="$(pixi exec -s libgbm -- sh -c 'echo $CONDA_PREFIX')/lib"
"""

from __future__ import annotations

import os
from pathlib import Path

from cyclopts import App
from loguru import logger
from playwright.sync_api import Browser, Playwright, sync_playwright


HERE = Path(__file__).resolve().parent
app = App(help="Render the HTML schematics in this folder to PNG.")


def _launch(p: Playwright) -> Browser:
    path = os.environ.get("CHROMIUM_PATH")
    if path:
        return p.chromium.launch(executable_path=path)
    return p.chromium.launch()


@app.default
def render(*names: str) -> None:
    """Screenshot each source's ``#diagram`` element into ``<stem>.png``.

    Args:
        names: HTML file names in this folder; none renders every one that
            does not start with an underscore.
    """
    files = [HERE / n for n in names] or sorted(HERE.glob("[!_]*.html"))
    with sync_playwright() as p:
        browser = _launch(p)
        page = browser.new_page(
            device_scale_factor=2, viewport={"width": 2200, "height": 1600}
        )
        for src in files:
            page.goto(src.as_uri())
            page.wait_for_function("document.body.dataset.ready === '1'")
            page.evaluate("document.fonts.ready")
            out = src.with_suffix(".png")
            page.locator("#diagram").screenshot(path=str(out), omit_background=True)
            logger.info("{} -> {}", src.name, out.name)
        browser.close()


if __name__ == "__main__":
    app()
