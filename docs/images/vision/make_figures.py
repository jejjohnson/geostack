"""Generate the vision page's figures as standalone SVG files.

Run from the repo root (any Python 3.12+, no dependencies)::

    python docs/images/vision/make_figures.py

Each figure is written next to this script as ``<name>.svg`` and referenced
from ``docs/design/vision.md``. Every SVG carries its own light card
background, so it reads the same in the light and dark site themes; the
palette matches ``docs/images/overview/make_figures.py``.
"""

from __future__ import annotations

from html import escape
from itertools import pairwise
from pathlib import Path


OUT = Path(__file__).resolve().parent

FONT = "Inter, 'Helvetica Neue', Arial, 'DejaVu Sans', sans-serif"
INK = "#1e293b"
MUTED = "#64748b"
LINE = "#94a3b8"
CARD = "#ffffff"
BORDER = "#cbd5e1"
ZONE = "#f1f5f9"
BLUE = "#1d4ed8"  # the control spectrum
TEAL = "#0f766e"  # outcomes, "easy to use"
PURPLE = "#6d28d9"  # "fully flexible"


def text(
    x: float,
    y: float,
    s: str,
    *,
    size: float = 13,
    fill: str = INK,
    anchor: str = "middle",
    weight: int = 400,
) -> str:
    """One line of text."""
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>'
    )


def svg(w: int, h: int, label: str, body: list[str], defs: str = "") -> str:
    """A complete SVG document with a card background."""
    card = (
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="14" '
        f'fill="{CARD}" stroke="{BORDER}"/>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" aria-label="{escape(label)}" '
        f'font-family="{escape(FONT)}"><defs>{defs}</defs>{card}'
        + "".join(body)
        + "</svg>\n"
    )


def geo_workflows() -> str:
    """Progressive disclosure: four stops from packaged to fully flexible."""
    ly, r = 160, 9
    xs = (110, 290, 470, 650)
    arrow = (
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        f'<path d="M0 0L10 5L0 10z" fill="{BLUE}"/></marker>'
    )
    body: list[str] = [
        text(
            24,
            34,
            "Geo workflows: from packaged to arbitrarily flexible",
            size=15,
            anchor="start",
            weight=600,
        )
    ]
    # the spectrum line, broken at each stop, ending in an arrow
    ends = [24, *xs, 728]
    for i, (x0, x1) in enumerate(pairwise(ends)):
        start = x0 if i == 0 else x0 + 14
        stop = x1 if i == len(ends) - 2 else x1 - 14
        marker = ' marker-end="url(#arrow)"' if i == len(ends) - 2 else ""
        body.append(
            f'<line x1="{start}" y1="{ly}" x2="{stop}" y2="{ly}" '
            f'stroke="{BLUE}" stroke-width="3"{marker}/>'
        )
    stops = [
        ("L3", ["Packaged pipeline", "+ built-in operators"],
         ["Standard workflows,", "fast and easy"], "emit_plumes.v1.run(...)"),
        ("L2 + pipekit", ["pipekit graph", "+ built-in operators"],
         ["New workflows,", "existing components"], "Graph(read, …, detect)"),
        ("L2, your own", ["pipekit graph", "+ your KernelOps", "+ your operators"],
         ["New components"], "SwathToGrid, Detector"),
        ("L1", ["Primitives on arrays:", "full control,", "your own loop"],
         ["Anything you want"], "vmap(matched_filter_score)"),
    ]  # fmt: skip
    for x, (tag, heads, outcomes, example) in zip(xs, stops, strict=True):
        body.append(
            f'<circle cx="{x}" cy="{ly}" r="{r}" fill="{CARD}" '
            f'stroke="{BLUE}" stroke-width="3"/>'
        )
        top = 136 - 18 * (len(heads) - 1)
        body.append(text(x, top - 22, tag, size=11.5, fill=MUTED))
        for k, line in enumerate(heads):
            body.append(text(x, top + 18 * k, line, weight=600))
        for k, line in enumerate(outcomes):
            body.append(text(x, 190 + 18 * k, line, fill=TEAL, weight=600))
        body.append(text(x, 234, example, size=11.5, fill=MUTED))
    label = (
        "Geo workflows from packaged to arbitrarily flexible: a packaged "
        "pipeline with built-in operators (L3) for standard workflows; a "
        "pipekit graph of built-in operators for new workflows from existing "
        "components; a graph with your own KernelOps and operators for new "
        "components; primitives on arrays in your own loop (L1) for anything."
    )
    return svg(760, 256, label, body, defs=arrow)


def computation_orchestration() -> str:
    """Two axes: computation and orchestration, each packaged to hand-written."""
    cx, cy, xl, xr, yt, yb = 380, 270, 170, 590, 96, 444
    body: list[str] = [
        text(
            24,
            30,
            "Computation and orchestration each go from packaged to hand-written",
            size=15,
            anchor="start",
            weight=600,
        ),
        f'<ellipse cx="{cx}" cy="{cy}" rx="206" ry="168" fill="{ZONE}"/>',
        f'<line x1="{xl}" y1="{cy}" x2="{xr}" y2="{cy}" stroke="{LINE}" '
        'stroke-width="1.25"/>',
        f'<line x1="{cx}" y1="{yt}" x2="{cx}" y2="{yb}" stroke="{LINE}" '
        'stroke-width="1.25"/>',
        # axis ends
        text(cx, 66, "High-level", weight=600),
        text(cx, 84, "orchestration APIs", weight=600),
        text(cx, 468, "Low-level", weight=600),
        text(cx, 486, "orchestration APIs", weight=600),
        text(xl - 10, cy - 4, "Low-level", anchor="end", weight=600),
        text(xl - 10, cy + 14, "computation APIs", anchor="end", weight=600),
        text(xr + 10, cy - 4, "High-level", anchor="start", weight=600),
        text(xr + 10, cy + 14, "computation APIs", anchor="start", weight=600),
        # the two corners
        text(540, 146, "Easy to use", size=18, fill=TEAL, weight=600),
        text(150, 420, "Fully flexible", size=18, fill=PURPLE, weight=600),
        # what sits where
        text(cx - 10, 126, "Built-in runners:", size=12, anchor="end"),
        text(cx - 10, 142, "ETL, train, assimilate", size=12, anchor="end"),
        text(470, 196, "Packaged pipelines", size=12),
        text(470, 212, "emit_plumes.v1.run(...)", size=11.5, fill=MUTED),
        text(cx - 10, 214, "Custom pipekit graph,", size=12, anchor="end"),
        text(cx - 10, 230, "stage by stage", size=12, anchor="end"),
        text(cx - 10, cy - 10, "KernelOp(your kernel)", size=12, anchor="end"),
        text(196, cy + 30, "Primitives on arrays", size=12, anchor="start"),
        text(196, cy + 46, "+ derived geometry", size=12, anchor="start"),
        text(cx + 14, cy + 30, "Operator chains", size=12, anchor="start"),
        text(cx + 14, cy + 46, "Mask | NDVI", size=11.5, fill=MUTED, anchor="start"),
        text(508, cy + 30, "Built-in", size=12, anchor="start"),
        text(508, cy + 46, "operators", size=12, anchor="start"),
        text(cx - 10, 392, "Your own loop:", size=12, anchor="end"),
        text(cx - 10, 408, "lax.scan, async I/O", size=12, anchor="end"),
    ]
    label = (
        "Computation (low-level primitives to high-level built-in operators) "
        "and orchestration (your own loop to high-level runners and packaged "
        "pipelines) are two axes. Packaged pipelines and built-in runners sit "
        "at the easy-to-use end; primitives in your own loop at the fully "
        "flexible end; custom pipekit graphs, KernelOps and operator chains "
        "in between."
    )
    return svg(760, 512, label, body)


FIGURES = {
    "geo_workflows": geo_workflows,
    "computation_orchestration": computation_orchestration,
}

if __name__ == "__main__":
    for name, make in FIGURES.items():
        path = OUT / f"{name}.svg"
        path.write_text(make())
        print("wrote", path.relative_to(OUT.parents[2]))
