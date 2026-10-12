"""Figures for the scoping reports under docs/reports/.

Run ``python docs/images/reports/make_figures.py``; each figure is written next
to this script as ``<name>.svg``. Drawing primitives and the palette come from
the overview's figure script, so every figure in the docs shares one style.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


OUT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "overview_figures", OUT.parent / "overview" / "make_figures.py"
)
assert _spec is not None and _spec.loader is not None
_ov = importlib.util.module_from_spec(_spec)
sys.modules["overview_figures"] = _ov
_spec.loader.exec_module(_ov)

Svg = _ov.Svg
INK, MUTED, LINE, BORDER, PANEL = _ov.INK, _ov.MUTED, _ov.LINE, _ov.BORDER, _ov.PANEL
ML, STATE, ENG, EXT = _ov.ML, _ov.STATE, _ov.ENG, _ov.EXT

STAGES = ["1 · Functions / Any", "2 · Protocols", "3 · Concrete types"]


def r1_trajectories() -> Svg:
    """Report 1: the operator and carrier trajectories, side by side."""
    f = Svg(
        840,
        420,
        "Two trajectories. Operators moved from toolz functions through"
        " Protocols to pipekit's concrete Operator base class, and are settled."
        " Carriers sit at all three stages at once: Any in pipekit's core,"
        " Array API Protocols in pipekit-array, and concrete GeoTensor or"
        " xarray types in geotoolz and xrtoolz.",
    )
    x0, col_w, gap = 140, 210, 24
    # stage headers
    for i, stage in enumerate(STAGES):
        cx = x0 + i * (col_w + gap) + col_w / 2
        f.text(cx, 38, stage.upper(), size=12.5, color=MUTED, weight=700)
    lanes = [
        (
            "Operators",
            "the steps",
            [
                ("toolz functions", ["pipe(x, f, g, h)", "any callable"], EXT, False),
                (
                    "OperatorLike",
                    ["a Protocol: a check,", "not a base class"],
                    EXT,
                    False,
                ),
                (
                    "pipekit Operator",
                    ["config · dispatch · flags", "settled"],
                    ML,
                    True,
                ),
            ],
            True,
        ),
        (
            "Carriers",
            "the data",
            [
                ("Carrier = Any", ["pipekit core"], STATE, False),
                (
                    "Array API",
                    ["pipekit-array: NumPy,", "JAX, CuPy, PyTorch"],
                    STATE,
                    False,
                ),
                ("GeoTensor · xarray", ["geotoolz · xrtoolz"], STATE, False),
            ],
            False,
        ),
    ]
    y = 60
    for name, sub, boxes, progression in lanes:
        f.rect(16, y, 808, 150, stroke=BORDER, fill=PANEL)
        f.text(32, y + 70, name, size=16.5, color=INK, anchor="start", weight=700)
        f.text(32, y + 92, sub, size=13, color=MUTED, anchor="start")
        for i, (title, lines, hue, strong) in enumerate(boxes):
            bx = x0 + i * (col_w + gap)
            f.box(
                bx,
                y + 30,
                col_w,
                90,
                title,
                lines,
                hue=hue,
                dashed=hue is EXT,
                title_size=16,
                sub_size=13.5,
            )
            if strong:
                f.rect(
                    bx - 3, y + 27, col_w + 6, 96, stroke=hue[0], fill="none", width=2.2
                )
            if progression and i < len(boxes) - 1:
                f.arrow(
                    [(bx + col_w + 2, y + 75), (bx + col_w + gap - 2, y + 75)],
                    color=INK,
                )
        if not progression:
            f.text(
                x0 + 1.5 * col_w + gap,
                y + 140,
                "all three in use at once: each library picks the stage that serves it",
                size=13,
                color=MUTED,
                italic=True,
            )
        y += 170
    f.h = int(y + 6)
    return f


R2_STATUS = {
    "as planned": (STATE, False),
    "differs": (ENG, False),
    "new": (ML, False),
    "reserved": (EXT, True),
}
R2_GROUPS = [  # (letter, name, module, status)
    ("A", "Foundations", "_base/", "differs"),
    ("B", "Composition", "compose.py", "as planned"),
    ("C", "Building blocks", "blocks.py", "as planned"),
    ("D", "Control flow", "control.py", "differs"),
    ("E", "Observers", "observe.py", "differs"),
    ("F", "Combination", "combine.py", "as planned"),
    ("G", "Caching", "cache.py", "as planned"),
    ("H", "Quality control", "qc.py", "as planned"),
    ("I", "Shape inference", "signature.py", "differs"),
    ("J", "Parallelism", "parallel.py", "differs"),
    ("K", "Disk cache", "(nothing yet)", "reserved"),
    ("L", "Serialisation", "serial.py", "differs"),
    ("M", "State", "state.py", "differs"),
    ("N", "Protocols", "protocols.py", "new"),
    ("", "Hashing (internal)", "hashing.py", "new"),
]


def r2_groups() -> Svg:
    """Report 2: pipekit 0.0.2's modules, coloured by how each matches the plan."""
    f = Svg(
        840,
        470,
        "pipekit 0.0.2 by group. As planned: composition, building blocks,"
        " combination, caching and quality control. Differs from the plan:"
        " foundations, control flow, observers, shape inference, parallelism,"
        " serialisation and state. New: protocols and the shared hashing module."
        " Group K is reserved for a disk-cache backend and ships nothing yet.",
    )
    x = 24.0
    for label, (hue, dashed) in R2_STATUS.items():
        x += f.chip(x, 18, label, hue=hue, dashed=dashed, size=13) + 12
    cols, w, h, gx, gy, y0 = 4, 186, 92, 16, 16, 64
    for i, (letter, name, module, status) in enumerate(R2_GROUPS):
        hue, dashed = R2_STATUS[status]
        bx = 24 + (i % cols) * (w + gx)
        by = y0 + (i // cols) * (h + gy)
        f.box(
            bx,
            by,
            w,
            h,
            f"{letter} · {name}" if letter else name,
            [module, status],
            hue=hue,
            dashed=dashed,
            title_size=15,
            sub_size=13,
        )
    f.h = int(y0 + 4 * (h + gy) + 6)
    return f


FIGURES = {
    "r1_trajectories": r1_trajectories,
    "r2_groups": r2_groups,
}


def main() -> None:
    for name, build in FIGURES.items():
        (OUT / f"{name}.svg").write_text(build().render(), encoding="utf-8")
        print(f"wrote {name}.svg")


if __name__ == "__main__":
    main()
