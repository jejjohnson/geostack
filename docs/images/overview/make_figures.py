"""Generate the overview's figures as standalone SVG files.

Run from the repo root (any Python 3.12+, no dependencies)::

    python docs/images/overview/make_figures.py

Each figure is written next to this script as ``<name>.svg`` and referenced
from ``docs/overview.md``. Every SVG carries its own light card background,
so it reads the same in the light and dark site themes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from itertools import pairwise
from pathlib import Path


OUT = Path(__file__).resolve().parent

FONT = "Inter, 'Helvetica Neue', Arial, 'DejaVu Sans', sans-serif"
INK = "#1e293b"
MUTED = "#64748b"
LINE = "#94a3b8"
CARD = "#ffffff"
PANEL = "#f8fafc"
BORDER = "#cbd5e1"

# One hue per meaning, used the same way in every figure.
OBS = ("#1d4ed8", "#eff6ff")  # observations, L0-L2, GeoTensor world
STATE = ("#0f766e", "#f0fdfa")  # model states, L3-L4, xarray world
ML = ("#6d28d9", "#f5f3ff")  # learning / ML layers
ENG = ("#b45309", "#fffbeb")  # engineering tools, storage
ORACLE = ("#b45309", "#fff7ed")  # the classical oracle column / step
EXT = (LINE, PANEL)  # external, not owned

DOMAINS = {
    "Ocean": ("#0369a1", "#e0f2fe"),
    "Atmosphere": ("#7e22ce", "#f3e8ff"),
    "Land": ("#15803d", "#dcfce7"),
    "Ice": ("#475569", "#f1f5f9"),
}


@dataclass
class Svg:
    """A figure under construction: a list of SVG elements on a W x H canvas."""

    w: int
    h: int
    label: str
    parts: list[str] = field(default_factory=list)
    markers: set[str] = field(default_factory=set)

    # -- primitives -------------------------------------------------------

    def text(
        self,
        x: float,
        y: float,
        s: str,
        *,
        size: float = 13,
        color: str = INK,
        anchor: str = "middle",
        weight: int = 400,
        italic: bool = False,
    ) -> None:
        style = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}"'
            f' text-anchor="{anchor}" font-weight="{weight}"{style}>'
            f"{escape(s)}</text>"
        )

    def lines(self, x: float, y: float, rows: list[str], *, gap: float = 15, **kw):
        for i, row in enumerate(rows):
            self.text(x, y + i * gap, row, **kw)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        stroke: str = BORDER,
        fill: str = CARD,
        rx: float = 8,
        dashed: bool = False,
        width: float = 1.4,
    ) -> None:
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"'
            f' rx="{rx}" fill="{fill}" stroke="{stroke}"'
            f' stroke-width="{width}"{dash}/>'
        )

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        sub: list[str] | None = None,
        *,
        hue: tuple[str, str] = (BORDER, CARD),
        dashed: bool = False,
        title_size: float = 14,
        sub_size: float = 11.5,
    ) -> None:
        """A rounded box with a bold title and optional muted lines below."""
        stroke, fill = hue
        self.rect(x, y, w, h, stroke=stroke, fill=fill, dashed=dashed)
        sub = sub or []
        block = title_size + len(sub) * (sub_size + 3.5)
        top = y + (h - block) / 2 + title_size * 0.85
        self.text(x + w / 2, top, title, size=title_size, weight=600, color=INK)
        self.lines(
            x + w / 2,
            top + title_size * 0.35 + sub_size + 2,
            sub,
            gap=sub_size + 3.5,
            size=sub_size,
            color=MUTED,
        )

    def _marker(self, color: str) -> str:
        mid = "m" + color.lstrip("#")
        self.markers.add(color)
        return mid

    def arrow(
        self,
        pts: list[tuple[float, float]],
        *,
        color: str = LINE,
        dashed: bool = False,
        width: float = 1.6,
        head: bool = True,
        label: str | None = None,
        at: tuple[float, float] | None = None,
        label_anchor: str = "middle",
        label_color: str = MUTED,
        label_size: float = 11.5,
        curve: bool = False,
        both: bool = False,
    ) -> None:
        """A polyline (or one quadratic curve through the middle point)."""
        if curve and len(pts) == 3:
            (x0, y0), (cx, cy), (x1, y1) = pts
            d = f"M{x0:.1f},{y0:.1f} Q{cx:.1f},{cy:.1f} {x1:.1f},{y1:.1f}"
        else:
            d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        end = f' marker-end="url(#{self._marker(color)})"' if head else ""
        if both:
            end += f' marker-start="url(#{self._marker(color)})"'
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
            f' stroke-linejoin="round"{dash}{end}/>'
        )
        if label:
            lx, ly = at if at else ((pts[0][0] + pts[-1][0]) / 2, pts[0][1] - 7)
            self.text(
                lx, ly, label, size=label_size, color=label_color, anchor=label_anchor
            )

    def chip(
        self,
        x: float,
        y: float,
        s: str,
        *,
        hue: tuple[str, str] = (BORDER, CARD),
        dashed: bool = False,
        size: float = 12.5,
        h: float = 26,
        pad: float = 11,
    ) -> float:
        """A pill with one label; returns its width so callers can flow chips."""
        w = _width(s, size) + 2 * pad
        self.rect(x, y, w, h, stroke=hue[0], fill=hue[1], rx=h / 2, dashed=dashed)
        self.text(x + w / 2, y + h / 2 + size * 0.36, s, size=size, color=INK)
        return w

    # -- output -----------------------------------------------------------

    def render(self) -> str:
        defs = "".join(
            f'<marker id="m{c.lstrip("#")}" viewBox="0 0 10 10" refX="9" refY="5"'
            f' markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in sorted(self.markers)
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}"'
            f' width="{self.w}" height="{self.h}" role="img"'
            f' aria-label="{escape(self.label)}" font-family="{FONT}">'
            f"<defs>{defs}</defs>"
            f'<rect x="0.5" y="0.5" width="{self.w - 1}" height="{self.h - 1}"'
            f' rx="14" fill="{CARD}" stroke="{BORDER}"/>'
            + "".join(self.parts)
            + "</svg>\n"
        )


def _width(s: str, size: float) -> float:
    """Approximate rendered width of ``s`` in a sans font at ``size`` px."""
    narrow = sum(c in "ilj.,:;|' ·()[]-ftrI1" for c in s)
    wide = sum(c in "mwMW@" for c in s)
    return size * (0.6 * (len(s) - narrow - wide) + 0.32 * narrow + 0.86 * wide)


# -- figures ------------------------------------------------------------------


def modeling_cycle() -> Svg:
    f = Svg(
        680,
        710,
        "The modeling cycle: the real-world system is both modelled (a"
        " conceptual model, written as equations and code, makes predictions)"
        " and observed (a sensor records measurements, processed into"
        " observations). The tracks meet where predictions are compared with"
        " observations. A random misfit feeds the inner loop, which estimates"
        " the state and parameters and runs the model again; a systematic"
        " misfit feeds the outer loop, which changes the model's structure or"
        " the observing system.",
    )
    t = 20  # everything below the outer loop's label
    left, right = 168, 476  # column centres: modeling, observing
    # the outer loop's label, above everything it encloses
    f.text(
        28,
        34,
        "outer loop · slow: change the structure only when validation demands it",
        size=12,
        color=INK,
        weight=500,
        anchor="start",
    )
    f.box(247, 30 + t, 150, 52, "Real-world", ["system"])
    f.arrow([(247, 56 + t), (left, 56 + t), (left, 124 + t)])
    f.arrow([(397, 56 + t), (right, 56 + t), (right, 124 + t)])
    f.text(44, 112 + t, "MODELING", size=11, color=ML[0], anchor="start", weight=700)
    f.text(561, 112 + t, "OBSERVING", size=11, color=OBS[0], anchor="end", weight=700)
    # row 1: what is modelled and what observes it
    f.box(78, 126 + t, 180, 52, "Conceptual model")
    f.box(391, 126 + t, 170, 52, "Sensor", hue=OBS)
    # row 2; every step label sits just right of its arrow
    f.box(44, 226 + t, 118, 50, "Equations", ["analysis"], hue=ML)
    f.box(174, 226 + t, 118, 50, "Code", ["simulation"], hue=ML)
    f.box(391, 226 + t, 170, 50, "Measurements", hue=OBS)
    f.arrow([(left, 178 + t), (left, 196 + t), (103, 196 + t), (103, 224 + t)])
    f.arrow([(left, 196 + t), (233, 196 + t), (233, 224 + t)])
    f.text(109, 214 + t, "formulate", size=11.5, color=MUTED, anchor="start")
    f.text(239, 214 + t, "implement", size=11.5, color=MUTED, anchor="start")
    f.arrow([(right, 178 + t), (right, 224 + t)])
    f.text(right + 6, 206 + t, "record", size=11.5, color=MUTED, anchor="start")
    # row 3: the two things compared
    f.box(78, 324 + t, 180, 50, "Predictions", hue=ML)
    f.box(391, 324 + t, 170, 50, "Observations", hue=OBS)
    f.arrow([(103, 276 + t), (103, 300 + t), (160, 300 + t), (160, 322 + t)])
    f.arrow([(233, 276 + t), (233, 300 + t), (176, 300 + t), (176, 322 + t)])
    f.text(109, 293 + t, "derive", size=11.5, color=MUTED, anchor="start")
    f.text(239, 293 + t, "simulate", size=11.5, color=MUTED, anchor="start")
    f.arrow([(right, 276 + t), (right, 322 + t)])
    f.text(right + 6, 304 + t, "process", size=11.5, color=MUTED, anchor="start")
    # compare
    f.box(232, 420 + t, 180, 56, "Compare", ["predictions vs observations"], hue=ENG)
    f.arrow([(left, 374 + t), (left, 448 + t), (230, 448 + t)], color=ML[0])
    f.arrow([(right, 374 + t), (right, 448 + t), (414, 448 + t)], color=OBS[0])
    # the misfit decides the loop
    f.box(
        44,
        536 + t,
        250,
        72,
        "Estimate",
        ["state x: data assimilation", "parameters θ: calibration"],
        hue=STATE,
    )
    f.box(
        372,
        536 + t,
        210,
        72,
        "Refine the model",
        ["equations, code,", "sensors, processing"],
    )
    f.arrow([(300, 476 + t), (300, 500 + t), (169, 500 + t), (169, 534 + t)])
    f.arrow([(344, 476 + t), (344, 500 + t), (477, 500 + t), (477, 534 + t)])
    f.text(234, 494 + t, "random misfit", size=11.5, color=MUTED)
    f.text(410, 494 + t, "systematic misfit", size=11.5, color=MUTED)
    # inner loop: same model, new state and parameters, run again
    f.arrow(
        [(44, 572 + t), (24, 572 + t), (24, 349 + t), (76, 349 + t)],
        color=STATE[0],
    )
    f.text(
        44,
        634 + t,
        "inner loop · fast: update x and θ, run the same model again",
        size=12,
        color=STATE[0],
        weight=500,
        anchor="start",
    )
    # outer loop: round the whole figure, back to the conceptual model
    f.arrow(
        [
            (582, 572 + t),
            (664, 572 + t),
            (664, 14),
            (10, 14),
            (10, 152 + t),
            (76, 152 + t),
        ],
        color=INK,
    )
    # ...and, dashed, to the observing system
    f.arrow([(664, 152 + t), (563, 152 + t)], color=INK, dashed=True)
    f.text(612, 124 + t, "more or", size=11.5, color=MUTED)
    f.text(612, 138 + t, "better data", size=11.5, color=MUTED)
    return f


DERIVED = ("#be123c", "#fff1f2")  # L5 derived products (extremes, indicators)
LEVELS = [
    ("L0", "Unstructured obs", ["raw instrument output"]),
    ("L1", "Structured obs", ["calibrated,", "georeferenced"]),
    ("L2", "Retrieved obs", ["per-pixel geophysical", "variables"]),
    ("L3", "Analysis", ["regular grid,", "gap-filled"]),
    ("L4", "Reanalysis", ["forecast; dynamical", "model output"]),
    ("L5", "Derived products", ["extremes, indicators,", "impacts"]),
]
STEPS = [
    "read + calibrate",
    "retrieve",
    "grid + gap-fill",
    "model + assimilate",
    "derive",
]


# lifecycle columns: label gutter, column width, gap (wide enough to name the step)
LC_X0, LC_CW, LC_GAP = 150, 134, 62
LC_CX = [LC_X0 + i * (LC_CW + LC_GAP) for i in range(6)]
LC_HUES = [OBS, OBS, OBS, STATE, STATE, DERIVED]
LC_W = LC_CX[-1] + LC_CW + 24


def _lifecycle_levels(f: Svg, top: float) -> float:
    """The six levels with each step named on its own arrow; returns the bottom."""
    cx, cw = LC_CX, LC_CW
    for a, b, label, hue in [
        (0, 2, "observations  (GeoTensor world)", OBS),
        (3, 4, "model states  (xarray world)", STATE),
        (5, 5, "derived products", DERIVED),
    ]:
        f.text((cx[a] + cx[b] + cw) / 2, top, label, size=12, color=hue[0], weight=600)
    y = top + 16
    for i, (lvl, name, sub) in enumerate(LEVELS):
        hue = LC_HUES[i]
        f.rect(cx[i], y, cw, 76, stroke=hue[0], fill=hue[1], dashed=i == 5)
        f.text(
            cx[i] + 11, y + 18, lvl, size=11, color=hue[0], anchor="start", weight=700
        )
        f.text(cx[i] + 11, y + 36, name, size=13, color=INK, anchor="start", weight=600)
        f.lines(cx[i] + 11, y + 53, sub, gap=14, size=11, color=MUTED, anchor="start")
    for i, step in enumerate(STEPS):
        a, b = cx[i] + cw, cx[i + 1]
        f.arrow([(a + 2, y + 52), (b - 2, y + 52)], color=INK)
        words = step.split(" + ")
        rows = [words[0] + " +", words[1]] if len(words) == 2 else words
        f.lines((a + b) / 2, y + 22, rows, gap=13, size=11, color=INK, weight=500)
    return y + 76


def lifecycle_science() -> Svg:
    f = Svg(
        LC_W,
        420,
        "The L0 to L5 data lifecycle, science axis: telemetry becomes one"
        " stream of radiances for every geoscience, retrieval fans it out into"
        " ocean, atmosphere, land and ice, each domain carries its own analysis"
        " and reanalysis or forecast, and derived products such as extremes"
        " come last (dashed: examples still to be defined).",
    )
    cx, cw, gap = LC_CX, LC_CW, LC_GAP
    head = _lifecycle_levels(f, 30)
    top, lane_h, lane_gap = head + 40, 46, 12
    lanes = [top + i * (lane_h + lane_gap) for i in range(4)]
    bottom = lanes[-1] + lane_h
    mid = (top + bottom) / 2
    f.text(24, top + 4, "SCIENCE", size=11, color=MUTED, anchor="start", weight=700)
    f.lines(
        24,
        top + 22,
        ["what is", "measured and", "modelled"],
        size=11.5,
        color=MUTED,
        anchor="start",
    )
    # one stream before retrieval, at lane height on the axis of the fan
    f.box(cx[0], mid - 30, cw, 60, "Telemetry", ["counts, packets"])
    f.box(cx[1], mid - 30, cw, 60, "Radiances", ["one stream for", "every geoscience"])
    f.arrow([(cx[0] + cw, mid), (cx[1] - 2, mid)])
    rows = {
        "Ocean": [
            "SST · SSH · colour",
            "gridded SST / SSH",
            "ocean reanalysis",
            "marine heatwaves",
        ],
        "Atmosphere": [
            "CH₄ · CO₂ · aerosol",
            "gridded columns",
            "chemistry forecast",
            "heatwaves · storms",
        ],
        "Land": [
            "LST · soil moisture",
            "gridded land state",
            "land reanalysis",
            "drought · floods",
        ],
        "Ice": ["sea ice · snow", "ice concentration", "ice forecast", "melt extremes"],
    }
    fan_x = cx[1] + cw
    for (name, cells), y in zip(rows.items(), lanes, strict=True):
        hue = DOMAINS[name]
        ym = y + lane_h / 2
        f.arrow(
            [(fan_x, mid), (fan_x + gap * 0.55, ym), (cx[2] - 2, ym)],
            color=hue[0],
            curve=True,
        )
        for i, cell in enumerate(cells):
            x = cx[2 + i]
            f.rect(
                x,
                y,
                cw,
                lane_h,
                stroke=hue[0],
                fill=hue[1] if i < 3 else CARD,
                dashed=i == 3,
            )
            f.text(
                x + 11, y + 19, name, size=11, color=hue[0], anchor="start", weight=700
            )
            f.text(x + 11, y + 36, cell, size=11.5, color=INK, anchor="start")
            if i:
                f.arrow([(x - gap + 2, ym), (x - 2, ym)], color=hue[0], dashed=i == 3)
    f.text(
        fan_x + 4,
        bottom + 24,
        "retrieval fans out: radiances → ocean, atmosphere, land, ice",
        size=11.5,
        color=INK,
        anchor="start",
        italic=True,
    )
    f.text(
        cx[5] + cw,
        bottom + 24,
        "L5: examples, to be defined",
        size=11.5,
        color=DERIVED[0],
        anchor="end",
        italic=True,
    )
    f.h = int(bottom + 44)
    return f


def lifecycle_engineering() -> Svg:
    f = Svg(
        LC_W,
        560,
        "The L0 to L5 data lifecycle, engineering axis: pipekit composes every"
        " level; radtrax retrieves, GeoML grids and gap-fills, GeoModels with"
        " data assimilation model and assimilate, and xtremax derives; geotoolz"
        " and xrtoolz transform, geopatcher patches, GeoTensor and xarray"
        " carry, georeader and xrreader read, geocatalog indexes and geocloud"
        " stores, with the carrier switching from GeoTensor to xarray between"
        " L2 and L3.",
    )
    cx, cw, gap = LC_CX, LC_CW, LC_GAP
    etop = _lifecycle_levels(f, 30) + 40
    rh, rg = 34, 12
    f.text(
        24, etop - 16, "ENGINEERING", size=11, color=MUTED, anchor="start", weight=700
    )
    # compute: what performs the step into each level, two lines per cell
    f.text(24, etop + 27, "compute", size=12.5, color=INK, anchor="start", weight=600)
    f.lines(
        24,
        etop + 42,
        ["the step into", "each level"],
        size=11,
        color=MUTED,
        anchor="start",
    )
    compute = [
        (2, "radtrax", "retrieval", OBS),
        (3, "GeoML", "pyrox · kernellib · vardax OI", ML),
        (4, "GeoModels + DA", "somax · plumax · filterax · vardax", STATE),
        (5, "xtremax", "extremes", DERIVED),
    ]
    for lvl, title, sub, hue in compute:
        x = cx[lvl]
        f.rect(x, etop, cw, 56, stroke=hue[0], fill=hue[1], rx=10)
        f.text(x + cw / 2, etop + 23, title, size=12.5, color=INK, weight=600)
        words, line, out = sub.split(" · "), "", []
        for w in words:  # wrap on the dots to fit the column
            cand = f"{line} · {w}" if line else w
            if _width(cand, 10.5) > cw - 14 and line:
                out.append(line)
                cand = w
            line = cand
        out.append(line)
        f.lines(x + cw / 2, etop + 39, out, gap=12, size=10.5, color=MUTED)
    rows_top = etop + 56 + rg + 6
    split = cx[3] - gap / 2
    span = {  # role: [(first level, last level, label, hue)]
        "compose": [(0, 5, "pipekit  (operators, graphs, cycles, training)", ENG)],
        "transform": [(1, 2, "geoops", OBS), (3, 5, "xrtoolz", STATE)],
        "patch": [(1, 4, "geopatcher  (carrier-neutral fields)", ENG)],
        "carry": [(0, 2, "GeoTensor", OBS), (3, 5, "xarray", STATE)],
        "read": [(0, 2, "georeader", OBS), (3, 5, "xrreader", STATE)],
        "index": [
            (0, 2, "geocatalog  (GeoSlice)", OBS),
            (3, 5, "geocatalog.states  (state rows)", STATE),
        ],
        "store": [
            (
                0,
                5,
                "geocloud  (object storage: COG for observations, Zarr / NetCDF"
                " for states)",
                ENG,
            ),
        ],
    }
    end = rows_top + len(span) * (rh + rg) - rg
    f.arrow(
        [(split, etop - 8), (split, end + 6)], color=STATE[0], dashed=True, head=False
    )
    for r, (role, bars) in enumerate(span.items()):
        y = rows_top + r * (rh + rg)
        f.text(
            24, y + rh / 2 + 4, role, size=12.5, color=INK, anchor="start", weight=600
        )
        for a, b, label, hue in bars:
            x, w = cx[a], cx[b] + cw - cx[a]
            f.rect(x, y, w, rh, stroke=hue[0], fill=hue[1], rx=rh / 2)
            f.text(x + w / 2, y + rh / 2 + 4.5, label, size=12.5, color=INK, weight=500)
    f.text(
        split,
        end + 24,
        "carrier switch: GeoTensor → xarray",
        size=11.5,
        color=STATE[0],
        weight=600,
    )
    f.h = int(end + 44)
    return f


IN_SITU = {
    "Ocean": "Argo · CTD · moorings · gliders",
    "Atmosphere": "radiosondes · TCCON · flux towers",
    "Land": "soil-moisture probes · river gauges",
    "Ice": "mass-balance buoys · snow pits",
}


def two_streams() -> Svg:
    f = Svg(
        1000,
        612,
        "Two streams converge: the satellite chain runs L0 to L5; in-situ"
        " networks rise from below, join it by spatiotemporal matchup between"
        " L2 and L3, and enter L4 directly through assimilation. Matchup also"
        " yields the pairs that validate and bias-correct the satellite stream.",
    )
    x, w, h = 40, 250, 56
    ys = [80, 170, 260, 350, 440, 530]
    f.text(x + w / 2, 40, "Satellite  (top-down)", size=13, color=OBS[0], weight=700)
    f.text(x + w / 2, 58, "GeoTensor to L2, xarray from L3", size=11, color=MUTED)
    for i, ((lvl, name, _), y) in enumerate(zip(LEVELS, ys, strict=True)):
        hue = LC_HUES[i]
        f.rect(x, y, w, h, stroke=hue[0], fill=hue[1], dashed=i == 5)
        f.text(x + 14, y + 22, lvl, size=11, color=hue[0], anchor="start", weight=700)
        f.text(x + 14, y + 41, name, size=13.5, color=INK, anchor="start", weight=600)
    for i, step in enumerate(STEPS):
        y0, y1 = ys[i] + h, ys[i + 1]
        f.arrow(
            [(x + w / 2, y0 + 2), (x + w / 2, y1 - 2)],
            color=INK,
            label=step,
            at=(x + w / 2 + 12, (y0 + y1) / 2 + 4),
            label_anchor="start",
            label_color=INK,
        )
    # The matchup node sits on the L2 -> L3 step.
    mx, my = 450, (ys[2] + h + ys[3]) / 2
    f.rect(mx - 82, my - 24, 164, 48, stroke=ENG[0], fill=ENG[1], rx=24)
    f.text(mx, my - 3, "matchup", size=13.5, color=INK, weight=600)
    f.text(mx, my + 13, "space + time window", size=11, color=MUTED)
    f.arrow(
        [(x + w, ys[2] + h / 2), (mx - 40, ys[2] + h / 2), (mx - 40, my - 26)],
        color=ENG[0],
    )
    f.arrow(
        [(mx - 40, my + 24), (mx - 40, ys[3] + h / 2), (x + w + 2, ys[3] + h / 2)],
        color=ENG[0],
    )
    # Matchup pairs: validation and bias correction.
    px, py, pw, ph = 590, 196, 370, 70
    f.rect(px, py, pw, ph, stroke=ENG[0], fill=ENG[1])
    f.text(
        px + 18,
        py + 27,
        "Matchup pairs",
        size=14,
        color=INK,
        anchor="start",
        weight=700,
    )
    f.text(
        px + 18,
        py + 50,
        "validation · bias correction → the misfits of the cycle",
        size=11.5,
        color=INK,
        anchor="start",
    )
    f.arrow(
        [(mx + 40, my - 24), (mx + 40, py + ph / 2), (px - 2, py + ph / 2)],
        color=ENG[0],
    )
    # In-situ networks rise from below.
    ix, iw, iy, ih = 590, 370, 400, 186
    f.text(
        ix + iw / 2,
        iy - 30,
        "In-situ  (bottom-up)",
        size=13,
        color=STATE[0],
        weight=700,
    )
    f.text(ix + iw / 2, iy - 12, "point tables, xarray", size=11, color=MUTED)
    f.rect(ix, iy, iw, ih, stroke=STATE[0], fill=STATE[1])
    f.text(
        ix + 18,
        iy + 28,
        "In-situ networks",
        size=14,
        color=INK,
        anchor="start",
        weight=700,
    )
    for k, (name, nets) in enumerate(IN_SITU.items()):
        y = iy + 56 + k * 24
        f.text(
            ix + 18,
            y,
            name,
            size=11.5,
            color=DOMAINS[name][0],
            anchor="start",
            weight=700,
        )
        f.text(ix + 108, y, nets, size=12, color=INK, anchor="start")
    f.text(
        ix + 18,
        iy + ih - 32,
        "calibrated and sparse at source: no L0 / L1",
        size=11,
        color=MUTED,
        anchor="start",
    )
    f.text(
        ix + 18,
        iy + ih - 14,
        "often with a depth or height axis",
        size=11,
        color=MUTED,
        anchor="start",
    )
    f.arrow([(ix + 40, iy), (ix + 40, my), (mx + 84, my)], color=ENG[0])
    f.arrow(
        [(ix, ys[4] + h / 2), (x + w + 2, ys[4] + h / 2)],
        color=STATE[0],
        dashed=True,
        label="assimilated directly (DA)",
        at=(ix - 12, ys[4] + h / 2 - 8),
        label_anchor="end",
    )
    return f


def three_layers() -> Svg:
    f = Svg(
        1140,
        344,
        "Three layers of transformation: scientific data (observations as"
        " GeoTensors, states as xarray) is harmonised and patched into ML-ready"
        " tensors with provenance, which encoders compress into latent"
        " embeddings; decoders map latents back to fields.",
    )
    w, h, y = 300, 200, 64
    xs = [20, 420, 820]
    # Brackets name the two halves: data engineering, then representation.
    for (x0, x1), label, hue in [
        ((xs[0] + w / 2, xs[1] + w / 2), "ML processing pipeline · GeoStack", ENG),
        ((xs[1] + w / 2, xs[2] + w / 2), "representation learning · GeoML", ML),
    ]:
        f.parts.append(
            f'<path d="M{x0 + 6:.1f},{y - 10} L{x0 + 6:.1f},{y - 18}'
            f' L{x1 - 6:.1f},{y - 18} L{x1 - 6:.1f},{y - 10}" fill="none"'
            f' stroke="{hue[0]}" stroke-width="1.4"/>'
        )
        f.text((x0 + x1) / 2, y - 26, label, size=12, color=hue[0], weight=700)
    layers = [
        (
            "Scientific data",
            "L0 → L4, both streams",
            (LINE, CARD),
            None,
            "(C, H, W)  ·  (time, lat, lon)",
            ["georeader · xrreader · geocatalog", "geoops · xrtoolz"],
        ),
        (
            "ML-ready",
            "tensors with provenance",
            ML,
            "CF-compliant, content-addressed splits",
            "(B, C, h, w)  + metadata",
            ["pipekit-train CachedDataset"],
        ),
        (
            "Embedding",
            "latent representations",
            ML,
            "encoder · decoder · latent dynamics",
            "z : (B, d), one d-vector per sample",
            ["geonnax · pyrox-nn", "pipekit-train, in the registry"],
        ),
    ]
    for x, (name, sub, hue, what, shape, pkgs) in zip(xs, layers, strict=True):
        f.rect(x, y, w, h, stroke=hue[0], fill=hue[1])
        f.text(x + 16, y + 28, name, size=15, color=INK, anchor="start", weight=700)
        sub_color = MUTED if what is None else hue[0]
        f.text(
            x + 16, y + 47, sub, size=11.5, color=sub_color, anchor="start", weight=600
        )
        if what is None:
            cx = x + 16
            cx += f.chip(
                cx, y + 62, "observations · GeoTensor", hue=OBS, size=10.5, h=22, pad=8
            )
            f.chip(cx + 6, y + 62, "states · xarray", hue=STATE, size=10.5, h=22, pad=8)
        else:
            f.text(x + 16, y + 78, what, size=12, color=INK, anchor="start")
        f.text(
            x + 16, y + 104, "shape", size=10.5, color=MUTED, anchor="start", weight=700
        )
        f.text(x + 16, y + 121, shape, size=12.5, color=INK, anchor="start")
        f.text(
            x + 16,
            y + 148,
            "packages",
            size=10.5,
            color=MUTED,
            anchor="start",
            weight=700,
        )
        f.lines(x + 16, y + 165, pkgs, gap=17, size=12, color=INK, anchor="start")
    for a, b, l1, l2, tool in [
        (xs[0] + w, xs[1], "harmonise,", "tensorise", "geopatcher"),
        (xs[1] + w, xs[2], "compress,", "map to latents", "geonnax"),
    ]:
        f.arrow([(a + 2, y + h / 2), (b - 2, y + h / 2)], color=INK)
        f.text((a + b) / 2, y + h / 2 - 22, l1, size=11.5, color=INK)
        f.text((a + b) / 2, y + h / 2 - 8, l2, size=11.5, color=INK)
        f.text((a + b) / 2, y + h / 2 + 18, tool, size=11, color=MUTED, italic=True)
    # The way back: decoders map latents to fields (emulators, generative models).
    by = y + h + 40
    f.arrow(
        [
            (xs[2] + w / 2, y + h + 2),
            (xs[2] + w / 2, by),
            (xs[0] + w / 2, by),
            (xs[0] + w / 2, y + h + 4),
        ],
        color=ML[0],
        dashed=True,
        label="decode: latents back to fields (emulators, generative models)",
        at=(xs[1] + w / 2, by - 8),
        label_color=ML[0],
    )
    return f


SIX = [
    ("1", "Simple model", ["a generative story;", "physics you can simulate"]),
    (
        "2",
        "Model-based inference",
        ["slow, trusted reference;", "classical DA / inversion"],
    ),
    ("3", "Emulator", ["neural surrogate trained", "on step 1; skip if cheap"]),
    ("4", "Emulator-based inference", ["step 2 in seconds,", "not hours"]),
    ("5", "Amortized inference", ["a direct map:", "observations → posterior"]),
]


SIX_MATH = [
    "y = H(f(θ)) + ε",
    "p(θ | y) = p(y | θ) p(θ) / p(y)",
    "f(θ; φ) ≈ f(θ)",
    "p(θ | y), with f(·; φ) for f",
    "q(θ | y; ψ) ≈ p(θ | y)",
]


def six_steps() -> Svg:
    f = Svg(
        1180,
        500,
        "The six-step cycle: five steps from simple model to amortized"
        " inference, each with its defining equation. Step 1 simulates the"
        " training data for steps 3 and 5; step 2 is the oracle that validates"
        " steps 3 to 5; step 6 improves any step against the one before it."
        " Cost per query falls from hours to milliseconds as training cost moves"
        " up front.",
    )
    w, h, y, gap = 204, 124, 116, 28
    xs = [24 + i * (w + gap) for i in range(5)]
    cxs = [x + w / 2 for x in xs]
    for i, ((n, name, sub), x) in enumerate(zip(SIX, xs, strict=True)):
        hue = ORACLE if n == "2" else (ML if n in "345" else STATE)
        f.rect(x, y, w, h, stroke=hue[0], fill=hue[1])
        f.text(
            x + 14,
            y + 24,
            f"STEP {n}",
            size=10.5,
            color=MUTED,
            anchor="start",
            weight=700,
        )
        if n == "2":
            f.text(
                x + w - 14,
                y + 24,
                "inner loop: x, θ",
                size=10.5,
                color=STATE[0],
                anchor="end",
                weight=700,
            )
        f.text(x + 14, y + 44, name, size=12.5, color=INK, anchor="start", weight=700)
        f.lines(x + 14, y + 64, sub, gap=14.5, size=11.5, color=MUTED, anchor="start")
        f.parts.append(
            f'<line x1="{x + 14}" y1="{y + 94}" x2="{x + w - 14}" y2="{y + 94}"'
            f' stroke="{hue[0]}" stroke-opacity="0.25"/>'
        )
        f.text(x + 14, y + 113, SIX_MATH[i], size=12.5, color=hue[0], anchor="start")
        if i < 4:
            f.arrow([(x + w + 2, y + h / 2), (x + w + gap - 2, y + h / 2)], color=INK)
    # Step 2 validates 3, 4 and 5.
    sx, bus = cxs[1], 72
    f.arrow(
        [(sx, y - 2), (sx, bus), (cxs[4], bus)],
        color=ORACLE[0],
        dashed=True,
        head=False,
    )
    for k in (2, 3, 4):
        f.arrow([(cxs[k], bus), (cxs[k], y - 4)], color=ORACLE[0], dashed=True)
    f.text(
        sx + 12,
        bus - 12,
        "oracle: step 2 is the trusted reference that validates steps 3\u20135",
        size=12,
        color=ORACLE[0],
        anchor="start",
        weight=600,
    )
    # Step 1 simulates the training data for the emulator and the amortized map.
    sim = y + h + 30
    f.arrow(
        [(cxs[0], y + h + 2), (cxs[0], sim), (cxs[4], sim)],
        color=STATE[0],
        dashed=True,
        head=False,
    )
    for k in (2, 4):
        f.arrow([(cxs[k], sim), (cxs[k], y + h + 4)], color=STATE[0], dashed=True)
    f.text(
        cxs[0] + 12,
        sim + 18,
        "simulate (θ, y) pairs: the training data for steps 3 and 5",
        size=12,
        color=STATE[0],
        anchor="start",
        weight=600,
    )
    # Step 6: improve any step k against step k - 1 as ground truth.
    iy, ih = sim + 40, 58
    f.rect(xs[0], iy, xs[4] + w - xs[0], ih, stroke=INK, fill=PANEL)
    f.text(
        xs[0] + 14,
        iy + 22,
        "STEP 6",
        size=10.5,
        color=MUTED,
        anchor="start",
        weight=700,
    )
    f.text(
        xs[0] + 14, iy + 42, "Improve", size=14, color=INK, anchor="start", weight=700
    )
    f.text(
        xs[0] + 92,
        iy + 42,
        "outer loop: structure",
        size=10.5,
        color=INK,
        anchor="start",
        weight=700,
    )
    for k in range(1, 5):
        f.text(
            cxs[k], iy + 24, f"improve step {k + 1}", size=11.5, color=INK, weight=600
        )
        f.text(cxs[k], iy + 42, f"ground truth: step {k}", size=11.5, color=MUTED)
    # Cost: per query it falls; training is paid once, up front.
    ay = iy + ih + 40
    f.arrow([(xs[0], ay), (xs[4] + w, ay)], color=MUTED)
    f.text(
        xs[0],
        ay - 10,
        "cost per query",
        size=11,
        color=MUTED,
        anchor="start",
        weight=700,
    )
    for k, label in [(1, "hours"), (3, "seconds"), (4, "milliseconds")]:
        f.parts.append(
            f'<line x1="{cxs[k]}" y1="{ay - 4}" x2="{cxs[k]}" y2="{ay + 4}"'
            f' stroke="{MUTED}" stroke-width="1.4"/>'
        )
        f.text(cxs[k], ay - 10, label, size=11.5, color=INK, weight=600)
    f.text(
        (xs[0] + xs[4] + w) / 2,
        ay + 22,
        "training cost moves up front: steps 3 and 5 train once,"
        " then every query is cheap",
        size=11.5,
        color=MUTED,
    )
    return f


# (transition, step name, running example, hue, cells for steps 1-5), top row first;
# each cell is (method, what it does on the running example)
MATRIX_ROWS = [
    (
        "L4 → L5",
        "derive",
        "heatwave return levels",
        DERIVED,
        [
            ("extreme-value model", "distribution of yearly maxima"),
            ("Bayesian sampling", "tail posterior per cell"),
            ("return-level emulator", "trained on model draws"),
            ("fast posterior", "every grid cell, seconds"),
            ("amortized extremes", "series → tail posterior"),
        ],
    ),
    (
        "L3 → L4",
        "model + assimilate",
        "ocean state from sea-level maps",
        STATE,
        [
            ("quasi-geostrophic model", "steps the sea-level field"),
            ("variational / ensemble", "data assimilation"),
            ("neural ocean model", "learned time step"),
            ("assimilation on emulator", "same cost function, fast"),
            ("direct analysis", "observations → state"),
        ],
    ),
    (
        "L2 → L3",
        "grid + gap-fill",
        "sea level from altimeter tracks",
        STATE,
        [
            ("Gaussian-process prior", "covariance of sea level"),
            ("optimal interpolation", "tracks → map + error"),
            ("neural field", "fit to interpolated maps"),
            ("fast interpolation", "emulated covariance"),
            ("amortized gap-filler", "tracks → map posterior"),
        ],
    ),
    (
        "L1 → L2",
        "retrieve",
        "methane from shortwave infrared",
        OBS,
        [
            ("radiative-transfer model", "radiance = F(methane) + noise"),
            ("optimal estimation", "Gauss-Newton on F"),
            ("radiative-transfer emulator", "F learned from runs"),
            ("estimation on emulator", "same inversion, fast"),
            ("amortized retrieval", "radiance → methane"),
        ],
    ),
    (
        "L0 → L1",
        "read + calibrate",
        "detector counts to radiance",
        OBS,
        [
            ("instrument model", "gain, dark current, nonlinearity"),
            ("vicarious calibration", "fit gains to targets"),
            ("sensor emulator", "learned instrument"),
            ("fast recalibration", "per orbit, on emulator"),
            ("amortized calibration", "counts → radiance"),
        ],
    ),
]


def _cap(s: str) -> str:
    """Sentence case: capitalise the first letter only."""
    return s[:1].upper() + s[1:]


def axes_matrix() -> Svg:
    f = Svg(
        1260,
        640,
        "The axes as a matrix: each lifecycle step from L0 to L5 (rows, each with"
        " a running example) crossed with the six-step cycle (columns) is a"
        " concrete, benchmarkable method. Step 2 is the oracle column; step 3 is"
        " scored against step 1, and steps 4 and 5 against step 2.",
    )
    x0, y0, cw, rh = 236, 196, 204, 70
    heads = ["Simple model", "Model-based", "Emulator", "Emulator-based", "Amortized"]
    hues = [STATE, ORACLE, ML, ML, ML]
    cx = [x0 + j * cw + cw / 2 for j in range(5)]
    # The benchmark rule: each cell is scored against the earlier step that
    # computes the same thing.
    for j0, j1, top, label in [
        (0, 2, 58, "Step 3 scored against step 1"),
        (1, 3, 30, "Steps 4 and 5 scored against step 2"),
    ]:
        f.arrow(
            [(cx[j0], 116), (cx[j0], top), (cx[j1], top), (cx[j1], 112)], color=MUTED
        )
        if j0 == 1:
            f.arrow([(cx[j1], top), (cx[4], top), (cx[4], 112)], color=MUTED)
        f.text(
            cx[j0] + 10,
            top - 7,
            label,
            size=11.5,
            color=INK,
            anchor="start",
            weight=600,
        )
    for j, (head, math) in enumerate(zip(heads, SIX_MATH, strict=True)):
        f.text(cx[j], 134, f"STEP {j + 1}", size=10.5, color=MUTED, weight=700)
        f.text(cx[j], 153, head, size=13, color=INK, weight=700)
        f.text(cx[j], 172, math, size=11.5, color=hues[j][0])
    f.rect(
        x0 + cw + 3, y0 - 4, cw - 6, rh * 5 + 8, stroke=ORACLE[0], fill=ORACLE[1], rx=10
    )
    f.text(
        cx[1], y0 + 5 * rh + 24, "Oracle column", size=11.5, color=ORACLE[0], weight=600
    )
    for i, (trans, task, example, hue, cells) in enumerate(MATRIX_ROWS):
        y = y0 + i * rh
        f.text(
            x0 - 196, y + 22, trans, size=13, color=hue[0], anchor="start", weight=700
        )
        f.text(
            x0 - 196, y + 40, _cap(task), size=12, color=INK, anchor="start", weight=600
        )
        f.text(
            x0 - 196,
            y + 56,
            _cap(example),
            size=11,
            color=MUTED,
            anchor="start",
            italic=True,
        )
        for j, (method, what) in enumerate(cells):
            x = x0 + j * cw
            f.rect(x + 8, y + 6, cw - 16, rh - 12, stroke=BORDER, fill=CARD, rx=6)
            f.text(x + cw / 2, y + 31, _cap(method), size=12.5, color=INK, weight=600)
            f.text(x + cw / 2, y + 49, _cap(what), size=11, color=MUTED)
    f.arrow([(24, y0 + 5 * rh), (24, y0)], color=INK)
    f.text(
        14, y0 + 2.5 * rh, "L0 → L5  (data maturity)", size=12, color=INK, weight=600
    )
    f.parts[-1] = f.parts[-1].replace(
        "<text ", f'<text transform="rotate(-90 14 {y0 + 2.5 * rh})" ', 1
    )
    f.text(
        x0 + 2.5 * cw,
        y0 + 5 * rh + 60,
        "The modeling cycle runs in every cell: simulate, compare, estimate, refine",
        size=12,
        color=MUTED,
        italic=True,
    )
    return f


NUMERICS = ("#475569", "#f1f5f9")  # GeoNumerics: structured numerics

# (name, job, hue, packages); a package in READ_ONLY_PKGS is third-party
LAYERS = {
    "GeoModels": (
        "Physics: forward models of the ocean, plumes and radiation",
        STATE,
        ["plumax", "somax", "radtrax"],
    ),
    "GeoML": (
        "Learning and inference: networks, kernels, GPs, filters, extremes",
        ML,
        [
            "geonnax",
            "kernellib",
            "pyrox",
            "vardax",
            "filterax",
            "gauss_flows",
            "xtremax",
            "optax_bayes",
        ],
    ),
    "GeoNumerics": (
        "Structured linear algebra, PDE operators, spectral methods",
        NUMERICS,
        ["gaussx", "finitevolX", "spectraldiffx"],
    ),
    "GeoStack": (
        "Data in, data out: read, transform, patch, compose",
        OBS,
        ["pipekit", "geotoolz", "xrtoolz", "georeader"],
    ),
}
READ_ONLY_PKGS = {"georeader"}


def _layer(f: Svg, x: float, y: float, w: float, h: float, name: str) -> None:
    job, hue, pkgs = LAYERS[name]
    f.rect(x, y, w, h, stroke=hue[0], fill=hue[1])
    f.text(x + 20, y + 30, name, size=16, color=hue[0], anchor="start", weight=700)
    f.text(x + 20, y + 52, job, size=12.5, color=INK, anchor="start")
    cx, cy = x + 20, y + 66
    for p in pkgs:
        label = f"{p} · third-party, read-only" if p in READ_ONLY_PKGS else p
        cw = _width(label, 11.5) + 18
        if cx + cw > x + w - 16:
            cx, cy = x + 20, cy + 32
        f.chip(
            cx,
            cy,
            label,
            hue=(LINE, CARD),
            dashed=p in READ_ONLY_PKGS,
            size=11.5,
            h=24,
            pad=9,
        )
        cx += cw + 7


def _external(
    f: Svg, x: float, y: float, w: float, title: str, pkgs: list[str]
) -> None:
    f.rect(x, y, w, 60, stroke=EXT[0], fill=EXT[1], dashed=True)
    f.text(x + 20, y + 35, title, size=12.5, color=MUTED, anchor="start", weight=700)
    cx = x + 20 + _width(title, 12.5) + 24
    for p in pkgs:
        cx += f.chip(cx, y + 18, p, hue=(BORDER, CARD), size=11.5, h=24, pad=9) + 7


def tier_stack() -> Svg:
    f = Svg(
        1100,
        608,
        "The stack in four owned layers between external foundations and"
        " external orchestration: GeoStack (data) and GeoNumerics (numerics)"
        " side by side at the bottom, GeoML (learning) on both, GeoModels"
        " (physics) on top. Imports point down; GeoStack and GeoNumerics never"
        " import each other.",
    )
    x, w = 64, 1012
    half = (w - 20) / 2
    _external(
        f,
        x,
        20,
        w,
        "External · orchestration and serving",
        ["Prefect", "DVC", "Hydra", "MLflow", "FastAPI"],
    )
    _layer(f, x, 112, w, 104, "GeoModels")
    _layer(f, x, 248, w, 104, "GeoML")
    _layer(f, x, 400, half, 104, "GeoNumerics")
    _layer(f, x + half + 20, 400, half, 104, "GeoStack")
    _external(
        f,
        x,
        530,
        w,
        "External · foundations",
        ["JAX", "Equinox", "lineax", "NumPyro", "xarray", "rasterio"],
    )
    # Imports point down: GeoModels -> GeoML -> both bottom layers.
    mid = x + w / 2
    f.arrow([(mid, 218), (mid, 246)], color=INK)
    for cx in (x + half / 2, x + half + 20 + half / 2):
        f.arrow([(cx, 354), (cx, 398)], color=INK)
    f.text(
        mid,
        380,
        "side by side: neither imports the other",
        size=11,
        color=MUTED,
        italic=True,
    )
    f.arrow([(36, 112), (36, 504)], color=MUTED)
    f.text(28, 308, "imports point down", size=11.5, color=MUTED)
    f.parts[-1] = f.parts[-1].replace(
        "<text ", '<text transform="rotate(-90 28 308)" ', 1
    )
    return f


# GeoStack zoomed in: (job, what it does, GeoTensor side, xarray side); a None
# xarray side means the GeoTensor-side packages serve both carriers.
GEOSTACK_ROWS = [
    (
        "Read",
        "files into a carrier",
        ["georeader", "geoproducts"],
        ["xrreader"],
    ),
    ("Index", "by space and time", ["geocatalog"], ["geocatalog.states"]),
    (
        "Transform",
        "operators on the carrier",
        ["geoops"],
        ["xrtoolz", "xrcore", "xrgrad", "xreinx", "xrsklearn"],
    ),
    ("Patch", "split, process, stitch", ["geopatcher"], None),
    ("Store", "COG, Zarr, NetCDF", ["geocloud"], None),
]
PROPOSED_PKGS = {"geocatalog.states"}


def _pkg_chip(f: Svg, x: float, y: float, p: str) -> float:
    label = p
    if p in READ_ONLY_PKGS:
        label = f"{p} · third-party, read-only"
    elif p in PROPOSED_PKGS:
        label = f"{p} · proposed"
    dashed = p in READ_ONLY_PKGS or p in PROPOSED_PKGS
    return f.chip(x, y, label, hue=(LINE, CARD), dashed=dashed, size=11.5, h=24, pad=9)


def tier4_map() -> Svg:
    f = Svg(
        1100,
        578,
        "GeoStack zoomed in: one pipekit core at the bottom; above it two"
        " carriers side by side, GeoTensor for observations and xarray for"
        " states, row by row by job (read, index, transform, patch, store);"
        " carrier-neutral ML loops on top.",
    )
    x0, w = 24, 1052
    lw = 200  # row-label gutter
    # ML loops on top: carrier-neutral.
    f.rect(x0, 20, w, 76, stroke=ML[0], fill=ML[1])
    f.text(x0 + 18, 46, "ML loops", size=14, color=ML[0], anchor="start", weight=700)
    f.text(x0 + 18, 66, "carrier-neutral", size=11.5, color=INK, anchor="start")
    cx = x0 + lw
    for p, what in [
        ("pipekit-cycle", "DA, time stepping"),
        ("pipekit-train", "training loops"),
        ("pipekit-experiment", "registry, tracking"),
        ("pipekit-evaluate", "benchmark ladder"),
    ]:
        cx += (
            f.chip(cx, 46, f"{p} · {what}", hue=(LINE, CARD), size=11.5, h=24, pad=9)
            + 8
        )
    # Two carrier columns.
    gx, gw = x0 + lw, 420
    xx, xw = gx + gw + 12, w - lw - gw - 12
    y0, rh = 150, 58
    for x, cw, hue, title, sub in [
        (
            gx,
            gw,
            OBS,
            "GeoTensor",
            "rasters with georeferencing · observations, L0\u2013L2",
        ),
        (xx, xw, STATE, "xarray", "labelled cubes · states, L3\u2013L5"),
    ]:
        f.rect(x, y0 - 30, cw, len(GEOSTACK_ROWS) * rh + 36, stroke=hue[0], fill=hue[1])
        f.text(
            x + 16, y0 - 10, title, size=13, color=hue[0], anchor="start", weight=700
        )
        f.text(
            x + 16 + _width(title, 13) + 10,
            y0 - 10,
            sub,
            size=11,
            color=MUTED,
            anchor="start",
        )
    for i, (job, what, geo, xr) in enumerate(GEOSTACK_ROWS):
        y = y0 + i * rh
        f.parts.append(
            f'<line x1="{x0}" y1="{y + 2}" x2="{x0 + w}" y2="{y + 2}"'
            f' stroke="{BORDER}" stroke-opacity="0.6"/>'
        )
        f.text(x0 + 18, y + 26, job, size=13, color=INK, anchor="start", weight=700)
        f.text(x0 + 18, y + 44, what, size=11, color=MUTED, anchor="start")
        if xr is None:  # one package serves both carriers: centre it on the seam
            label = f"{geo[0]} · either carrier"
            cw = _width(label, 11.5) + 18
            f.chip(xx - 6 - cw / 2, y + 17, label, hue=ENG, size=11.5, h=24, pad=9)
            continue
        for x, pkgs in [(gx, geo), (xx, xr)]:
            cx = x + 16
            for p in pkgs:
                cx += _pkg_chip(f, cx, y + 17, p) + 7
    # pipekit core at the bottom.
    by = y0 + len(GEOSTACK_ROWS) * rh + 26
    f.rect(x0, by, w, 76, stroke=ENG[0], fill=ENG[1])
    f.text(
        x0 + 18,
        by + 26,
        "pipekit core",
        size=14,
        color=ENG[0],
        anchor="start",
        weight=700,
    )
    f.text(
        x0 + 18,
        by + 46,
        "Operator, Sequential, Graph:",
        size=11.5,
        color=INK,
        anchor="start",
    )
    f.text(
        x0 + 18,
        by + 62,
        "the algebra both sides share",
        size=11.5,
        color=INK,
        anchor="start",
    )
    cx = x0 + lw
    for p, what in [
        ("pipekit", "operators and graphs"),
        ("pipekit-array", "Array API operators"),
        ("pipekit-jax", "JAX model wrapper, diffrax forward model"),
    ]:
        cx += (
            f.chip(
                cx, by + 26, f"{p} · {what}", hue=(LINE, CARD), size=11.5, h=24, pad=9
            )
            + 8
        )
    f.h = int(by + 96)
    return f


DG_PAGE = OUT.parents[1] / "api" / "dependency_graph.md"
# (layer, hue, x0 and x1 as fractions of the width, rows from the top tier down);
# every row lists (repo, x as a fraction of the width)
DG_LAYERS = [
    (
        "GeoModels",
        STATE,
        0.0,
        1.0,
        [[("plumax", 0.58)], [("radtrax", 0.22), ("somax", 0.66)]],
    ),
    (
        "GeoML",
        ML,
        0.0,
        1.0,
        [
            [("vardax", 0.84)],
            [("pyrox", 0.2), ("filterax", 0.42)],
            [("optax_bayes", 0.06), ("kernellib", 0.28), ("xtremax", 0.95)],
            [("gauss_flows", 0.12), ("geonnax", 0.47)],
        ],
    ),
    (
        "GeoNumerics",
        NUMERICS,
        0.0,
        0.49,
        [[("finitevolX", 0.38)], [("spectraldiffx", 0.28)], [("gaussx", 0.16)]],
    ),
    (
        "GeoStack",
        OBS,
        0.51,
        1.0,
        [
            [],
            [("xrtoolz", 0.6), ("geotoolz", 0.86)],
            [("pipekit", 0.72), ("georeader", 0.94)],
        ],
    ),
]


def _dg_edges() -> list[tuple[str, str, bool]]:
    """(source, target, solid) for every edge the generated page draws."""
    text = DG_PAGE.read_text(encoding="utf-8")
    block = text[text.index("```{mermaid}") :]
    block = block[: block.index("\n```")]
    out = []
    for line in block.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in ("-->", "-.->"):
            out.append((parts[0], parts[2], parts[1] == "-->"))
    return out


def dependency_graph() -> Svg:
    f = Svg(
        1200,
        760,
        "How the repositories import each other, by layer: imports point down,"
        " GeoStack and GeoNumerics sit side by side at the bottom. Solid edges"
        " are dependencies or imports, dashed edges optional extras; red edges"
        " form a cycle, a known violation. Drawn from the generated graph on"
        " the dependency-graph page.",
    )
    W, nw, nh, rg = 1200 - 48, 112, 30, 62
    pos: dict[str, tuple[float, float]] = {}
    y = 20
    bottom_top = None
    for name, hue, f0, f1, rows in DG_LAYERS:
        if name == "GeoStack":
            y = bottom_top  # beside GeoNumerics
        if name == "GeoNumerics":
            bottom_top = y
        bh = 40 + len(rows) * rg
        bx, bw = 24 + f0 * W, (f1 - f0) * W
        f.rect(bx, y, bw, bh, stroke=hue[0], fill=hue[1], rx=12)
        f.text(bx + 16, y + 24, name, size=13, color=hue[0], anchor="start", weight=700)
        for r, row in enumerate(rows):
            for repo, fx in row:
                pos[repo] = (24 + fx * W, y + 38 + r * rg)
        y += bh + 22
    edges = _dg_edges()
    missing = {r for e in edges for r in e[:2]} - set(pos)
    if missing:
        raise SystemExit(f"dependency_graph: place {sorted(missing)} in DG_LAYERS")
    adj: dict[str, set[str]] = {}
    for a, b, _ in edges:
        adj.setdefault(a, set()).add(b)

    def reaches(a: str, b: str) -> bool:
        seen, todo = set(), [a]
        while todo:
            v = todo.pop()
            if v == b:
                return True
            if v not in seen:
                seen.add(v)
                todo.extend(adj.get(v, ()))
        return False

    for a, b, solid in edges:
        (ax, ay), (bx2, by2) = pos[a], pos[b]
        red = reaches(b, a)
        color = "#c62828" if red else LINE
        if by2 > ay:  # downward: bottom of the source to the top of the target
            x0, y0, x1, y1 = ax, ay + nh, bx2, by2 - 2
            if red:
                x0, x1 = x0 - 22, x1 + 22
        else:  # upward (a cycle): the source's side up to the target's bottom
            x0, y0, x1, y1 = ax + nw / 2, ay + nh / 2, bx2 + 22, by2 + nh + 2
        dy = (y1 - y0) / 2
        dash = "" if solid else ' stroke-dasharray="5 4"'
        f.parts.append(
            f'<path d="M{x0:.1f},{y0:.1f} C{x0:.1f},{y0 + dy:.1f} {x1:.1f},'
            f'{y1 - dy:.1f} {x1:.1f},{y1:.1f}" fill="none" stroke="{color}"'
            f' stroke-width="{2.2 if red else 1.3}"{dash}'
            f' marker-end="url(#{f._marker(color)})"/>'
        )
    for repo, (x, yy) in pos.items():
        third = repo in READ_ONLY_PKGS
        f.rect(x - nw / 2, yy, nw, nh, stroke=BORDER, fill=CARD, dashed=third)
        f.text(x, yy + 21, repo, size=12.5, color=INK, weight=600)
        if third:
            f.text(x, yy + nh + 14, "third-party", size=10.5, color=MUTED)
    ly = y + 4
    for k, (label, color, dashed) in enumerate(
        [
            ("depends on or imports", LINE, False),
            ("optional extra", LINE, True),
            ("cycle: a known violation", "#c62828", False),
        ]
    ):
        lx = 40 + k * 250
        f.arrow([(lx, ly), (lx + 40, ly)], color=color, dashed=dashed, width=1.4)
        f.text(lx + 50, ly + 4, label, size=11.5, color=MUTED, anchor="start")
    f.h = int(ly + 24)
    return f


# (step, title, method, packages, level, ML role); a package in PROPOSED_PKGS
# is drawn dashed
WALK = [
    (
        "1",
        "Catalog the scenes",
        "find EMIT scenes over the area; mask clouds",
        ["geocatalog"],
        None,
        None,
    ),
    (
        "2",
        "Retrieve methane",
        "matched filter (fast); optimal estimation is its oracle",
        ["georeader", "geoops", "radtrax"],
        "L1 → L2",
        "learned retrieval, optional",
    ),
    (
        "3",
        "Match with in-situ",
        "nearest retrieval within 3 h and 25 km",
        ["geocatalog"],
        "L2 + in-situ",
        None,
    ),
    (
        "4",
        "Grid + gap-fill",
        "Gaussian-process prior on the methane field",
        ["pyrox", "pipekit"],
        "L2 → L3",
        "Gaussian process",
    ),
    (
        "5",
        "Assimilate hourly",
        "ensemble Kalman filter, plume forward model",
        ["pipekit-cycle", "plumax", "filterax"],
        "L3 → L4",
        "emulator, optional",
    ),
]
WALK_ENG = [
    ("6", "Index outputs", "states by space and time", ["geocatalog.states"]),
    ("7", "Evaluate", "against CAMS, aircraft, held-out TCCON", ["pipekit-evaluate"]),
    (
        "8",
        "Register + serve",
        "content-addressed model, an API",
        ["pipekit-experiment", "FastAPI"],
    ),
]
# what flows down the science column, between steps k and k + 1
WALK_DATA = [
    "scene list: paths + footprints",
    "enhancement map: L2 · GeoTensor (H, W)",
    "matchup pairs · M rows",
    "gridded field: L3 · xarray (T, N)",
    "analysis: L4 ensemble · (Nₑ, T, N)",
]


def _walk_box(f, x, y, w, h, n, title, method, pkgs, hue, level=None, ml=None):
    f.rect(x, y, w, h, stroke=hue[0], fill=hue[1])
    f.text(x + 16, y + 26, n, size=14, color=hue[0], anchor="start", weight=700)
    f.text(x + 38, y + 26, title, size=14, color=INK, anchor="start", weight=700)
    if level:
        f.text(
            x + w - 14, y + 26, level, size=11, color=hue[0], anchor="end", weight=700
        )
    f.text(x + 38, y + 45, method, size=11.5, color=MUTED, anchor="start")
    cx = x + 38
    for p in pkgs:
        label = f"{p} · proposed" if p in PROPOSED_PKGS else p
        cx += (
            f.chip(
                cx,
                y + 54,
                label,
                hue=(LINE, CARD),
                dashed=p in PROPOSED_PKGS,
                size=10.5,
                h=20,
                pad=7,
            )
            + 6
        )
    if ml:
        label = f"ML · {ml}"
        f.chip(
            x + w - 14 - _width(label, 10.5) - 14,
            y + 54,
            label,
            hue=ML,
            size=10.5,
            h=20,
            pad=7,
        )


def methane_walkthrough() -> Svg:
    f = Svg(
        1200,
        930,
        "The methane walkthrough: five science steps down the left (catalog,"
        " retrieve, match with in-situ, gap-fill, assimilate), each labelled with"
        " the data that flows to the next, its level, its packages and where ML"
        " enters; three engineering steps along the bottom (index, evaluate,"
        " register and serve). Evaluation feeds the misfit back to assimilation.",
    )
    x, w, h, gap = 230, 500, 80, 60
    ys = [60 + k * (h + gap) for k in range(len(WALK))]
    for (n, title, method, pkgs, level, ml), y in zip(WALK, ys, strict=True):
        hue = OBS if n in "123" else STATE
        _walk_box(f, x, y, w, h, n, title, method, pkgs, hue, level, ml)
    for k, label in enumerate(WALK_DATA[:-1]):
        y0, y1 = ys[k] + h, ys[k + 1]
        f.arrow([(x + 60, y0 + 2), (x + 60, y1 - 2)], color=INK)
        f.text(x + 74, (y0 + y1) / 2 + 4, label, size=11.5, color=INK, anchor="start")
    # Stages, as brackets on the left.
    for label, sub, k0, k1 in [
        ("Stage 1", "retrieve", 0, 1),
        ("Stage 2", "match", 2, 2),
        ("Stage 3", "gap-fill", 3, 3),
        ("Stage 4", "assimilate", 4, 4),
    ]:
        top, bot = ys[k0] + 4, ys[k1] + h - 4
        f.parts.append(
            f'<path d="M{x - 14},{top} L{x - 22},{top} L{x - 22},{bot} L{x - 14},{bot}"'
            f' fill="none" stroke="{MUTED}" stroke-width="1.4"/>'
        )
        mid = (top + bot) / 2
        f.text(x - 34, mid - 2, label, size=12, color=INK, anchor="end", weight=700)
        f.text(x - 34, mid + 14, sub, size=11.5, color=MUTED, anchor="end")
    # Inputs from the right.
    ix = 800
    for k, label in [
        (0, "EMIT scenes · L1 radiances (B, H, W)"),
        (2, "TCCON columns · training / held-out sites"),
    ]:
        cy = ys[k] + h / 2
        f.chip(ix, cy - 13, label, hue=(LINE, CARD), size=11.5, h=26, pad=10)
        f.arrow([(ix - 2, cy), (x + w + 2, cy)], color=MUTED)
    # Engineering lane along the bottom.
    ey, ew, eh = ys[-1] + h + 80, 300, 80
    exs = [x - 10, x - 10 + ew + 20, x - 10 + 2 * (ew + 20)]
    f.arrow([(x + 60, ys[-1] + h + 2), (x + 60, ey - 2)], color=INK)
    f.text(x + 74, ys[-1] + h + 44, WALK_DATA[-1], size=11.5, color=INK, anchor="start")
    for (n, title, method, pkgs), ex in zip(WALK_ENG, exs, strict=True):
        _walk_box(f, ex, ey, ew, eh, n, title, method, pkgs, ENG)
    for a, b in pairwise(exs):
        f.arrow([(a + ew + 2, ey + eh / 2), (b - 2, ey + eh / 2)], color=INK)
    f.parts.append(
        f'<path d="M{exs[0]},{ey - 12} L{exs[0]},{ey - 20} L{exs[2] + ew},{ey - 20}'
        f' L{exs[2] + ew},{ey - 12}" fill="none" stroke="{ENG[0]}" stroke-width="1.4"/>'
    )
    f.text(
        exs[2] + ew,
        ey - 28,
        "Stage 5 · index, evaluate, register: engineering",
        size=12,
        color=ENG[0],
        anchor="end",
        weight=700,
    )
    # References for evaluation.
    rx, ry = x + w + 40, ys[-1] + 4
    f.chip(
        rx,
        ry,
        "CAMS reanalysis · aircraft campaigns",
        hue=(LINE, CARD),
        size=11.5,
        h=26,
        pad=10,
    )
    f.arrow([(exs[1] + ew - 20, ry + 28), (exs[1] + ew - 20, ey - 2)], color=MUTED)
    # The inner loop: evaluation feeds the misfit back to assimilation.
    lx = x + w + 22
    f.arrow(
        [(lx, ey - 2), (lx, ys[-1] + h / 2), (x + w + 2, ys[-1] + h / 2)],
        color=STATE[0],
        dashed=True,
    )
    f.text(
        lx - 8,
        ys[-1] + h + 30,
        "misfit: inner loop, update x and θ",
        size=11.5,
        color=STATE[0],
        anchor="end",
        weight=600,
    )
    f.h = int(ey + eh + 24)
    return f


# (month, phase, name, lands, gates, status); status is shipped, in progress or
# planned, as the month tabs on the overview page say
MONTHS = [
    (
        "1",
        "4a",
        "Framework core",
        "pipekit · pipekit-array · workspace",
        "A1 config round-trip",
        "shipped",
    ),
    (
        "2",
        "4b",
        "Indexes and patchers",
        "geocatalog · geopatcher · carriers",
        "B1 backend parity · C1 patch identity",
        "in progress",
    ),
    (
        "3",
        "4c",
        "States and cycles",
        "geocatalog.states · pipekit-cycle · DA adapters",
        "B2 state rows · D1 kriging oracle · D2 Kalman oracle",
        "in progress",
    ),
    (
        "4",
        "4d",
        "Training and registry",
        "pipekit-train · pipekit-experiment",
        "E1 registry round-trip",
        "in progress",
    ),
    (
        "5",
        "5a",
        "Evaluation",
        "pipekit-evaluate · splitters · baselines",
        "F1 metric identities",
        "in progress",
    ),
    (
        "6",
        "5b",
        "Benchmarks, first cycle",
        "contracts · methane cycle · SSH",
        "G1 end to end",
        "planned",
    ),
]
STATUS = {
    "shipped": ("#15803d", "#dcfce7", False),
    "in progress": (ENG[0], ENG[1], False),
    "planned": (LINE, CARD, True),
}
ALONGSIDE = [  # phases 1-3, one per lower layer
    ("1", "GeoNumerics", NUMERICS, "gaussx · spectraldiffx · finitevolX"),
    ("2", "GeoML", ML, "geonnax · kernellib · pyrox · vardax · filterax · …"),
    ("3", "GeoModels", STATE, "somax · plumax · radtrax"),
]


def roadmap() -> Svg:
    f = Svg(
        1100,
        820,
        "The roadmap: phases 4 and 5 as six months from top to bottom, each with"
        " its status and closed by its gates; phases 1 to 3, one per lower"
        " layer, run alongside and feed the DA adapters in month 3 and the models"
        " trained in month 4; phases 6 and 7 follow the last gate.",
    )
    x, w, bh, step = 24, 640, 64, 116
    ys = [24 + i * step for i in range(len(MONTHS))]
    for i, (month, phase, name, lands, gate, status) in enumerate(MONTHS):
        y = ys[i]
        hue = STATE if i >= 4 else OBS
        f.rect(x, y, w, bh, stroke=hue[0], fill=hue[1])
        f.text(
            x + 18,
            y + 24,
            f"{phase} · MONTH {month}",
            size=10.5,
            color=MUTED,
            anchor="start",
            weight=700,
        )
        f.text(x + 18, y + 46, name, size=14, color=INK, anchor="start", weight=700)
        f.text(x + w - 18, y + 46, lands, size=11.5, color=MUTED, anchor="end")
        sc, sf, sd = STATUS[status]
        sw = _width(status, 10.5) + 16
        f.rect(x + w - 18 - sw, y + 11, sw, 20, stroke=sc, fill=sf, rx=10, dashed=sd)
        f.text(x + w - 18 - sw / 2, y + 25, status, size=10.5, color=INK, weight=600)
        # the gate closes the month: a diamond on the arrow, named
        gy = y + bh + (step - bh) / 2
        f.arrow(
            [(x + 60, y + bh + 2), (x + 60, ys[i] + step - 2 if i < 5 else gy + 26)],
            color=LINE,
        )
        d = 9
        f.parts.append(
            f'<path d="M{x + 60},{gy - d} L{x + 60 + d},{gy} L{x + 60},{gy + d}'
            f' L{x + 60 - d},{gy} Z" fill="{ORACLE[1]}" stroke="{ORACLE[0]}"'
            f' stroke-width="1.4"/>'
        )
        f.text(
            x + 80,
            gy + 4,
            f"gate {gate}",
            size=11.5,
            color=ORACLE[0],
            anchor="start",
            weight=600,
        )
    # Phases 1-3 alongside, one per lower layer.
    px, pw = 700, 376
    f.text(
        px,
        40,
        "PHASES 1\u20133 · ALONGSIDE, PROJECT-DRIVEN",
        size=10.5,
        color=MUTED,
        anchor="start",
        weight=700,
    )
    py = 52
    for n, layer, hue, pkgs in ALONGSIDE:
        f.rect(px, py, pw, 58, stroke=hue[0], fill=hue[1])
        f.text(
            px + 16,
            py + 24,
            f"{n} · {layer}",
            size=13.5,
            color=hue[0],
            anchor="start",
            weight=700,
        )
        f.text(px + 16, py + 44, pkgs, size=11.5, color=INK, anchor="start")
        py += 68
    col_bot = py - 10
    # They meet the chain twice: the DA adapters (month 3), the models (month 4).
    for k, (sx, label) in enumerate(
        [
            (px + 24, "DA adapters: filterax, vardax, plumax"),
            (px + pw - 24, "GeoML models to train"),
        ]
    ):
        ty = ys[2 + k] + bh / 2
        f.arrow(
            [(sx, col_bot + 2), (sx, ty), (x + w + 4, ty)], color=ML[0], dashed=True
        )
        f.text(
            sx + 12 if k == 0 else x + w + 16,
            ty - 8,
            label,
            size=11.5,
            color=ML[0],
            anchor="start",
            weight=600,
        )
    # Phases 6 and 7, after the last gate.
    ay = ys[-1] + step + 6
    f.rect(x, ay, 1052, 74, stroke=EXT[0], fill=EXT[1], dashed=True)
    f.text(
        x + 18,
        ay + 22,
        "AFTER G1 · PULLED BY PROJECTS",
        size=10.5,
        color=MUTED,
        anchor="start",
        weight=700,
    )
    for k, (title, sub) in enumerate(
        [
            (
                "6 · Deferred, each with a trigger",
                "pipekit-jax operator layer · deploy adapters · Slice refactor",
            ),
            (
                "7 · Polish and community",
                "docs site · research journal · benchmark adoption",
            ),
        ]
    ):
        bx = x + 18 + k * 516
        f.text(bx, ay + 46, title, size=13, color=INK, anchor="start", weight=700)
        f.text(bx, ay + 64, sub, size=11.5, color=MUTED, anchor="start")
    f.h = int(ay + 94)
    return f


FIGURES = {
    "modeling_cycle": modeling_cycle,
    "lifecycle_science": lifecycle_science,
    "lifecycle_engineering": lifecycle_engineering,
    "two_streams": two_streams,
    "three_layers": three_layers,
    "six_steps": six_steps,
    "axes_matrix": axes_matrix,
    "tier_stack": tier_stack,
    "tier4_map": tier4_map,
    "dependency_graph": dependency_graph,
    "methane_walkthrough": methane_walkthrough,
    "roadmap": roadmap,
}


def main() -> None:
    for name, build in FIGURES.items():
        (OUT / f"{name}.svg").write_text(build().render(), encoding="utf-8")
        print(f"wrote {name}.svg")


if __name__ == "__main__":
    main()
