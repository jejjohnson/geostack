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
        "transform": [(1, 2, "geotoolz", OBS), (3, 5, "xrtoolz", STATE)],
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
            ["georeader · xrreader · geocatalog", "geotoolz · xrtoolz"],
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


def tier4_map() -> Svg:
    f = Svg(
        1100,
        470,
        "Tier 4 zoomed in: the pipekit framework packages, the ML"
        " loop packages built on them, and the infrastructure peers that index"
        " and patch data.",
    )
    groups = [
        (
            "Framework",
            ENG,
            24,
            [
                ("pipekit", ["operators, Sequential,", "Graph; the core"]),
                ("pipekit-array", ["Array API ops;", "NumPy · JAX · CuPy · Torch"]),
                ("pipekit-jax", ["Equinox carrier;", "deferred"]),
            ],
        ),
        (
            "ML loops",
            ML,
            382,
            [
                ("pipekit-cycle", ["DA, time-stepping,", "iterative inference"]),
                ("pipekit-train", ["training loops for", "emulators, amortized inf."]),
                ("pipekit-experiment", ["model registry; MLflow,", "W&B adapters"]),
                ("pipekit-evaluate", ["multidim. evaluation;", "benchmark contracts"]),
            ],
        ),
        (
            "Infrastructure peers",
            OBS,
            740,
            [
                ("geocatalog", ["observation index,", "GeoSlice, L0\u2013L2"]),
                ("geopatcher", ["four-axis patcher;", "Field / Domain"]),
                (
                    "geocatalog.states",
                    ["model-state profile,", "same index, L3\u2013L4"],
                ),
            ],
        ),
    ]
    gw = 336
    for title, hue, gx, items in groups:
        f.rect(gx, 24, gw, 420, stroke=hue[0], fill=hue[1], rx=12)
        f.text(
            gx + 18,
            52,
            title.upper(),
            size=11.5,
            color=hue[0],
            anchor="start",
            weight=700,
        )
        for k, (name, sub) in enumerate(items):
            y = 70 + k * 92
            f.rect(gx + 16, y, gw - 32, 78, stroke=BORDER, fill=CARD)
            f.text(
                gx + 32, y + 26, name, size=14, color=INK, anchor="start", weight=700
            )
            f.lines(
                gx + 32, y + 46, sub, gap=15, size=11.5, color=MUTED, anchor="start"
            )
    f.rect(740 + 16, 70 + 3 * 92, gw - 32, 64, stroke=LINE, fill=PANEL, dashed=True)
    f.text(
        740 + 32,
        70 + 3 * 92 + 26,
        "georeader",
        size=14,
        color=INK,
        anchor="start",
        weight=700,
    )
    f.text(
        740 + 32,
        70 + 3 * 92 + 46,
        "external substrate: GeoTensor, I/O",
        size=11.5,
        color=MUTED,
        anchor="start",
    )
    return f


def dependency_graph() -> Svg:
    f = Svg(
        1320,
        560,
        "How the reference packages import each other: every"
        " pipekit package depends on pipekit, xrtoolz builds on xrtoolz-core"
        " and pipekit, and the geotoolz packages build on georeader;"
        " dashed edges are optional extras.",
    )
    bw, bh = 150, 38
    pos: dict[str, tuple[float, float]] = {}

    def node(name: str, x: float, y: float, hue=(BORDER, CARD), dashed=False):
        pos[name] = (x, y)
        f.rect(
            x - bw / 2, y - bh / 2, bw, bh, stroke=hue[0], fill=hue[1], dashed=dashed
        )
        f.text(x, y + 4.5, name, size=12.5, color=INK, weight=600)

    def edge(a: str, b: str, *, dashed: bool = False) -> None:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        if y0 == y1:
            s = 1 if x1 > x0 else -1
            pts = [(x0 + s * bw / 2, y0), (x1 - s * (bw / 2 + 2), y1)]
        else:
            pts = [(x0, y0 + bh / 2), (x1, y1 - bh / 2 - 2)]
        f.arrow(pts, color=LINE, dashed=dashed, width=1.4)

    for x, w, title, hue in [
        (20, 480, "pipekit", ENG),
        (520, 380, "xrtoolz", STATE),
        (920, 380, "geotoolz", OBS),
    ]:
        f.rect(x, 20, w, 400, stroke=hue[0], fill=hue[1], rx=12)
        f.text(
            x + 16,
            44,
            title.upper() + "  FAMILY",
            size=11.5,
            color=hue[0],
            anchor="start",
            weight=700,
        )

    # pipekit family: every member depends on pipekit (one bus), plus these.
    node("pipekit-jax", 100, 100)
    node("pipekit-evaluate", 260, 100)
    node("pipekit-train", 420, 100)
    node("pipekit-array", 100, 210)
    node("pipekit-cycle", 260, 210)
    node("pipekit-experiment", 420, 210)
    node("pipekit", 260, 370, hue=ENG)
    f.arrow([(100, 119), (260, 191 - 2)], color=LINE, width=1.4)  # jax -> cycle
    edge("pipekit-evaluate", "pipekit-cycle", dashed=True)
    f.arrow([(420, 119), (260 + 20, 191 - 2)], color=LINE, dashed=True, width=1.4)
    edge("pipekit-train", "pipekit-experiment", dashed=True)
    edge("pipekit-cycle", "pipekit-array", dashed=True)
    f.arrow([(60, 290), (460, 290)], color=ENG[0], head=False, width=1.6)
    for x in (100, 260, 420):
        f.arrow([(x, 229), (x, 290)], color=ENG[0], head=False, width=1.6)
    f.arrow([(260, 290), (260, 370 - bh / 2 - 2)], color=ENG[0], width=1.6)
    f.text(
        272,
        316,
        "every pipekit-* package",
        size=11.5,
        color=ENG[0],
        anchor="start",
        weight=600,
    )

    # xrtoolz family.
    node("xrtoolz", 710, 100, hue=STATE)
    node("xrtoolz-einx", 615, 210)
    node("xrtoolz-sklearn", 805, 210)
    node("xrtoolz-core", 710, 310)
    f.arrow([(690, 119), (615, 191 - 2)], color=LINE, width=1.4)
    f.arrow([(730, 119), (805, 191 - 2)], color=LINE, width=1.4)
    edge("xrtoolz", "xrtoolz-core")
    f.arrow([(615, 229), (690, 291 - 2)], color=LINE, width=1.4)
    f.arrow([(805, 229), (730, 291 - 2)], color=LINE, width=1.4)
    f.text(
        710, 372, "xrtoolz also → xrtoolz-grad, xrtoolz-reader", size=11.5, color=MUTED
    )
    f.arrow(
        [(710 - bw / 2, 310), (560, 310), (560, 370), (260 + bw / 2 + 2, 370)],
        color=STATE[0],
        width=1.6,
    )
    f.text(
        345,
        362,
        "xrtoolz-core → pipekit",
        size=11.5,
        color=STATE[0],
        anchor="start",
        weight=600,
    )

    # geotoolz family (all dashed edges are optional extras).
    node("geotoolz", 1020, 100, hue=OBS)
    node("geoproducts", 1200, 100)
    node("geopatcher", 1020, 210)
    node("geocloud", 1200, 210)
    node("geocatalog", 1110, 310)
    edge("geoproducts", "geotoolz", dashed=True)
    edge("geotoolz", "geopatcher", dashed=True)
    f.arrow([(1050, 119), (1170, 191 - 2)], color=LINE, dashed=True, width=1.4)
    edge("geoproducts", "geocloud", dashed=True)
    edge("geopatcher", "geocloud", dashed=True)
    f.arrow([(1085, 291), (1035, 229 + 2)], color=LINE, dashed=True, width=1.4)
    f.arrow([(1135, 291), (1185, 229 + 2)], color=LINE, dashed=True, width=1.4)
    f.arrow(
        [
            (1020 - bw / 2, 100),
            (910, 100),
            (910, 450),
            (230, 450),
            (230, 370 + bh / 2 + 2),
        ],
        color=OBS[0],
        width=1.6,
    )
    f.text(560, 442, "geotoolz → pipekit", size=11.5, color=OBS[0], weight=600)
    node("georeader", 1110, 490, hue=EXT, dashed=True)
    f.arrow([(1110, 420), (1110, 490 - bh / 2 - 2)], color=OBS[0], width=1.6)
    f.text(
        1124,
        450,
        "every geotoolz-family",
        size=11.5,
        color=OBS[0],
        anchor="start",
        weight=600,
    )
    f.text(
        1124,
        465,
        "package → georeader",
        size=11.5,
        color=OBS[0],
        anchor="start",
        weight=600,
    )

    f.arrow([(40, 530), (80, 530)], color=LINE, width=1.4)
    f.text(88, 534, "depends on", size=11.5, color=MUTED, anchor="start")
    f.arrow([(190, 530), (230, 530)], color=LINE, dashed=True, width=1.4)
    f.text(238, 534, "optional extra", size=11.5, color=MUTED, anchor="start")
    return f


WALK = [  # (step, name, level, packages)
    ("1", "Catalog the scenes", "", "geocatalog"),
    ("2", "Read + retrieve", "L1 → L2", "georeader + geotoolz"),
    ("3", "Match with in-situ", "", "geocatalog.matchup"),
    ("4", "Grid + gap-fill", "L2 → L3", "pyrox via pipekit"),
    (
        "5",
        "Data assimilation",
        "L3 → L4",
        "pipekit-cycle.EnsembleDACycle + filterax, plumax",
    ),
    ("6", "Index outputs", "", "geocatalog.states"),
    ("7", "Evaluate", "multi-track", "pipekit-evaluate  (vs CAMS, vs aircraft)"),
    ("8", "Register + serve", "", "pipekit-experiment + FastAPI / LitServe"),
]


def methane_walkthrough() -> Svg:
    f = Svg(
        1000,
        560,
        "The methane attribution walkthrough: eight steps from"
        " cataloguing scenes to serving estimates, each handled by one"
        " package, crossing L1 to L4.",
    )
    y0, rh = 30, 64
    for i, (n, name, lvl, pkg) in enumerate(WALK):
        y = y0 + i * rh
        hue = STATE if lvl in ("L3 → L4",) or i >= 5 else OBS
        f.rect(24, y, 300, 46, stroke=hue[0], fill=hue[1])
        f.text(44, y + 29, n, size=15, color=hue[0], anchor="start", weight=700)
        f.text(70, y + 29, name, size=13.5, color=INK, anchor="start", weight=600)
        if lvl:
            f.chip(342, y + 10, lvl, hue=(BORDER, PANEL), size=11.5, h=24, pad=9)
        f.arrow([(464, y + 23), (522, y + 23)], color=LINE, dashed=True)
        f.rect(526, y, 450, 46, stroke=BORDER, fill=CARD)
        f.text(544, y + 28, pkg, size=13, color=INK, anchor="start")
        if i < len(WALK) - 1:
            f.arrow([(174, y + 46 + 1), (174, y + rh - 1)], color=INK)
    f.h = y0 + len(WALK) * rh + 6
    return f


MONTHS = [  # (month, phase, name, lands, gates)
    ("1", "4a", "Framework core", "pipekit · pipekit-array · workspace", "A1"),
    (
        "2",
        "4b",
        "Indexes and patchers",
        "geocatalog · geopatcher · carriers",
        "B1 · C1",
    ),
    (
        "3",
        "4c",
        "States and cycles",
        "geocatalog.states · pipekit-cycle · DA adapters",
        "B2 · D1 · D2",
    ),
    ("4", "4d", "Training and registry", "pipekit-train · pipekit-experiment", "E1"),
    ("5", "5a", "Evaluation", "pipekit-evaluate · splitters · baselines", "F1"),
    ("6", "5b", "Benchmarks, first cycle", "contracts · methane cycle · SSH", "G1"),
]


def roadmap() -> Svg:
    f = Svg(
        1100,
        900,
        "The roadmap: six monthly phases from top to bottom, each closed by a"
        " gate diamond; phases 1 to 3 run alongside and feed month 3; phases 6"
        " and 7 follow the last gate.",
    )
    x, w, bh, step = 24, 640, 64, 116
    cx, y = x + w / 2, 24
    for i, (month, phase, name, lands, gate) in enumerate(MONTHS):
        hue = ENG if i == 0 else (STATE if i >= 4 else OBS)
        f.rect(x, y, w, bh, stroke=hue[0], fill=hue[1], width=2 if i == 0 else 1.4)
        f.text(
            x + 18,
            y + 26,
            f"{phase} · MONTH {month}",
            size=10.5,
            color=MUTED,
            anchor="start",
            weight=700,
        )
        f.text(x + 18, y + 47, name, size=14, color=INK, anchor="start", weight=700)
        f.text(x + w - 18, y + 39, lands, size=12, color=MUTED, anchor="end")
        gy = y + bh + 26  # centre of the gate diamond
        f.arrow([(cx, y + bh), (cx, gy - 14)], color=LINE)
        d = 14
        f.parts.append(
            f'<polygon points="{cx},{gy - d} {cx + d},{gy} {cx},{gy + d} {cx - d},{gy}"'
            f' fill="{ORACLE[1]}" stroke="{ORACLE[0]}" stroke-width="1.4"/>'
        )
        f.text(
            cx + d + 10,
            gy + 4.5,
            f"gate {gate}",
            size=12,
            color=ORACLE[0],
            anchor="start",
            weight=600,
        )
        f.arrow([(cx, gy + d), (cx, y + step - 2)], color=LINE)
        y += step
    # Phases 1-3 run alongside and meet the chain at month 3.
    px, pw = 700, 376
    f.rect(px, 24, pw, 300, stroke=BORDER, fill=PANEL)
    f.text(
        px + 18,
        50,
        "PHASES 1\u20133 · ALONGSIDE",
        size=10.5,
        color=MUTED,
        anchor="start",
        weight=700,
    )
    side = [
        ("1 · GeoML foundations", "gaussx · geonnax · kernellib · optax_bayes"),
        ("2 · GeoML inference", "pyrox · filterax · vardax · gauss_flows"),
        ("3 · GeoModels", "spectraldiffx · finitevolx · somax · plumax"),
    ]
    for j, (title, pkgs) in enumerate(side):
        by = 66 + j * 82
        f.box(px + 16, by, pw - 32, 66, title, [pkgs], hue=ML)
    m3 = 24 + 2 * step + bh / 2
    f.arrow([(px, m3), (x + w + 2, m3)], color=ML[0], dashed=True)
    f.text(
        px + 18,
        316,
        "← adapters land in month 3",
        size=11.5,
        color=ML[0],
        anchor="start",
        weight=600,
    )
    # After the last gate: phases pulled by projects.
    f.rect(x, y, 1052, 92, stroke=BORDER, fill=CARD, dashed=True)
    f.text(
        x + 18,
        y + 26,
        "AFTER G1 · PULLED BY PROJECTS",
        size=10.5,
        color=MUTED,
        anchor="start",
        weight=700,
    )
    f.box(
        x + 16,
        y + 36,
        500,
        46,
        "6 · Deferred, each with a trigger",
        ["pipekit-jax · deploy adapters · Slice refactor"],
        hue=EXT,
        dashed=True,
        title_size=13,
        sub_size=11,
    )
    f.box(
        x + 536,
        y + 36,
        500,
        46,
        "7 · Polish and community",
        ["docs site · research journal · benchmark adoption"],
        hue=EXT,
        dashed=True,
        title_size=13,
        sub_size=11,
    )
    f.h = int(y + 92 + 24)
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
