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
        1120,
        380,
        "The modeling cycle: the real-world system is both modelled"
        " (abstract, then equations and code, then predictions) and observed"
        " (sensor, measurements, observations); the two tracks meet where"
        " predictions are compared with observations, and the model is refined.",
    )
    f.box(24, 185, 150, 80, "Real-world", ["system"])
    f.text(214, 176, "MODELING", size=11, color=ML[0], anchor="start", weight=700)
    f.text(214, 366, "OBSERVING", size=11, color=OBS[0], anchor="start", weight=700)
    f.box(214, 102, 120, 56, "Abstract")
    f.box(380, 60, 150, 48, "Equations", ["analysis"], hue=ML)
    f.box(380, 152, 150, 48, "Code", ["simulation"], hue=ML)
    f.box(580, 102, 140, 56, "Predictions", hue=ML)
    f.box(214, 292, 120, 56, "Sensor", hue=OBS)
    f.box(380, 292, 150, 56, "Measurements", hue=OBS)
    f.box(580, 292, 140, 56, "Observations", hue=OBS)
    f.box(790, 197, 170, 56, "Compare", ["predictions vs observations"], hue=ENG)
    f.box(1000, 197, 96, 56, "Refine", ["the model"])
    f.arrow([(174, 210), (194, 210), (194, 130), (212, 130)])
    f.arrow([(174, 240), (194, 240), (194, 320), (212, 320)])
    f.arrow([(334, 120), (357, 120), (357, 84), (378, 84)])
    f.arrow([(334, 140), (357, 140), (357, 176), (378, 176)])
    f.arrow(
        [(530, 84), (555, 84), (555, 120), (578, 120)],
        label="derive",
        at=(560, 76),
        label_anchor="start",
    )
    f.arrow(
        [(530, 176), (555, 176), (555, 140), (578, 140)],
        label="simulate",
        at=(560, 194),
        label_anchor="start",
    )
    f.arrow([(334, 320), (378, 320)], label="record", at=(356, 312))
    f.arrow([(530, 320), (578, 320)], label="process", at=(554, 312))
    f.arrow([(720, 130), (875, 130), (875, 195)], color=ML[0])
    f.arrow([(720, 320), (875, 320), (875, 255)], color=OBS[0])
    f.arrow([(960, 225), (998, 225)])
    f.arrow([(1048, 197), (1048, 34), (274, 34), (274, 100)], color=INK)
    f.text(
        661,
        26,
        "iterate: add complexity only when validation demands it",
        size=12,
        color=INK,
        weight=500,
    )
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


def lifecycle() -> Svg:
    f = Svg(
        1250,
        800,
        "The L0 to L5 data lifecycle on two perpendicular axes:"
        " the science axis fans retrievals out into ocean, atmosphere, land"
        " and ice and ends in derived products such as extremes; the"
        " engineering axis maps each level to the GeoStack tools that"
        " transform, patch, carry, read, index and store it.",
    )
    x0, cw, gap = 186, 152, 24  # first column x, column width, gap
    cx = [x0 + i * (cw + gap) for i in range(6)]
    hues = [OBS, OBS, OBS, STATE, STATE, DERIVED]

    # Column headers: the six maturity levels and the step between each pair.
    for i, (lvl, name, sub) in enumerate(LEVELS):
        hue = hues[i]
        f.rect(cx[i], 46, cw, 76, stroke=hue[0], fill=hue[1], dashed=i == 5)
        f.text(cx[i] + 12, 64, lvl, size=11, color=hue[0], anchor="start", weight=700)
        f.text(cx[i] + 12, 82, name, size=13.5, color=INK, anchor="start", weight=600)
        f.lines(cx[i] + 12, 99, sub, gap=14, size=11, color=MUTED, anchor="start")
    for i, step in enumerate(STEPS):
        a, b = cx[i] + cw, cx[i + 1]
        f.arrow([(a + 2, 84), (b - 2, 84)], color=INK)
        f.text((a + b) / 2, 140, step, size=11.5, color=INK, weight=500)
    for a, b, label, hue in [
        (0, 2, "observations  (GeoTensor world)", OBS),
        (3, 4, "model states  (xarray world)", STATE),
        (5, 5, "derived products", DERIVED),
    ]:
        f.text((cx[a] + cx[b] + cw) / 2, 30, label, size=12, color=hue[0], weight=600)

    # Science axis: one stream until retrieval, then one lane per domain.
    top, lane_h, lane_gap = 178, 46, 14
    lanes = [top + i * (lane_h + lane_gap) for i in range(4)]
    bottom = lanes[-1] + lane_h
    f.text(
        24, top - 8, "SCIENCE AXIS", size=11, color=MUTED, anchor="start", weight=700
    )
    f.lines(
        24,
        top + 12,
        ["what is measured", "and modelled"],
        size=11.5,
        color=MUTED,
        anchor="start",
    )
    f.box(cx[0], top, cw, bottom - top, "Telemetry", ["counts, packets"])
    f.box(
        cx[1],
        top,
        cw,
        bottom - top,
        "Radiances",
        ["one stream for", "every geoscience"],
    )
    f.arrow([(cx[0] + cw, (top + bottom) / 2), (cx[1] - 2, (top + bottom) / 2)])
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
            [(fan_x, (top + bottom) / 2), (fan_x + 6, ym), (cx[2] - 2, ym)],
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
                x + 12, y + 19, name, size=11, color=hue[0], anchor="start", weight=700
            )
            f.text(x + 12, y + 36, cell, size=12, color=INK, anchor="start")
            if i:
                f.arrow([(x - gap + 2, ym), (x - 2, ym)], color=hue[0], dashed=i == 3)
    f.text(
        fan_x + 4,
        bottom + 22,
        "retrieval fans out: radiances → ocean, atmosphere, land, ice",
        size=11.5,
        color=INK,
        anchor="start",
        italic=True,
    )
    f.text(
        cx[5] + cw,
        bottom + 22,
        "L5: examples, to be defined",
        size=11.5,
        color=DERIVED[0],
        anchor="end",
        italic=True,
    )

    # Engineering axis: which GeoStack tool owns each level, down to storage.
    etop, rh, rg = 470, 34, 12
    f.text(
        24,
        etop - 14,
        "ENGINEERING AXIS",
        size=11,
        color=MUTED,
        anchor="start",
        weight=700,
    )
    split = cx[3] - gap / 2
    f.arrow(
        [(split, etop - 6), (split, etop + 6 * (rh + rg) + 2)],
        color=STATE[0],
        dashed=True,
        head=False,
    )
    f.text(
        split,
        etop + 6 * (rh + rg) + 18,
        "carrier switch: GeoTensor → xarray",
        size=11.5,
        color=STATE[0],
        weight=600,
    )
    span = {  # role: [(first level, last level, label, hue, dashed)]
        "transform": [
            (1, 2, "geotoolz", OBS, False),
            (3, 4, "xrtoolz", STATE, False),
            (5, 5, "xtremax", DERIVED, False),
        ],
        "patch": [(1, 4, "geopatcher  (carrier-neutral fields)", ENG, False)],
        "carry": [(0, 2, "GeoTensor", OBS, False), (3, 5, "xarray", STATE, False)],
        "read": [(0, 2, "georeader", OBS, False), (3, 5, "xrreader", STATE, False)],
        "index": [
            (0, 2, "geocatalog  (GeoSlice)", OBS, False),
            (3, 5, "statecatalog  (StateSlice)", STATE, False),
        ],
        "store": [
            (0, 2, "geocloud  (object storage, COG)", ENG, False),
            (3, 5, "Zarr / NetCDF on object storage", EXT, True),
        ],
    }
    for r, (role, bars) in enumerate(span.items()):
        y = etop + r * (rh + rg)
        f.text(
            24, y + rh / 2 + 4, role, size=12.5, color=INK, anchor="start", weight=600
        )
        for a, b, label, hue, dashed in bars:
            x, w = cx[a], cx[b] + cw - cx[a]
            f.rect(x, y, w, rh, stroke=hue[0], fill=hue[1], rx=rh / 2, dashed=dashed)
            f.text(x + w / 2, y + rh / 2 + 4.5, label, size=12.5, color=INK, weight=500)
    f.arrow([(150, etop + 2), (150, etop + 6 * (rh + rg) - 14)], color=MUTED)
    f.text(150, etop + 6 * (rh + rg) + 4, "toward storage", size=11, color=MUTED)
    return f


IN_SITU = {
    "Ocean": "Argo · CTD · moorings · gliders",
    "Atmosphere": "radiosondes · TCCON · flux towers",
    "Land": "soil-moisture probes · river gauges",
    "Ice": "mass-balance buoys · snow pits",
}


def two_streams() -> Svg:
    f = Svg(
        1040,
        560,
        "Two streams converge: the satellite chain runs L0 to L4;"
        " in-situ networks in every domain join it by spatiotemporal matchup"
        " between L2 and L3, and enter L4 through assimilation.",
    )
    x, w, h = 60, 250, 58
    ys = [70, 166, 262, 358, 470]
    rows = [
        ("L0", "Raw telemetry"),
        ("L1", "Calibrated radiances"),
        ("L2", "Geophysical variables"),
        ("L3", "Gridded products"),
        ("L4", "Analysed / fused fields"),
    ]
    f.text(x + w / 2, 44, "Satellite  (top-down)", size=13, color=OBS[0], weight=700)
    for i, ((lvl, name), y) in enumerate(zip(rows, ys, strict=True)):
        hue = OBS if i < 3 else STATE
        f.rect(x, y, w, h, stroke=hue[0], fill=hue[1])
        f.text(x + 14, y + 23, lvl, size=11, color=hue[0], anchor="start", weight=700)
        f.text(x + 14, y + 42, name, size=13.5, color=INK, anchor="start", weight=600)
    steps = ["cal + geoloc", "retrieval", "grid + composite", "DA + fusion"]
    for i, step in enumerate(steps):
        y0, y1 = ys[i] + h, ys[i + 1]
        f.arrow(
            [(x + w / 2, y0 + 2), (x + w / 2, y1 - 2)],
            color=INK,
            label=step,
            at=(x + w / 2 + 12, (y0 + y1) / 2 + 4),
            label_anchor="start",
            label_color=INK,
        )
    ix, iw, iy, ih = 650, 360, 214, 176
    f.text(ix + iw / 2, 44, "In-situ  (bottom-up)", size=13, color=STATE[0], weight=700)
    f.text(
        ix + iw / 2,
        (ys[0] + ys[1] + h) / 2 + 4,
        "no L0 / L1 equivalent",
        size=11.5,
        color=MUTED,
        italic=True,
    )
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
        iy + ih - 16,
        "calibrated and sparse; often with a depth or height axis",
        size=11,
        color=MUTED,
        anchor="start",
    )
    # The matchup node sits on the L2 -> L3 step.
    mx, my = 470, (ys[2] + h + ys[3]) / 2
    f.rect(mx - 78, my - 24, 156, 48, stroke=ENG[0], fill=ENG[1], rx=24)
    f.text(mx, my - 3, "matchup", size=13.5, color=INK, weight=600)
    f.text(mx, my + 13, "space + time", size=11, color=MUTED)
    f.arrow([(x + w, ys[2] + h / 2), (mx, ys[2] + h / 2), (mx, my - 26)], color=ENG[0])
    f.arrow([(ix, my), (mx + 80, my)], color=ENG[0])
    f.arrow(
        [(mx, my + 24), (mx, ys[3] + h / 2), (x + w + 2, ys[3] + h / 2)],
        color=ENG[0],
        label="the convergence operation",
        at=(mx + 10, ys[3] + h / 2 + 20),
        label_anchor="start",
    )
    f.arrow(
        [
            (ix + iw / 2, iy + ih),
            (ix + iw / 2, ys[4] + h / 2),
            (x + w + 2, ys[4] + h / 2),
        ],
        color=STATE[0],
        dashed=True,
        label="enters through assimilation",
        at=(ix + iw / 2 - 12, ys[4] + h / 2 - 8),
        label_anchor="end",
    )
    return f


def three_layers() -> Svg:
    f = Svg(
        1100,
        250,
        "Three layers of transformation: scientific data is"
        " harmonised into ML-ready datasets, which representation learning"
        " compresses into embeddings.",
    )
    w, h, y = 280, 150, 40
    xs = [24, 410, 796]
    layers = [
        (
            "Scientific data",
            "L0 → L4, both streams",
            OBS,
            ["georeader · geocatalog", "geotoolz / xrtoolz", "statecatalog"],
        ),
        (
            "ML-ready",
            "tensors with provenance",
            ML,
            [
                "pipekit-train CachedDataset",
                "CF-compliant datasets",
                "content-addressed splits",
            ],
        ),
        (
            "Embedding",
            "latent representations",
            ML,
            [
                "pipekit-train · pipekit-cycle",
                "encoder / decoder / latent",
                "dynamics, in the registry",
            ],
        ),
    ]
    for x, (name, sub, hue, owners) in zip(xs, layers, strict=True):
        f.rect(x, y, w, h, stroke=hue[0], fill=hue[1])
        f.text(x + 16, y + 28, name, size=15, color=INK, anchor="start", weight=700)
        f.text(x + 16, y + 47, sub, size=11.5, color=hue[0], anchor="start", weight=600)
        f.text(
            x + 16,
            y + 74,
            "owned by",
            size=10.5,
            color=MUTED,
            anchor="start",
            weight=700,
        )
        f.lines(x + 16, y + 92, owners, gap=17, size=12, color=INK, anchor="start")
    for a, b, l1, l2 in [
        (xs[0] + w, xs[1], "harmonise,", "tensorise"),
        (xs[1] + w, xs[2], "compress,", "map to latents"),
    ]:
        f.arrow([(a + 2, y + h / 2), (b - 2, y + h / 2)], color=INK)
        f.text((a + b) / 2, y + h / 2 - 22, l1, size=11.5, color=INK)
        f.text((a + b) / 2, y + h / 2 - 8, l2, size=11.5, color=INK)
    f.text(
        550,
        228,
        "ML processing pipeline  →  representation learning",
        size=11.5,
        color=MUTED,
    )
    return f


SIX = [
    ("1", "Simple model", ["a generative story;", "physics you can simulate"]),
    ("2", "Model-based inference", ["slow but exact;", "classical DA / inversion"]),
    ("3", "Emulator", ["neural surrogate trained", "on step 1; skip if cheap"]),
    ("4", "Emulator-based inference", ["step 2 in seconds,", "not hours"]),
    ("5", "Amortized inference", ["a direct map:", "observations → posterior"]),
]


def six_steps() -> Svg:
    f = Svg(
        1180,
        372,
        "The six-step cycle: five steps from simple model to"
        " amortized inference; step 2 is the oracle that validates steps 3"
        " to 5, and improvement runs alongside steps 2 to 5 as one shared"
        " step.",
    )
    w, h, y, gap = 204, 100, 110, 28
    xs = [24 + i * (w + gap) for i in range(5)]
    for i, ((n, name, sub), x) in enumerate(zip(SIX, xs, strict=True)):
        hue = ORACLE if n == "2" else (ML if n in "345" else (BORDER, CARD))
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
        f.text(x + 14, y + 44, name, size=12.5, color=INK, anchor="start", weight=700)
        f.lines(x + 14, y + 64, sub, gap=14.5, size=11.5, color=MUTED, anchor="start")
        if i < 4:
            f.arrow([(x + w + 2, y + h / 2), (x + w + gap - 2, y + h / 2)], color=INK)
    # Step 2 validates 3, 4 and 5.
    sx, bus = xs[1] + w / 2, 66
    f.arrow(
        [(sx, y - 2), (sx, bus), (xs[4] + w / 2, bus)],
        color=ORACLE[0],
        dashed=True,
        head=False,
    )
    for k in (2, 3, 4):
        tx = xs[k] + w / 2
        f.arrow([(tx, bus), (tx, y - 4)], color=ORACLE[0], dashed=True)
    f.text(
        sx + 12,
        bus - 12,
        "oracle: step 2 is the ground truth that validates steps 3\u20135",
        size=12,
        color=ORACLE[0],
        anchor="start",
        weight=600,
    )
    # Step 6: one shared improvement step under all of them, linked to 2-5.
    iy, ih = 270, 72
    f.rect(xs[0], iy, xs[4] + w - xs[0], ih, stroke=INK, fill=PANEL)
    f.text(
        xs[0] + 16,
        iy + 26,
        "STEP 6",
        size=10.5,
        color=MUTED,
        anchor="start",
        weight=700,
    )
    f.text(
        xs[0] + 16, iy + 48, "Improve", size=14, color=INK, anchor="start", weight=700
    )
    f.text(
        xs[0] + w + gap / 2,
        iy + 42,
        "upgrade the model, data, emulator or"
        " predictor, using the previous step as ground truth",
        size=12,
        color=INK,
        anchor="start",
    )
    for k in (1, 2, 3, 4):
        cx = xs[k] + w / 2
        f.arrow([(cx, y + h + 4), (cx, iy - 4)], color=INK, both=True)
    return f


MATRIX_ROWS = [  # (transition, task, cells for steps 1-5)
    (
        "L3 → L4",
        "forecast",
        [
            "PDE forward",
            "EnKF / 4D-Var",
            "neural forward",
            "fast DA",
            "direct forecast",
        ],
    ),
    (
        "L3 → L3",
        "analysis",
        [
            "forward + prior",
            "classical DA",
            "neural forward",
            "neural DA",
            "direct analysis",
        ],
    ),
    (
        "L2 → L3",
        "gap-fill",
        [
            "GP prior",
            "kriging / DINEOF",
            "neural map",
            "fast gap-fill",
            "neural posterior",
        ],
    ),
    (
        "L1 → L2",
        "retrieval",
        [
            "RT model",
            "matched filter",
            "neural RT",
            "fast retrieval",
            "direct retrieval",
        ],
    ),
    (
        "L0 → L1",
        "calibration",
        ["cal model", "classical cal", "neural cal", "fast cal", "direct cal"],
    ),
]


def axes_matrix() -> Svg:
    f = Svg(
        1060,
        490,
        "The axes as a matrix: each L0 to L4 transition (rows)"
        " crossed with the six-step cycle (columns) is a concrete,"
        " benchmarkable method; step 2 is the oracle column.",
    )
    x0, y0, cw, rh = 190, 96, 166, 56
    heads = ["Simple model", "Model-based", "Emulator", "Emulator-based", "Amortized"]
    f.text(
        x0 + 2.5 * cw,
        30,
        "six-step cycle  (production maturity)  →",
        size=12.5,
        color=INK,
        weight=600,
    )
    for j, head in enumerate(heads):
        x = x0 + j * cw
        f.text(x + cw / 2, 62, f"STEP {j + 1}", size=10.5, color=MUTED, weight=700)
        f.text(x + cw / 2, 80, head, size=12.5, color=INK, weight=600)
    f.rect(
        x0 + cw + 3, y0 - 4, cw - 6, rh * 5 + 8, stroke=ORACLE[0], fill=ORACLE[1], rx=10
    )
    f.text(
        x0 + 1.5 * cw,
        y0 + 5 * rh + 24,
        "oracle column",
        size=11.5,
        color=ORACLE[0],
        weight=600,
    )
    for i, (trans, task, cells) in enumerate(MATRIX_ROWS):
        y = y0 + i * rh
        hue = STATE if trans.startswith("L3") else OBS
        f.text(
            x0 - 120,
            y + rh / 2 - 2,
            trans,
            size=13,
            color=hue[0],
            anchor="start",
            weight=700,
        )
        f.text(x0 - 120, y + rh / 2 + 14, task, size=11.5, color=MUTED, anchor="start")
        for j, cell in enumerate(cells):
            x = x0 + j * cw
            f.rect(x + 8, y + 7, cw - 16, rh - 14, stroke=BORDER, fill=CARD, rx=6)
            f.text(x + cw / 2, y + rh / 2 + 4.5, cell, size=12, color=INK)
    f.arrow([(x0 - 150, y0 + 5 * rh), (x0 - 150, y0)], color=INK)
    f.text(
        x0 - 160,
        y0 + 2.5 * rh,
        "L0 → L4  (data maturity)",
        size=12,
        color=INK,
        weight=600,
        anchor="middle",
    )
    f.parts[-1] = f.parts[-1].replace(
        "<text ", f'<text transform="rotate(-90 {x0 - 160} {y0 + 2.5 * rh})" ', 1
    )
    f.text(
        x0 + 2.5 * cw,
        y0 + 5 * rh + 56,
        "the modeling cycle (axis 1) runs"
        " through every cell: abstract, simulate, validate, refine",
        size=12,
        color=MUTED,
        italic=True,
    )
    return f


TIERS = [  # (tier, name, note, owned, packages)
    ("6", "Serving", "external", False, ["FastAPI", "LitServe", "Fused", "Modal"]),
    (
        "5",
        "Orchestration",
        "external",
        False,
        ["Prefect", "MLflow", "DVC", "Hydra", "Metaflow", "W&B"],
    ),
    (
        "4",
        "Pipeline infrastructure",
        "carrier-neutral",
        True,
        [
            "pipekit",
            "pipekit-array",
            "pipekit-jax",
            "pipekit-cycle",
            "pipekit-train",
            "pipekit-evaluate",
            "pipekit-experiment",
            "geocatalog",
            "geopatcher",
            "xrpatcher",
            "statecatalog",
        ],
    ),
    (
        "3",
        "Domain libraries",
        "carrier-specific",
        True,
        ["geotoolz  (GeoTensor)", "xrtoolz  (xarray)", "geopandas  (vector)"],
    ),
    (
        "2",
        "Algorithms",
        "JAX-native",
        True,
        ["pyrox-gp", "pyrox-nn", "filterX", "vardaX", "gaussFlowX"],
    ),
    (
        "1",
        "Numerical foundation",
        "JAX-native",
        True,
        ["gaussX", "spectralDiffX", "finiteVolX", "optax-bayes", "eqx-trainer"],
    ),
    (
        "0",
        "Ecosystem",
        "external",
        False,
        [
            "JAX",
            "equinox",
            "lineax",
            "optax",
            "diffrax",
            "numpyro",
            "xarray",
            "pandas",
            "rasterio",
            "scipy",
            "georeader",
        ],
    ),
]


def tier_stack() -> Svg:
    f = Svg(
        1100,
        700,
        "The stack in seven tiers: external ecosystem at the"
        " bottom, owned numerical, algorithm, domain and pipeline tiers in"
        " the middle, external orchestration and serving on top.",
    )
    x, lw, w, y = 64, 230, 1012, 24
    for tier, name, note, owned, pkgs in TIERS:
        hue = ML if owned else EXT
        if tier == "4":
            hue = ENG
        rows = 2 if len(pkgs) > 8 else 1
        h = max(84, 30 + rows * 38)
        f.rect(x, y, w, h, stroke=hue[0], fill=hue[1], dashed=not owned)
        f.text(
            x + 18,
            y + 28,
            f"TIER {tier}",
            size=10.5,
            color=MUTED,
            anchor="start",
            weight=700,
        )
        f.text(x + 18, y + 47, name, size=14, color=INK, anchor="start", weight=700)
        f.text(
            x + 18,
            y + 64,
            ("owned · " if owned else "") + note,
            size=11.5,
            color=hue[0] if owned else MUTED,
            anchor="start",
        )
        cx, cy = x + lw, y + 18
        for p in pkgs:
            ext = p == "georeader"
            cw = _width(p, 12.5) + 22
            if cx + cw > x + w - 14:
                cx, cy = x + lw, cy + 38
            f.chip(cx, cy, p, hue=(OBS if ext else (BORDER, CARD)), dashed=False)
            cx += cw + 8
        y += h + 14
    f.arrow([(36, y - 24), (36, 40)], color=MUTED)
    f.text(28, (y + 30) / 2, "builds on", size=11.5, color=MUTED)
    f.parts[-1] = f.parts[-1].replace(
        "<text ", f'<text transform="rotate(-90 28 {(y + 30) / 2})" ', 1
    )
    f.h = int(y + 10)
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
                ("statecatalog", ["model-state index,", "StateSlice, L3\u2013L4"]),
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
    ("4", "Grid + gap-fill", "L2 → L3", "pyrox-gp via pipekit"),
    ("5", "Data assimilation", "L3 → L4", "pipekit-cycle.DACycle + filterX, plumax"),
    ("6", "Index outputs", "", "statecatalog"),
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


FIGURES = {
    "modeling_cycle": modeling_cycle,
    "lifecycle": lifecycle,
    "two_streams": two_streams,
    "three_layers": three_layers,
    "six_steps": six_steps,
    "axes_matrix": axes_matrix,
    "tier_stack": tier_stack,
    "tier4_map": tier4_map,
    "dependency_graph": dependency_graph,
    "methane_walkthrough": methane_walkthrough,
}


def main() -> None:
    for name, build in FIGURES.items():
        (OUT / f"{name}.svg").write_text(build().render(), encoding="utf-8")
        print(f"wrote {name}.svg")


if __name__ == "__main__":
    main()
