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
CARD = _ov.CARD
ML, STATE, ENG, EXT, OBS = _ov.ML, _ov.STATE, _ov.ENG, _ov.EXT, _ov.OBS

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


CORE = ("#334155", "#f1f5f9")


def r3_layers() -> Svg:
    """Report 3: how the sister libraries actually depend on pipekit."""
    f = Svg(
        840,
        520,
        "Shipped dependencies. geotoolz depends on pipekit core directly."
        " xrtoolz depends on xrcore, which subclasses pipekit's Operator."
        " pipekit-array and pipekit-jax depend on pipekit core. The planned"
        " links from geotoolz and xrtoolz down to pipekit-array do not exist.",
    )
    f.arrow([(24, 30), (64, 30)], color=INK)
    f.text(72, 34, "depends on", size=13, color=MUTED, anchor="start")
    f.arrow([(200, 30), (240, 30)], color=LINE, dashed=True)
    f.text(
        248,
        34,
        "planned in this report, not wired",
        size=13,
        color=MUTED,
        anchor="start",
    )
    f.box(
        24,
        60,
        380,
        110,
        "geotoolz family · GeoTensor",
        ["geotoolz 0.10.2", "catalog · cloud · patcher · products"],
        hue=OBS,
        title_size=16,
        sub_size=13.5,
    )
    f.box(
        436,
        60,
        380,
        110,
        "xrtoolz family · xarray",
        ["xrtoolz 0.0.5", "einx · grad · reader · sklearn"],
        hue=STATE,
        title_size=16,
        sub_size=13.5,
    )
    f.box(
        120,
        250,
        230,
        86,
        "pipekit-array",
        ["array_namespace, MeanScalar,", "StackAlong, ConcatenateAlong"],
        hue=ML,
        title_size=15,
        sub_size=13,
    )
    f.box(
        375,
        250,
        200,
        86,
        "pipekit-jax",
        ["JaxModelOp, diffrax,", "adjoint helpers"],
        hue=ML,
        title_size=15,
        sub_size=13,
    )
    f.box(
        600,
        250,
        216,
        86,
        "xrcore",
        ["Operator subclass:", "DataTree dispatch, Signature"],
        hue=STATE,
        title_size=15,
        sub_size=13,
    )
    f.box(
        24,
        420,
        792,
        72,
        "pipekit core",
        ["Operator · Sequential · Graph · observe · control · qc · parallel"],
        hue=CORE,
        title_size=16,
        sub_size=13.5,
    )
    f.arrow([(70, 170), (70, 418)], color=INK)
    f.arrow([(235, 170), (235, 248)], color=LINE, dashed=True)
    f.arrow([(520, 170), (342, 248)], color=LINE, dashed=True)
    f.arrow([(708, 170), (708, 248)], color=INK)
    f.arrow([(708, 336), (708, 418)], color=INK)
    f.arrow([(235, 336), (235, 418)], color=INK)
    f.arrow([(475, 336), (475, 418)], color=INK)
    f.text(483, 385, "+ pipekit-cycle", size=12.5, color=MUTED, anchor="start")
    f.text(80, 214, "direct", size=12.5, color=MUTED, anchor="start")
    f.h = 506
    return f


# geocatalog's own palette (docs/assets/make_diagrams.py in the geocatalog repo)
GC_SOURCE = ("#4C72B0", "#e8f0fb")
GC_BUNDLE = ("#55A868", "#e9f5ec")
GC_CATALOG = ("#C44E52", "#fbe8e9")
GC_SLICE = ("#8172B2", "#efeaf6")
GC_LOADER = ("#CCB974", "#fbf6e3")


def r6_dataflow() -> Svg:
    """Report 6: geocatalog's data flow, after the repo's concepts diagram."""
    f = Svg(
        840,
        760,
        "geocatalog data flow, top to bottom. Discovery sources (STAC, NASA"
        " EarthAccess and CMR, Earth Engine, local files) feed a CatalogBundle"
        " or a builder. Both produce a GeoCatalog, backed by InMemoryGeoCatalog"
        " or DuckDBGeoCatalog. A catalog yields GeoSlices; loaders turn a slice"
        " plus a catalog into a GeoTensor or an xarray Dataset. Catalogs"
        " persist as GeoParquet 1.1, which open_catalog reads back lazily.",
    )
    # 1 discovery
    f.rect(24, 20, 792, 110, stroke=BORDER, fill=PANEL)
    f.text(44, 46, "1 · Discover", size=14, weight=700, anchor="start")
    x = 44.0
    for name in ["STACSource", "EarthAccessSource", "CMRSource", "GEESource"]:
        x += f.chip(x, 62, name, hue=GC_SOURCE, size=13) + 10
    f.chip(600, 62, "local files", hue=(BORDER, CARD), size=13)
    f.text(
        44,
        116,
        "yield SourceRows with the query that produced them",
        size=12,
        color=MUTED,
        anchor="start",
        italic=True,
    )
    # bundle + builders
    f.box(
        60,
        170,
        330,
        76,
        "CatalogBundle",
        ["items + queries + matchups", "storage.CatalogBundle"],
        hue=GC_BUNDLE,
        title_size=15,
        sub_size=12.5,
    )
    f.box(
        450,
        170,
        330,
        76,
        "Builders",
        ["build_raster / _vector / _xarray_catalog", "build.append_files"],
        hue=(BORDER, CARD),
        title_size=15,
        sub_size=12.5,
    )
    f.arrow([(225, 130), (225, 168)], color=INK)
    f.arrow([(650, 130), (650, 168)], color=INK)
    # 2 index
    f.rect(24, 286, 600, 150, stroke=GC_CATALOG[0], fill=GC_CATALOG[1])
    f.text(
        44, 312, "2 · Index: GeoCatalog Protocol", size=14, weight=700, anchor="start"
    )
    f.box(
        44,
        330,
        270,
        86,
        "InMemoryGeoCatalog",
        ["GeoDataFrame + R-tree", "+ IntervalIndex · eager"],
        hue=(GC_CATALOG[0], CARD),
        title_size=14,
        sub_size=12.5,
    )
    f.box(
        334,
        330,
        270,
        86,
        "DuckDBGeoCatalog",
        ["lazy SQL over GeoParquet 1.1", "bbox predicate pushdown"],
        hue=(GC_CATALOG[0], CARD),
        title_size=14,
        sub_size=12.5,
    )
    f.arrow([(225, 246), (225, 284)], color=INK)
    f.arrow([(560, 246), (560, 284)], color=INK)
    f.text(
        44,
        454,
        "query · intersect · union return new catalogs",
        size=12,
        color=MUTED,
        anchor="start",
        italic=True,
    )
    # GeoParquet store
    f.box(
        660,
        330,
        156,
        86,
        "GeoParquet 1.1",
        ["local · s3:// · hf://", "Hive-partitioned"],
        hue=(LINE, CARD),
        dashed=True,
        title_size=14,
        sub_size=12,
    )
    f.arrow(
        [(626, 300), (740, 300), (740, 328)],
        color=LINE,
        dashed=True,
        label="to_geoparquet",
        at=(683, 292),
        label_size=12,
    )
    f.arrow([(660, 400), (606, 400)], color=LINE, dashed=True)
    f.text(738, 434, "open_catalog → DuckDB", size=12, color=MUTED)
    # 3 slice
    f.box(
        170,
        490,
        500,
        80,
        "GeoSlice",
        ["bounds + interval + resolution + CRS", "frozen · hashable · iter_slices()"],
        hue=GC_SLICE,
        title_size=15,
        sub_size=12.5,
    )
    f.arrow([(324, 436), (324, 488)], color=INK)
    f.text(332, 474, "query(slice) → rows", size=12, color=MUTED, anchor="start")
    # 4 materialise
    loaders = [
        ("load_raster", "GeoTensor"),
        ("load_xarray", "xr.Dataset"),
        ("load_vector", "GeoTensor"),
    ]
    for i, (fn, out) in enumerate(loaders):
        bx = 90 + i * 230
        f.box(
            bx,
            612,
            200,
            56,
            fn,
            ["catalog + slice"],
            hue=GC_LOADER,
            title_size=14,
            sub_size=12,
        )
        f.chip(bx + 100 - 52, 700, out, hue=(BORDER, PANEL), size=13)
        f.arrow([(bx + 100, 668), (bx + 100, 698)], color=INK)
    f.arrow([(420, 570), (190, 610)], color=INK)
    f.arrow([(420, 570), (420, 610)], color=INK)
    f.arrow([(420, 570), (650, 610)], color=INK)
    f.text(24, 640, "3 · Load", size=14, weight=700, anchor="start")
    f.h = 742
    return f


def r9_stack() -> Svg:
    """Report 9: the stack the use cases run on, with only real dependencies."""
    f = Svg(
        840,
        560,
        "The stack at the pinned revisions. User code (notebooks, services,"
        " orchestrators, geoapps) calls four libraries side by side: geotoolz,"
        " xrtoolz, geopatcher and geocatalog. geotoolz and xrtoolz depend on"
        " pipekit; geopatcher reaches pipekit only through its pipekit extra;"
        " geocatalog does not depend on pipekit at all. pipekit's sister"
        " packages (array, train, experiment, evaluate, cycle, jax) sit beside"
        " it. georeader, geocloud and xarray are the substrate underneath.",
    )
    f.rect(24, 20, 792, 44, stroke=BORDER, fill=PANEL)
    f.text(
        420,
        48,
        "user code · notebooks · services · orchestrators · geoapps",
        size=14,
        color=INK,
        weight=600,
    )
    boxes = [
        ("geotoolz", ["GeoTensor operators"], OBS),
        ("xrtoolz", ["xarray operators", "(on xrcore)"], STATE),
        ("geopatcher", ["patches: split, merge"], ML),
        ("geocatalog", ["the index: GeoSlice"], ENG),
    ]
    xs = [24, 228, 432, 636]
    for (name, sub, hue), x in zip(boxes, xs, strict=True):
        f.box(x, 104, 180, 80, name, sub, hue=hue, title_size=15, sub_size=12.5)
        f.arrow([(x + 90, 64), (x + 90, 102)], color=INK)
    # pipekit panel
    f.rect(24, 250, 384, 150, stroke=CORE[0], fill=CORE[1])
    f.text(44, 276, "pipekit", size=16, weight=700, anchor="start")
    f.text(
        44,
        296,
        "Operator · Sequential · Graph · parallel · qc",
        size=12.5,
        color=MUTED,
        anchor="start",
    )
    chips = ["-array", "-train", "-experiment", "-evaluate", "-cycle", "-jax"]
    cx, cy = 44.0, 314.0
    for c in chips:
        w = f.chip(cx, cy, "pipekit" + c, hue=(BORDER, CARD), size=12)
        cx += w + 8
        if cx > 330:
            cx, cy = 44.0, cy + 34
    f.arrow([(114, 184), (114, 248)], color=INK)
    f.arrow([(318, 184), (318, 248)], color=INK)
    f.arrow([(480, 184), (480, 330), (410, 330)], color=LINE, dashed=True)
    f.text(488, 300, "[pipekit] extra", size=12, color=MUTED, anchor="start")
    # substrate
    f.box(
        24,
        460,
        792,
        70,
        "georeader · geocloud · xarray",
        ["I/O, GeoTensor, object storage: underneath geotoolz too"],
        hue=(LINE, PANEL),
        title_size=15,
        sub_size=12.5,
    )
    f.arrow([(580, 184), (580, 458)], color=INK)
    f.arrow([(726, 184), (726, 458)], color=INK)
    f.text(734, 230, "no pipekit", size=12, color=MUTED, anchor="start", italic=True)
    f.h = 546
    return f


def r13_ownership() -> Svg:
    """Report 13: who produces, indexes and scores model states."""
    f = Svg(
        840,
        470,
        "Ownership of model states. A script runs pipekit-cycle, writes one"
        " file per run and member, indexes the files with geocatalog and scores"
        " the loaded cube with xrtoolz. pipekit-cycle produces states, geocatalog"
        " indexes them (the new geocatalog.states namespace is the only new"
        " surface), xrtoolz verifies them, and pipekit-experiment supplies the"
        " producer hash. None of the four imports another; NetCDF, Zarr and"
        " GeoParquet files sit underneath.",
    )
    f.box(
        24,
        20,
        792,
        58,
        "scripts and notebooks (geostack)",
        ["run the cycle · write one file per (run, member) · build the catalog"],
        hue=(BORDER, PANEL),
        title_size=15,
        sub_size=12.5,
    )
    boxes = [
        ("pipekit-cycle", ["produces states", "Cycle.history"], ML, False),
        ("geocatalog.states", ["indexes states", "rows · load · queries"], ENG, True),
        ("xrtoolz.metrics", ["scores states", "SkillByLeadTime · CRPS"], STATE, False),
        ("pipekit-experiment", ["names the producer", "Run.pipeline_hash"], OBS, False),
    ]
    xs = [24, 228, 432, 636]
    for (name, sub, hue, dashed), x in zip(boxes, xs, strict=True):
        f.box(
            x,
            140,
            180,
            90,
            name,
            sub,
            hue=hue,
            dashed=dashed,
            title_size=14.5,
            sub_size=12,
        )
        f.arrow([(x + 90, 78), (x + 90, 138)], color=INK)
    f.text(
        340,
        270,
        "no arrows between the four: none imports another",
        size=12.5,
        color=MUTED,
        anchor="start",
        italic=True,
    )
    f.box(
        24,
        300,
        792,
        64,
        "geocatalog core · GeoSlice · backends · GeoParquet",
        ["the existing index: footprint, valid-time interval, path, extra columns"],
        hue=(BORDER, CARD),
        title_size=14.5,
        sub_size=12,
    )
    f.arrow([(318, 230), (318, 298)], color=INK)
    f.box(
        24,
        394,
        792,
        56,
        "NetCDF · Zarr · GeoParquet (xarray, georeader)",
        ["the state files and the catalog file"],
        hue=(LINE, PANEL),
        title_size=14.5,
        sub_size=12,
    )
    f.arrow([(420, 364), (420, 392)], color=INK)
    return f


def r13_roadmap() -> Svg:
    """Report 13: what ships first, and the gates after each row."""
    f = Svg(
        840,
        470,
        "Roadmap. Phase 1 runs three independent pieces side by side: the"
        " reserved state columns in geocatalog (A1), forecast alignment,"
        " member broadcasting and area weights in xrtoolz (C1 to C3) and"
        " ensemble histories in"
        " pipekit-cycle (D1); their gates must pass before phase 2. Phase 2"
        " adds the state builder (A2), run-aware loading (B1) and the typed"
        " queries (B2) on top of A1. Phase 3 is the consumers, which are"
        " upcoming work rather than workstreams.",
    )
    rows = [
        (
            "1",
            [
                ("1a · A1", ["reserved state columns", "geocatalog schema v1"], ENG),
                (
                    "1b · C1 C2 C3",
                    ["align_reference · broadcasting", "cell-area weights"],
                    STATE,
                ),
                ("1c · D1", ["ensemble cycles", "save_history"], ML),
            ],
            "A1 · C1 · C2 · C3 · D1",
        ),
        (
            "2",
            [
                ("2a · A2", ["build_state_catalog", "CF and cfgrib names"], ENG),
                ("2b · B1 B2", ["load_states · producer trees", "typed queries"], ENG),
            ],
            "A2 · B1 · B2 · B3",
        ),
        (
            "3",
            [
                (
                    "3 · consumers",
                    ["CatalogDataset · L4 references", "forecast presets"],
                    EXT,
                ),
            ],
            None,
        ),
    ]
    y = 24.0
    for _phase, items, gate in rows:
        n = len(items)
        gap = 18
        w = (792 - gap * (n - 1)) / n
        for i, (name, sub, hue) in enumerate(items):
            x = 24 + i * (w + gap)
            f.box(x, y, w, 76, name, sub, hue=hue, title_size=14.5, sub_size=12)
        if gate:
            cx, gy, d = 420, y + 76 + 30, 14
            f.arrow([(cx, y + 76), (cx, gy - d - 1)], color=LINE)
            f.parts.append(
                f'<polygon points="{cx},{gy - d} {cx + d},{gy} {cx},{gy + d}'
                f' {cx - d},{gy}" fill="{ENG[1]}" stroke="{ENG[0]}"'
                ' stroke-width="1.4"/>'
            )
            f.text(
                cx + d + 10,
                gy + 4.5,
                f"gates {gate}",
                size=12.5,
                color=ENG[0],
                anchor="start",
                weight=600,
            )
            f.arrow([(cx, gy + d), (cx, gy + 58)], color=LINE)
            y = gy + 60
    f.h = int(y + 76 + 16)
    return f


def r10_ownership() -> Svg:
    """Report 10: pipekit-cycle between pipekit core and the libraries it serves."""
    f = Svg(
        840,
        500,
        "Ownership around pipekit-cycle. Scripts build cycles from pipekit-cycle"
        " and from algorithm libraries; the libraries satisfy pipekit-cycle's"
        " protocols by structural typing and never import it. pipekit-cycle"
        " groups the cycles, the DA cycles, the protocols, the generic"
        " observation and forward helpers, the carry states and the adjoint"
        " specs. It builds on pipekit core (StatefulOperator, CarryState,"
        " Sequential); pipekit-jax reads its adjoint specs.",
    )
    f.box(
        24,
        20,
        380,
        64,
        "scripts and notebooks",
        ["build a cycle · supply the carrier and state"],
        hue=(BORDER, PANEL),
        title_size=15,
        sub_size=12.5,
    )
    f.box(
        436,
        20,
        380,
        64,
        "algorithm libraries",
        ["filterax · vardax · plumax adapters (not checked)"],
        hue=EXT,
        dashed=True,
        title_size=15,
        sub_size=12.5,
    )
    f.rect(24, 120, 792, 230, stroke=ML[0], fill=ML[1])
    f.text(44, 148, "pipekit-cycle 0.0.2", size=16, weight=700, anchor="start")
    groups = [
        ("cycles", ["Cycle", "EnsembleCycle", "WindowedCycle", "Recurrence"]),
        ("DA cycles", ["DACycle", "EnsembleDACycle", "SmootherCycle", "BFNCycle"]),
        ("protocols", ["ForwardModel", "ObservationOperator", "AnalysisStep", "+ 11"]),
        ("helpers", ["IdentityObs", "LinearObs", "CallableObs", "CompositeObs"]),
        ("", ["CallableForward", "CompositeForward", "NeuralForward"]),
        ("states, specs", ["DAState", "IterationState", "WindowState", "adjoints"]),
    ]
    y = 166.0
    for label, chips in groups:
        f.text(44, y + 17, label, size=12.5, color=MUTED, anchor="start")
        x = 150.0
        for c in chips:
            x += f.chip(x, y, c, hue=(BORDER, CARD), size=12) + 8
        y += 30
    f.arrow([(214, 84), (214, 118)], color=INK)
    f.arrow([(626, 84), (626, 118)], color=LINE, dashed=True)
    f.text(634, 106, "structural typing", size=12, color=MUTED, anchor="start")
    f.box(
        24,
        400,
        560,
        76,
        "pipekit core",
        ["StatefulOperator · CarryState · Sequential (threads state)"],
        hue=(BORDER, CARD),
        title_size=15,
        sub_size=12.5,
    )
    f.box(
        616,
        400,
        200,
        76,
        "pipekit-jax",
        ["reads the adjoint specs"],
        hue=(BORDER, CARD),
        title_size=15,
        sub_size=12.5,
    )
    f.arrow([(304, 350), (304, 398)], color=INK)
    f.arrow([(716, 398), (716, 352)], color=LINE, dashed=True)
    f.h = 492
    return f


def r10_roadmap() -> Svg:
    """Report 10: the additions, in order, with their gates."""
    f = Svg(
        840,
        330,
        "Roadmap. Phase 1 runs three small additions side by side: cycles that"
        " accept a stateful Sequential as their step (A1), an observation clock"
        " that survives restarts (B1), and ensemble histories (Report 13, D1)."
        " Their gates pass before phase 2, which refreshes the README and"
        " decides the latent-space design.",
    )
    items = [
        ("1a · A1", ["stateful Sequential", "as a cycle step"], ML),
        ("1b · B1", ["obs_clock", "survives restarts"], ML),
        ("1c · R13 D1", ["ensemble histories", "(Report 13)"], ENG),
    ]
    w = (792 - 2 * 18) / 3
    for i, (name, sub, hue) in enumerate(items):
        f.box(24 + i * (w + 18), 24, w, 76, name, sub, hue=hue, title_size=14.5)
    cx, gy, d = 420, 130, 14
    f.arrow([(cx, 100), (cx, gy - d - 1)], color=LINE)
    f.parts.append(
        f'<polygon points="{cx},{gy - d} {cx + d},{gy} {cx},{gy + d}'
        f' {cx - d},{gy}" fill="{ENG[1]}" stroke="{ENG[0]}" stroke-width="1.4"/>'
    )
    f.text(
        cx + d + 10,
        gy + 4.5,
        "gates A1 · B1 · R13-D1",
        size=12.5,
        color=ENG[0],
        anchor="start",
        weight=600,
    )
    f.arrow([(cx, gy + d), (cx, 188)], color=LINE)
    f.box(
        24,
        190,
        792,
        76,
        "2 · docs and design",
        ["README for 0.0.2 · latent.md: build it or retire it"],
        hue=EXT,
        title_size=14.5,
    )
    f.h = 284
    return f


def r11_ownership() -> Svg:
    """Report 11: pipekit-train's layers, its backends and its two handoffs."""
    f = Svg(
        840,
        520,
        "pipekit-train 0.0.2. Data sources (pipekit-cycle forward models, Zarr"
        " stores, plain iterables; geocatalog is planned) feed its datasets."
        " TrainingLoop runs on one of six backends: Equinox and the three"
        " Bayesian ones (NumPyro SVI, NumPyro MCMC, BlackJAX) are real, Lightning"
        " and Keras are scaffolds. The loop returns a trained operator, which"
        " drops into pipekit-cycle as a NeuralForward, and a TrainingArtifact for"
        " pipekit-experiment.",
    )
    srcs = [
        ("pipekit-cycle", ["ForwardModel → SimulationDataset"], ML, False),
        ("Zarr stores", ["XarrayWindowDataset"], STATE, False),
        ("geocatalog", ["CatalogDataset (planned)"], EXT, True),
    ]
    for i, (name, sub, hue, dashed) in enumerate(srcs):
        f.box(24 + i * 270, 20, 252, 60, name, sub, hue=hue, dashed=dashed)
        f.arrow([(150 + i * 270, 80), (150 + i * 270, 118)], color=LINE, dashed=dashed)
    f.rect(24, 120, 792, 210, stroke=ENG[0], fill=ENG[1])
    f.text(44, 148, "pipekit-train 0.0.2", size=16, weight=700, anchor="start")
    rows = [
        (
            "datasets",
            [
                "TrainingDataset",
                "SimulationDataset",
                "IterableDataset",
                "CachedDataset",
                "XarrayWindowDataset",
            ],
        ),
        ("losses", ["Loss", "MSE", "NLL", "KL", "Composite"]),
        (
            "components",
            [
                "Callback",
                "EarlyStopping",
                "Checkpoint",
                "LogToExperiment",
                "JSONLWriter",
                "HyperSweep",
            ],
        ),
        ("loop", ["TrainingLoop", "TrainerCarryState", "ValidationStep"]),
    ]
    y = 164.0
    for label, chips in rows:
        f.text(44, y + 17, label, size=12.5, color=MUTED, anchor="start")
        x = 140.0
        for c in chips:
            x += f.chip(x, y, c, hue=(BORDER, CARD), size=11.5) + 7
        y += 32
    f.text(44, 312, "backends", size=12.5, color=MUTED, anchor="start")
    x = 140.0
    for c, dashed in [
        ("equinox", False),
        ("numpyro-svi", False),
        ("numpyro-mcmc", False),
        ("blackjax", False),
        ("lightning", True),
        ("keras", True),
    ]:
        hue = (LINE, PANEL) if dashed else (ENG[0], CARD)
        x += f.chip(x, 296, c, hue=hue, dashed=dashed, size=11.5) + 7
    f.box(
        24,
        380,
        380,
        76,
        "trained operator",
        ["EquinoxModelOp · NumpyroPredictiveOp", "→ pc.NeuralForward in a cycle"],
        hue=ML,
    )
    f.box(
        436,
        380,
        380,
        76,
        "TrainingArtifact",
        ["dataset_hash · backend_info · deps_lock", "→ pipekit-experiment registry"],
        hue=OBS,
    )
    f.arrow([(214, 330), (214, 378)], color=INK)
    f.arrow([(626, 330), (626, 378)], color=INK)
    f.text(
        420,
        490,
        "dashed: planned or scaffold only",
        size=12,
        color=MUTED,
        italic=True,
    )
    f.h = 506
    return f


def r11_roadmap() -> Svg:
    """Report 11: fixes first, then the missing features, then the backends."""
    f = Svg(
        840,
        470,
        "Roadmap. Phase 1 fixes what the running example hit: the SimulationDataset"
        " trajectory path and its hash (A2, A3), the window import (A5), the registry"
        " round trip and the model hash (B1, B2), checkpoints, sweep selection,"
        " ValidationStep in the loop and the SVI step (C1 to C4). Phase 2 adds the"
        " missing pieces the scoping report asked for:"
        " CatalogDataset (A1), a faster cache (A4) and the amortized-inference recipe"
        " (D1). Phase 3 builds the Lightning and Keras backends (E1, E2).",
    )
    rows = [
        (
            [
                ("1a · A2 A3 A5", ["simulation path, hash,", "window import"], ENG),
                ("1b · B1 B2", ["registry round trip,", "model hash"], ML),
                ("1c · C1 to C4", ["checkpoints, sweeps,", "validation, SVI jit"], OBS),
            ],
            "A2 · A3 · A5 · B1 · B2 · C1 · C2 · C3 · C4",
        ),
        (
            [
                ("2a · A1", ["CatalogDataset", "over geocatalog"], ENG),
                ("2b · A4", ["CachedDataset", "speed and formats"], ENG),
                ("2c · D1", ["amortized-inference", "recipe and heads"], ML),
            ],
            "A1 · A4 · D1",
        ),
        (
            [
                ("3a · E1", ["Lightning adapter", "(ADR D5, v0.2)"], EXT),
                ("3b · E2", ["Keras 3 adapter", "(ADR D5, v0.3)"], EXT),
            ],
            "E1 · E2",
        ),
    ]
    y = 24.0
    for items, gate in rows:
        n = len(items)
        w = (792 - 18 * (n - 1)) / n
        for i, (name, sub, hue) in enumerate(items):
            f.box(24 + i * (w + 18), y, w, 70, name, sub, hue=hue, title_size=14)
        cx, gy, d = 420, y + 70 + 26, 13
        f.arrow([(cx, y + 70), (cx, gy - d - 1)], color=LINE)
        f.parts.append(
            f'<polygon points="{cx},{gy - d} {cx + d},{gy} {cx},{gy + d}'
            f' {cx - d},{gy}" fill="{ENG[1]}" stroke="{ENG[0]}" stroke-width="1.4"/>'
        )
        f.text(
            cx + d + 10,
            gy + 4.5,
            f"gates {gate}",
            size=12,
            color=ENG[0],
            anchor="start",
            weight=600,
        )
        if items is not rows[-1][0]:
            f.arrow([(cx, gy + d), (cx, gy + 48)], color=LINE)
        y = gy + 50
    f.h = int(y - 8)
    return f


def r12_ownership() -> Svg:
    """Report 12: pipekit-experiment between the stack and external tools."""
    f = Svg(
        840,
        500,
        "pipekit-experiment 0.0.2 between pipekit's training and inference"
        " packages and external tools. It ships two protocols, run records, a"
        " content-addressed model registry with local and fsspec backends,"
        " training and inference artifacts, and adapters for Hydra, DVC and"
        " Metaflow. The MLflow and W&B adapters and an MLflow registry are"
        " planned but not built. pipekit-train calls its trackers and registry;"
        " serving and orchestration stay external.",
    )
    tools = [
        ("Hydra", False),
        ("DVC", False),
        ("Metaflow", False),
        ("MLflow", True),
        ("W&B", True),
    ]
    f.rect(24, 20, 792, 60, stroke=BORDER, fill=PANEL)
    f.text(44, 46, "external tools", size=14, weight=600, anchor="start")
    x = 200.0
    for name, dashed in tools:
        hue = (LINE, PANEL) if dashed else (BORDER, CARD)
        x += f.chip(x, 34, name, hue=hue, dashed=dashed, size=12.5) + 10
    f.text(
        600,
        52,
        "dashed: no adapter yet",
        size=12,
        color=MUTED,
        anchor="start",
        italic=True,
    )
    f.rect(24, 120, 792, 200, stroke=OBS[0], fill=OBS[1])
    f.text(44, 148, "pipekit-experiment 0.0.2", size=16, weight=700, anchor="start")
    rows = [
        ("protocols", ["ExperimentTracker", "ModelRegistry"], False),
        ("records", ["Run", "RunMetrics", "RunArtifacts"], False),
        ("registry", ["LocalModelRegistry", "S3ModelRegistry (fsspec)"], False),
        ("artifacts", ["TrainingArtifact", "InferenceArtifact"], False),
        (
            "adapters",
            ["HydraConfigLoader", "DVCDatasetVersioning", "MetaflowStepAdapter"],
            False,
        ),
    ]
    y = 162.0
    for label, chips, _ in rows:
        f.text(44, y + 17, label, size=12.5, color=MUTED, anchor="start")
        x = 150.0
        for c in chips:
            x += f.chip(x, y, c, hue=(BORDER, CARD), size=11.5) + 8
        y += 30
    f.chip(
        560, 162, "no tracker implementation", hue=(LINE, PANEL), dashed=True, size=11.5
    )
    f.arrow([(420, 80), (420, 118)], color=LINE, both=True)
    f.box(
        24,
        360,
        380,
        76,
        "pipekit-train",
        ["LogToExperiment → a tracker", "Checkpoint → registry.store"],
        hue=ENG,
    )
    f.box(
        436,
        360,
        380,
        76,
        "inference pipelines · pipekit-cycle",
        ["registry.load or from_registry", "→ NeuralForward, serving"],
        hue=ML,
    )
    f.arrow([(214, 358), (214, 322)], color=INK)
    f.arrow([(626, 322), (626, 358)], color=INK)
    f.text(
        420,
        470,
        "registry storage: <hash>/operator.json · weights.bin ·"
        " metadata.json · _tags/<name>",
        size=12,
        color=MUTED,
    )
    f.h = 490
    return f


def r12_roadmap() -> Svg:
    """Report 12: correctness first, then trackers, then the round trips."""
    f = Svg(
        840,
        470,
        "Roadmap. Phase 1 fixes the registry and artifact paths the running"
        " example broke: weight rebinding on load (A1), hash identity (A2), tag"
        " history and safe naming (A3), honest artifact fields (C2, C3) and the"
        " tracker run id (B4). Phase 2 adds what the scoping report asked for and"
        " nothing ships yet: a file tracker, MLflow and W&B trackers, an MLflow"
        " registry, registry gc and robust fsspec writes. Phase 3 closes the round"
        " trips: retraining from an artifact, nested Hydra YAML, the Metaflow step"
        " and the run records.",
    )
    rows = [
        (
            [
                (
                    "1a · A1 A2 A3",
                    ["load with weights, hash", "identity, tag history"],
                    OBS,
                ),
                (
                    "1b · B4 C2 C3",
                    ["tracker run id, pinned", "hashes, registry URI"],
                    OBS,
                ),
            ],
            "A1 · A2 · A3 · B4 · C2 · C3",
        ),
        (
            [
                ("2a · B1 B2 B3", ["file, MLflow and", "W&B trackers"], ENG),
                ("2b · A4 A5 A6", ["gc, fsspec writes,", "MLflow registry"], ENG),
            ],
            "B1 · B2 · B3 · A4 · A5 · A6",
        ),
        (
            [
                ("3a · C1", ["TrainingArtifact.rerun"], ML),
                ("3b · D1 D2 D3", ["Hydra nested, Metaflow", "step, DVC pointer"], ML),
                ("3c · E1", ["run records", "RunMetrics, RunArtifacts"], ML),
            ],
            "C1 · D1 · D2 · D3 · E1",
        ),
    ]
    y = 24.0
    for items, gate in rows:
        n = len(items)
        w = (792 - 18 * (n - 1)) / n
        for i, (name, sub, hue) in enumerate(items):
            f.box(24 + i * (w + 18), y, w, 70, name, sub, hue=hue, title_size=14)
        cx, gy, d = 420, y + 70 + 26, 13
        f.arrow([(cx, y + 70), (cx, gy - d - 1)], color=LINE)
        f.parts.append(
            f'<polygon points="{cx},{gy - d} {cx + d},{gy} {cx},{gy + d}'
            f' {cx - d},{gy}" fill="{ENG[1]}" stroke="{ENG[0]}" stroke-width="1.4"/>'
        )
        f.text(
            cx + d + 10,
            gy + 4.5,
            f"gates {gate}",
            size=12,
            color=ENG[0],
            anchor="start",
            weight=600,
        )
        if items is not rows[-1][0]:
            f.arrow([(cx, gy + d), (cx, gy + 48)], color=LINE)
        y = gy + 50
    f.h = int(y - 8)
    return f


def r14_ownership() -> Svg:
    """Report 14: metrics live with their carriers; pipekit-evaluate orchestrates."""
    f = Svg(
        840,
        520,
        "Ownership of evaluation. Scripts and benchmarks call pipekit-evaluate,"
        " which ships the benchmark ladder (cells, the reference rule, Benchmark,"
        " Scorer) and, as the one new surface, would add the Unit, Lens and Stage"
        " tags, a scorer adapter and an evaluation report. The metrics themselves"
        " live with their carriers: xrtoolz.metrics for fields, statistics,"
        " events and physical checks, with the lead-time, region and band"
        " wrappers; geotoolz for detection building blocks; geocatalog for"
        " matchups. pipekit-cycle, pipekit-train and pipekit-experiment are the"
        " neighbours it connects to.",
    )
    f.box(
        24,
        20,
        792,
        56,
        "scripts · benchmarks · pilots (ocean, plume, land extremes, fusion)",
        ["build tasks, estimators and scorers; read the report"],
        hue=(BORDER, PANEL),
        title_size=14.5,
        sub_size=12,
    )
    f.rect(24, 110, 792, 120, stroke=ENG[0], fill=ENG[1])
    f.text(44, 136, "pipekit-evaluate", size=16, weight=700, anchor="start")
    x = 44.0
    for c in ["Cell", "ReferenceRule", "Benchmark", "Scorer", "Oracle"]:
        x += f.chip(x, 150, c, hue=(BORDER, CARD), size=12) + 8
    f.text(44, 206, "proposed:", size=12.5, color=MUTED, anchor="start")
    x = 120.0
    for c in ["unit · lens · stage tags", "MetricScorer", "EvaluationReport"]:
        x += f.chip(x, 190, c, hue=(ENG[0], CARD), dashed=True, size=12) + 8
    f.arrow([(420, 76), (420, 108)], color=INK)
    boxes = [
        (
            "xrtoolz.metrics",
            ["fields · statistics · events", "physical · by lead, region, band"],
            STATE,
            24,
            272,
        ),
        (
            "geotoolz",
            ["Threshold · LabelConnected", "PlumeMask · RegionProps"],
            OBS,
            300,
            216,
        ),
        (
            "geocatalog",
            ["matchup: IouAtLeast,", "NearestInTime, WithinWindow"],
            OBS,
            540,
            276,
        ),
    ]
    for name, sub, hue, bx, bw in boxes:
        f.box(bx, 280, bw, 80, name, sub, hue=hue, title_size=14.5, sub_size=12)
        f.arrow([(bx + bw / 2, 230), (bx + bw / 2, 278)], color=INK)
    f.box(
        24,
        410,
        792,
        66,
        "pipekit-cycle · pipekit-train · pipekit-experiment",
        ["forward models and cycles · losses and validation · runs and the registry"],
        hue=(LINE, PANEL),
        title_size=14.5,
        sub_size=12,
    )
    f.arrow([(528, 232), (528, 408)], color=LINE, dashed=True)
    f.text(536, 392, "planned links", size=12, color=MUTED, anchor="start", italic=True)
    f.h = 496
    return f


def r14_roadmap() -> Svg:
    """Report 14: the taxonomy and report first, then the missing metrics."""
    f = Svg(
        840,
        560,
        "Roadmap. Phase 1 gives pipekit-evaluate its taxonomy tags, a scorer"
        " adapter over xrtoolz metrics and an evaluation report, and fixes the"
        " ladder (A1 to A4). Phase 2 fills the metric gaps where their names are"
        " already reserved: FSS and spectrum comparison, the detection scores, the"
        " missing point-wise and probabilistic scores, and the seasonal, event and"
        " member wrappers (B1 to B4, C1 to C3). Phase 3 adds the two new units:"
        " trajectories (D1, D2) and events as catalog rows (E1, E2). Phase 4"
        " connects evaluation to training, cycles, the registry and presets (F1"
        " to F4).",
    )
    rows = [
        (
            [
                (
                    "1 · A1 to A4",
                    ["tags, MetricScorer,", "EvaluationReport, ladder fixes"],
                    ENG,
                )
            ],
            "A1 · A2 · A3 · A4",
        ),
        (
            [
                (
                    "2a · B1 to B4",
                    ["FSS, spectrum comparison,", "POD/FAR/CSI, NSE, Brier"],
                    STATE,
                ),
                ("2b · C1 to C3", ["by season, by event,", "by member"], STATE),
            ],
            "B1 to B4 · C1 to C3",
        ),
        (
            [
                ("3a · D1 D2", ["xrtoolz.lagrangian,", "trajectory scores"], ML),
                ("3b · E1 E2", ["event detectors,", "events as catalog rows"], ML),
            ],
            "D1 · D2 · E1 · E2",
        ),
        (
            [
                (
                    "4 · F1 to F4",
                    ["losses, per-step checks,", "registry links, presets"],
                    EXT,
                )
            ],
            "F1 to F4",
        ),
    ]
    y = 24.0
    for items, gate in rows:
        n = len(items)
        w = (792 - 18 * (n - 1)) / n
        for i, (name, sub, hue) in enumerate(items):
            f.box(24 + i * (w + 18), y, w, 66, name, sub, hue=hue, title_size=14)
        cx, gy, d = 420, y + 66 + 24, 12
        f.arrow([(cx, y + 66), (cx, gy - d - 1)], color=LINE)
        f.parts.append(
            f'<polygon points="{cx},{gy - d} {cx + d},{gy} {cx},{gy + d}'
            f' {cx - d},{gy}" fill="{ENG[1]}" stroke="{ENG[0]}" stroke-width="1.4"/>'
        )
        f.text(
            cx + d + 10,
            gy + 4.5,
            f"gates {gate}",
            size=12,
            color=ENG[0],
            anchor="start",
            weight=600,
        )
        if items is not rows[-1][0]:
            f.arrow([(cx, gy + d), (cx, gy + 42)], color=LINE)
        y = gy + 44
    f.h = int(y - 10)
    return f


FIGURES = {
    "r1_trajectories": r1_trajectories,
    "r2_groups": r2_groups,
    "r3_layers": r3_layers,
    "r6_dataflow": r6_dataflow,
    "r9_stack": r9_stack,
    "r13_ownership": r13_ownership,
    "r13_roadmap": r13_roadmap,
    "r10_ownership": r10_ownership,
    "r10_roadmap": r10_roadmap,
    "r11_ownership": r11_ownership,
    "r11_roadmap": r11_roadmap,
    "r12_ownership": r12_ownership,
    "r12_roadmap": r12_roadmap,
    "r14_ownership": r14_ownership,
    "r14_roadmap": r14_roadmap,
}


def main() -> None:
    for name, build in FIGURES.items():
        (OUT / f"{name}.svg").write_text(build().render(), encoding="utf-8")
        print(f"wrote {name}.svg")


if __name__ == "__main__":
    main()
