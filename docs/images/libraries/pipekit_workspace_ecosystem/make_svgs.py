"""Write the pipekit workspace's dependency figure as light and dark SVGs.

The layout is pyrox's (``pyrox_workspace_ecosystem/``): the seven packages of
the pipekit workspace and the umbrella, with an arrow for each import between
them, and the GeoStack libraries that import the workspace underneath. The
edges are each package's ``from pipekit_<x>… import`` and ``import pipekit_<x>``
lines in its ``src/`` on pipekit's ``main``, outside tests and outside
``TYPE_CHECKING`` blocks; an import through an optional extra is dashed.
pipekit-train reaches pipekit-experiment with
``import_module("pipekit_experiment.artifacts")`` in ``TrainingLoop``, behind
its ``experiment`` extra, as ``pipekit_experiment_ecosystem/`` draws it.
Left out, as the per-package figures leave them out: pipekit-cycle's ``array``
extra and pipekit-train's ``cycle`` extra, which install a sibling and import
nothing from it, and pipekit-evaluate's ``cycle`` extra, imported only for type
checking. The libraries on each card are its ``pyproject.toml``: what it
requires, then its extras. The umbrella imports pipekit-ops only in its
``__getattr__``, to say where a moved name went. ``_icon*.svg`` are pipekit's
``docs/assets/icon.svg`` and ``docs/assets/packages/pipekit-<pkg>/icon.svg``,
copied unchanged. Re-run after editing the data below, from the repo root::

    uv run --no-project --with loguru --with cyclopts \\
        python docs/images/libraries/pipekit_workspace_ecosystem/make_svgs.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

from cyclopts import App
from loguru import logger


HERE = Path(__file__).resolve().parent
app = App(help="Write the pipekit workspace's dependency figure as two SVGs.")

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"


@dataclass(frozen=True)
class Package:
    """One card: the package, what it holds and the libraries it builds on."""

    name: str
    icon: str
    what: str
    needs: tuple[str, ...]
    extras: tuple[str, ...] = ()
    foundation: bool = False


UPPER = [
    Package(
        "pipekit-train",
        "train",
        "training as an operator",
        (),
        (
            "extras: Equinox · optax · JAX",
            "jaxtyping · Grain · Orbax · Keras",
            "NumPyro · BlackJAX · zarr",
            "fsspec · s3fs · gcsfs",
        ),
    ),
    Package(
        "pipekit",
        "",
        "the umbrella, no code",
        (),
        (
            "installs all seven packages and",
            "imports pipekit-ops only to say",
            "where a moved name went",
        ),
    ),
    Package(
        "pipekit-jax",
        "jax",
        "operators as pytrees",
        ("JAX · Equinox · jaxtyping",),
        ("extra array: pipekit-array",),
    ),
]
LOWER = [
    Package(
        "pipekit-experiment",
        "experiment",
        "models under their hash",
        (),
        (
            "extras: fsspec · s3fs · DVC",
            "Hydra · hydra-zen · Metaflow",
            "mlflow · wandb (pins only)",
        ),
    ),
    Package(
        "pipekit-evaluate",
        "evaluate",
        "scores and reports",
        ("NumPy · SciPy",),
        ("extras: xskillscore", "properscoring"),
    ),
    Package(
        "pipekit-cycle",
        "cycle",
        "time stepping and DA",
        (),
        ("extra array installs", "pipekit-array"),
    ),
    Package(
        "pipekit-array",
        "array",
        "arrays by dim name",
        (),
        (
            "extras: NumPy · JAX · PyTorch",
            "CuPy · dask · xarray · einx",
            "scikit-learn · array-api-compat",
        ),
        foundation=True,
    ),
]
OPS = Package(
    "pipekit-ops",
    "ops",
    "the operator graph",
    ("the Python standard library alone: no dependencies",),
    foundation=True,
)
# (importer, imported, through an optional extra)
EDGES = [
    ("pipekit-train", "pipekit-experiment", True),
    ("pipekit-train", "pipekit-ops", False),
    ("pipekit", "pipekit-ops", False),
    ("pipekit-jax", "pipekit-cycle", False),
    ("pipekit-jax", "pipekit-ops", False),
    ("pipekit-jax", "pipekit-array", True),
    ("pipekit-experiment", "pipekit-ops", False),
    ("pipekit-evaluate", "pipekit-ops", False),
    ("pipekit-cycle", "pipekit-ops", False),
    ("pipekit-array", "pipekit-ops", False),
]
USERS = (
    "geotoolz · gridax · geophysix · radiomax · tracerax · diagnostix · plumax · "
    "radtrax · sesmax · somax · vardax"
)
NOTE = (
    "Not drawn: pipekit-cycle's array extra and pipekit-train's cycle extra install "
    "a sibling and import nothing from it,",
    "and pipekit-evaluate's cycle extra imports pipekit-cycle for type checking only.",
)

THEMES = {
    "light": {
        "pipekit-ops": "#EA580C",
        "pipekit-array": "#6D28D9",
        "pipekit-cycle": "#0369A1",
        "pipekit-experiment": "#334155",
        "ops_fill": "#FFF7ED",
        "array_fill": "#F5F3FF",
        "card": "#FFFFFF",
        "line": "#CBD5E1",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "box": ("#F9FAFB", "#E5E7EB"),
        "uses": "#334155",
        "base": ("#F3F4F6", "#D1D5DB"),
    },
    "dark": {
        "pipekit-ops": "#FB923C",
        "pipekit-array": "#A78BFA",
        "pipekit-cycle": "#38BDF8",
        "pipekit-experiment": "#94A3B8",
        "ops_fill": "#2A1408",
        "array_fill": "#1E1338",
        "card": "#111827",
        "line": "#374151",
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "box": ("#0F172A", "#1F2937"),
        "uses": "#CBD5E1",
        "base": ("#1F2937", "#374151"),
    },
}

WIDTH = 1000
BOX_X, BOX_Y, BOX_W = 16, 16, 968
INNER = BOX_X + 20
LOW_W, LOW_GAP = 211, 28
UP_W = 218
UP_Y, UP_H = 66, 140
LOW_Y, LOW_H = UP_Y + UP_H + 56, 128
OPS_Y, OPS_H = LOW_Y + LOW_H + 44, 64
BOX_H = OPS_Y + OPS_H + 20 - BOX_Y
# where an arrow leaves its importer, as an x offset from the card's left edge;
# pipekit-array's leaves left of the FOUNDATION badge on pipekit-ops
LEAVE = {
    ("pipekit-train", "pipekit-experiment"): 48,
    ("pipekit-jax", "pipekit-cycle"): 50,
    ("pipekit-jax", "pipekit-array"): 170,
    ("pipekit-array", "pipekit-ops"): 50,
}


def text(
    x: float,
    y: float,
    body: str,
    size: float,
    fill: str,
    *,
    font: str = SANS,
    bold: bool = False,
    anchor: str = "start",
) -> str:
    weight = ' font-weight="700"' if bold else ""
    return (
        f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" font-family="{font}" '
        f'font-size="{size}"{weight} fill="{fill}">{escape(body)}</text>'
    )


def icon(name: str, x: float, y: float, size: float, theme: str) -> str:
    """One package's icon, with its gradient id made unique."""
    suffix = f"-{name}" if name else ""
    src = (HERE / f"_icon{suffix}.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.S)
    assert inner is not None
    body = re.sub(r"<title>.*?</title>", "", inner.group(1), flags=re.S)
    (gid,) = re.findall(r'linearGradient id="([^"]+)"', body)
    body = body.replace(gid, f"pkw{suffix}-{theme}").strip()
    return (
        f'<svg x="{x:g}" y="{y:g}" width="{size}" height="{size}" '
        f'viewBox="0 0 128 128">{body}</svg>'
    )


def pill(x: float, y: float, colour: str) -> str:
    """The FOUNDATION badge, its right end at ``x`` on the card's top edge."""
    w = 92
    return (
        f'<rect x="{x - w:g}" y="{y - 9:g}" width="{w}" height="18" rx="9" '
        f'fill="{colour}"/>'
        + text(
            x - w / 2, y + 4, "FOUNDATION", 10.5, "#FFFFFF", bold=True, anchor="middle"
        )
    )


def card(p: Package, x: float, y: float, w: float, h: float, theme: str) -> str:
    t = THEMES[theme]
    fill, stroke, width = t["card"], t["line"], 1.5
    if p.foundation:
        fill, stroke, width = t["array_fill"], t[p.name], 2
    out = [
        f'<rect x="{x:g}" y="{y:g}" width="{w}" height="{h}" rx="12" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>',
        icon(p.icon, x + 12, y + 14, 36, theme),
        text(x + 56, y + 30, p.name, 13, t["ink"], font=MONO, bold=True),
        text(x + 56, y + 47, p.what, 12, t["muted"]),
    ]
    ly = y + 76
    for line in p.needs:
        out.append(text(x + 14, ly, line, 12, t["uses"]))
        ly += 18
    for line in p.extras:
        out.append(text(x + 14, ly, line, 11.5, t["muted"]))
        ly += 17
    if p.foundation:
        out.append(pill(x + w - 14, y, t[p.name]))
    return "".join(out)


def ops_bar(y: float, theme: str) -> str:
    t = THEMES[theme]
    x, w = INNER, BOX_W - 40
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{OPS_H}" rx="12" '
        f'fill="{t["ops_fill"]}" stroke="{t["pipekit-ops"]}" stroke-width="2"/>'
        + icon(OPS.icon, x + 12, y + 12, 40, theme)
        + text(x + 62, y + 30, OPS.name, 14, t["ink"], font=MONO, bold=True)
        + text(x + 62, y + 48, OPS.what, 12, t["muted"])
        + text(x + w - 18, y + 40, OPS.needs[0], 12, t["uses"], anchor="end")
        + pill(x + w - 14, y, t["pipekit-ops"])
    )


def build(theme: str) -> str:
    t = THEMES[theme]
    fill, stroke = t["box"]
    targets = ("pipekit-ops", "pipekit-array", "pipekit-cycle", "pipekit-experiment")
    markers = "".join(
        f'<marker id="w-{k}-{theme}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{t[k]}"/></marker>'
        for k in targets
    )
    out = [
        f"<defs>{markers}</defs>",
        f'<rect x="{BOX_X}" y="{BOX_Y}" width="{BOX_W}" height="{BOX_H}" rx="16" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        text(BOX_X + 20, BOX_Y + 31, "The pipekit workspace", 16, t["ink"], bold=True),
        text(
            BOX_X + BOX_W - 20,
            BOX_Y + 31,
            "an arrow points at the package that is imported",
            12.5,
            t["muted"],
            anchor="end",
        ),
    ]
    # the cards: the lower row fills the box, the upper row sits over its gaps
    low_x = {p.name: INNER + i * (LOW_W + LOW_GAP) for i, p in enumerate(LOWER)}
    gaps = [INNER + LOW_W + LOW_GAP / 2 + i * (LOW_W + LOW_GAP) for i in range(3)]
    up_x = {p.name: g - UP_W / 2 for p, g in zip(UPPER, gaps, strict=True)}
    for src, dst, extra in EDGES:
        if src in up_x:
            y0 = UP_Y + UP_H
            x = up_x[src] + LEAVE.get((src, dst), UP_W / 2)
        else:
            y0 = LOW_Y + LOW_H
            x = low_x[src] + LEAVE.get((src, dst), LOW_W / 2)
        y1 = OPS_Y if dst == "pipekit-ops" else LOW_Y
        dash = ' stroke-dasharray="6 4"' if extra else ""
        out.append(
            f'<path d="M{x:g} {y0} L{x:g} {y1 - 3}" fill="none" stroke="{t[dst]}" '
            f'stroke-width="2"{dash} marker-end="url(#w-{dst}-{theme})"/>'
        )
    for p in UPPER:
        out.append(card(p, up_x[p.name], UP_Y, UP_W, UP_H, theme))
    for p in LOWER:
        out.append(card(p, low_x[p.name], LOW_Y, LOW_W, LOW_H, theme))
    out.append(ops_bar(OPS_Y, theme))
    # the GeoStack libraries that import the workspace
    by = BOX_Y + BOX_H + 16
    bfill, bstroke = t["base"]
    out.append(
        f'<rect x="{BOX_X}" y="{by}" width="{BOX_W}" height="58" rx="12" '
        f'fill="{bfill}" stroke="{bstroke}"/>'
        + text(BOX_X + 18, by + 24, "Imported on main by", 13.5, t["ink"], bold=True)
        + text(BOX_X + 18, by + 44, USERS, 13, t["muted"])
    )
    # legend: an import, and an import through an extra
    ly = by + 58 + 30
    grey = t["muted"]
    x = 40.0
    for dash, label in (
        ("", "imports it"),
        (' stroke-dasharray="6 4"', "imports it through an optional extra"),
    ):
        out.append(
            f'<path d="M{x:g} {ly - 4} L{x + 26:g} {ly - 4}" stroke="{grey}" '
            f'stroke-width="2"{dash}/>' + text(x + 34, ly, label, 12.5, grey)
        )
        x += 34 + 6.6 * len(label) + 40
    for i, line in enumerate(NOTE):
        out.append(text(40, ly + 24 + i * 18, line, 12, grey))
    height = ly + 24 + 18 * (len(NOTE) - 1) + 18
    label = (
        "The seven packages of the pipekit workspace and the umbrella. Every one "
        "imports pipekit-ops, the first foundation. pipekit-jax also imports "
        "pipekit-cycle, and pipekit-array through its array extra; pipekit-train "
        "imports pipekit-experiment through its experiment extra. pipekit-array, "
        "the second foundation, imports only pipekit-ops, and pipekit-ops imports "
        f"nothing. Imported on main by {USERS.replace(' · ', ', ')}."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(label)}">\n'
        "<title>The packages of the pipekit workspace</title>\n"
        + "\n".join(out)
        + "\n</svg>\n"
    )


@app.default
def main() -> None:
    """Write ``pipekit_workspace_ecosystem-light.svg`` and ``-dark.svg``."""
    for theme in THEMES:
        path = HERE / f"pipekit_workspace_ecosystem-{theme}.svg"
        path.write_text(build(theme))
        logger.info("wrote {}", path.name)


if __name__ == "__main__":
    app()
