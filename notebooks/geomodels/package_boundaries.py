# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GeoModels · package boundaries, run
#
# Regenerates every number and figure in `docs/geomodels/package_boundaries.md`:
# where plume and radiative-transfer code lives today in geotoolz and plumax,
# what moves to plumax and to a new radtrax, and whether the package graph
# stays acyclic.
#
# 1. plumax's subpackage graph, from its imports (stage 1);
# 2. what geotoolz's `plume` and `matched_filter` import from geotoolz
#    (stage 2);
# 3. the implementations that exist more than once (stage 3);
# 4. the package graph today and after the moves, from each `pyproject.toml`,
#    with a strongly-connected-components check (stage 4).
#
# Nothing is imported or installed: the notebook reads source files and
# metadata only. geotoolz, xrtoolz, pipekit and georeader come from
# `references/<layer>/` at their pinned revisions, and so do plumax, gaussx,
# finitevolX and spectraldiffx, whose commits are asserted below.

# %%
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

root = Path.cwd()
while not (root / "myst.yml").exists() and root != root.parent:
    root = root.parent
REFS = root / "references"
GEOSTACK, NUMERICS, GEOMODELS = REFS / "geostack", REFS / "numerics", REFS / "geomodels"
LAYER = {"plumax": GEOMODELS, "gaussx": NUMERICS, "finitevolX": NUMERICS,
         "spectraldiffx": NUMERICS}
FIG = root / "docs/images/package_boundaries"
FIG.mkdir(parents=True, exist_ok=True)


def head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short=7", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


commits = {
    name: head(LAYER[name] / name)
    for name in ("plumax", "gaussx", "finitevolX", "spectraldiffx")
}
commits

# %%
assert commits == {
    "plumax": "28f257e",
    "gaussx": "7fa1a38",
    "finitevolX": "32ffb45",
    "spectraldiffx": "b5f3dc4",
}

# %% [markdown]
# ## Stage 1 · plumax's subpackages and their imports


# %%
def imports_of(path: Path) -> set[str]:
    """Absolute module names imported anywhere in a file, lazy ones included."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
    return out


PLX = GEOMODELS / "plumax/src/plumax"
SUBS = sorted(
    p.name for p in PLX.iterdir() if p.is_dir() and not p.name.startswith("_")
)
plx_edges: dict[str, set[str]] = defaultdict(set)
plx_loc: dict[str, int] = defaultdict(int)
for f in PLX.rglob("*.py"):
    rel = f.relative_to(PLX).parts
    sub = rel[0] if len(rel) > 1 else "(top level)"
    plx_loc[sub] += len(f.read_text(encoding="utf-8").splitlines())
    for mod in imports_of(f):
        parts = mod.split(".")
        if (
            parts[0] == "plumax"
            and len(parts) > 1
            and parts[1] in SUBS
            and parts[1] != sub
        ):
            plx_edges[sub].add(parts[1])

RADTRAX = {"radtran", "hapi_lut", "assimilation", "matched_filter"}
stays = set(SUBS) - RADTRAX
out_of_cluster = {s: sorted(plx_edges[s] - RADTRAX) for s in RADTRAX}
into_cluster = {
    s: sorted(plx_edges[s] & RADTRAX)
    for s in [*sorted(stays), "(top level)"]
    if plx_edges[s] & RADTRAX
}
stage1 = {
    "subs": SUBS,
    "edges": {s: sorted(v) for s, v in sorted(plx_edges.items()) if v},
    "loc_radtrax": sum(plx_loc[s] for s in RADTRAX),
    "loc_total": sum(plx_loc.values()),
    "out_of_cluster": out_of_cluster,
    "into_cluster": into_cluster,
}
stage1

# %%
assert stage1["subs"] == [
    "assimilation",
    "coupled",
    "gauss_plume",
    "gauss_puff",
    "hapi_lut",
    "lagrangian",
    "les_fvm",
    "matched_filter",
    "met",
    "population",
    "radtran",
]
assert stage1["edges"] == {
    "(top level)": ["gauss_plume", "gauss_puff", "lagrangian", "les_fvm", "radtran"],
    "assimilation": ["radtran"],
    "coupled": ["gauss_plume", "lagrangian", "radtran"],
    "gauss_puff": ["gauss_plume"],
    "les_fvm": ["gauss_puff", "lagrangian"],
    "matched_filter": ["assimilation"],
    "met": ["les_fvm"],
    "population": ["lagrangian"],
}
# the radiative-transfer cluster imports nothing else in plumax: it can move
assert all(v == [] for v in stage1["out_of_cluster"].values())
assert stage1["into_cluster"] == {"coupled": ["radtran"], "(top level)": ["radtran"]}
assert (stage1["loc_radtrax"], stage1["loc_total"]) == (6_100, 18_880)
assert sum(plx_loc[s] for s in stays) == 12_307  # the dispersion side

# %% [markdown]
# ## Stage 2 · What geotoolz's `plume` and `matched_filter` import from geotoolz

# %%
GZ = GEOSTACK / "geotoolz/packages/geotoolz-ops/src/geoops"


def internal_imports(pkg: str) -> dict[str, list[str]]:
    """geotoolz modules (outside the package itself) imported by `pkg`, with names."""
    out: dict[str, set[str]] = defaultdict(set)
    for f in (GZ / pkg).rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("geotoolz.")
                and not node.module.startswith(f"geotoolz.{pkg}")
            ):
                out[node.module] |= {a.name for a in node.names}
    return {k: sorted(v) for k, v in sorted(out.items())}


gz_plume = internal_imports("plume")
gz_mf = internal_imports("matched_filter")
gz_loc = {
    p: sum(len(f.read_text().splitlines()) for f in (GZ / p).rglob("*.py"))
    for p in ("plume", "matched_filter")
}
private = lambda d: sorted(m for m in d if "._src" in m)  # noqa: E731
# who imports them: other geotoolz modules mention them in docstrings only
callers = sorted(
    str(f.relative_to(GZ))
    for f in GZ.rglob("*.py")
    if not f.is_relative_to(GZ / "plume")
    and not f.is_relative_to(GZ / "matched_filter")
    and any(
        m.startswith(("geoops.plume", "geoops.matched_filter"))
        for m in imports_of(f)
    )
)
root_reexports = [
    m
    for m in sorted(imports_of(GZ / "__init__.py"))
    if m in ("geoops.plume", "geoops.matched_filter")
]
stage2 = {
    "plume": gz_plume,
    "mf": gz_mf,
    "loc": gz_loc,
    "n_private_plume": len(private(gz_plume)),
    "n_private_mf": len(private(gz_mf)),
    "callers": callers,
    "root_reexports": root_reexports,
}
stage2

# %%
assert stage2["loc"] == {"plume": 1_724, "matched_filter": 2_622}
assert stage2["n_private_plume"] == 9 and stage2["n_private_mf"] == 6
assert set(gz_plume) == {
    "geoops._src.bands",
    "geoops._src.config",
    "geoops._src.geo",
    "geoops._src.labels",
    "geoops._src.optional",
    "geoops._src.shape",
    "geoops._src.valid",
    "geoops._src.wrap",
    "geoops.segment._src.array",
}
# only the package root imports them: nothing else in geotoolz calls them
assert stage2["callers"] == ["__init__.py"]
assert stage2["root_reexports"] == ["geoops.matched_filter", "geoops.plume"]

# %% [markdown]
# ## Stage 3 · Implementations that exist more than once


# %%
def defs(path: Path) -> set[str]:
    return {
        n.name
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef | ast.ClassDef)
    }


def find(root_dir: Path, name: str) -> list[str]:
    return sorted(
        str(f.relative_to(root_dir.parent))
        for f in root_dir.rglob("*.py")
        if name in defs(f)
    )


dupes = {
    "matched_filter_snr": find(GZ, "matched_filter_snr")
    + find(PLX, "matched_filter_snr"),
    "matched filter, image": find(GZ, "apply_image")
    + find(PLX, "apply_image")
    + find(PLX, "matched_filter_image"),
    "Gaussian SRF": find(GEOSTACK / "georeader/georeader", "srf")
    + find(PLX, "SpectralResponseFunction"),
    "air-mass factor": find(PLX, "air_mass_factor"),
}
stage3 = {k: v for k, v in dupes.items()}
stage3

# %%
assert stage3["matched_filter_snr"] == [
    "geoops/matched_filter/_src/array.py",
    "plumax/matched_filter/core.py",
    "plumax/radtran/matched_filter.py",
]
assert stage3["matched filter, image"] == [
    "geoops/matched_filter/_src/array.py",
    "plumax/matched_filter/core.py",
    "plumax/radtran/matched_filter.py",
]
assert stage3["Gaussian SRF"] == ["georeader/reflectance.py", "plumax/radtran/srf.py"]
assert stage3["air-mass factor"] == [
    "plumax/hapi_lut/beers.py",
    "plumax/radtran/config.py",
]

# %% [markdown]
# ## Stage 4 · The package graph, today and after the moves

# %%
NODES = {  # distribution name → (label, stack)
    "pipekit": ("pipekit", "GeoStack"),
    "pipekit-cycle": ("pipekit-cycle", "GeoStack"),
    "georeader-spaceml": ("georeader", "GeoStack"),
    "geotoolz": ("geotoolz", "GeoStack"),
    "xrtoolz": ("xrtoolz", "GeoStack"),
    "gaussx": ("gaussx", "GeoNumerics"),
    "spectraldiffx": ("spectraldiffx", "GeoNumerics"),
    "finitevolx": ("finitevolx", "GeoNumerics"),
    "plumax": ("plumax", "GeoModels"),
    "radtrax": ("radtrax", "GeoModels"),
}


# the dependency parser and the cycle check are the ones
# tests/test_dependency_graph.py enforces
dg_spec = importlib.util.spec_from_file_location(
    "dependency_graph", root / "scripts/dependency_graph.py"
)
dg = importlib.util.module_from_spec(dg_spec)
sys.modules["dependency_graph"] = dg  # dataclasses look their module up
dg_spec.loader.exec_module(dg)


def deps(pyproject: Path) -> tuple[set[str], set[str]]:
    """(base, extras) stack dependencies declared in a pyproject.toml."""
    d = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    p = d.get("project") or d["tool"]["poetry"]
    base, extra = dg.declared(p)
    # an extra that names the package itself (an "all" extra) is not an edge
    keep = set(NODES) - {dg._norm(p["name"])}
    return base & keep, extra & keep


META = {
    "pipekit": GEOSTACK / "pipekit/packages/pipekit/pyproject.toml",
    "pipekit-cycle": GEOSTACK / "pipekit/packages/pipekit-cycle/pyproject.toml",
    "georeader-spaceml": GEOSTACK / "georeader/pyproject.toml",
    "geotoolz": GEOSTACK / "geotoolz/packages/geotoolz-ops/pyproject.toml",
    "xrtoolz": GEOSTACK / "xrtoolz/packages/xrtoolz/pyproject.toml",
    "gaussx": NUMERICS / "gaussx/pyproject.toml",
    "spectraldiffx": NUMERICS / "spectraldiffx/pyproject.toml",
    "finitevolx": NUMERICS / "finitevolX/pyproject.toml",
    "plumax": GEOMODELS / "plumax/pyproject.toml",
}
today = {n: deps(p) for n, p in META.items()}

# after the moves: radtrax takes plumax's radiative-transfer cluster and
# geotoolz's matched filter and SBMP; plumax takes geotoolz's plume operators
after = dict(today)
after["radtrax"] = ({"pipekit", "gaussx"}, {"geotoolz", "georeader-spaceml"})
after["plumax"] = (today["plumax"][0] | {"radtrax"}, today["plumax"][1] | {"geotoolz"})
# the counterfactual: geotoolz keeps re-exports of what moved, behind extras
shim = dict(after)
shim["geotoolz"] = (today["geotoolz"][0], today["geotoolz"][1] | {"plumax", "radtrax"})


def edges(graph, with_extras=True):
    return {
        (a, b)
        for a, (base, extra) in graph.items()
        for b in (base | extra if with_extras else base)
    }


def sccs(graph) -> list[list[str]]:
    """The strongly connected components with more than one node."""
    nodes = set(graph) | {b for _, b in edges(graph)}
    return dg.sccs(sorted(nodes), sorted(edges(graph)))


def layers(graph) -> dict[str, int]:
    # longest path from a node with no stack dependencies
    memo: dict[str, int] = {}

    def depth(v):
        if v not in memo:
            ds = graph.get(v, (set(), set()))
            memo[v] = 1 + max((depth(w) for w in ds[0] | ds[1]), default=-1)
        return memo[v]

    return {v: depth(v) for v in graph}


cross = sorted(
    (NODES[a][1], NODES[b][1]) for a, b in edges(after) if NODES[a][1] != NODES[b][1]
)
stage4 = {
    "today": {k: (sorted(b), sorted(e)) for k, (b, e) in sorted(today.items())},
    "cycles_today": sccs(today),
    "cycles_after": sccs(after),
    "cycles_shim": sccs(shim),
    "layers_after": layers(after),
    "cross_stack": sorted(set(cross)),
}
stage4

# %%
assert stage4["today"] == {
    "finitevolx": (["gaussx", "spectraldiffx"], []),
    "gaussx": ([], []),
    "geotoolz": (["georeader-spaceml", "pipekit"], []),
    "georeader-spaceml": ([], []),
    "pipekit": ([], []),
    "pipekit-cycle": (["pipekit"], []),
    "plumax": (["finitevolx", "gaussx", "pipekit"], []),
    "spectraldiffx": (["gaussx"], []),
    "xrtoolz": (["pipekit"], []),
}
assert stage4["cycles_today"] == [] and stage4["cycles_after"] == []
# re-exporting the moved code from geoops closes two loops through the extras
assert stage4["cycles_shim"] == [["geotoolz", "plumax", "radtrax"]]
assert stage4["layers_after"] == {
    "pipekit": 0,
    "pipekit-cycle": 1,
    "georeader-spaceml": 0,
    "geotoolz": 1,
    "xrtoolz": 1,
    "gaussx": 0,
    "spectraldiffx": 1,
    "finitevolx": 2,
    "plumax": 3,
    "radtrax": 2,
}
# edges between stacks point from GeoModels down to GeoNumerics and GeoStack only
assert stage4["cross_stack"] == [("GeoModels", "GeoStack"), ("GeoModels", "GeoNumerics")]

# %% [markdown]
# ## The figures

# %%
spec = importlib.util.spec_from_file_location(
    "overview_figures", root / "docs/images/overview/make_figures.py"
)
ov = importlib.util.module_from_spec(spec)
sys.modules["overview_figures"] = ov
spec.loader.exec_module(ov)
HUE = {"GeoStack": ov.OBS, "GeoNumerics": ov.ENG, "GeoModels": ov.STATE}

# (a) the package graph after the moves, by layer
W, bw, bh = 760, 132, 46
lay = stage4["layers_after"]
rows: dict[int, list[str]] = defaultdict(list)
order = [
    "pipekit",
    "georeader-spaceml",
    "gaussx",
    "pipekit-cycle",
    "xrtoolz",
    "geotoolz",
    "spectraldiffx",
    "finitevolx",
    "radtrax",
    "plumax",
]
for n in order:
    rows[lay[n]].append(n)
top = 64
H = top + (max(rows) + 1) * 108 + 20
svg = ov.Svg(
    W,
    H,
    "The stack's package graph after the moves, by layer: pipekit,"
    " georeader and gaussx at the bottom; pipekit-cycle, geotoolz, xrtoolz and"
    " spectraldiffx above them; finitevolx and radtrax next; plumax on top."
    " Solid arrows are base dependencies and dashed arrows extras. radtrax is"
    " new; the graph has no cycle. An arrow implied by a longer path is not drawn.",
)
# x positions staggered between layers, so no edge runs through a box
XS = {
    "pipekit": 24,
    "georeader-spaceml": 314,
    "gaussx": 470,
    "pipekit-cycle": 24,
    "xrtoolz": 186,
    "geotoolz": 400,
    "spectraldiffx": 614,
    "finitevolx": 169,
    "radtrax": 480,
    "plumax": 314,
}
pos = {n: (XS[n], H - 30 - bh - lay[n] * 108) for n in order}


def reaches(graph, a, b, skip, base_only):
    # is there a path a → … → b that does not use the edge `skip`? A base
    # edge is implied only by a path of base edges
    seen, todo = set(), [a]
    while todo:
        v = todo.pop()
        base, extra = graph.get(v, (set(), set()))
        for w in base if base_only else base | extra:
            if (v, w) == skip or w in seen:
                continue
            if w == b:
                return True
            seen.add(w)
            todo.append(w)
    return False


# the transitive reduction: an arrow implied by a longer path is not drawn
drawn = {
    (a, b)
    for a, b in edges(after)
    if not reaches(after, a, b, (a, b), base_only=b in after[a][0])
}
for a, (base, extra) in after.items():
    for b in sorted(base | extra):
        if (a, b) not in drawn:
            continue
        (xa, ya), (xb, yb) = pos[a], pos[b]
        p0, p1 = (xa + bw / 2, ya + bh), (xb + bw / 2, yb)
        dash = ' stroke-dasharray="5 4"' if b in extra and b not in base else ""
        if (a, b) == ("radtrax", "pipekit"):  # two layers down: route round the boxes
            svg.arrow(
                [
                    (p0[0] - 30, p0[1] + 2),
                    (p0[0] - 30, p0[1] + 34),
                    (171, p0[1] + 34),
                    (171, yb + bh / 2),
                    (xb + bw + 4, yb + bh / 2),
                ],
                color=ov.LINE,
            )
            continue
        svg.markers.add(ov.LINE)
        svg.parts.append(
            f'<line x1="{p0[0]:.1f}" y1="{p0[1] + 2:.1f}" x2="{p1[0]:.1f}"'
            f' y2="{p1[1] - 4:.1f}" stroke="{ov.LINE}" stroke-width="1.3"{dash}'
            f' marker-end="url(#m{ov.LINE.lstrip("#")})"/>'
        )
for n, (x, y) in pos.items():
    label, stack = NODES[n]
    svg.rect(
        x, y, bw, bh, stroke=HUE[stack][0], fill=HUE[stack][1], dashed=(n == "radtrax")
    )
    svg.text(x + bw / 2, y + 20, label, size=13.5, weight=700, color=ov.INK)
    svg.text(x + bw / 2, y + 36, stack, size=10.5, color=HUE[stack][0])
svg.text(
    24,
    30,
    "after the moves · solid: base · dashed: extra · implied arrows omitted",
    size=12.5,
    color=ov.MUTED,
    anchor="start",
)
(FIG / "package_graph.svg").write_text(svg.render(), encoding="utf-8")

# (b) plumax's subpackages, with the cluster that becomes radtrax
# the package root re-exports four models and radtran; it is left out of the
# drawing, which shows the subpackages
grid = {
    "hapi_lut": (0, 0),
    "radtran": (1, 0),
    "assimilation": (2, 0),
    "matched_filter": (3, 0),
    "gauss_plume": (0, 1),
    "coupled": (1, 1),
    "lagrangian": (2, 1),
    "gauss_puff": (0, 2),
    "les_fvm": (1, 2),
    "met": (2, 2),
    "population": (3, 2),
}
cw, ch, gx, gy = 150, 44, 184, 96
svg2 = ov.Svg(
    W,
    70 + 3 * gy,
    "plumax's subpackages and their imports. radtran,"
    " hapi_lut, assimilation and matched_filter import nothing else in plumax"
    " and become radtrax; coupled imports radtran, and so does the package"
    " root, which is not drawn.",
)
svg2.rect(14, 44, 4 * gy * 1.92, 64, stroke=ov.STATE[0], fill="none", dashed=True)
svg2.text(
    24,
    34,
    "becomes radtrax: 6,100 lines, no import out of the group",
    size=12.5,
    color=ov.STATE[0],
    anchor="start",
    weight=600,
)
cpos = {n: (24 + c * gx, 54 + r * gy) for n, (c, r) in grid.items()}
for a, bs in stage1["edges"].items():
    if a not in grid:
        continue
    for b in bs:
        (xa, ya), (xb, yb) = cpos[a], cpos[b]
        ca, cb = (xa + cw / 2, ya + ch / 2), (xb + cw / 2, yb + ch / 2)
        if ya == yb:
            p0 = (xa + (cw if xb > xa else 0), ca[1])
            p1 = (xb + (0 if xb > xa else cw) + (-4 if xb > xa else 4), cb[1])
        else:
            p0 = (ca[0], ya if yb < ya else ya + ch)
            p1 = (cb[0], yb + ch + 4 if yb < ya else yb - 4)
        svg2.arrow([p0, p1], color=ov.LINE)
for n, (x, y) in cpos.items():
    hue = ov.STATE if n in RADTRAX else (ov.BORDER, ov.CARD)
    svg2.rect(x, y, cw, ch, stroke=hue[0], fill=hue[1])
    svg2.text(x + cw / 2, y + 20, n, size=12.5, weight=600, color=ov.INK)
    svg2.text(x + cw / 2, y + 35, f"{plx_loc[n]:,} lines", size=10.5, color=ov.MUTED)
svg2.h = 54 + 3 * gy
(FIG / "plumax_modules.svg").write_text(svg2.render(), encoding="utf-8")
sorted(p.name for p in FIG.iterdir())

# %%
assert sorted(p.name for p in FIG.iterdir()) == [
    "package_graph.svg",
    "plumax_modules.svg",
]
