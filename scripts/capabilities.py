"""Generate, or check, the capability index: every public name we build on.

``docs/api/capabilities.md`` lists each public object of the project package
(when there is one) and of every ``UPSTREAM`` library once, under the shortest
module path that exports it, with the first sentence of its docstring. Agents
and people search it before writing a loader, a model, a training step or a
helper in a script or a notebook ("Reuse before you write" in ``AGENTS.md``).

Usage::

    make capabilities                                   # rewrite the index
    uv run python scripts/capabilities.py --check       # fail if stale

``tests/test_capabilities.py`` runs the check. The script is generic: it
walks every non-private module of ``PACKAGE`` (``None`` when the project
has no package of its own, so only ``UPSTREAM`` is listed). A module's public names are
its ``__all__`` when it defines one, otherwise the classes and functions it
defines itself (imports are skipped; list constants in ``__all__``).

``--check`` also fails when two modules (of the package or of ``UPSTREAM``)
export the same name for different objects, unless ``ALLOWED_SHARED_NAMES``
gives the reason.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import pkgutil
import re
import sys
import types
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "api" / "capabilities.md"

# The import name of the project's own package (src/<PACKAGE>/), or None:
# geostack has no package of its own, it builds on the UPSTREAM libraries.
PACKAGE: str | None = None

# The reference libraries geostack builds on (git submodules in references/,
# installed from there by pixi), as {distribution: [public modules]}, with a
# one-line blurb each.
UPSTREAM: dict[str, list[str]] = {
    "pipekit": ["pipekit"],
    "pipekit-array": ["pipekit_array"],
    "pipekit-cycle": ["pipekit_cycle"],
    "pipekit-evaluate": ["pipekit_evaluate"],
    "pipekit-experiment": ["pipekit_experiment"],
    "pipekit-jax": ["pipekit_jax"],
    "pipekit-train": ["pipekit_train"],
    "georeader-spaceml": [
        "georeader.geotensor",
        "georeader.read",
        "georeader.save",
        "georeader.rasterio_reader",
        "georeader.mosaic",
    ],
    "geotoolz": ["geotoolz"],
    "geotoolz-catalog": ["geocatalog"],
    "geotoolz-cloud": ["geocloud"],
    "geotoolz-patcher": ["geopatcher"],
    "geotoolz-products": ["geoproducts"],
    "xrtoolz": ["xrtoolz"],
    "xrtoolz-core": ["xrcore"],
    "xrtoolz-einx": ["xreinx"],
    "xrtoolz-grad": ["xrgrad"],
    "xrtoolz-reader": ["xrreader"],
    "xrtoolz-sklearn": ["xrsklearn"],
}
UPSTREAM_BLURB: dict[str, str] = {
    "pipekit": "Carrier-agnostic operator-graph framework for pipelines.",
    "pipekit-array": "Array API operators (NumPy, JAX, CuPy, PyTorch).",
    "pipekit-cycle": "Time-stepping, data assimilation and observation operators.",
    "pipekit-evaluate": "Multidimensional evaluation (Unit x Lens x Stage).",
    "pipekit-experiment": "Experiment tracking adapters and a model registry.",
    "pipekit-jax": "JAX / Equinox carrier integration for pipekit.",
    "pipekit-train": "Training pipelines for emulators and amortized inference.",
    "georeader-spaceml": "Raster reading and the GeoTensor carrier.",
    "geotoolz": "Composable remote-sensing operators on GeoTensor.",
    "geotoolz-catalog": "Spatiotemporal catalog over geospatial files.",
    "geotoolz-cloud": "Cloud object storage and batched async COG reads.",
    "geotoolz-patcher": "Four-axis patcher: split, run per patch, stitch.",
    "geotoolz-products": "Readers for Earth-observation data products.",
    "xrtoolz": "Composable operators for Earth System Data Cubes.",
    "xrtoolz-core": "xarray-aware Operator base and Signature descriptor.",
    "xrtoolz-einx": "Labeled named-tensor algebra bridging xarray and einx.",
    "xrtoolz-grad": "Finite-difference calculus on labeled xarray grids.",
    "xrtoolz-reader": "Authenticated archive readers (CMEMS, CDS, AEMET).",
    "xrtoolz-sklearn": "scikit-learn to xarray bridge (XarrayEstimator, SklearnOp).",
}

# Names two homes may share on purpose, and why: {name: reason}.
ALLOWED_SHARED_NAMES: dict[str, str] = {
    "Operator": (
        "pipekit.Operator is the base (geotoolz re-exports it); xrcore.Operator"
        " (re-exported by xrtoolz) subclasses it with xarray DataTree dispatch."
    ),
    "Signature": (
        "pipekit.Signature records array shape + dtype; xrcore.Signature is the"
        " xarray-aware descriptor used by xrtoolz operators."
    ),
    "Composite": (
        "geotoolz.Composite builds a band composite; pipekit_train.Composite is"
        " a weighted sum of losses."
    ),
    "SklearnOp": (
        "One scikit-learn adapter per carrier: geotoolz.SklearnOp for"
        " GeoTensor, xrsklearn.SklearnOp for xarray."
    ),
    "register": (
        "pipekit.register marks a class as loadable by loads_sandboxed;"
        " xrreader.register adds a Variable to the CF variable registry."
    ),
}

_AUTOREF = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
# A full stop that ends a sentence (not "e.g." / "i.e." / "etc.").
_SENTENCE_END = re.compile(r"(?<!e\.g)(?<!i\.e)(?<!etc)\.\s+(?=[A-Z])")
UPSTREAM_MARKER = "<!-- upstream -->"


def _first_line(text: str) -> str:
    """The docstring's summary: its first sentence, on one line."""
    paragraph = text.strip().split("\n\n")[0]
    line = " ".join(part.strip() for part in paragraph.splitlines())
    line = _AUTOREF.sub(r"\1", line)
    sentence = _SENTENCE_END.split(line, maxsplit=1)[0]
    if sentence != line:
        line = sentence + "."
    if len(line) > 160:
        line = line[:157].rstrip() + "..."
    return line.replace("|", "\\|")


def _kind(obj: object) -> str:
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "constant"


def _origin(obj: object) -> str:
    module = getattr(obj, "__module__", None)
    if not isinstance(module, str) or not (callable(obj) or inspect.isclass(obj)):
        module = type(obj).__module__
    return module


def _summary(obj: object, *, home: str) -> str:
    origin = _origin(obj).split(".")[0]
    if origin != home and (callable(obj) or inspect.isclass(obj)):
        return f"Re-exported from `{origin}`."
    if not callable(obj):
        value = repr(obj)
        return f"`{value if len(value) <= 60 else value[:57] + '...'}`".replace(
            "|", "\\|"
        )
    return _first_line(inspect.getdoc(obj) or "")


class _NeedsExtra:
    """Stands in for a lazily exported name whose optional dependency is absent."""

    def __init__(self, err: ImportError) -> None:
        self.missing = err.name


def _get(module: types.ModuleType, name: str) -> object:
    """``getattr``, tolerating a lazy export whose optional dependency is absent."""
    try:
        return getattr(module, name)
    except ImportError as err:
        return _NeedsExtra(err)


def _row(name: str, obj: object, *, home: str) -> tuple[str, str, str]:
    if isinstance(obj, _NeedsExtra):
        dep = f"`{obj.missing}`" if obj.missing else "an optional dependency"
        return (name, "lazy", f"Needs {dep}.")
    return (name, _kind(obj), _summary(obj, home=home))


def _public(module: types.ModuleType) -> list[str]:
    """``__all__``, else the classes and functions the module defines."""
    names = getattr(module, "__all__", None)
    if names is None:
        names = [
            n
            for n, obj in vars(module).items()
            if not n.startswith("_")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
            and obj.__module__ == module.__name__
        ]
    return [
        n
        for n in names
        if not n.startswith("__") and not isinstance(_get(module, n), types.ModuleType)
    ]


def modules() -> list[str]:
    """``PACKAGE`` and every submodule with no ``_``-prefixed component."""
    package = PACKAGE
    if package is None:
        return []
    root = importlib.import_module(package)
    found = [package]
    for info in pkgutil.walk_packages(root.__path__, prefix=f"{package}."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        found.append(info.name)
    return sorted(found)


def collect() -> tuple[dict[str, list[tuple]], list[str]]:
    """``{module: [(name, kind, summary)]}`` and the list of name clashes.

    Each object is listed once, at the first module (in sorted order, so the
    shortest public path) that exports it.
    """
    index: dict[str, list[tuple]] = defaultdict(list)
    owners: dict[str, dict[str, object]] = defaultdict(dict)
    seen: set[int] = set()
    for module_name in modules():
        module = importlib.import_module(module_name)
        for name in _public(module):
            obj = getattr(module, name)
            owners[name][f"{module_name}.{name}"] = obj
            if id(obj) in seen and not isinstance(obj, int | float | str):
                continue
            seen.add(id(obj))
            index[module_name].append(
                (name, _kind(obj), _summary(obj, home=str(PACKAGE)))
            )
    for upstream_modules in UPSTREAM.values():
        for module_name in upstream_modules:
            try:
                upstream = importlib.import_module(module_name)
            except ImportError:
                continue
            for name in _public(upstream):
                owners[name][f"{module_name}.{name}"] = _get(upstream, name)
    clashes = []
    for name, homes in sorted(owners.items()):
        if name in ALLOWED_SHARED_NAMES:
            continue
        if len({id(o) for o in homes.values()}) > 1:
            clashes.append(f"{name}: {', '.join(sorted(homes))}")
    return index, clashes


def _table(rows: list[tuple]) -> list[str]:
    out = ["| Name | Kind | What it does |", "|---|---|---|"]
    out += [f"| `{r[0]}` | {r[1]} | {r[2]} |" for r in rows]
    return [*out, ""]


def _package_part() -> tuple[list[str], int]:
    index, _ = collect()
    out: list[str] = []
    total = 0
    for module_name in modules():
        rows = index.get(module_name, [])
        if not rows:
            continue
        total += len(rows)
        title = _first_line(importlib.import_module(module_name).__doc__ or "")
        out += [f"## `{module_name}`", "", title, "", *_table(sorted(rows))]
    return out, total


def _upstream_part() -> list[str]:
    """Each upstream object once, at the first module (in ``UPSTREAM`` order)."""
    out: list[str] = []
    seen: set[int] = set()
    for package, upstream_modules in UPSTREAM.items():
        out += [f"## Upstream: `{package}`", "", UPSTREAM_BLURB[package], ""]
        for module_name in upstream_modules:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            home = module_name.split(".")[0]
            rows = []
            for name in sorted(_public(module)):
                obj = _get(module, name)
                if id(obj) in seen and not isinstance(obj, int | float | str):
                    continue
                seen.add(id(obj))
                rows.append(_row(name, obj, home=home))
            if rows:
                out += [f"### `{module_name}`", "", *_table(rows)]
    return out


def _versions() -> str:
    return ", ".join(f"{p} {importlib.metadata.version(p)}" for p in UPSTREAM)


def render() -> str:
    package_part, total = _package_part()
    out = [
        "# Capability index",
        "",
        "<!-- Generated by scripts/capabilities.py — do not edit by hand. -->",
        "",
        (
            f"Every public name in `{PACKAGE}` ({total} of them), each listed once"
            if PACKAGE
            else "Every public name in the libraries geostack builds on, each listed"
        ),
        "once under the shortest module path that exports it, with the first",
        "sentence of its docstring. Search this page before writing a",
        "loader, a model, a training step or a helper in a script or a",
        "notebook: if what you need is here, use it; if it is almost here,",
        "extend it where it lives. Regenerate with `make capabilities` after",
        "changing a public API (`tests/test_capabilities.py` checks it is",
        "current).",
        "",
        *package_part,
    ]
    if UPSTREAM:
        out += [
            UPSTREAM_MARKER,
            "",
            f"Listed at {_versions()}.",
            "",
            *_upstream_part(),
        ]
    return "\n".join(out).rstrip() + "\n"


def stale(current: str, text: str) -> bool:
    """Whether ``current`` differs from ``text``.

    The upstream libraries are pinned by the submodules in ``references/``
    and ``pixi.lock``, so the whole page, versions included, must match:
    moving a submodule without ``make capabilities`` fails the check.
    """
    return current != text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if stale")
    args = parser.parse_args()
    text = render()
    status = 0
    _, clashes = collect()
    if clashes:
        print("Names bound to different objects in two homes:")
        print("\n".join(f"  {c}" for c in clashes))
        print("Rename one, or add it to ALLOWED_SHARED_NAMES with a reason.")
        status = 1
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if stale(current, text):
            print(f"{INDEX.relative_to(ROOT)} is stale; run `make capabilities`")
            status = 1
    else:
        INDEX.write_text(text, encoding="utf-8")
        print(f"wrote {INDEX.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
