"""Generate, or check, the capability index: every public name we build on.

``docs/api/capabilities.md`` lists each public object of the project package
(when there is one) and of every ``UPSTREAM`` library once, under the shortest
module path that exports it, with the first sentence of its docstring. Agents
and people search it before writing a loader, a model, a training step or a
helper in a script or a notebook ("Reuse before you write" in ``AGENTS.md``).

The libraries sit in four layers (``LAYERS``), one submodule folder and one
index page each: GeoStack in ``docs/api/capabilities.md`` (the hub page),
GeoNumerics, GeoML and GeoModels in ``docs/api/capabilities_<layer>.md``. Each
layer is installed by its own pixi environment, so each index is generated
there.

Usage::

    make capabilities                                   # every layer
    pixi run -e geoml python scripts/capabilities.py --layer geoml
    python scripts/capabilities.py --layer geostack --check   # fail if stale

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

# The reference libraries geostack builds on (git submodules in
# references/<layer>/, installed from there by pixi), as
# {distribution: [public modules]}, with a one-line blurb each. UPSTREAM is the
# GeoStack layer; NUMERICS, GEOML and GEOMODELS are the layers above it.
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
    "geotoolz-ops": ["geoops"],
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
    "geotoolz-ops": "Composable remote-sensing operators on GeoTensor.",
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
    # GeoNumerics
    "gaussx": "Structured linear algebra and Gaussian primitives on lineax.",
    "finitevolx": "Arakawa C-grid finite-volume operators.",
    "spectraldiffx": "Pseudospectral differentiation, filters and PDE solvers.",
    # GeoML
    "geonnax": "Basis functions, neural layers and coordinate transforms in Equinox.",
    "kernellib": "Covariance kernels and scalable kernel methods on gaussx.",
    "optax-bayes": "The Bayesian learning rule as optax gradient transformations.",
    "filterax": "Differentiable ensemble Kalman filters and smoothers.",
    "vardax": "Variational data assimilation: BLUE/OI, 3D-Var, 4D-Var and kin.",
    "gauss-flows": "Gaussianization (RBIG) and normalizing flows.",
    "xtremax": "Extreme-value and point-process models in NumPyro.",
    "pyrox": "Equinox modules with NumPyro sample and param sites.",
    "pyrox-gp": "Gaussian processes for pyrox.",
    "pyrox-lgm": "Latent Gaussian models for pyrox.",
    "pyrox-nn": "Bayesian neural networks for pyrox.",
    # GeoModels
    "plumax": "Plume transport: Gaussian plume and puff, Lagrangian and LES.",
    "somax": "Simple ocean and atmosphere models in JAX.",
    "radtrax": "Differentiable radiative transfer for methane remote sensing.",
}

NUMERICS: dict[str, list[str]] = {
    "gaussx": ["gaussx"],
    "finitevolx": ["finitevolx"],
    "spectraldiffx": ["spectraldiffx"],
}
GEOML: dict[str, list[str]] = {
    "geonnax": ["geonnax"],
    "kernellib": ["kernellib"],
    "optax-bayes": ["optax_bayes"],
    # filterax, xtremax and pyrox export little at the top level, so their
    # public subpackages are listed (as georeader's modules are above)
    "filterax": [
        "filterax",
        "filterax.filters",
        "filterax.smoothers",
        "filterax.processes",
        "filterax.differentiable",
        "filterax.optax",
        "filterax.pipekit",
    ],
    "vardax": ["vardax"],
    "gauss-flows": ["gauss_flows"],
    "xtremax": [
        "xtremax",
        "xtremax.distributions",
        "xtremax.extraction",
        "xtremax.point_processes",
        "xtremax.primitives",
        "xtremax.simulations",
    ],
    "pyrox": ["pyrox", "pyrox.inference"],
    "pyrox-gp": ["pyrox_gp"],
    "pyrox-lgm": ["pyrox_lgm"],
    "pyrox-nn": ["pyrox_nn"],
}
# The GeoModels export their APIs from subpackages, which are listed one by one.
GEOMODELS: dict[str, list[str]] = {
    "plumax": [
        "plumax",
        *(
            f"plumax.{sub}"
            for sub in (
                "gauss_plume",
                "gauss_puff",
                "lagrangian",
                "les_fvm",
                "met",
                "population",
                "coupled",
                "operators",
                "adapters",
                "radtran",
                "hapi_lut",
                "assimilation",
                "matched_filter",
            )
        ),
    ],
    "somax": [
        "somax",
        *(
            f"somax.{sub}"
            for sub in (
                "core",
                "domain",
                "models",
                "operators",
                "solvers",
                "da",
                "flows",
                "eval",
                "io",
                "monitor",
            )
        ),
    ],
    "radtrax": [
        "radtrax",
        *(
            f"radtrax.{sub}"
            for sub in (
                "atmosphere",
                "spectroscopy",
                "solvers",
                "surface",
                "instrument",
                "slab",
                "retrieval",
                "geo",
            )
        ),
    ],
}

# The four layers, bottom to top, as {layer: (title, libraries, index page,
# layers it may import)}. Edges between layers point down: GeoModels may import
# GeoML, GeoNumerics and GeoStack; GeoML may import GeoNumerics and GeoStack;
# GeoNumerics and GeoStack import neither of the others.
LAYERS: dict[str, tuple[str, dict[str, list[str]], Path, tuple[str, ...]]] = {
    "geostack": ("GeoStack", UPSTREAM, INDEX, ()),
    "numerics": (
        "GeoNumerics",
        NUMERICS,
        ROOT / "docs" / "api" / "capabilities_numerics.md",
        (),
    ),
    "geoml": (
        "GeoML",
        GEOML,
        ROOT / "docs" / "api" / "capabilities_geoml.md",
        ("numerics", "geostack"),
    ),
    "geomodels": (
        "GeoModels",
        GEOMODELS,
        ROOT / "docs" / "api" / "capabilities_geomodels.md",
        ("geoml", "numerics", "geostack"),
    ),
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
        "geoops.Composite builds a band composite; pipekit_train.Composite is"
        " a weighted sum of losses."
    ),
    "SklearnOp": (
        "One scikit-learn adapter per carrier: geoops.SklearnOp for"
        " GeoTensor, xrsklearn.SklearnOp for xarray."
    ),
    "register": (
        "pipekit.register marks a class as loadable by loads_sandboxed;"
        " xrreader.register adds a Variable to the CF variable registry."
    ),
}

# Name clashes the layers above GeoStack ship today, as {layer: {name: why}}.
# Unlike ALLOWED_SHARED_NAMES these are not endorsed: most are duplicate
# implementations. The check fails on a clash missing from this list and on an
# entry that no longer clashes, so the list can only shrink.
_PORTED = "ported from plumax to radtrax; leaves plumax with plumax#133"
_KERNELS = "kernels in both kernellib and pyrox-gp"
_LIKELIHOODS = "likelihoods in both gaussx and pyrox-gp"
_NN_LAYERS = "layers in both geonnax and pyrox-nn"
_SOMAX = "somax's own name for a different object"
_NUMERICS = "the same solver or grid in several GeoNumerics libraries"
_CALCULUS = "kernel operators in kernellib, finite differences in xrtoolz-grad"
_VARDAX = "vardax's own copy of a pipekit-cycle name"
_PLUME_PUFF = "the same name in plumax's plume and puff models"
_DIVERGENCES = "Gaussian divergences in both gauss_flows and gaussx"

KNOWN_CLASHES: dict[str, dict[str, str]] = {
    "numerics": {
        "CapacitanceSolver": _NUMERICS,
        "SphericalGrid2D": _NUMERICS,
        "build_capacitance_solver": _NUMERICS,
    },
    "geoml": {
        "Augment": "different objects in gauss_flows and xrtoolz",
        "BernoulliLikelihood": _LIKELIHOODS,
        "Constant": _KERNELS,
        "Cosine": _KERNELS,
        "DenseRank1": _NN_LAYERS,
        "GaussianLikelihood": _LIKELIHOODS,
        "Graph": "kernellib's Graph beside pipekit's operator Graph",
        "HeteroscedasticGaussianLikelihood": _LIKELIHOODS,
        "ImplicitAdjoint": _VARDAX,
        "Kronecker": "Kronecker structures in both gaussx and pyrox-lgm",
        "LayerNormEnsemble": _NN_LAYERS,
        "Linear": _KERNELS,
        "LinearObs": _VARDAX,
        "MCSigmoidDenseFA": _NN_LAYERS,
        "MCSoftmaxDenseFA": _NN_LAYERS,
        "Matern": _KERNELS,
        "MultiHeadAttentionBE": _NN_LAYERS,
        "OrthogonalRandomFeatures": "random features in geonnax, kernellib, pyrox-nn",
        "Periodic": _KERNELS,
        "PoissonLikelihood": _LIKELIHOODS,
        "Polynomial": _KERNELS,
        "Prior": "different objects in pipekit-evaluate and vardax",
        "RBF": _KERNELS,
        "RandomFeatureGaussianProcess": _NN_LAYERS,
        "RationalQuadratic": _KERNELS,
        "RecursiveCheckpointAdjoint": _VARDAX,
        "ReducedBasis": _VARDAX,
        "SoftmaxLikelihood": _LIKELIHOODS,
        "SphericalHarmonicTransform": "spherical harmonics, geonnax and spectraldiffx",
        "StudentTLikelihood": _LIKELIHOODS,
        "White": _KERNELS,
        "curl": _CALCULUS,
        "divergence": _CALCULUS,
        "funk_hecke_coefficients": _KERNELS,
        "gaussian_entropy": _DIVERGENCES,
        "gradient": _CALCULUS,
        "kl_divergence": _DIVERGENCES,
        "laplacian": _CALCULUS,
    },
    "geomodels": {
        "ATM_TO_PA": _PORTED,
        "BOLTZMANN_J_PER_K": _PORTED,
        "CM3_PER_M3": _PORTED,
        "Diagnostics": _SOMAX,
        "Domain": _SOMAX,
        "ForwardResult": _PORTED,
        "GroundSamplingDistance": _PORTED,
        "InstrumentSpec": _PORTED,
        "NBLookup": _PORTED,
        "ObservationGeometry": _PORTED,
        "PointSpreadFunction": _PORTED,
        "STABILITY_CLASSES": _PLUME_PUFF,
        "Scaled": _SOMAX,
        "SpectralResponseFunction": _PORTED,
        "Sum": _SOMAX,
        "absorption_coefficient": _PORTED,
        "air_mass_factor": _PORTED,
        "beers_law_from_lut": _PORTED,
        "build_nb_lut": _PORTED,
        "forward_maclaurin": _PORTED,
        "forward_maclaurin_normalized": _PORTED,
        "forward_nonlinear": _PORTED,
        "forward_nonlinear_normalized": _PORTED,
        "forward_taylor": _PORTED,
        "forward_taylor_normalized": _PORTED,
        "infer_emission_rate": _PLUME_PUFF,
        "inject_plume": _PORTED,
        "interpolate_cross_section": _PORTED,
        "kinetic_energy": _SOMAX,
        "lookup_nb": _PORTED,
        "matched_filter_snr": "plumax's two matched filters; plumax#131 retires one",
        "number_density": _PORTED,
        "number_density_cm3": _PORTED,
        "plume_ratio_spectrum": _PORTED,
        "total_enstrophy": _SOMAX,
        "transmittance": _PORTED,
    },
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


def _libraries(layer: str) -> dict[str, list[str]]:
    """The layer's libraries and those of every layer it may import."""
    _, own, _, below = LAYERS[layer]
    out = dict(own)
    for lower in below:
        out.update(LAYERS[lower][1])
    return out


def collect(layer: str = "geostack") -> tuple[dict[str, list[tuple]], list[str]]:
    """``{module: [(name, kind, summary)]}`` and the list of name clashes.

    Each object is listed once, at the first module (in sorted order, so the
    shortest public path) that exports it. Clashes are checked across the
    layer and the layers it may import.
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
    for upstream_modules in _libraries(layer).values():
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


def known_clashes(layer: str) -> dict[str, str]:
    """``KNOWN_CLASHES`` for the layer and every layer it may import."""
    _, _, _, below = LAYERS[layer]
    out: dict[str, str] = {}
    for name in (*below, layer):
        out.update(KNOWN_CLASHES.get(name, {}))
    return out


def check_clashes(layer: str) -> tuple[list[str], list[str]]:
    """Clashes missing from ``KNOWN_CLASHES``, and known ones that are gone."""
    _, clashes = collect(layer)
    known = known_clashes(layer)
    names = {c.split(":", 1)[0] for c in clashes}
    new = [c for c in clashes if c.split(":", 1)[0] not in known]
    gone = sorted(set(known) - names)
    return new, gone


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


def _upstream_part(libraries: dict[str, list[str]]) -> list[str]:
    """Each upstream object once, at the first module (in ``libraries`` order)."""
    out: list[str] = []
    seen: set[int] = set()
    for package, upstream_modules in libraries.items():
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


def _versions(libraries: dict[str, list[str]]) -> str:
    return ", ".join(f"{p} {importlib.metadata.version(p)}" for p in libraries)


def _layer_links(layer: str) -> list[str]:
    """One line linking every layer's index page, this one in bold."""
    parts = []
    for name, (title, _, page, _) in LAYERS.items():
        parts.append(f"**{title}**" if name == layer else f"[{title}]({page.name})")
    return ["Layers: " + " · ".join(parts) + ".", ""]


def render(layer: str = "geostack") -> str:
    title, libraries, _, below = LAYERS[layer]
    package_part, total = _package_part() if layer == "geostack" else ([], 0)
    imports = (
        "It may import " + ", ".join(LAYERS[b][0] for b in below) + "."
        if below
        else "It imports no other layer."
    )
    out = [
        f"# Capability index: {title}",
        "",
        "<!-- Generated by scripts/capabilities.py — do not edit by hand. -->",
        "",
        *_layer_links(layer),
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
        f"This page is the {title} layer. {imports}",
        "",
        *package_part,
    ]
    if libraries:
        out += [
            UPSTREAM_MARKER,
            "",
            f"Listed at {_versions(libraries)}.",
            "",
            *_upstream_part(libraries),
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
    parser.add_argument(
        "--layer", choices=list(LAYERS), default="geostack", help="which index"
    )
    args = parser.parse_args()
    index = LAYERS[args.layer][2]
    text = render(args.layer)
    status = 0
    clashes, gone = check_clashes(args.layer)
    if clashes:
        print("Names bound to different objects in two homes:")
        print("\n".join(f"  {c}" for c in clashes))
        print("Rename one, or add it to ALLOWED_SHARED_NAMES with a reason.")
        status = 1
    if gone:
        print("No longer clash; remove them from KNOWN_CLASHES:", ", ".join(gone))
        status = 1
    if args.check:
        current = index.read_text(encoding="utf-8") if index.exists() else ""
        if stale(current, text):
            print(f"{index.relative_to(ROOT)} is stale; run `make capabilities`")
            status = 1
    else:
        index.write_text(text, encoding="utf-8")
        print(f"wrote {index.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
