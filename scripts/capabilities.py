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
    "pipekit-ops": ["pipekit_ops"],
    # the umbrella installs the seven workspace packages and holds no code
    "pipekit": [],
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
    "geotoolz-array": ["geoarray"],
    "geotoolz-catalog": ["geocatalog"],
    "geotoolz-cloud": ["geocloud"],
    "geotoolz-patcher": ["geopatcher"],
    "geotoolz-products": ["geoproducts"],
    "xrtoolz-ops": ["xrops"],
    "xrtoolz-products": ["xrproducts"],
}
UPSTREAM_BLURB: dict[str, str] = {
    "pipekit-ops": "The core: Operator, Graph and the ready-made operators.",
    "pipekit": "The pipekit workspace in one install; an umbrella that holds no code.",
    "pipekit-array": "Array API operators (NumPy, JAX, CuPy, PyTorch).",
    "pipekit-cycle": "Time-stepping, data assimilation and observation operators.",
    "pipekit-evaluate": "Multidimensional evaluation (Unit x Lens x Stage).",
    "pipekit-experiment": "Experiment tracking adapters and a model registry.",
    "pipekit-jax": "JAX / Equinox carrier integration for pipekit.",
    "pipekit-train": "Training pipelines for emulators and amortized inference.",
    "georeader-spaceml": "Raster reading and the GeoTensor carrier.",
    "geotoolz-ops": "Composable remote-sensing operators on GeoTensor.",
    "geotoolz-array": "The GeoArray carrier: an array with its grid (transform, CRS).",
    "geotoolz-catalog": "Spatiotemporal catalog over geospatial files.",
    "geotoolz-cloud": "Cloud object storage and batched async COG reads.",
    "geotoolz-patcher": "Four-axis patcher: split, run per patch, stitch.",
    "geotoolz-products": "Readers for Earth-observation data products.",
    "xrtoolz-ops": "Composable operators for Earth System Data Cubes (xrops).",
    "xrtoolz-products": "Archive readers (CMEMS, CDS, AEMET) and the CF registry.",
    # GeoNumerics
    "gaussx": "Structured linear algebra and Gaussian primitives on lineax.",
    "finitevolx": "Arakawa C-grid finite-volume operators.",
    "spectraldiffx": "Pseudospectral differentiation, filters and PDE solvers.",
    "functax": "Function approximation: bases, quadrature rules, interpolation.",
    "argminax": "Optimisation with uncertainty at the minimum: solvers, curvature.",
    "integrix": "Integration: rules, cumulative scans, time marching, expectations.",
    # GeoML
    "geonnax": "Basis functions, neural layers and coordinate transforms in Equinox.",
    "kernellib": "Covariance kernels and scalable kernel methods on gaussx.",
    "filterax": "Differentiable ensemble Kalman filters and smoothers.",
    "vardax": "Variational data assimilation: BLUE/OI, 3D-Var, 4D-Var and kin.",
    "gauss-flows": "Gaussianization (RBIG) and normalizing flows.",
    "xtremax": "Extreme-value and point-process models in NumPyro.",
    "pyrox": "Equinox modules with NumPyro sample and param sites.",
    "pyrox-gp": "Gaussian processes for pyrox.",
    "pyrox-lgm": "Latent Gaussian models for pyrox.",
    "pyrox-nn": "Bayesian neural networks for pyrox.",
    "pyrox-ssm": "State-space models for pyrox.",
    "pyrox-gauss": "Gaussian NumPyro distributions on gaussx operators.",
    "manipy-jax": "Manifold learning and alignment, and hyperspectral workflows.",
    # GeoModels
    "plumax": "Plume transport: Gaussian plume and puff, Lagrangian and LES.",
    "radtrax": "Differentiable radiative transfer for methane remote sensing.",
    "sesmax": "The sesmax workspace in one install; an umbrella that holds no code.",
    "sesmax-core": "Simple Earth-system models: state, terms, steppers, forcing.",
}

NUMERICS: dict[str, list[str]] = {
    "gaussx": ["gaussx"],
    "finitevolx": ["finitevolx"],
    "spectraldiffx": ["spectraldiffx"],
    # functax, argminax and integrix export only their errors at the top
    # level, so their public submodules are listed
    "functax": [
        "functax",
        *(
            f"functax.{sub}"
            for sub in (
                "basis",
                "diffops",
                "geometry",
                "interp",
                "project",
                "quad",
                "transform",
            )
        ),
    ],
    "argminax": [
        "argminax",
        *(f"argminax.{sub}" for sub in ("adjoint", "curvature", "diagnose", "solve")),
    ],
    "integrix": [
        "integrix",
        *(
            f"integrix.{sub}"
            for sub in ("adapt", "adjoint", "expect", "march", "rule", "scan")
        ),
    ],
}
GEOML: dict[str, list[str]] = {
    "geonnax": ["geonnax"],
    "kernellib": ["kernellib"],
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
    "pyrox-ssm": ["pyrox_ssm"],
    "pyrox-gauss": ["pyrox_gauss"],
    # manipy.sklearn needs scikit-learn (manipy's sklearn extra), which no
    # environment installs
    "manipy-jax": ["manipy", "manipy.datasets", "manipy.hsi", "manipy.metrics"],
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
    # the umbrella installs the workspace packages and holds no code
    "sesmax": [],
    "sesmax-core": [
        "sesmax_core",
        *(
            f"sesmax_core.{sub}"
            for sub in (
                "checkpoint",
                "da",
                "domain",
                "flows",
                "forcing",
                "gfd",
                "io",
                "monitor",
                "operators",
                "scales",
                "solvers",
                "terms",
                "testing",
                "train",
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
_GP_LIKELIHOOD = (
    "pyrox-gp's observation model builds the gaussx likelihood of the same name"
    " (pyrox#375); the numerics are gaussx's"
)
_GP_KERNEL = (
    "pyrox-gp's kernel is kernellib's with priors and guides on its"
    " hyperparameters, and freezes into the kernellib kernel of the same name"
)
ALLOWED_SHARED_NAMES: dict[str, str] = {
    "BernoulliLikelihood": _GP_LIKELIHOOD,
    "GaussianLikelihood": _GP_LIKELIHOOD,
    "HeteroscedasticGaussianLikelihood": _GP_LIKELIHOOD,
    "PoissonLikelihood": _GP_LIKELIHOOD,
    "SoftmaxLikelihood": _GP_LIKELIHOOD,
    "StudentTLikelihood": _GP_LIKELIHOOD,
    "Constant": _GP_KERNEL,
    "Cosine": _GP_KERNEL,
    "Matern": _GP_KERNEL,
    "Periodic": _GP_KERNEL,
    "Polynomial": _GP_KERNEL,
    "RationalQuadratic": _GP_KERNEL,
    "White": _GP_KERNEL,
    "Operator": (
        "pipekit_ops.Operator is the base (geotoolz re-exports it); xrops.Operator"
        " subclasses it with xarray DataTree dispatch for the xrops operators."
    ),
    "stack": (
        "pipekit_array.stack adds a leading dim on any named carrier and never"
        " aligns; geoarray.stack stacks GeoArrays on one grid."
    ),
    "Reduce": (
        "pipekit_ops.Reduce is a left fold of an operator over a sequence;"
        " pipekit_array.Reduce reduces a named array over named dims."
    ),
    "Rename": (
        "pipekit_ops.Rename renames the keys of a dict carrier;"
        " pipekit_array.Rename renames the dims of a named array."
    ),
    "pad": (
        "pipekit_array.pad pads named dims of any named carrier;"
        " geoarray.pad pads a GeoArray's spatial axes and moves its transform."
    ),
    "CallableObs": (
        "vardax.CallableObs subclasses pipekit_cycle.CallableObs as an Equinox"
        " module, so vardax's solvers and pipekit-cycle share one family."
    ),
    "Shift": (
        "pipekit_array.Shift shifts data along named dims; kernellib.Shift is"
        " the input warp x - offset of a kernel."
    ),
    "Squeeze": (
        "pipekit_array.Squeeze drops named length-1 dims; gauss_flows.Squeeze is"
        " a volume-preserving bijection that trades spatial extent for channels."
    ),
    "compose": (
        "pipekit_ops.compose composes operators right to left;"
        " functax.interp.compose composes two interpolation stencils into one."
    ),
    "window": (
        "pipekit_array.window makes sliding windows along a named dim;"
        " functax.geometry.window selects a window of an affine grid."
    ),
    "ForwardModel": (
        "pipekit_cycle.ForwardModel (re-exported by vardax) is the protocol for"
        " advancing a state by dt; radtrax.ForwardModel is the radiative-transfer"
        " chain from state to band values."
    ),
    "GaussianNoise": (
        "geoops.GaussianNoise adds sensor noise to imagery;"
        " plumax.gauss_plume.GaussianNoise is a homoscedastic likelihood."
    ),
    "Instrument": (
        "radtrax.instrument.Instrument is the spectral-response protocol;"
        " plumax.coupled.Instrument is one satellite's geometry and error budget."
    ),
    "GridError": (
        "radtrax and vardax each raise their own GridError, as with"
        " ValidationError, so a caller can catch one library's failures."
    ),
    "ShapeError": (
        "radtrax and vardax each raise their own ShapeError, as with"
        " ValidationError, so a caller can catch one library's failures."
    ),
    "Composite": (
        "geoops.Composite builds a band composite; pipekit_train.Composite is"
        " a weighted sum of losses."
    ),
    "register": (
        "pipekit_ops.register marks a class as loadable by loads_sandboxed;"
        " xrproducts.register adds a Variable to the CF variable registry."
    ),
    "Estimator": (
        "pipekit_array.Estimator is the scikit-learn operator on a named"
        " carrier; pipekit_evaluate.Estimator is the benchmark protocol for a"
        " rung that maps observations to an Estimate."
    ),
    "Stage": (
        "geonnax.Stage is one encoder/decoder stage of a U-Net;"
        " pipekit_evaluate.Stage is the enum of where in a model's life a score"
        " may run."
    ),
    "Divergence2D": (
        "geoops.Divergence2D is a pipekit Operator on a GeoArray's collocated"
        " map axes; finitevolx.Divergence2D is the finite-volume divergence on"
        " an Arakawa C-grid. GeoStack may not import GeoNumerics."
    ),
    "ValidationError": (
        "functax, argminax and integrix each raise their own ValidationError,"
        " a subclass of the library's base error, so a caller can catch one"
        " library's failures without the others'."
    ),
    "MissingExtraError": (
        "functax and argminax each raise their own MissingExtraError, naming"
        " the extra of that library to install."
    ),
    "GaussianRegressionHead": (
        "geonnax.GaussianRegressionHead is the mean plus log-scale head;"
        " pyrox_nn.GaussianRegressionHead wraps it and samples its four weight"
        " arrays under Gaussian priors."
    ),
    "SphericalHarmonicFeatures": (
        "kernellib.SphericalHarmonicFeatures is the deterministic feature map"
        " of a zonal kernel truncated at degree L;"
        " pyrox_nn.SphericalHarmonicFeatures is the weight-space GP on the same"
        " harmonics and Funk-Hecke coefficients, with a standard normal prior"
        " on the weights."
    ),
}

# Name clashes the layers above GeoStack ship today, as {layer: {name: why}}.
# Unlike ALLOWED_SHARED_NAMES these are not endorsed: most are duplicate
# implementations. The check fails on a clash missing from this list and on an
# entry that no longer clashes, so the list can only shrink.
_PORTED = "ported from plumax to radtrax; leaves plumax with plumax#133"
# the pins of finitevolX and spectraldiffx predate the moves; their main already
# re-exports integrix's steppers and functax's transforms, so these go when the
# pins move
_FX_KERNELS = "functax.interp's RBF kernels beside kernellib's covariance kernels"
_FX_DIFFOPS = "functax.diffops on fields beside kernellib's operators on kernels"
_PIPEKIT_JAX = (
    "moved from pipekit-jax to integrix.adjoint (pipekit#226); the pipekit pin"
    " predates it"
)
_TOY_MODELS = "toy models in both sesmax_core.testing and vardax"
_STEPPERS = "moved from finitevolX to integrix.march (finitevolX#407)"
_TRANSFORMS = "moved from spectraldiffx to functax.transform (spectraldiffx#166)"
_NN_LAYERS = "layers in both geonnax and pyrox-nn"
_SAME_NAME = "different objects under one name in several GeoML and GeoStack libraries"
_NUMERICS = "the same solver or grid in several GeoNumerics libraries"
_VARDAX = "vardax's own copy of a pipekit-cycle name"
_PLUME_PUFF = "the same name in plumax's plume and puff models"
_RADTRAX_MF = (
    "the matched-filter operators moved from geoops to radtrax.geo; geoops keeps"
    " its copy until geotoolz#482 removes it"
)
_PLUMAX_LOGQUAD = (
    "deprecated plumax.population aliases forwarding to integrix.expect"
    " (plumax#253), until plumax#283 removes them"
)
_DIVERGENCES = "Gaussian divergences in both gauss_flows and gaussx"

KNOWN_CLASHES: dict[str, dict[str, str]] = {
    "numerics": {
        "AB2Solver": _STEPPERS,
        "CapacitanceSolver": _NUMERICS,
        "ChebyshevTransform1D": _TRANSFORMS,
        "ChebyshevTransform2D": _TRANSFORMS,
        "LeapfrogRAFSolver": _STEPPERS,
        "SphericalGrid2D": _NUMERICS,
        "SphericalHarmonicTransform": _TRANSFORMS,
        "SplitExplicitRKSolver": _STEPPERS,
        "build_capacitance_solver": _NUMERICS,
        "laplace_mode": (
            "the loop moved to argminax.solve; gaussx deprecates its own copy"
            " without forwarding and removes it a release later (gaussx#814)"
        ),
    },
    "geoml": {
        "Augment": "different objects in gauss_flows and xrtoolz",
        "Axis": "functax.geometry.Axis beside pipekit-evaluate's evaluation Axis",
        "Cubic": _FX_KERNELS,
        "DenseRank1": _NN_LAYERS,
        "Graph": "kernellib's Graph beside pipekit's operator Graph",
        "Identity": "functax.interp.Identity beside geoops' Identity operator",
        "ImplicitAdjoint": _VARDAX,
        "Kernel": "functax.interp's RBF kernel protocol beside pyrox-gp's Kernel",
        "Kronecker": "Kronecker structures in both gaussx and pyrox-lgm",
        "LayerNormEnsemble": _NN_LAYERS,
        "Linear": _FX_KERNELS,
        "LinearBasis": "functax.interp.LinearBasis beside vardax's LinearBasis",
        "LinearObs": _VARDAX,
        "MCSigmoidDenseFA": _NN_LAYERS,
        "MCSoftmaxDenseFA": _NN_LAYERS,
        "MultiHeadAttentionBE": _NN_LAYERS,
        "OrthogonalRandomFeatures": "random features in geonnax, kernellib, pyrox-nn",
        "Prior": "different objects in pipekit-evaluate and vardax",
        "RBF": _FX_KERNELS,
        "RandomFeatureGaussianProcess": _NN_LAYERS,
        "RecursiveCheckpointAdjoint": _VARDAX,
        "ReducedBasis": _VARDAX,
        "SphericalHarmonicTransform": "spherical harmonics, geonnax and spectraldiffx",
        "Wendland": _FX_KERNELS,
        "divergence": _FX_DIFFOPS,
        "funk_hecke_coefficients": (
            "pyrox-gp's deprecated copy forwards to kernellib's; it goes when pyrox"
            " removes it"
        ),
        "gaussian_entropy": _DIVERGENCES,
        "gradient": _FX_DIFFOPS,
        "kl_divergence": _DIVERGENCES,
        "laplacian": _FX_DIFFOPS,
        "to_diffrax_adjoint": _PIPEKIT_JAX,
        "to_optimistix_adjoint": (
            "moved from vardax to argminax.adjoint (vardax#278); the vardax pin"
            " predates it"
        ),
        "truncated_scan": _PIPEKIT_JAX,
    },
    "geomodels": {
        "AdaptiveWindowBackground": _RADTRAX_MF,
        "ApplyAdaptiveMF": _RADTRAX_MF,
        "ApplyClusterMF": _RADTRAX_MF,
        "ColumnEnhancement": _RADTRAX_MF,
        "DetectionThreshold": _RADTRAX_MF,
        "EstimateCovEmpirical": _RADTRAX_MF,
        "EstimateCovLowRank": _RADTRAX_MF,
        "EstimateCovShrunk": _RADTRAX_MF,
        "EstimateMean": _RADTRAX_MF,
        "GMMClusterBackground": _RADTRAX_MF,
        "LinearTargetFromObs": _RADTRAX_MF,
        "MatchedFilter": _RADTRAX_MF,
        "MatchedFilterPixel": _RADTRAX_MF,
        "MatchedFilterSNR": _RADTRAX_MF,
        "NonlinearTargetFromObs": _RADTRAX_MF,
        "SBMP": _RADTRAX_MF,
        "StreamingBackground": _RADTRAX_MF,
        "ValidateMFInputs": _RADTRAX_MF,
        "effective_sample_size": _PLUMAX_LOGQUAD,
        "log_gauss_hermite": _PLUMAX_LOGQUAD,
        "log_simpson": _PLUMAX_LOGQUAD,
        "pareto_smoothed_log_weights": _PLUMAX_LOGQUAD,
        "Diagnostics": _SAME_NAME,
        "Diffusion2D": (
            "sesmax_core.testing's toy model beside finitevolX's diffusion operator"
        ),
        "Domain": _SAME_NAME,
        "InstrumentSpec": _PORTED,
        "Lorenz63": _TOY_MODELS,
        "Lorenz96": _TOY_MODELS,
        "ModelOp": "geoops' ModelOp beside sesmax_core.operators' ModelOp",
        "PosteriorCovariance": (
            "functax.interp's kriging covariance beside plumax.les_fvm's"
        ),
        "Scaled": _SAME_NAME,
        "Sum": _SAME_NAME,
        "infer_emission_rate": _PLUME_PUFF,
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


def _repr(obj: object) -> str:
    """``repr(obj)``, with a set's members sorted.

    A set's order follows string hashing, which Python salts per process,
    so a plain ``repr`` would change the index from one run to the next.
    """
    if not isinstance(obj, set | frozenset) or not obj:
        return repr(obj)
    members = "{" + ", ".join(sorted(repr(m) for m in obj)) + "}"
    return members if type(obj) is set else f"{type(obj).__name__}({members})"


def _summary(obj: object, *, home: str) -> str:
    origin = _origin(obj).split(".")[0]
    if origin != home and (callable(obj) or inspect.isclass(obj)):
        return f"Re-exported from `{origin}`."
    if not callable(obj):
        value = _repr(obj)
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
