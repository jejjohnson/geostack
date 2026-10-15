"""The capability index (docs/api/capabilities.md) is current.

Agents and contributors search the index before writing a loader, a model or
a helper in a script or a notebook ("Reuse before you write" in AGENTS.md),
so a stale index sends them to re-implement something that exists.
``scripts/capabilities.py`` regenerates it; this runs its check. It imports
every public module of the package, so it skips (rather than fails) when an
environment lacks a third-party dependency one of them imports.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capabilities.py"
if not SCRIPT.is_file() or not (ROOT / "docs" / "api" / "capabilities.md").is_file():
    pytest.skip("scripts/capabilities.py is not present", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("capabilities", SCRIPT)
assert _spec is not None and _spec.loader is not None
capabilities = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capabilities)


def _absent(layer: str) -> list[str]:
    """The distributions of ``layer`` that are not installed here."""
    absent = []
    for dist in capabilities.LAYERS[layer][1]:
        try:
            importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            absent.append(dist)
    return absent


def _skip_reason(layer: str) -> str | None:
    """Why this layer's index cannot be checked here, or None.

    Each index is checked only in its own environment, the one whose top
    layer it is: a lower layer's render depends on what else is installed
    (diffrax in the geomodels environment adds a line to the GeoStack index).
    """
    env = "default" if layer == "geostack" else layer
    if absent := _absent(layer):
        return f"{layer}: not installed ({', '.join(absent)}); run in `pixi -e {env}`"
    above = [up for up in LAYERS[LAYERS.index(layer) + 1 :] if not _absent(up)]
    if above:
        return f"{layer}: checked in `pixi -e {env}`, not under {above[-1]}"
    try:
        capabilities.collect(layer)
    except ModuleNotFoundError as err:
        if (err.name or "").split(".")[0] == capabilities.PACKAGE:
            raise
        return f"{layer}: a public module imports {err.name!r}, which is not installed"
    return None


LAYERS = list(capabilities.LAYERS)


@pytest.mark.parametrize("layer", LAYERS)
def test_capability_index_is_current(layer: str) -> None:
    if reason := _skip_reason(layer):
        pytest.skip(reason)
    page = capabilities.LAYERS[layer][2]
    current = page.read_text(encoding="utf-8") if page.exists() else ""
    assert not capabilities.stale(current, capabilities.render(layer)), (
        f"docs/api/{page.name} is stale: run `make capabilities` and commit it."
    )


@pytest.mark.parametrize("layer", LAYERS)
def test_no_name_is_bound_to_two_objects(layer: str) -> None:
    if reason := _skip_reason(layer):
        pytest.skip(reason)
    clashes, gone = capabilities.check_clashes(layer)
    assert not clashes, (
        "a name is exported for different objects by two modules, or shadows "
        f"an upstream name: {clashes}. Rename it, or add it to "
        "ALLOWED_SHARED_NAMES in scripts/capabilities.py with a reason."
    )
    assert not gone, (
        f"{gone} no longer clash: remove them from KNOWN_CLASHES in "
        "scripts/capabilities.py, so the list only shrinks."
    )
