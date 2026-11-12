"""The shelf pages' tables match what the submodules give.

``scripts/skill_shelf.py`` counts a library's public names by unit and the
names each dependent package takes from it, at the submodule pins, and writes
the tables of ``docs/design/skill_tree_<library>.md``. These tests fail on a
stale table and on a public name that belongs to no unit. It reads files
only, so it runs in every environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "skill_shelf.py"
if not (ROOT / "references" / "geostack" / "pipekit" / "pyproject.toml").exists():
    pytest.skip("the submodules are not checked out", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("skill_shelf", SCRIPT)
assert _spec is not None and _spec.loader is not None
ss = importlib.util.module_from_spec(_spec)
sys.modules["skill_shelf"] = ss  # dataclasses look their module up
_spec.loader.exec_module(ss)


@pytest.fixture(scope="module")
def pkgs():
    return ss.dg.packages()


def test_every_shelf_page_is_current(pkgs) -> None:
    assert ss.stale(pkgs) == [], "run python scripts/skill_shelf.py"


@pytest.mark.parametrize("library", sorted(ss.SHELVES))
def test_every_public_name_belongs_to_a_unit(library: str, pkgs) -> None:
    shelf = ss.SHELVES[library]
    measured = ss.measure(shelf, pkgs)
    assert measured.names, f"{library} exports no public name"
    assert ss.unassigned(shelf, measured) == []


@pytest.mark.parametrize("library", sorted(ss.SHELVES))
def test_every_unit_holds_a_name(library: str, pkgs) -> None:
    shelf = ss.SHELVES[library]
    measured = ss.measure(shelf, pkgs)
    held = {ss.unit_of(shelf, origin) for origin in measured.names.values()}
    assert held == {unit.name for unit in shelf.units}


def test_uses_counts_from_imports_and_aliased_attributes() -> None:
    source = (
        "import lib as lb\n"
        "import other\n"
        "from lib import solve, missing\n"
        "from lib.sub import logdet\n"
        "y = lb.Kronecker(a, b)\n"
        "w = lb.sub.trace(a)\n"
        "z = other.cholesky(a)\n"
    )
    names = {"solve", "logdet", "Kronecker", "cholesky", "trace"}
    found = ss.uses(source, {"lib"}, names)
    assert found == {"solve", "logdet", "Kronecker", "trace"}


def test_uses_ignores_a_file_that_does_not_parse() -> None:
    assert ss.uses("def broken(:\n", {"lib"}, {"solve"}) == set()


def test_unit_of_prefers_the_longest_pattern() -> None:
    shelf = ss.Shelf(
        page="",
        packages=("lib",),
        units=(
            ss.Unit("whole", "⓪", "numerics", ("lib",)),
            ss.Unit("part", "①", "numerics", ("lib.sub",)),
        ),
    )
    assert ss.unit_of(shelf, "lib.sub") == "part"
    assert ss.unit_of(shelf, "lib.other") == "whole"
    assert ss.unit_of(shelf, "elsewhere.sub") is None


def test_render_replaces_only_the_marked_block() -> None:
    shelf = ss.SHELVES["gaussx"]
    measured = ss.Measured(names={"solve": "gaussx._primitives"}, used={})
    text = "before\n<!-- shelf:units:begin -->\nold\n<!-- shelf:units:end -->\nafter\n"
    out = ss.render(text, shelf, measured)
    assert out.startswith("before\n<!-- shelf:units:begin -->\n| Unit |")
    assert out.endswith("<!-- shelf:units:end -->\nafter\n")
    assert "old" not in out


def test_exports_reads_the_root_and_its_public_submodules(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    (root / "_src").mkdir(parents=True)
    (root / "__init__.py").write_text(
        "from lib._src.solvers import solve\n__all__ = ['solve']\n"
    )
    (root / "curvature.py").write_text("__all__ = ['laplace', 'solve']\n")
    (root / "_private.py").write_text("__all__ = ['hidden']\n")
    pkg = ss.dg.Package("lib", "lib", "numerics", tmp_path, modules=[root])
    assert ss.exports(pkg) == {"solve": "lib.solvers", "laplace": "lib.curvature"}
