"""The dependency graph between the stack's repositories obeys its rules.

``scripts/dependency_graph.py`` reads every submodule's ``pyproject.toml`` and
imports and checks them against the rules on ``docs/api/dependency_graph.md``.
These tests fail on a broken rule not in ``KNOWN``, on a ``KNOWN`` entry that
is fixed (so the list only shrinks), and on a stale diagram in the page or the
README. It reads files only, so it runs in every environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dependency_graph.py"
if not (ROOT / "references" / "geostack" / "pipekit" / "pyproject.toml").exists():
    pytest.skip("the submodules are not checked out", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("dependency_graph", SCRIPT)
assert _spec is not None and _spec.loader is not None
dg = importlib.util.module_from_spec(_spec)
sys.modules["dependency_graph"] = dg  # dataclasses look their module up
_spec.loader.exec_module(dg)

GRAPH = dg.build()


def test_no_new_violation_and_none_fixed() -> None:
    new, gone = dg.check(GRAPH)
    assert not new, f"the graph breaks a rule (docs/api/dependency_graph.md): {new}"
    assert not gone, f"{gone} are fixed: remove them from KNOWN in {SCRIPT.name}"


def test_every_repository_has_a_tier() -> None:
    for repo, layer in GRAPH.repos.items():
        assert repo in dg.TIERS[layer], f"{repo}: add it to TIERS ({layer})"


@pytest.mark.parametrize("page", dg.PAGES, ids=lambda p: p.name)
def test_page_is_current(page: Path) -> None:
    assert dg.render(page, GRAPH) == page.read_text(encoding="utf-8"), (
        f"{page.relative_to(ROOT)} is stale: run `make dependency-graph`"
    )


def test_the_cycle_check_is_live() -> None:
    # a re-export of plumax from geotoolz (Package boundaries' counterfactual)
    # must close a cycle through plumax → pipekit → geotoolz
    edges = [(e.source, e.target) for e in GRAPH.edges] + [("geotoolz", "plumax")]
    assert ["geotoolz", "pipekit", "plumax"] in dg.sccs(list(GRAPH.repos), edges)


def test_a_reverse_layer_edge_is_caught() -> None:
    upward = dg.Edge("gaussx", "plumax", "import", "gaussx → plumax")
    broken = dg.Graph(GRAPH.pkgs, GRAPH.imports, [*GRAPH.edges, upward])
    assert ("upward", "gaussx", "plumax") in dg.violations(broken)


@pytest.mark.parametrize("repo", sorted(GRAPH.repos))
def test_every_guard_renders(repo: str) -> None:
    path, source = dg.guard(repo, GRAPH)
    compile(source, path, "exec")
    assert path.endswith("test_dependency_rules.py")
