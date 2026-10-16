"""The dependency graph between the stack's repositories obeys its rules.

``scripts/dependency_graph.py`` reads every submodule's ``pyproject.toml`` and
imports and checks them against the rules on ``docs/api/dependency_graph.md``.
These tests fail on a broken rule not in ``KNOWN``, on a ``KNOWN`` entry that
is fixed (so the list only shrinks), and on a stale diagram in the page or the
README. It reads files only, so it runs in every environment.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import replace
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


def _with_dependency(source: str, target: str) -> dg.Graph:
    """The graph with one more base dependency, source → target."""
    pkg = GRAPH.pkgs[source]
    pkgs = {**GRAPH.pkgs, source: replace(pkg, base=pkg.base | {target})}
    return dg.graph(pkgs, GRAPH.imports)


def _with_import(
    source: str, target: str, where: str, *, eager: bool = False
) -> dg.Graph:
    """The graph with one more module-level import, in a new file."""
    imp = {p: dict(found) for p, found in GRAPH.imports.items()}
    new = dg.Import(target, where, "module", eager)
    imp[source][target] = [*imp[source].get(target, []), new]
    return dg.graph(GRAPH.pkgs, imp)


def test_a_reverse_layer_edge_is_caught() -> None:
    broken = _with_dependency("gaussx", "plumax")
    assert ("upward", "gaussx", "plumax") in dg.violations(broken)


def test_the_foundation_layers_do_not_import_each_other() -> None:
    assert ("upward", "gaussx", "pipekit") in dg.violations(
        _with_dependency("gaussx", "pipekit")
    )
    assert ("upward", "pipekit", "gaussx") in dg.violations(
        _with_dependency("pipekit", "gaussx")
    )
    _, source = dg.guard("gaussx", GRAPH)
    assert '"pipekit": "pipekit"' in source.split("ALLOWED")[0]  # forbidden


def test_a_second_package_edge_between_known_repositories_is_new() -> None:
    new, _ = dg.check(_with_dependency("pipekit-cycle", "geotoolz-cloud"))
    assert ("tier", "pipekit-cycle", "geotoolz-cloud") in new


def test_a_second_undeclared_import_is_new() -> None:
    new, _ = dg.check(_with_import("somax", "filterax", "somax/_src/new.py"))
    assert set(new) == {("undeclared", "somax:somax/_src/new.py", "filterax")}


def test_an_extra_does_not_declare_what_import_runs() -> None:
    # gaussx is an extra of optax-bayes: enough for an optional module only
    where = "src/optax_bayes/new.py"
    key = ("undeclared", f"optax-bayes:{where}", "gaussx")
    assert key not in dg.violations(_with_import("optax-bayes", "gaussx", where))
    eager = _with_import("optax-bayes", "gaussx", where, eager=True)
    assert key in dg.violations(eager)


def test_eager_modules_follow_module_level_imports(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("from . import a\nfrom .sub import b\n")
    (pkg / "a.py").write_text("def f():\n    from . import lazy\n")
    (pkg / "sub" / "__init__.py").write_text("")
    (pkg / "sub" / "b.py").write_text("from ..c import x\n")
    for name in ("lazy", "c", "optional"):
        (pkg / f"{name}.py").write_text("")
    parsed = {}
    for f in dg._files(pkg):
        module, package = dg._dotted(f, pkg)
        parsed[module] = dg._imports(ast.parse(f.read_text()), package)
    assert dg._eager(parsed, [pkg]) == {"pkg", "pkg.a", "pkg.sub", "pkg.sub.b", "pkg.c"}


def test_poetry_optional_dependencies_are_extras() -> None:
    project = {
        "dependencies": {
            "python": ">=3.11",
            "numpy": ">=1",
            "Pydantic": {"version": ">=2", "optional": True},
        },
        "extras": {"web": ["requests"]},
    }
    assert dg._declared(project) == ({"python", "numpy"}, {"pydantic", "requests"})


def test_a_stale_tier_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(dg.TIERS["geoml"], "gone", 0)
    assert ("stale-tier", "gone", "geoml") in dg.violations(GRAPH)


def test_every_edge_of_a_longer_cycle_is_red() -> None:
    # the counterfactual re-export closes geotoolz → plumax → pipekit → geotoolz
    g = _with_dependency("geotoolz", "plumax")
    lines = dg.mermaid(g, "```mermaid").splitlines()
    drawn = [line.split() for line in lines if "-->" in line or "-.->" in line]
    loop = {"geotoolz", "pipekit", "plumax"}
    on_loop = {str(i) for i, (a, _, b) in enumerate(drawn) if {a, b} <= loop}
    style = next(line for line in lines if "linkStyle" in line)
    assert set(style.split()[1].split(",")) == on_loop
    assert len(on_loop) >= 3


IMPORTS = """
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import a
else:
    import b
if not TYPE_CHECKING:
    import c
else:
    import d
try:
    import e
except ImportError:
    import f
else:
    import g
finally:
    import h
def lazy():
    import i
"""


def test_imports_are_classified_by_branch() -> None:
    found = dict(dg._imports(ast.parse(IMPORTS)))
    del found["typing.TYPE_CHECKING"]  # from m import a gives m and m.a
    assert found == {
        "typing": "module",
        "a": "typing",
        "b": "module",
        "c": "module",
        "d": "typing",
        "e": "guarded",
        "f": "guarded",
        "g": "guarded",
        "h": "module",
        "i": "lazy",
    }


def test_namespace_and_single_file_roots_are_found(tmp_path: Path) -> None:
    (tmp_path / "src" / "ns" / "inner").mkdir(parents=True)
    (tmp_path / "src" / "ns" / "inner" / "mod.py").write_text("")
    (tmp_path / "src" / "single.py").write_text("")
    (tmp_path / "src" / "tests").mkdir()
    (tmp_path / "src" / "tests" / "test_x.py").write_text("")
    roots = dg._roots({}, tmp_path)
    assert [dg._module_name(r) for r in roots] == ["ns", "single"]
    hatch = {
        "tool": {"hatch": {"build": {"targets": {"wheel": {"packages": ["src/ns"]}}}}}
    }
    assert dg._roots(hatch, tmp_path) == [tmp_path / "src" / "ns"]


@pytest.mark.parametrize("repo", sorted(GRAPH.repos))
def test_every_guard_renders(repo: str) -> None:
    path, source = dg.guard(repo, GRAPH)
    assert path.endswith("test_dependency_rules.py")
    # run its freshness check as if it sat at its path in the submodule: it
    # rediscovers the packages and their roots the way this script does
    where = dg.REFS / GRAPH.repos[repo] / repo / path
    namespace = {"__file__": str(where), "__name__": "guard"}
    exec(compile(source, str(where), "exec"), namespace)
    namespace["test_packages_are_current"]()


def test_a_guard_catches_a_stale_import_root() -> None:
    path, source = dg.guard("pipekit", GRAPH)
    where = dg.REFS / "geostack" / "pipekit" / path
    namespace = {"__file__": str(where), "__name__": "guard"}
    exec(compile(source, str(where), "exec"), namespace)
    pyproject, roots = namespace["PACKAGES"]["pipekit-train"]
    namespace["PACKAGES"]["pipekit-train"] = (pyproject, roots[:-1])
    with pytest.raises(AssertionError, match="regenerate"):
        namespace["test_packages_are_current"]()
