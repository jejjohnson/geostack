"""The dependency graph between the stack's repositories, and its rules.

Reads every submodule under ``references/<layer>/``: each package's
``pyproject.toml`` (base dependencies and extras; not the dev, docs or
tutorial dependency groups) and the
imports in its source (module-level, lazy, ``try``-guarded and
``TYPE_CHECKING``-only). It folds packages into repositories, checks the
graph against the rules on ``docs/api/dependency_graph.md``, and writes the
diagram between the ``dependency-graph`` markers of that page and of
``README.md``.

Usage::

    python scripts/dependency_graph.py          # rewrite the diagrams
    python scripts/dependency_graph.py --check  # exit 1 if stale or broken
    python scripts/dependency_graph.py --guard REPO  # write REPO's guard test

``tests/test_dependency_graph.py`` runs the check. It needs only the standard
library and the submodules, so it runs in every environment.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import textwrap
import tomllib
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "references"
PAGES = (ROOT / "docs" / "api" / "dependency_graph.md", ROOT / "README.md")
BEGIN, END = "<!-- dependency-graph:begin -->", "<!-- dependency-graph:end -->"

#: layers from the bottom up
LAYERS = ("geostack", "numerics", "geoml", "geomodels")
#: an edge between layers points to a lower rank; GeoStack and Numerics share
#: the bottom rank, so neither may import the other
LAYER_RANK = {"geostack": 0, "numerics": 0, "geoml": 1, "geomodels": 2}
TITLES = {
    "geostack": "GeoStack",
    "numerics": "Numerics",
    "geoml": "GeoML",
    "geomodels": "GeoModels",
}
#: within a layer, a repository may import only repositories of a lower tier
TIERS: dict[str, dict[str, int]] = {
    "geostack": {"pipekit": 0, "georeader": 0, "xrtoolz": 1, "geotoolz": 1},
    "numerics": {"gaussx": 0, "spectraldiffx": 1, "finitevolX": 2},
    "geoml": {
        "geonnax": 0,
        "gauss_flows": 0,
        "kernellib": 1,
        "optax_bayes": 1,
        "xtremax": 1,
        "filterax": 2,
        "pyrox": 2,
        "vardax": 3,
    },
    "geomodels": {"radtrax": 0, "somax": 0, "plumax": 1},
}
#: what each tier holds, for the page
TIER_ROLES = {
    "geostack": ("pipelines and readers", "operators on arrays and GeoTensors"),
    "numerics": ("linear algebra and Gaussians", "spectral", "finite volume"),
    "geoml": (
        "layers and flows",
        "kernels, optimisers and extremes",
        "ensemble filters and probabilistic models",
        "variational assimilation (may build on filterax)",
    ),
    "geomodels": (
        "radiative transfer and dynamics",
        "plume models, which may build on radtrax",
    ),
}
#: violations the graph has today, with why; the check fails on a new one and
#: on one that is gone, so the list only shrinks
#: Rules between packages name the two packages, and ``undeclared`` names the
#: importing file, so a new edge or a new import is a new entry
KNOWN: dict[tuple[str, str, str], str] = {
    ("cycle", "geotoolz", "pipekit"): (
        "pipekit-train's _xreader and xarray_window import geopatcher.time (which"
        " no longer exists); Report 16 D2 moves the dataset to"
        " geopatcher.integrations.train"
    ),
    ("tier", "pipekit-train", "geotoolz-patcher"): "the same imports",
    (
        "undeclared",
        "pipekit-train:src/pipekit_train/_xreader.py",
        "geotoolz-patcher",
    ): "the same imports",
    (
        "undeclared",
        "pipekit-train:src/pipekit_train/xarray_window.py",
        "geotoolz-patcher",
    ): "the same imports",
    ("undeclared", "somax:somax/_src/da/filterax_bridge.py", "filterax"): (
        "module-level imports in somax/_src/da, declared only in the da"
        " dependency group; make it an extra"
    ),
    ("undeclared", "somax:somax/_src/da/obs.py", "filterax"): "the same extra",
}
#: imports that stay undeclared on purpose, and why: a guarded integration
#: with a library that cannot be installed beside the package. Each repository's
#: guard test carries the same list. {(package, imported distribution): reason}
UNDECLARED_OK: dict[tuple[str, str], str] = {
    ("gauss-flows", "gaussx"): (
        "gaussx is not on PyPI, and gauss_flows' [interp] extra (interpax) caps"
        " lineax below what gaussx needs; a [gaussx] extra forks the lock and"
        " moves the base install to a jax its numpyro cannot import"
    ),
}
#: repositories that are not ours: a leaf we read, never write (AGENTS.md)
THIRD_PARTY = {"georeader": "georeader-spaceml"}
SKIP_DIRS = {"tests", "test", "docs", "examples", "notebooks", "scripts", "benchmarks"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _req(spec: str) -> str:
    match = re.match(r"[A-Za-z0-9._-]+", spec.strip())
    return _norm(match.group(0)) if match else ""


@dataclass
class Package:
    name: str
    repo: str
    layer: str
    path: Path
    base: set[str] = field(default_factory=set)
    extras: set[str] = field(default_factory=set)
    modules: list[Path] = field(default_factory=list)  # import roots


def _module_name(root: Path) -> str:
    """The top-level import name of an import root: a directory or a ``.py``."""
    return root.stem if root.suffix == ".py" else root.name


def _files(root: Path) -> list[Path]:
    """The source files under an import root, shipped type stubs included."""
    if root.is_file():
        return [root]
    return sorted(f for f in root.rglob("*.py*") if f.suffix in (".py", ".pyi"))


def _roots(meta: dict, path: Path) -> list[Path]:
    """A package's import roots: from its build metadata, else from its tree.

    Hatch's ``packages`` and poetry's ``packages`` name them; without either, a
    ``src/`` layout holds packages (namespace ones too) and single-file
    modules, and a flat layout holds the directories with an ``__init__.py``.
    """
    tool = meta.get("tool", {})
    wheel = tool.get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})
    named = list(wheel.get("packages", []))
    for entry in tool.get("poetry", {}).get("packages", []):
        named.append(str(Path(entry.get("from", ".")) / entry["include"]))
    if named:
        return sorted(path / n for n in named if (path / n).exists())
    src = path / "src"
    if src.is_dir():
        return sorted(
            p
            for p in src.iterdir()
            if p.name not in SKIP_DIRS
            and _module_name(p).isidentifier()
            and ((p.is_dir() and any(p.rglob("*.py"))) or p.suffix == ".py")
        )
    return sorted(
        d
        for d in path.iterdir()
        if (d / "__init__.py").exists() and d.name not in SKIP_DIRS
    )


def declared(project: dict) -> tuple[set[str], set[str]]:
    """(base dependencies, extras) of a PEP 621 or a poetry project table."""
    deps = project.get("dependencies") or []
    if isinstance(deps, dict):  # poetry: a name -> constraint table
        optional = {
            _norm(n)
            for n, c in deps.items()
            if isinstance(c, dict) and c.get("optional")
        }
        extras = optional | {
            _req(r) for reqs in (project.get("extras") or {}).values() for r in reqs
        }
        return {_norm(n) for n in deps} - optional, extras
    extras = {
        _req(r)
        for reqs in (project.get("optional-dependencies") or {}).values()
        for r in reqs
    }
    return {_req(r) for r in deps if isinstance(r, str)}, extras


def packages() -> dict[str, Package]:
    """Every distribution in the submodules, by normalised name."""
    out: dict[str, Package] = {}
    for layer in LAYERS:
        for repo in sorted(p for p in (REFS / layer).iterdir() if p.is_dir()):
            members = sorted((repo / "packages").glob("*/pyproject.toml"))
            for pyproject in members or [repo / "pyproject.toml"]:
                meta = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                project = meta.get("project") or meta["tool"]["poetry"]
                pkg = Package(
                    _norm(project["name"]), repo.name, layer, pyproject.parent
                )
                pkg.base, pkg.extras = declared(project)
                pkg.modules = _roots(meta, pkg.path)
                out[pkg.name] = pkg
    names = set(out)
    for pkg in out.values():
        for attr in ("base", "extras"):
            setattr(pkg, attr, (getattr(pkg, attr) & names) - {pkg.name})
    return out


@dataclass(frozen=True)
class Import:
    target: str  # distribution
    where: str  # file, relative to the package
    how: str  # "module", "lazy", "guarded" or "typing"
    eager: bool = False  # the file runs on ``import <package>``


def _type_checking(test: ast.expr) -> bool:
    """Whether a test is ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _imports(tree: ast.AST, package: str = "") -> list[tuple[str, str]]:
    """(module, how) for every import in a module, as absolute dotted names.

    ``from m import a`` gives ``m`` and ``m.a`` (``a`` may be a submodule).
    ``package`` is the module's own package, which resolves relative imports;
    without it they are skipped.
    """
    out: list[tuple[str, str]] = []

    def visit(node: ast.AST, how: str) -> None:
        if isinstance(node, ast.Import):
            out.extend((a.name, how) for a in node.names)
            return
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                # level 1 is the module's own package, level 2 its parent, …
                base = package.split(".") if package else []
                keep = len(base) - node.level + 1
                if keep < 1:
                    return
                module = ".".join([*base[:keep], *([module] if module else [])])
            if module:
                out.append((module, how))
                names = (a.name for a in node.names if a.name != "*")
                out.extend((f"{module}.{n}", how) for n in names)
            return
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            how = "lazy" if how == "module" else how
        elif isinstance(node, ast.Try | ast.TryStar):
            # the body, handlers and else may not run; finally always does
            guarded = "guarded" if how == "module" else how
            for child in [*node.body, *node.handlers, *node.orelse]:
                visit(child, guarded)
            for child in node.finalbody:
                visit(child, how)
            return
        elif isinstance(node, ast.If):
            test, body, orelse = node.test, "typing", how
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                test, body, orelse = test.operand, how, "typing"
            if _type_checking(test):
                for child in node.body:
                    visit(child, body)
                for child in node.orelse:
                    visit(child, orelse)
                return
        for child in ast.iter_child_nodes(node):
            visit(child, how)

    visit(tree, "module")
    return out


RANK = {"module": 0, "guarded": 1, "lazy": 2, "typing": 3}


def _dotted(f: Path, root: Path) -> tuple[str, str]:
    """(module, its package) of a file under an import root, as dotted names."""
    parts = f.relative_to(root.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        return ".".join(parts[:-1]), ".".join(parts[:-1])
    return ".".join(parts), ".".join(parts[:-1])


def _eager(parsed: dict[str, list[tuple[str, str]]], roots: list[Path]) -> set[str]:
    """The modules ``import <package>`` runs: the roots, then every module a
    module it runs imports at module level, with the packages on the way."""
    todo = [_module_name(r) for r in roots]
    seen: set[str] = set()
    while todo:
        module = todo.pop()
        if module in seen or module not in parsed:
            continue
        seen.add(module)
        for name, how in parsed[module]:
            if how == "module":
                parts = name.split(".")
                todo.extend(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    return seen


def imports(pkgs: dict[str, Package]) -> dict[str, dict[str, list[Import]]]:
    """For each package, the other packages it imports: one ``Import`` per file
    (the strongest way that file imports it), strongest first."""
    owner = {_module_name(m): p.name for p in pkgs.values() for m in p.modules}
    out: dict[str, dict[str, list[Import]]] = {}
    for pkg in pkgs.values():
        parsed: dict[str, list[tuple[str, str]]] = {}
        wheres: dict[str, str] = {}
        for root in pkg.modules:
            for f in _files(root):
                # an upstream invalid escape warns; under warnings-as-errors that
                # would turn into a SyntaxError and silently drop the file
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tree = ast.parse(f.read_text(encoding="utf-8"))
                module, package = _dotted(f, root)
                names = _imports(tree, package)
                if f.suffix == ".pyi":
                    # a stub never runs: its imports are edges, typing-only
                    module += " (stub)"
                    names = [(name, "typing") for name, _ in names]
                parsed[module] = names
                wheres[module] = str(f.relative_to(pkg.path))
        eager = _eager(parsed, pkg.modules)
        found: dict[tuple[str, str], Import] = {}
        for module, names in parsed.items():
            where = wheres[module]
            for name, how in names:
                dist = owner.get(name.split(".")[0])
                if dist is None or dist == pkg.name:
                    continue
                old = found.get((dist, where))
                if old is None or RANK[how] < RANK[old.how]:
                    found[(dist, where)] = Import(dist, where, how, module in eager)
        by_target: dict[str, list[Import]] = defaultdict(list)
        for im in sorted(found.values(), key=lambda i: (RANK[i.how], i.where)):
            by_target[im.target].append(im)
        out[pkg.name] = dict(by_target)
    return out


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str  # "base", "extra" or "import": the strongest evidence
    via: str  # between repositories, one package-level example; between
    # packages, the importing file of an import edge


KIND_RANK = {"base": 0, "extra": 1, "import": 2}


def package_edges(
    pkgs: dict[str, Package], imp: dict[str, dict[str, list[Import]]]
) -> list[Edge]:
    """Every dependency between two packages, with its strongest evidence."""
    best: dict[tuple[str, str], Edge] = {}
    for p in sorted(pkgs.values(), key=lambda p: p.name):
        # dependency groups (dev, docs, tutorials) are not library dependencies
        evidence = [Edge(p.name, t, "base", "") for t in p.base]
        evidence += [Edge(p.name, t, "extra", "") for t in p.extras]
        evidence += [
            Edge(p.name, t, "import", f"{ims[0].where} ({ims[0].how})")
            for t, ims in imp[p.name].items()
        ]
        for e in evidence:
            old = best.get((e.source, e.target))
            if old is None or KIND_RANK[e.kind] < KIND_RANK[old.kind]:
                best[(e.source, e.target)] = e
    return sorted(best.values(), key=lambda e: (e.source, e.target))


def repo_edges(pkgs: dict[str, Package], pkg_edges: list[Edge]) -> list[Edge]:
    """The package edges folded into repositories, each with its strongest."""
    best: dict[tuple[str, str], Edge] = {}
    for e in pkg_edges:
        a, b = pkgs[e.source].repo, pkgs[e.target].repo
        if a == b:
            continue
        old = best.get((a, b))
        if old is None or KIND_RANK[e.kind] < KIND_RANK[old.kind]:
            best[(a, b)] = Edge(a, b, e.kind, f"{e.source} → {e.target}")
    return sorted(best.values(), key=lambda e: (e.source, e.target))


def sccs(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Tarjan: the strongly connected components with more than one node."""
    adj: dict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on: set[str] = set()
    out: list[list[str]] = []

    def visit(v: str) -> None:
        index[v] = low[v] = len(index)
        stack.append(v)
        on.add(v)
        for w in sorted(adj[v]):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(sorted(comp))

    for v in sorted(nodes):
        if v not in index:
            visit(v)
    return sorted(out)


@dataclass
class Graph:
    pkgs: dict[str, Package]
    imports: dict[str, dict[str, list[Import]]]
    pkg_edges: list[Edge]
    edges: list[Edge]

    @property
    def repos(self) -> dict[str, str]:
        return {p.repo: p.layer for p in self.pkgs.values()}


def graph(pkgs: dict[str, Package], imp: dict[str, dict[str, list[Import]]]) -> Graph:
    pkg_edges = package_edges(pkgs, imp)
    return Graph(pkgs, imp, pkg_edges, repo_edges(pkgs, pkg_edges))


def build() -> Graph:
    pkgs = packages()
    return graph(pkgs, imports(pkgs))


def violations(g: Graph) -> dict[tuple[str, str, str], str]:
    """Every rule the graph breaks: {(rule, a, b): evidence}.

    The rules between repositories are checked on each package edge, so a
    second package edge between the same two repositories is a new entry.
    """
    out: dict[tuple[str, str, str], str] = {}
    layer = g.repos
    pairs = [(e.source, e.target) for e in g.edges]
    for comp in sccs(list(layer), pairs):
        out[("cycle", comp[0], comp[1] if len(comp) == 2 else "+".join(comp[1:]))] = (
            " ⇄ ".join(comp)
        )
    pkg_pairs = [(e.source, e.target) for e in g.pkg_edges]
    for comp in sccs(list(g.pkgs), pkg_pairs):
        out[("package-cycle", comp[0], "+".join(comp[1:]))] = " ⇄ ".join(comp)
    for repo, lay in layer.items():
        if repo not in TIERS[lay]:
            out[("untiered", repo, lay)] = (
                "add it to TIERS in scripts/dependency_graph.py"
            )
    for lay, tiers in TIERS.items():
        for repo in tiers:
            if layer.get(repo) != lay:
                out[("stale-tier", repo, lay)] = (
                    "no such submodule in this layer; remove it from TIERS"
                )
    for e in g.pkg_edges:
        a, b = g.pkgs[e.source], g.pkgs[e.target]
        if a.repo == b.repo:
            continue
        evidence = e.kind + (f", {e.via}" if e.via else "")
        tiers = TIERS[a.layer]
        if a.layer != b.layer and LAYER_RANK[b.layer] >= LAYER_RANK[a.layer]:
            out[("upward", e.source, e.target)] = evidence
        elif a.layer == b.layer and tiers.get(b.repo, 0) >= tiers.get(a.repo, 0):
            out[("tier", e.source, e.target)] = evidence
        if a.repo in THIRD_PARTY:
            out[("third-party", e.source, e.target)] = evidence
    for p, found in g.imports.items():
        pkg = g.pkgs[p]
        for t, ims in found.items():
            if t in pkg.base or (p, t) in UNDECLARED_OK:
                continue
            for im in ims:
                if im.how == "typing":
                    continue
                # an extra declares an import that ``import <package>`` does
                # not run: an optional module, or a lazy or guarded import
                if t not in pkg.extras:
                    out[("undeclared", f"{p}:{im.where}", t)] = im.how
                elif im.how == "module" and im.eager:
                    out[("undeclared", f"{p}:{im.where}", t)] = (
                        f"module, run by `import {p}`: an extra is not enough"
                    )
    return out


def check(g: Graph) -> tuple[dict, list]:
    """(violations not in KNOWN, KNOWN entries that no longer occur)."""
    found = violations(g)
    new = {k: v for k, v in found.items() if k not in KNOWN}
    gone = sorted(k for k in KNOWN if k not in found)
    return new, gone


def _reduced(g: Graph) -> list[Edge]:
    """The edges no longer path implies: what the diagram draws.

    A solid edge (a base dependency or an import) is implied only by a path
    of solid edges, and no path may pass back through its source.
    """
    solid = {"base", "import"}

    def reaches(e: Edge) -> bool:
        kinds = solid if e.kind in solid else set(KIND_RANK)
        adj: dict[str, set[str]] = defaultdict(set)
        for f in g.edges:
            if f.kind in kinds and f != e and e.source not in (f.target,):
                adj[f.source].add(f.target)
        seen, todo = {e.source}, list(adj[e.source])
        while todo:
            v = todo.pop()
            if v == e.target:
                return True
            if v not in seen:
                seen.add(v)
                todo.extend(adj[v])
        return False

    return [e for e in g.edges if not reaches(e)]


def mermaid(g: Graph, fence: str) -> str:
    """The repository graph as a Mermaid flowchart, top layer first."""
    comps = sccs(list(g.repos), [(e.source, e.target) for e in g.edges])
    component = {repo: i for i, comp in enumerate(comps) for repo in comp}
    lines = [fence, "flowchart TB"]
    for layer in reversed(LAYERS):
        lines.append(f'  subgraph {layer}["{TITLES[layer]}"]')
        lines.append("    direction LR")
        repos = [r for r, lay in g.repos.items() if lay == layer]
        for repo in sorted(repos, key=lambda r: (-TIERS[layer].get(r, 0), r)):
            label = repo + (" (third party)" if repo in THIRD_PARTY else "")
            lines.append(f'    {repo}["{label}"]')
        lines.append("  end")
    red = []
    drawn = _reduced(g)
    for i, e in enumerate(drawn):
        arrow = "-->" if e.kind in ("base", "import") else "-.->"
        lines.append(f"  {e.source} {arrow} {e.target}")
        # an edge lies on a cycle when both ends are in one component
        if component.get(e.source, -1) == component.get(e.target, -2):
            red.append(str(i))
    if red:
        lines.append(f"  linkStyle {','.join(red)} stroke:#c62828,stroke-width:2.5px")
    lines.append("```")
    return "\n".join(lines)


def block(g: Graph, page: Path) -> str:
    # GitHub renders a mermaid code block; MyST renders the mermaid directive
    fence = "```mermaid" if page.name == "README.md" else "```{mermaid}"
    found = violations(g)
    legend = (
        f"{len(g.repos)} repositories, {len(g.pkgs)} packages and {len(g.edges)}"
        " dependencies between repositories, from each submodule's `pyproject.toml`"
        " and imports. Solid arrows are base dependencies or imports, dashed ones"
        " extras; an arrow a longer path implies is not drawn. Dev, docs and"
        " tutorial dependency groups are not library dependencies and are left out."
    )
    if any(rule == "cycle" for rule, _, _ in found):
        legend += " Red arrows form a cycle."
    return "\n".join(
        [
            BEGIN,
            "<!-- Generated by scripts/dependency_graph.py; do not edit by hand. -->",
            "",
            mermaid(g, fence),
            "",
            legend,
            "",
            END,
        ]
    )


def edge_table(g: Graph) -> str:
    layer = g.repos
    rows = ["| From | To | Strongest evidence | Example |", "|---|---|---|---|"]
    for e in sorted(
        g.edges,
        key=lambda e: (-LAYERS.index(layer[e.source]), e.source, e.target),
    ):
        rows.append(f"| {e.source} | {e.target} | {e.kind} | `{e.via}` |")
    return "\n".join(rows)


def tier_table(g: Graph) -> str:
    rows = ["| Layer | Tier | Repositories | Holds |", "|---|---|---|---|"]
    for lay in reversed(LAYERS):
        for tier, role in reversed(list(enumerate(TIER_ROLES[lay]))):
            repos = sorted(r for r, t in TIERS[lay].items() if t == tier)
            rows.append(f"| {TITLES[lay]} | {tier} | {', '.join(repos)} | {role} |")
    return "\n".join(rows)


def violation_table(g: Graph) -> str:
    found = violations(g)
    rows = ["| Rule | From | To | Where | Why, and the fix |", "|---|---|---|---|---|"]
    for key in sorted(found):
        rule, a, b = key
        why = KNOWN.get(key, "**new**")
        rows.append(f"| {rule} | {a} | {b} | `{found[key]}` | {why} |")
    if len(rows) == 2:
        rows.append("| none | | | | |")
    return "\n".join(rows)


NUMBERS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight")
#: how the summary names each rule's rows, singular and plural
RULE_NOUNS = {
    "cycle": ("cycle between repositories", "cycles between repositories"),
    "package-cycle": ("cycle between packages", "cycles between packages"),
    "upward": ("edge up a layer", "edges up a layer"),
    "tier": (
        "edge that does not point down a tier",
        "edges that do not point down a tier",
    ),
    "third-party": (
        "edge out of a third-party repository",
        "edges out of a third-party repository",
    ),
    "undeclared": (
        "import its package does not declare",
        "imports their package does not declare",
    ),
    "untiered": ("repository without a tier", "repositories without a tier"),
    "stale-tier": ("tier for a missing repository", "tiers for missing repositories"),
}


def violation_summary(g: Graph) -> str:
    """One sentence counting the rows of the violation table, by rule."""
    found = violations(g)
    counts: dict[str, int] = defaultdict(int)
    for rule, _, _ in found:
        counts[rule] += 1
    if not counts:
        return "Today the graph breaks none of them."
    cycles = [v for (rule, _, _), v in sorted(found.items()) if rule == "cycle"]
    parts = []
    for rule, (one, many) in RULE_NOUNS.items():
        n = counts.get(rule, 0)
        if n:
            word = NUMBERS[n] if n < len(NUMBERS) else str(n)
            noun = one if n == 1 else many
            if rule == "cycle":
                noun += f" ({'; '.join(cycles)})"
            parts.append(f"**{word} {noun}**")
    listed = (
        parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    )
    return f"Today it has {listed}. [](#tbl-dg-violations) lists each one with its fix."


TABLES = {
    "edge-table": edge_table,
    "tier-table": tier_table,
    "violation-table": violation_table,
    "violation-summary": violation_summary,
}


def render(page: Path, g: Graph) -> str:
    text = page.read_text(encoding="utf-8")
    start, stop = text.index(BEGIN), text.index(END) + len(END)
    out = text[:start] + block(g, page) + text[stop:]
    for name, make in TABLES.items():
        tb, te = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
        if tb in out:
            s, t = out.index(tb), out.index(te) + len(te)
            out = out[:s] + f"{tb}\n{make(g)}\n{te}" + out[t:]
    return out


#: where each repository's guard test goes, relative to its root
GUARD_PATHS = {
    "pipekit": "packages/pipekit/tests/test_dependency_rules.py",
    "geotoolz": "packages/geotoolz/tests/test_dependency_rules.py",
    "xrtoolz": "packages/xrtoolz-core/tests/test_dependency_rules.py",
    "pyrox": "packages/pyrox/tests/test_dependency_rules.py",
}
PAGE_URL = (
    "https://github.com/jejjohnson/geostack/blob/main/docs/api/dependency_graph.md"
)


def guard(repo: str, g: Graph | None = None) -> tuple[str, str]:
    """(path in the repository, source) of the repository's guard test."""
    g = g or build()
    layer = g.repos[repo]
    tier = TIERS[layer][repo]
    base = REFS / layer / repo
    path = GUARD_PATHS.get(repo, "tests/test_dependency_rules.py")

    def allowed(other: str) -> bool:
        lay = g.repos[other]
        if other == repo:
            return True
        if lay == layer:
            return TIERS[lay].get(other, 0) < tier
        return LAYER_RANK[lay] < LAYER_RANK[layer]

    mods = {_module_name(m): p for p in g.pkgs.values() for m in p.modules}
    forbidden = {m: p.name for m, p in mods.items() if not allowed(p.repo)}
    ok = {m: p.name for m, p in mods.items() if allowed(p.repo)}
    packages = {
        p.name: (
            str((p.path / "pyproject.toml").relative_to(base)),
            tuple(str(m.relative_to(base)) for m in p.modules),
        )
        for p in g.pkgs.values()
        if p.repo == repo
    }
    up = sorted({g.pkgs[d].repo for d in forbidden.values()})

    def q(v: object) -> str:
        if isinstance(v, tuple):
            inner = ", ".join(q(x) for x in v)
            return f"({inner},)" if len(v) == 1 else f"({inner})"
        return json.dumps(v)

    def literal(d: dict) -> str:
        if not d:
            return "{}"
        rows = "".join(f"\n    {q(k)}: {q(v)}," for k, v in sorted(d.items()))
        return "{" + rows + "\n}"

    def reasons(d: dict) -> str:
        # long strings as implicit concatenation, which ruff format keeps
        if not d:
            return "{}"
        rows = ""
        for k, v in sorted(d.items()):
            parts = textwrap.wrap(v, 66)
            body = "".join(
                f"\n        {json.dumps(part if i == 0 else ' ' + part)}"
                for i, part in enumerate(parts)
            )
            rows += f"\n    {q(k)}: ({body}\n    ),"
        return "{" + rows + "\n}"

    def table(d: dict) -> str:
        # one entry per pyproject, always expanded, as ruff format keeps it
        rows = "".join(
            f"\n    {q(k)}: (\n        {q(v[0])},\n        {q(v[1])},\n    ),"
            for k, v in sorted(d.items())
        )
        return "{" + rows + "\n}"

    intro = textwrap.fill(
        f"{repo} sits in {TITLES[layer]}, tier {tier} ({TIER_ROLES[layer][tier]})."
        f" Its packages must not import or depend on"
        f" {', '.join(up) if up else 'any other stack library'}: they sit above"
        f" it, or beside it in the same tier. Every import of a stack library is"
        f" declared, as a dependency or an extra, and what importing a package"
        f" runs needs a dependency. Its packages form no cycle. Dev, docs and"
        f" tutorial dependency groups are not checked.",
        width=79,
    )
    template = ROOT / "scripts" / "templates" / "test_dependency_rules.py.tmpl"
    text = template.read_text(encoding="utf-8").format(
        repo=repo,
        intro=intro,
        page=PAGE_URL,
        depth=len(Path(path).parts) - 1,
        packages=table(packages),
        forbidden=literal(forbidden),
        allowed=literal(ok),
        skip_dirs="{" + "".join(f"\n    {q(d)}," for d in sorted(SKIP_DIRS)) + "\n}",
        exceptions=reasons(
            {v[1]: why for v, why in UNDECLARED_OK.items() if v[0] in packages}
        ),
    )
    return path, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if stale")
    parser.add_argument(
        "--guard",
        metavar="REPO",
        help="write REPO's dependency-rules test into its submodule",
    )
    args = parser.parse_args(argv)
    g = build()
    if args.guard:
        if args.guard in THIRD_PARTY:
            print(f"{args.guard} is third party: geostack never writes to it")
            return 1
        path, text = guard(args.guard, g)
        dest = REFS / g.repos[args.guard] / args.guard / path
        dest.write_text(text, encoding="utf-8")
        print(f"wrote {dest.relative_to(ROOT)}")
        return 0
    status = 0
    for page in PAGES:
        new = render(page, g)
        if args.check:
            if new != page.read_text(encoding="utf-8"):
                print(f"{page.relative_to(ROOT)} is stale: run {Path(__file__).name}")
                status = 1
        else:
            page.write_text(new, encoding="utf-8")
    fresh, gone = check(g)
    for (rule, a, b), why in sorted(fresh.items()):
        print(f"new {rule}: {a} → {b} ({why})")
    for rule, a, b in gone:
        print(f"no longer {rule}: {a} → {b}; remove it from KNOWN")
    return 1 if status or fresh or gone else 0


if __name__ == "__main__":
    sys.exit(main())
