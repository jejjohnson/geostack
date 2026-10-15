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

#: layers from the bottom up; an edge may point to its own layer or one below
LAYERS = ("geostack", "numerics", "geoml", "geomodels")
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
    "geomodels": {"plumax": 0, "radtrax": 0, "somax": 0},
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
    "geomodels": ("models; never import each other",),
}
#: violations the graph has today, with why; the check fails on a new one and
#: on one that is gone, so the list only shrinks
KNOWN: dict[tuple[str, str, str], str] = {
    ("cycle", "geotoolz", "pipekit"): (
        "pipekit-train's _xreader lazily imports geopatcher.time (which no longer"
        " exists); Report 16 D2 moves the dataset to geopatcher.integrations.train"
    ),
    ("tier", "pipekit", "geotoolz"): "the same import",
    ("undeclared", "pipekit-train", "geotoolz-patcher"): "the same import",
    ("undeclared", "gauss-flows", "gaussx"): (
        "a try-guarded import in transform_filter.py; declare a [gaussx] extra"
    ),
    ("undeclared", "somax", "filterax"): (
        "module-level imports in somax/_src/da, declared only in the da"
        " dependency group; make it an extra"
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
    modules: list[Path] = field(default_factory=list)


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
                base = project.get("dependencies") or []
                pkg.base = {_req(r) for r in base if isinstance(r, str)}
                pkg.extras = {
                    _req(r)
                    for reqs in (project.get("optional-dependencies") or {}).values()
                    for r in reqs
                }
                src = pkg.path / "src" if (pkg.path / "src").is_dir() else pkg.path
                pkg.modules = sorted(
                    d
                    for d in src.iterdir()
                    if (d / "__init__.py").exists() and d.name not in SKIP_DIRS
                )
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


def _imports(tree: ast.AST) -> list[tuple[str, str]]:
    """(top-level module, how) for every absolute import in a module."""
    out: list[tuple[str, str]] = []

    def visit(node: ast.AST, how: str) -> None:
        for child in ast.iter_child_nodes(node):
            here = how
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                here = "lazy" if how == "module" else how
            elif isinstance(child, ast.Try):
                here = "guarded" if how == "module" else how
            elif isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(
                child.test
            ):
                here = "typing"
            if isinstance(child, ast.Import):
                out.extend((a.name.split(".")[0], here) for a in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module and not child.level:
                out.append((child.module.split(".")[0], here))
            visit(child, here)

    visit(tree, "module")
    return out


RANK = {"module": 0, "guarded": 1, "lazy": 2, "typing": 3}


def imports(pkgs: dict[str, Package]) -> dict[str, dict[str, Import]]:
    """For each package, the other packages it imports, the strongest way."""
    owner = {m.name: p.name for p in pkgs.values() for m in p.modules}
    out: dict[str, dict[str, Import]] = {}
    for pkg in pkgs.values():
        found: dict[str, Import] = {}
        for module in pkg.modules:
            for f in sorted(module.rglob("*.py")):
                # an upstream invalid escape warns; under warnings-as-errors that
                # would turn into a SyntaxError and silently drop the file
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tree = ast.parse(f.read_text(encoding="utf-8"))
                for top, how in _imports(tree):
                    dist = owner.get(top)
                    if dist is None or dist == pkg.name:
                        continue
                    old = found.get(dist)
                    if old is None or RANK[how] < RANK[old.how]:
                        found[dist] = Import(dist, str(f.relative_to(pkg.path)), how)
        out[pkg.name] = found
    return out


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str  # "base", "extra" or "import": the strongest evidence
    via: str  # one package-level example


KIND_RANK = {"base": 0, "extra": 1, "import": 2}


def repo_edges(
    pkgs: dict[str, Package], imp: dict[str, dict[str, Import]]
) -> list[Edge]:
    best: dict[tuple[str, str], Edge] = {}
    for p in sorted(pkgs.values(), key=lambda p: p.name):
        # dependency groups (dev, docs, tutorials) are not library dependencies
        evidence = [(t, "base") for t in p.base] + [(t, "extra") for t in p.extras]
        evidence += [(t, "import") for t in imp[p.name]]
        for t, kind in sorted(evidence):
            a, b = p.repo, pkgs[t].repo
            if a == b:
                continue
            edge = Edge(a, b, kind, f"{p.name} → {t}")
            old = best.get((a, b))
            if old is None or KIND_RANK[kind] < KIND_RANK[old.kind]:
                best[(a, b)] = edge
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
    imports: dict[str, dict[str, Import]]
    edges: list[Edge]

    @property
    def repos(self) -> dict[str, str]:
        return {p.repo: p.layer for p in self.pkgs.values()}


def build() -> Graph:
    pkgs = packages()
    imp = imports(pkgs)
    return Graph(pkgs, imp, repo_edges(pkgs, imp))


def violations(g: Graph) -> dict[tuple[str, str, str], str]:
    """Every rule the graph breaks: {(rule, a, b): evidence}."""
    out: dict[tuple[str, str, str], str] = {}
    layer = g.repos
    pairs = [(e.source, e.target) for e in g.edges]
    for comp in sccs(list(layer), pairs):
        out[("cycle", comp[0], comp[1] if len(comp) == 2 else "+".join(comp[1:]))] = (
            " ⇄ ".join(comp)
        )
    pkg_pairs = [
        (p, t)
        for p, pkg in g.pkgs.items()
        for t in pkg.base | pkg.extras | set(g.imports[p])
    ]
    for comp in sccs(list(g.pkgs), pkg_pairs):
        out[("package-cycle", comp[0], "+".join(comp[1:]))] = " ⇄ ".join(comp)
    for repo, lay in layer.items():
        if repo not in TIERS[lay]:
            out[("untiered", repo, lay)] = (
                "add it to TIERS in scripts/dependency_graph.py"
            )
    for e in g.edges:
        la, lb = LAYERS.index(layer[e.source]), LAYERS.index(layer[e.target])
        tiers = TIERS[layer[e.source]]
        if lb > la:
            out[("upward", e.source, e.target)] = e.via
        elif la == lb and tiers.get(e.target, 0) >= tiers.get(e.source, 0):
            out[("tier", e.source, e.target)] = e.via
        if e.source in THIRD_PARTY:
            out[("third-party", e.source, e.target)] = e.via
    for p, found in g.imports.items():
        pkg = g.pkgs[p]
        for t, im in found.items():
            if im.how == "typing":
                continue
            declared = pkg.base if im.how == "module" else pkg.base | pkg.extras
            if t not in declared:
                out[("undeclared", p, t)] = f"{im.where} ({im.how})"
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
    known_cycle = {(a, b) for (rule, a, b) in violations(g) if rule == "cycle"}
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
        if {(e.source, e.target), (e.target, e.source)} & known_cycle:
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


TABLES = {
    "edge-table": edge_table,
    "tier-table": tier_table,
    "violation-table": violation_table,
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
        return LAYERS.index(lay) < LAYERS.index(layer)

    mods = {m.name: p for p in g.pkgs.values() for m in p.modules}
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

    intro = textwrap.fill(
        f"{repo} sits in {TITLES[layer]}, tier {tier} ({TIER_ROLES[layer][tier]})."
        f" Its packages must not import or depend on"
        f" {', '.join(up) if up else 'any other stack library'}: they sit above"
        f" it, or beside it in the same tier. Every import of a stack library is"
        f" declared: a module-level import as a base dependency, a lazy or"
        f" ``try``-guarded one as a dependency or an extra. Dev, docs and"
        f" tutorial dependency groups are not checked.",
        width=79,
    )
    template = ROOT / "scripts" / "templates" / "test_dependency_rules.py.tmpl"
    text = template.read_text(encoding="utf-8").format(
        repo=repo,
        intro=intro,
        page=PAGE_URL,
        depth=len(Path(path).parts) - 1,
        packages=literal(packages),
        forbidden=literal(forbidden),
        allowed=literal(ok),
    )
    return path, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if stale")
    parser.add_argument(
        "--guard", metavar="REPO", help="print REPO's dependency-rules test"
    )
    args = parser.parse_args(argv)
    g = build()
    if args.guard:
        print(guard(args.guard, g)[1], end="")
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
