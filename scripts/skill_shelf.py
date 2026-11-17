"""Measure a library's skill shelf from the submodules, and write the tables.

A shelf page (``docs/design/skill_tree_<library>.md``) groups a library's
public names into units and says which units each dependent package uses.
Both are measured here, at the submodule pins, so the pages can be
regenerated and checked:

- the public names of each of the library's packages, from the static
  ``__all__`` of its ``__init__.py``, with the subpackage each is imported
  from;
- the names each other package in the stack uses, from its imports and its
  attribute accesses on the library's modules.

It reads files only (it imports and installs nothing), like
``dependency_graph.py``, whose package discovery it reuses.

    python scripts/skill_shelf.py            # rewrite the tables
    python scripts/skill_shelf.py --check    # exit 1 if a page is stale
"""

from __future__ import annotations

import argparse
import ast
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import dependency_graph as dg


ROOT = Path(__file__).resolve().parents[1]
#: a dependent receives a unit's card when it uses at least this many names
CARD_THRESHOLD = 3


@dataclass(frozen=True)
class Unit:
    """A group of a library's subpackages that the same dependents use.

    ``patterns`` are ``module`` (a whole import package) or
    ``module.subpackage``; the longest match wins.
    """

    name: str
    rungs: str
    band: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class Shelf:
    """One library's shelf: its page, its distributions and its units."""

    page: str
    packages: tuple[str, ...]  # distribution names, as dependency_graph has them
    units: tuple[Unit, ...]


SHELVES: dict[str, Shelf] = {
    "gaussx": Shelf(
        page="docs/design/skill_tree_gaussx.md",
        packages=("gaussx",),
        units=(
            Unit(
                "Structures and primitives",
                "⓪ ①",
                "numerics",
                (
                    "gaussx._operators",
                    "gaussx._tags",
                    "gaussx._gmrf",
                    "gaussx._sparse",
                    "gaussx._primitives",
                    "gaussx._linalg",
                ),
            ),
            Unit(
                "Strategies",
                "②, setup",
                "numerics, ML",
                (
                    "gaussx._strategies",
                    "gaussx._preconditioners",
                    "gaussx._solve_frontend",
                    "gaussx._randomized",
                    "gaussx._sketching",
                ),
            ),
            Unit("Gaussians", "③", "ML", ("gaussx._distributions", "gaussx._expfam")),
            Unit("Gaussian processes", "④", "ML", ("gaussx._gp",)),
            Unit("State space", "④", "ML", ("gaussx._ssm",)),
            Unit("Quadrature", "② to ④", "ML", ("gaussx._quadrature",)),
            Unit("Inference and ensembles", "④", "ML", ("gaussx._inference",)),
        ),
    ),
    "kernellib": Shelf(
        page="docs/design/skill_tree_kernellib.md",
        packages=("kernellib",),
        units=(
            Unit(
                "Kernels",
                "⓪ ①",
                "ML",
                (
                    "kernellib",
                    "kernellib._kernels",
                    "kernellib.functional",
                    "kernellib._derivatives",
                    "kernellib._heuristics",
                ),
            ),
            Unit("Geostatistical kernels", "⓪", "ML, geoscience", ("kernellib._geo",)),
            Unit(
                "Operators and feature maps",
                "⓪, setup",
                "numerics, ML",
                ("kernellib._operators", "kernellib._spectral"),
            ),
            Unit("Graphs", "⓪ ①", "ML", ("kernellib._graph",)),
            Unit(
                "Estimators",
                "③ ④",
                "ML",
                ("kernellib._regression", "kernellib.sklearn"),
            ),
            Unit("Decomposition", "③", "ML", ("kernellib._decomposition",)),
            Unit("Dependence measures", "④", "ML", ("kernellib._dependence",)),
        ),
    ),
}


def _parse(source: str) -> ast.Module | None:
    """Parse ``source``, quietly: a library's own syntax warnings are not ours."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return ast.parse(source)
        except SyntaxError:
            return None


def _origin(module: str, node: ast.ImportFrom, top: str) -> str | None:
    """``module.subpackage`` an ``__init__`` imports a name from, or ``None``.

    ``_src`` is skipped, so ``pkg._src.solvers`` and ``pkg.solvers`` agree. A
    name imported from the package root gets the subpackage ``.``.
    """
    parts = (node.module or "").split(".")
    if node.level == 0:
        if parts[0] != top:
            return None
        parts = parts[1:]
    elif node.level > 1:
        return None
    parts = [p for p in parts if p and p != "_src"]
    return f"{module}.{parts[0] if parts else '.'}"


def _static_all(tree: ast.Module) -> list[str] | None:
    """The names of a module's ``__all__`` when it is a literal list, else ``None``."""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "__all__" for t in node.targets)
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            return [
                e.value
                for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
    return None


def exports(pkg: dg.Package) -> dict[str, str]:
    """A package's public names, each with the ``module.subpackage`` it comes from.

    The public names are the static ``__all__`` of each import root's
    ``__init__.py``, then the static ``__all__`` of each public submodule (one
    whose name has no leading underscore) for names the root does not export.
    A name added to ``__all__`` at run time (behind an extra) is not counted.
    Without a static ``__all__``, every name the root imports that does not
    start with an underscore counts.
    """
    out: dict[str, str] = {}
    for root in pkg.modules:
        init = root / "__init__.py"
        tree = _parse(init.read_text(encoding="utf-8")) if init.exists() else None
        if tree is None:
            continue
        module = dg._module_name(root)
        origin: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                where = _origin(module, node, module)
                if where is None:
                    continue
                for alias in node.names:
                    origin[alias.asname or alias.name] = where
        public = _static_all(tree)
        names = public if public is not None else [n for n in origin if n[:1] != "_"]
        for name in names:
            out[name] = origin.get(name, f"{module}..")
        for child in sorted(root.iterdir()):
            sub = child.stem if child.suffix == ".py" else child.name
            source = child if child.suffix == ".py" else child / "__init__.py"
            if sub[:1] == "_" or not source.is_file():
                continue
            subtree = _parse(source.read_text(encoding="utf-8"))
            if subtree is None:
                continue
            for name in _static_all(subtree) or []:
                out.setdefault(name, f"{module}.{sub}")
    return out


def uses(source: str, modules: set[str], names: set[str]) -> set[str]:
    """The names of ``names`` that ``source`` takes from one of ``modules``.

    Counts ``from module import name``, ``from module.sub import name`` and
    ``alias.name`` or ``alias.sub.name`` for every alias of a module.
    """
    tree = _parse(source)
    if tree is None:
        return set()
    aliases: set[str] = set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in modules:
                    aliases.add(alias.asname or alias.name.split(".")[0])
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module.split(".")[0] in modules
        ):
            found |= {alias.name for alias in node.names}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base: ast.expr = node.value
        while isinstance(base, ast.Attribute):
            base = base.value
        if isinstance(base, ast.Name) and base.id in aliases:
            found.add(node.attr)
    return found & names


def unit_of(shelf: Shelf, origin: str) -> str | None:
    """The unit a ``module.subpackage`` belongs to; the longest pattern wins."""
    module = origin.split(".")[0]
    best: tuple[int, str] | None = None
    for unit in shelf.units:
        for pattern in unit.patterns:
            if pattern in (origin, module) and (best is None or len(pattern) > best[0]):
                best = (len(pattern), unit.name)
    return best[1] if best else None


@dataclass
class Measured:
    """A shelf's measurements: names by unit, and each dependent's use of them."""

    names: dict[str, str]  # public name -> module.subpackage
    used: dict[str, set[str]]  # dependent distribution -> names it uses


def measure(shelf: Shelf, pkgs: dict[str, dg.Package]) -> Measured:
    """Measure ``shelf`` over every package in the submodules."""
    names: dict[str, str] = {}
    modules: set[str] = set()
    for dist in shelf.packages:
        names |= exports(pkgs[dist])
        modules |= {dg._module_name(root) for root in pkgs[dist].modules}
    used: dict[str, set[str]] = {}
    for dist, pkg in sorted(pkgs.items()):
        if dist in shelf.packages or pkg.repo in dg.THIRD_PARTY:
            continue
        found: set[str] = set()
        for root in pkg.modules:
            for f in dg._files(root):
                found |= uses(f.read_text(encoding="utf-8"), modules, set(names))
        if found:
            used[dist] = found
    return Measured(names, used)


def unassigned(shelf: Shelf, m: Measured) -> list[str]:
    """Public names whose subpackage belongs to no unit."""
    return sorted(n for n, o in m.names.items() if unit_of(shelf, o) is None)


def _count(shelf: Shelf, m: Measured, found: set[str]) -> dict[str, int]:
    out = {unit.name: 0 for unit in shelf.units}
    for name in found:
        unit = unit_of(shelf, m.names[name])
        if unit is not None:
            out[unit] += 1
    return out


def units_table(shelf: Shelf, m: Measured) -> str:
    """Each unit with its rungs, its subpackages and how many names it holds."""
    rows = [
        "| Unit | Rungs | Home band | Subpackages | Names |",
        "|---|---|---|---|---|",
    ]
    total = 0
    for unit in shelf.units:
        per: dict[str, int] = {}
        for origin in m.names.values():
            if unit_of(shelf, origin) == unit.name:
                per[origin] = per.get(origin, 0) + 1
        subs = ", ".join(
            f"`{o.removesuffix('..')}` {n}"
            for o, n in sorted(per.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        count = sum(per.values())
        total += count
        rows.append(f"| {unit.name} | {unit.rungs} | {unit.band} | {subs} | {count} |")
    rows.append(f"| **All** | | | | **{total}** |")
    return "\n".join(rows)


def _dependents(m: Measured) -> list[str]:
    return sorted(m.used, key=lambda d: (-len(m.used[d]), d))


def uses_table(shelf: Shelf, m: Measured) -> str:
    """Each dependent package with the number of names it uses from each unit."""
    heads = " | ".join(unit.name for unit in shelf.units)
    rows = [
        f"| Dependent package | Names used | {heads} |",
        "|---|---|" + "---|" * len(shelf.units),
    ]
    for dist in _dependents(m):
        counts = _count(shelf, m, m.used[dist])
        cells = " | ".join(str(counts[unit.name]) for unit in shelf.units)
        rows.append(f"| {dist} | {len(m.used[dist])} | {cells} |")
    return "\n".join(rows)


def cards_table(shelf: Shelf, m: Measured) -> str:
    """The cards each dependent receives: units it uses ``CARD_THRESHOLD`` names of."""
    rows = ["| Dependent package | Cards | Also uses |", "|---|---|---|"]
    for dist in _dependents(m):
        counts = _count(shelf, m, m.used[dist])
        cards = [u.name for u in shelf.units if counts[u.name] >= CARD_THRESHOLD]
        below = [
            f"{u.name} ({counts[u.name]})"
            for u in shelf.units
            if 0 < counts[u.name] < CARD_THRESHOLD
        ]
        rows.append(f"| {dist} | {'; '.join(cards) or 'none'} | {'; '.join(below)} |")
    return "\n".join(rows)


TABLES = {"units": units_table, "uses": uses_table, "cards": cards_table}


def render(text: str, shelf: Shelf, m: Measured) -> str:
    """``text`` with each ``<!-- shelf:<table>:begin/end -->`` block regenerated."""
    for key, table in TABLES.items():
        begin, end = f"<!-- shelf:{key}:begin -->", f"<!-- shelf:{key}:end -->"
        if begin not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{table(shelf, m)}\n{end}{tail}"
    return text


def stale(pkgs: dict[str, dg.Package] | None = None) -> list[str]:
    """The shelf pages whose tables differ from what the submodules give."""
    pkgs = pkgs if pkgs is not None else dg.packages()
    out = []
    for shelf in SHELVES.values():
        page = ROOT / shelf.page
        text = page.read_text(encoding="utf-8")
        if render(text, shelf, measure(shelf, pkgs)) != text:
            out.append(shelf.page)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if stale")
    args = parser.parse_args(argv)
    pkgs = dg.packages()
    if args.check:
        pages = stale(pkgs)
        for page in pages:
            print(f"stale: {page} (run python scripts/skill_shelf.py)")
        return 1 if pages else 0
    for name, shelf in SHELVES.items():
        m = measure(shelf, pkgs)
        page = ROOT / shelf.page
        page.write_text(render(page.read_text(encoding="utf-8"), shelf, m), "utf-8")
        missing = unassigned(shelf, m)
        if missing:
            print(f"{name}: {len(missing)} names in no unit: {', '.join(missing[:8])}")
        print(f"{name}: {len(m.names)} names, {len(m.used)} dependent packages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
