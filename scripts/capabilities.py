"""Generate, or check, the capability index: every public name in the package.

``docs/api/capabilities.md`` lists each public object of the project package
once, under the shortest module path that exports it, with the first sentence
of its docstring. Agents and people search it before writing a
loader, a model, a training step or a helper in a script or a notebook
("Reuse before you write" in ``AGENTS.md``).

Usage::

    make capabilities                                   # rewrite the index
    uv run python scripts/capabilities.py --check       # fail if stale

``tests/test_capabilities.py`` runs the check. The script is generic: it
walks every non-private module of ``PACKAGE``. A module's public names are
its ``__all__`` when it defines one, otherwise the classes and functions it
defines itself (imports are skipped; list constants in ``__all__``).

``--check`` also fails when two modules export the same name for different
objects, or the package exports a name an ``UPSTREAM`` library exports for
something else, unless ``ALLOWED_SHARED_NAMES`` gives the reason.
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

# template-example: the import name of your package (src/<PACKAGE>/).
PACKAGE = "myproject"

# Libraries you build on whose public API should be listed after yours, as
# {distribution: [public modules]}, with a one-paragraph blurb each, e.g.
# {"gaussx": ["gaussx"]}. Leave it empty when there are none.
UPSTREAM: dict[str, list[str]] = {}
UPSTREAM_BLURB: dict[str, str] = {}

# Names two homes may share on purpose, and why: {name: reason}.
ALLOWED_SHARED_NAMES: dict[str, str] = {}

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
        if not n.startswith("__")
        and not isinstance(getattr(module, n, None), types.ModuleType)
    ]


def modules() -> list[str]:
    """``PACKAGE`` and every submodule with no ``_``-prefixed component."""
    package = PACKAGE
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
            index[module_name].append((name, _kind(obj), _summary(obj, home=PACKAGE)))
    clashes = []
    for name, homes in sorted(owners.items()):
        if name in ALLOWED_SHARED_NAMES:
            continue
        if len({id(o) for o in homes.values()}) > 1:
            clashes.append(f"{name}: {', '.join(sorted(homes))}")
    for upstream_modules in UPSTREAM.values():
        for module_name in upstream_modules:
            try:
                upstream = importlib.import_module(module_name)
            except ImportError:
                continue
            for name in _public(upstream):
                if name in ALLOWED_SHARED_NAMES:
                    continue
                for path, obj in owners.get(name, {}).items():
                    if obj is not getattr(upstream, name):
                        clashes.append(f"{name}: {path} shadows {module_name}.{name}")
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
    out: list[str] = []
    for package, upstream_modules in UPSTREAM.items():
        out += [f"## Upstream: `{package}`", "", UPSTREAM_BLURB[package], ""]
        for module_name in upstream_modules:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            rows = []
            for name in sorted(_public(module)):
                obj = getattr(module, name)
                rows.append((name, _kind(obj), _summary(obj, home=package)))
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
        f"Every public name in `{PACKAGE}` ({total} of them), each listed once",
        "under the shortest module path that exports it, with the first",
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
    """Whether ``current`` differs from ``text`` where it is comparable.

    The upstream sections are compared only when the recorded versions match
    the installed ones; the package part always is.
    """
    if current == text:
        return False
    if not UPSTREAM or f"Listed at {_versions()}." in current:
        return True
    return current.partition(UPSTREAM_MARKER)[0] != text.partition(UPSTREAM_MARKER)[0]


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
