"""The shelf check that every library with a shelf carries a copy of.

``scripts/templates/skill_shelf.py`` is written into a library by
``scripts/skill_tree.py --shelf-check``. These tests run it on a small library
built in a temporary directory: one with a single module, whose entries list
bare names, and one with two modules, whose entries list ``module.name``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "templates" / "skill_shelf.py"

_spec = importlib.util.spec_from_file_location("shelf_check", SCRIPT)
assert _spec is not None and _spec.loader is not None
sc = importlib.util.module_from_spec(_spec)
sys.modules["shelf_check"] = sc  # dataclasses look their module up
_spec.loader.exec_module(sc)

SKILL = """---
name: use-lib
description: Use lib.
---

# Use lib

<!-- shelf:begin -->
<!-- shelf:end -->
"""


def _library(root: Path, modules: dict[str, list[str]], names: list[str]) -> None:
    """A library with one skill, one unit and one entry that lists ``names``."""
    table = []
    for module, public in modules.items():
        init = root / "src" / module.replace(".", "/") / "__init__.py"
        init.parent.mkdir(parents=True)
        init.write_text(f"__all__ = {public!r}\n", encoding="utf-8")
        table.append(f'"{module}" = "{init.relative_to(root)}"')
    plugin = root / "plugins" / "lib"
    (plugin / "skills" / "use-lib" / "shelf").mkdir(parents=True)
    (plugin / "tree" / "cards").mkdir(parents=True)
    (plugin / "skills" / "use-lib" / "SKILL.md").write_text(SKILL, encoding="utf-8")
    words = " ".join(n.rsplit(".", 1)[-1] for n in names)
    (plugin / "skills" / "use-lib" / "shelf" / "all.md").write_text(
        f"# All\n\n{words}\n", encoding="utf-8"
    )
    first = next(iter(modules))
    (plugin / "tree" / "cards" / "core.md").write_text(
        f"- **Provides.** `{first}.{modules[first][0]}`.\n", encoding="utf-8"
    )
    (plugin / "tree" / "shelf.toml").write_text(
        "[modules]\n"
        + "\n".join(table)
        + "\n\n[skills.use-lib]\n\n"
        '[units.core]\ntitle = "Core"\nrungs = "①"\nskill = "use-lib"\n\n'
        '[[entries]]\nfile = "all"\ntitle = "All"\nrole = "primitive"\n'
        'unit = "core"\nskill = "use-lib"\ncovers = "everything"\n'
        f"names = {names!r}\n".replace("'", '"'),
        encoding="utf-8",
    )


def _write(shelf) -> None:
    for skill in shelf.skills:
        path = sc.skill_file(skill)
        path.write_text(sc.render(shelf, skill, path.read_text("utf-8")), "utf-8")


@pytest.fixture
def one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    _library(tmp_path, {"lib": ["solve", "logdet"]}, ["solve", "logdet"])
    return tmp_path


@pytest.fixture
def two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    _library(
        tmp_path,
        {"lib": ["solve", "geometry"], "lib.geometry": ["Affine"]},
        ["lib.solve", "lib.geometry.Affine"],
    )
    return tmp_path


def test_one_module_lists_bare_names_and_passes_once_written(one) -> None:
    shelf = sc.load()
    assert sc.plugin_dir() == one / "plugins" / "lib"
    assert sc.public_names(shelf) == {"solve", "logdet"}
    assert sc.violations(shelf, sc.public_names(shelf)) == [
        "stale: use-lib/SKILL.md (run scripts/skill_shelf.py)"
    ]
    _write(shelf)
    assert sc.violations(shelf, sc.public_names(shelf)) == []
    assert "| ① primitive | [All](shelf/all.md) | everything | 2 |" in (
        sc.skill_file("use-lib").read_text(encoding="utf-8")
    )


def test_several_modules_list_names_with_their_module(two) -> None:
    shelf = sc.load()
    # lib.geometry is a module of its own, so it is not a name of lib
    assert sc.public_names(shelf) == {"lib.solve", "lib.geometry.Affine"}
    _write(shelf)
    assert sc.violations(shelf, sc.public_names(shelf)) == []


def test_sh1_a_public_name_in_no_entry_and_an_entry_that_does_not_name_it(one) -> None:
    init = one / "src" / "lib" / "__init__.py"
    init.write_text('__all__ = ["solve", "logdet", "trace"]\n', encoding="utf-8")
    (sc.plugin_dir() / "skills/use-lib/shelf/all.md").write_text("# All\n\nsolve\n")
    shelf = sc.load()
    _write(shelf)
    assert sc.violations(shelf, sc.public_names(shelf)) == [
        "SH-1 all: does not name logdet",
        "SH-1 trace: public, in no entry",
    ]


def test_sh2_a_card_cites_only_public_names_of_the_longest_module(two) -> None:
    shelf = sc.load()
    _write(shelf)
    card = sc.card_file("core")
    card.write_text(
        "- `lib.geometry.Affine`, `lib.geometry`, `lib.geometry.Gone`, `lib.nope()`.\n"
    )
    assert sc.cited(shelf, card.read_text()) == [
        ("lib.geometry", "Affine"),
        ("lib.geometry", "Gone"),
        ("lib", "nope"),
    ]
    assert sc.violations(shelf, sc.public_names(shelf)) == [
        "SH-2 core: no public name lib.nope",
        "SH-2 core: no public name lib.geometry.Gone",
    ]


def test_sh3_a_card_over_its_budget(one) -> None:
    shelf = sc.load()
    _write(shelf)
    sc.card_file("core").write_text("\n".join(["- line"] * 16), encoding="utf-8")
    assert sc.violations(shelf, sc.public_names(shelf)) == [
        "SH-3 core: card over 15 lines"
    ]


def test_cards_below_a_skill_go_when_it_stops_building_on_units(one) -> None:
    path = sc.skill_file("use-lib")
    begin, end = sc.BELOW
    path.write_text(f"{path.read_text()}\n{begin}\nan old card\n{end}\n", "utf-8")
    shelf = sc.load()
    assert shelf.skills["use-lib"] == ()
    _write(shelf)
    text = path.read_text(encoding="utf-8")
    assert "an old card" not in text
    assert f"{begin}\n{end}" in text
    assert sc.violations(shelf, sc.public_names(shelf)) == []


def test_examples_are_the_python_blocks_of_the_plugin(one) -> None:
    page = sc.plugin_dir() / "skills/use-lib/shelf/all.md"
    page.write_text("# All\n\n```python\nx = 1\n```\n", encoding="utf-8")
    assert sc.examples() == [(page, "x = 1\n")]
