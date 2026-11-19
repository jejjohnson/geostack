"""The skill tree's generated blocks and its gates.

``scripts/skill_tree.py`` links each repository's use skills along the
dependency graph: a "Builds on" block at the end of each ``SKILL.md`` carries
one contract card per repository below it. These tests check the block and
the gates on a small tree built in a temporary directory, and then the gates
on the submodules at their pins. It reads files only.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "skill_tree.py"
HAVE_SUBMODULES = (ROOT / "references/geostack/pipekit/pyproject.toml").exists()

_spec = importlib.util.spec_from_file_location("skill_tree", SCRIPT)
assert _spec is not None and _spec.loader is not None
st = importlib.util.module_from_spec(_spec)
sys.modules["skill_tree"] = st  # dataclasses look their module up
_spec.loader.exec_module(st)

CARD = "- **Provides.** {name}.\n- **Never.** Densify."
SHELF = """
[skills.front]
[skills.recipes]
builds_on_units = ["core"]

[units.core]
title = "Core"
rungs = "⓪ ①"
skill = "front"

[units.state-space]
title = "State space"
rungs = "④"
skill = "recipes"
"""


def _plugin(path: Path, name: str, skills: list[str], body: str = "# Skill\n") -> None:
    (path / "plugins" / name / ".claude-plugin").mkdir(parents=True)
    for skill in skills:
        folder = path / "plugins" / name / "skills" / skill
        folder.mkdir(parents=True)
        text = f"---\nname: {skill}\ndescription: Use {name}.\n---\n\n{body}"
        (folder / "SKILL.md").write_text(text, encoding="utf-8")


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """base (a shelf of two units) ← mid ← top, each with one skill; bare has none."""
    monkeypatch.setattr(st, "CARDS", tmp_path / "held")
    (tmp_path / "held").mkdir()
    for repo in ("base", "mid", "top", "bare"):
        (tmp_path / repo).mkdir()
    _plugin(tmp_path / "base", "base", ["front", "recipes"])
    cards = tmp_path / "base" / "plugins" / "base" / "tree" / "cards"
    cards.mkdir(parents=True)
    (cards.parent / "shelf.toml").write_text(SHELF, encoding="utf-8")
    for unit in ("core", "state-space"):
        (cards / f"{unit}.md").write_text(CARD.format(name=unit), encoding="utf-8")
    _plugin(tmp_path / "mid", "mid", ["use-mid"])
    (tmp_path / "held" / "mid.md").write_text(CARD.format(name="mid"), "utf-8")
    _plugin(tmp_path / "top", "top", ["use-top"])
    nodes = {r: st.read_node(r, tmp_path / r) for r in ("bare", "base", "mid", "top")}
    edges = {
        "bare": (),
        "base": (),
        "mid": (st.Edge("base", "base", ("core",)),),
        "top": (
            st.Edge("base", "base", ("core", "state-space")),
            st.Edge("bare", "extra"),
            st.Edge("mid", "import"),
        ),
    }
    return st.Tree(nodes, edges)


def _adopt(tree, repo: str) -> None:
    for path, text in st.render(tree, repo).items():
        path.write_text(text, encoding="utf-8")


def test_a_shelf_gives_its_listed_skills_in_order_and_its_units(tree) -> None:
    base = tree.nodes["base"]
    assert base.skills == ("front", "recipes")
    assert list(base.units) == ["core", "state-space"]
    assert st.slug("State space") == "state-space"


def test_block_carries_only_the_units_the_edge_names(tree) -> None:
    block = st.render_block(tree, "mid")
    assert "### base · base dependency · 1 of 2 units" in block
    assert "#### Core · ⓪ ① · skill `base:front`" in block
    assert "State space" not in block
    assert "Used by (pointers, not skills to load): top." in block


def test_block_names_a_parent_with_no_skill_and_no_card(tree) -> None:
    block = st.render_block(tree, "top")
    assert "### bare · extra · no skill yet" in block
    assert "No card yet: read `AGENTS.md` in bare's repository." in block
    assert "### mid · guarded import · skill `mid:use-mid`" in block
    assert "- **Provides.** mid." in block  # the card geostack holds


def test_a_base_skill_says_so_and_lists_who_uses_it(tree) -> None:
    block = st.render_block(tree, "base")
    assert "Nothing else in the stack: this is a base skill." in block
    assert "Used by (pointers, not skills to load): mid, top." in block


def test_a_repo_below_the_card_threshold_gets_the_front_door(tree) -> None:
    tree.edges["mid"] = (st.Edge("base", "base", ()),)
    block = st.render_block(tree, "mid")
    assert "0 of 2 units" in block
    assert "The front door is `base:front`." in block


WORKSPACE = """
[skills.front]
packages = ["ws"]

[skills.use-gp]
packages = ["ws-gp"]

[skills.use-all]
"""


@pytest.fixture
def workspace(tmp_path: Path):
    """ws (packages ws, ws-gp) above the shelf of base and above mid."""
    for repo in ("base", "mid", "ws"):
        (tmp_path / repo).mkdir()
    _plugin(tmp_path / "base", "base", ["front", "recipes"])
    (tmp_path / "base/plugins/base/tree").mkdir()
    (tmp_path / "base/plugins/base/tree/shelf.toml").write_text(SHELF, "utf-8")
    _plugin(tmp_path / "mid", "mid", ["use-mid"])
    _plugin(tmp_path / "ws", "ws", ["front", "use-gp", "use-all"])
    (tmp_path / "ws/plugins/ws/tree").mkdir()
    (tmp_path / "ws/plugins/ws/tree/shelf.toml").write_text(WORKSPACE, "utf-8")
    pkgs = {
        name: st.dg.Package(name, repo, "geoml", tmp_path / repo)
        for name, repo in [
            ("base", "base"),
            ("mid", "mid"),
            ("ws", "ws"),
            ("ws-gp", "ws"),
        ]
    }
    pkg_edges = [
        st.dg.Edge("ws", "base", "extra", ""),
        st.dg.Edge("ws-gp", "base", "base", ""),
        st.dg.Edge("ws-gp", "mid", "import", ""),
        st.dg.Edge("ws-gp", "ws", "base", ""),
    ]
    graph = st.dg.Graph(pkgs, {}, pkg_edges, st.dg.repo_edges(pkgs, pkg_edges))
    paths = {repo: tmp_path / repo for repo in ("base", "mid", "ws")}
    return st.build(graph, paths=paths), graph


def test_a_skill_that_names_its_packages_builds_on_what_they_do(workspace) -> None:
    tree, _ = workspace
    assert [(e.parent, e.kind) for e in tree.builds_on("ws", "front")] == [
        ("base", "extra")
    ]
    assert [(e.parent, e.kind) for e in tree.builds_on("ws", "use-gp")] == [
        ("base", "base"),
        ("mid", "import"),
    ]
    # a skill that names no package stands for the repository
    assert tree.builds_on("ws", "use-all") == tree.edges["ws"]
    assert [(e.parent, e.kind) for e in tree.edges["ws"]] == [
        ("base", "base"),
        ("mid", "import"),
    ]


def test_a_package_skill_points_at_the_skill_of_a_sibling_it_depends_on(
    workspace,
) -> None:
    tree, _ = workspace
    gp = st.render_block(tree, "ws", "use-gp")
    assert "In this repository, load when the task reaches into it: `ws:front`." in gp
    assert "### mid · guarded import · skill `mid:use-mid`" in gp
    front = st.render_block(tree, "ws", "front")
    assert "In this repository" not in front
    assert "### mid" not in front
    blocks = {p.parent.name: text for p, text in st.render(tree, "ws").items()}
    assert blocks["use-gp"].endswith(gp + "\n")
    assert blocks["front"].endswith(front + "\n")


def test_units_are_folded_over_the_packages_of_a_skill(workspace) -> None:
    tree, graph = workspace
    units = {("ws", "base"): {"state-space"}, ("ws-gp", "base"): {"core"}}
    one = st._edges(graph, tree.nodes, units, "ws", {"ws-gp"})
    assert one[0] == st.Edge("base", "base", ("core",))
    both = st._edges(graph, tree.nodes, units, "ws", {"ws", "ws-gp"})
    assert both[0] == st.Edge("base", "base", ("core", "state-space"))
    # a measured unit that the parent's shelf lacks reaches ST-1
    units["ws-gp", "base"] = {"core", "gone"}
    tree.edges["ws"] = st._edges(graph, tree.nodes, units, "ws", {"ws-gp"})
    assert tree.edges["ws"][0].units == ("core", "gone")
    assert st.graph_violations(tree) == [
        "ST-1 ws → base: skill_shelf.py measures the unit 'gone', which base's"
        " shelf.toml lacks"
    ]


def test_st1_a_skill_names_only_packages_of_its_repository(workspace) -> None:
    tree, graph = workspace
    assert st.package_violations(tree, graph.pkgs) == []
    node = tree.nodes["ws"]
    tree.nodes["ws"] = st.Node(
        node.repo, node.path, node.plugin, node.skills, {}, {"front": ("mid", "nope")}
    )
    assert st.package_violations(tree, graph.pkgs) == [
        "ST-1 ws: shelf.toml gives the skill front the package mid,"
        " which is not one of ws's",
        "ST-1 ws: shelf.toml gives the skill front the package nope,"
        " which is not one of ws's",
    ]


def test_replace_block_is_idempotent_and_keeps_text_after_an_old_block() -> None:
    block = f"{st.BLOCK_BEGIN}\nnew\n{st.BLOCK_END}"
    old = f"body\n\n{st.BLOCK_MARK} old marker -->\nold\n{st.BLOCK_END}\n\nafter\n"
    once = st.replace_block(old, block)
    assert once == f"body\n\nafter\n\n{block}\n"
    assert st.replace_block(once, block) == once
    with pytest.raises(ValueError, match="without"):
        st.replace_block(f"body\n{st.BLOCK_MARK} -->\n", block)


def test_st2_holds_a_repo_to_its_block_only_once_it_carries_one(tree) -> None:
    assert st.violations(tree) == []
    assert st.pending(tree) == ["base", "mid", "top"]
    _adopt(tree, "mid")
    assert st.violations(tree) == []
    assert st.pending(tree) == ["base", "top"]
    card = tree.nodes["base"].plugin_dir / "tree" / "cards" / "core.md"
    card.write_text(CARD.format(name="changed"), encoding="utf-8")
    (stale,) = st.violations(tree)
    assert stale.startswith("ST-2 stale: mid plugins/mid/skills/use-mid/SKILL.md")


def test_st3_budgets(tree) -> None:
    card = tree.nodes["base"].plugin_dir / "tree" / "cards" / "core.md"
    card.write_text("\n".join(["- line"] * 16), encoding="utf-8")
    skill = tree.nodes["top"].skill_file("use-top")
    skill.write_text(
        "---\nname: use-top\ndescription: >-\n  " + "x" * 1537 + "\n---\n", "utf-8"
    )
    assert st.violations(tree) == [
        "ST-3 base: card core.md has 16 lines",
        "ST-3 top:use-top: description over 1536",
    ]


def test_st4_a_skill_never_names_a_skill_that_is_not_below_it(tree) -> None:
    skill = tree.nodes["mid"].skill_file("use-mid")
    skill.write_text(skill.read_text() + "Load `top:use-top` and `base:front`.\n")
    assert st.violations(tree) == [
        "ST-4 mid:use-mid: names top:use-top, which is not below it"
    ]


def test_st1_fails_on_a_cycle_and_on_a_unit_the_shelf_lacks(tree) -> None:
    tree.edges["mid"] = (st.Edge("base", "base", ("gone",)),)
    (message,) = st.graph_violations(tree)
    assert "'gone'" in message and message.startswith("ST-1 mid → base")
    tree.edges["base"] = (st.Edge("top", "base"),)
    assert any(m.startswith("ST-1 cycle") for m in st.graph_violations(tree))


def test_st7_a_card_names_only_public_names_by_a_path_that_exists(tmp_path) -> None:
    pkg = tmp_path / "lib"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "_core").mkdir()
    (pkg / "_hidden").mkdir()
    (pkg / "__init__.py").write_text(
        "__all__ = ['solve', '_core']\ntry:\n    __all__ += ['Extra']\n"
        "except ImportError:\n    pass\n"
    )
    (pkg / "sub" / "__init__.py").write_text("__all__ = ['Grid']\n")
    (pkg / "_core" / "__init__.py").write_text("__all__ = ['Base']\n")
    (pkg / "_hidden" / "__init__.py").write_text("__all__ = ['Secret']\n")
    (pkg / "plain.py").write_text("import os\n\ndef read():\n    pass\n")
    for ok in (["solve"], ["Extra"], ["sub"], ["sub", "Grid"], ["_core", "Base"]):
        assert st.resolves(pkg, ok), ok
    for ok in (["plain"], ["plain", "read"]):
        assert st.resolves(pkg, ok), ok
    for bad in (["nope"], ["sub", "nope"], ["_hidden", "Secret"], ["plain", "nope"]):
        assert not st.resolves(pkg, bad), bad
    assert not st.resolves(pkg, ["solve", "attr"])


def test_description_length_reads_plain_and_folded_scalars() -> None:
    assert st.description_length("---\nname: a\ndescription: four\n---\nbody") == 4
    folded = "---\ndescription: >-\n  two\n  words\nname: a\n---\n"
    assert st.description_length(folded) == len("two words")


def test_marketplace_lists_each_plugin_with_the_plugins_it_builds_on(tree) -> None:
    for repo in ("base", "mid", "top"):
        manifest = tree.nodes[repo].plugin_dir / ".claude-plugin" / "plugin.json"
        manifest.write_text(json.dumps({"name": repo, "description": f"Use {repo}."}))
    urls = {repo: f"owner/{repo}" for repo in tree.nodes}
    plugins = json.loads(st.marketplace(tree, urls))["plugins"]
    assert [p["name"] for p in plugins] == [st.ROOT_PLUGIN, "base", "mid", "top"]
    assert plugins[0]["source"] == f"./plugins/{st.ROOT_PLUGIN}"
    by_name = {p["name"]: p for p in plugins}
    assert by_name["mid"]["source"] == {
        "source": "git-subdir",
        "url": "owner/mid",
        "path": "plugins/mid",
    }
    # a shelf is written in the role ladder, so it depends on the root plugin
    assert by_name["base"]["dependencies"] == [st.ROOT_PLUGIN]
    assert by_name["mid"]["dependencies"] == ["base"]
    # bare has no plugin, so top cannot depend on it
    assert by_name["top"]["dependencies"] == ["base", "mid"]


@pytest.mark.skipif(not HAVE_SUBMODULES, reason="the submodules are not checked out")
def test_the_tree_at_the_pins_passes_its_gates() -> None:
    graph = st.dg.build()
    tree = st.build(graph)
    assert st.violations(tree, graph.pkgs) == [], "python scripts/skill_tree.py"
    # a block quotes a parent's card, so every repository with a dependent has one
    missing = [
        repo
        for repo, node in tree.nodes.items()
        if tree.children(repo) and not node.units and not st.card_file(tree, repo)
    ]
    assert missing == [], "a repository that others build on has no card"
    assert not st.marketplace_stale(tree), "python scripts/skill_tree.py"
    assert {(r, e.parent) for r, es in tree.edges.items() for e in es} == {
        (e.source, e.target) for e in graph.edges
    }


def test_write_refuses_a_checkout_behind_its_default_branch(tmp_path) -> None:
    def git(cwd, *args) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
        )

    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    git(origin, "commit", "-q", "--allow-empty", "-m", "one")
    git(tmp_path, "clone", "-q", str(origin), "clone")
    node = st.Node(repo="lib", path=tmp_path / "clone", plugin=None, skills=())
    tree = st.Tree(nodes={"lib": node}, edges={})
    assert st.behind(tree) == []
    git(origin, "commit", "-q", "--allow-empty", "-m", "two")
    git(tmp_path / "clone", "fetch", "-q")
    assert st.behind(tree) == []  # nothing the generator reads changed
    (origin / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    git(origin, "add", "pyproject.toml")
    git(origin, "commit", "-q", "-m", "three")
    git(tmp_path / "clone", "fetch", "-q")
    assert st.behind(tree) == ["lib is behind its default branch in 1 files"]
    # a feature checkout's own changes are not the default branch's
    clone = tmp_path / "clone"
    git(clone, "checkout", "-q", "-b", "feature", "origin/main")
    (clone / "new.py").write_text("", encoding="utf-8")
    git(clone, "add", "new.py")
    git(clone, "commit", "-q", "-m", "four")
    git(origin, "commit", "-q", "--allow-empty", "-m", "five")
    git(clone, "fetch", "-q")
    assert st.behind(tree) == []
    # with --into, the pinned checkout that supplies the graph is judged too
    git(tmp_path, "clone", "-q", str(origin), "pin")
    (origin / "mod.py").write_text("", encoding="utf-8")
    git(origin, "add", "mod.py")
    git(origin, "commit", "-q", "-m", "six")
    git(tmp_path / "pin", "fetch", "-q")
    assert st.behind(tree, {"lib": tmp_path / "pin"}) == [
        "lib's pin is behind its default branch in 1 files"
    ]
