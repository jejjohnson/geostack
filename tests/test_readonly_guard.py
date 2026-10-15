"""The guards that keep georeader, a third-party submodule, read-only."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".claude" / "hooks" / "guard_readonly.py"
GEOREADER = "references/geostack/georeader"


def _hook():
    spec = importlib.util.spec_from_file_location("guard_readonly", HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "command",
    [
        f"git -C {GEOREADER} commit -am fix",
        f"git -C {GEOREADER} push origin main",
        f"cd {GEOREADER} && git add . && git commit -m x",
        f"cd {GEOREADER}; gh pr create --fill",
        f"echo x > {GEOREADER}/georeader/io.py",
        f"sed -i s/a/b/ {GEOREADER}/README.md",
        f"cp patch.py {GEOREADER}/georeader/",
        "gh issue create --repo spaceml-org/georeader --title dedupe",
        "gh pr comment 12 -R spaceml-org/georeader --body hi",
        "gh api repos/spaceml-org/georeader/issues -X POST -f title=x",
    ],
)
def test_refuses_writes(command: str) -> None:
    assert _hook().refusal(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        f"git -C {GEOREADER} log --oneline -5",
        f"git -C {GEOREADER} status --porcelain",
        f"git -C {GEOREADER} stash -u",
        f"git submodule update {GEOREADER}",
        f"git add {GEOREADER} && git commit -m 'chore: move the georeader pin'",
        f"grep -rn read_srf {GEOREADER}/georeader",
        "gh issue list --repo spaceml-org/georeader",
        "gh api repos/spaceml-org/georeader/issues",
        "git -C references/geostack/geotoolz commit -m ours",
    ],
)
def test_allows_reads_and_the_pin(command: str) -> None:
    assert _hook().refusal(command) is None


def test_hook_exit_codes() -> None:
    def run(command: str) -> int:
        event = {"tool_name": "Bash", "tool_input": {"command": command}}
        return subprocess.run(
            [sys.executable, str(HOOK)],
            input=json.dumps(event),
            text=True,
            capture_output=True,
            check=False,
        ).returncode

    assert run(f"git -C {GEOREADER} push") == 2
    assert run("git status") == 0


def test_settings_deny_edits() -> None:
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    assert f"Edit(/{GEOREADER}/**)" in settings["permissions"]["deny"]
