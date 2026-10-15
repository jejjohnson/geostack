"""Fail when a read-only submodule has been written to.

georeader (``references/geostack/georeader``) is spaceml-org's library, not
ours: geostack pins it and reads it, and nothing here edits it. This check is
a local pre-commit hook (``.pre-commit-config.yaml``) and a step of the Lint
workflow (``.github/workflows/lint.yml``). It fails when a read-only
submodule's working tree has modified, staged or untracked files, or when its
checkout carries a commit no remote branch of its own repository has (a local
commit on top of upstream; skipped in a shallow clone, which cannot tell).
Moving the pin to another upstream commit is allowed.

The other guards: ``.claude/settings.json`` denies edits there and
``.claude/hooks/guard_readonly.py`` refuses agent shell commands that write
there or open issues and pull requests on its repository; ``make install``
disables its push URL.

Usage::

    python scripts/check_readonly_submodules.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

#: submodules nothing in this repository may write to: {path: owner}
READ_ONLY: dict[str, str] = {
    "references/geostack/georeader": "spaceml-org/georeader",
}


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def problems() -> list[str]:
    """One line per read-only submodule that has been written to."""
    out = []
    for rel, owner in READ_ONLY.items():
        path = ROOT / rel
        if not (path / ".git").exists():
            continue  # not checked out: nothing can have been written
        dirty = _git(path, "status", "--porcelain")
        if dirty:
            files = ", ".join(line[3:] for line in dirty.splitlines()[:5])
            out.append(f"{rel} ({owner}) has local changes: {files}")
        if _git(path, "rev-parse", "--is-shallow-repository") == "true":
            continue  # a shallow clone cannot tell an old pin from a local commit
        remotes = _git(path, "branch", "-r", "--contains", "HEAD")
        known = _git(path, "for-each-ref", "--count=1", "refs/remotes")
        if known and not remotes:
            out.append(
                f"{rel} ({owner}) is checked out at a commit no remote branch has"
                " (a local commit?)"
            )
    return out


def main() -> int:
    found = problems()
    for line in found:
        print(line)
    if found:
        print(
            "These submodules are read-only: they belong to someone else. Set the"
            " changes aside (git -C <path> stash -u), restore the pin"
            " (git submodule update <path>) and propose them upstream instead."
        )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
