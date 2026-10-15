"""PreToolUse hook: refuse agent shell commands that write to a read-only submodule.

The read-only submodules are listed in ``READ_ONLY`` in
``scripts/check_readonly_submodules.py`` (today only georeader, which belongs
to spaceml-org). This hook refuses a Bash command that

- runs a git command that writes, or a gh command, inside one of them
  (``git -C <path> …``, ``cd <path> && …``);
- writes a file inside one of them (redirection, ``sed -i``, ``tee``, ``cp``,
  ``mv``, ``rm``, ``touch``, ``patch``);
- opens, edits or comments on an issue, a pull request or a release of its
  repository, or calls ``gh api`` on it with a writing method.

Reading is allowed, and so is moving the pin from the parent repository
(``git add references/geostack/georeader``, ``git submodule update``). It
matches command text, so it guards against slips, not against someone set on
getting round it; ``scripts/check_readonly_submodules.py`` in pre-commit and
CI is the backstop. Exit code 2 refuses the command and shows the reason.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts"))

from check_readonly_submodules import READ_ONLY


GIT_WRITES = (
    r"commit|push|add|rm|mv|am|apply|rebase|merge|cherry-pick|revert|tag"
    r"|reset|commit-tree|update-ref|remote|config"
)
GH_WRITES = r"(issue|pr|release)\s+(create|edit|comment|close|reopen|merge|review)"
API_WRITES = r"(-X|--method)\s*(POST|PATCH|PUT|DELETE)|\s-(f|F)\s|--field|--input"
FILE_WRITES = r"(sed\s+-i|\btee\b|\bcp\b|\bmv\b|\brm\b|\btouch\b|\bpatch\b|>)"


def refusal(command: str) -> str | None:
    """Why ``command`` is refused, or ``None`` when it may run."""
    for rel, owner in READ_ONLY.items():
        path = re.escape(rel)
        inside = rf"(git\s+-C\s+\S*{path}/?(\s|$)|\bcd\s+\S*{path}/?(\s|;|&|$))"
        if re.search(inside, command) and (
            re.search(rf"\bgit\b[^;&|]*\s({GIT_WRITES})\b", command)
            or re.search(r"\bgh\s", command)
        ):
            return f"{rel} belongs to {owner}: no git writes or gh calls inside it"
        target = rf"{path}/\S"
        if re.search(target, command) and re.search(
            rf"{FILE_WRITES}[^;&|]*{target}", command
        ):
            return f"{rel} belongs to {owner}: no file inside it may be written"
        if owner in command and (
            re.search(rf"\bgh\s+{GH_WRITES}\b", command)
            or (re.search(r"\bgh\s+api\b", command) and re.search(API_WRITES, command))
        ):
            return f"{owner} is not ours: no issues, pull requests or API writes there"
    return None


def main() -> int:
    event = json.load(sys.stdin)
    if event.get("tool_name") != "Bash":
        return 0
    reason = refusal(event.get("tool_input", {}).get("command", ""))
    if reason is None:
        return 0
    print(
        f"{reason}. It is read-only in geostack (AGENTS.md, Boundaries); propose"
        " the change to its maintainers instead.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
