---
applyTo: "**"
---

# Code Review Instructions

When performing code review, use `/CODE_REVIEW.md` as the source of truth for:

- The review checklist (reuse and where code lives, configs, the DVC
  pipeline, reproducibility and data hygiene, notebooks / docs / citation,
  tests, Python / dependencies / security)
- The research-specific checks (settings from the config, one seed, no
  leakage from the test split, complete DVC stages)
- Output format and priority levels
- Suggestion type emojis and review tone

Key principles:
- Read "Boundaries", "Reuse before you write" and "The contracts" in
  `/AGENTS.md` first; the review procedure is
  `.claude/skills/myproject-review/SKILL.md`, with the checklists
  in `.claude/agents/reuse-reviewer.md` and
  `.claude/agents/reproducibility-reviewer.md`.
- Sacrifice *cleverness* for *clarity*. Sacrifice *brevity* for *explicitness*.
- Don't worry about formatting — CI (ruff format, pre-commit) handles that automatically.
- Be **constructive** and **specific**. Acknowledge good patterns with 👍.
- Every suggestion must include a concrete alternative, not just criticism.
