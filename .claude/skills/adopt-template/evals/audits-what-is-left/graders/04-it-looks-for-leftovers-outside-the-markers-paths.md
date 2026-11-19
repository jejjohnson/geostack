---
type: llm
focus: trace
weight: 1
---

It looks for leftovers outside the markers (paths of the template's example, such as src/ or tests/test_models.py, named in .github/ or the agent files), checks them against what exists, and says the repo has no local package by design
