---
type: llm
focus: trace
weight: 1
---

Problem 2: the stage depends on `outputs`, Hydra's run directory, which changes on every run; the fix is to take it out of deps
