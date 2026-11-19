---
type: llm
focus: trace
weight: 1
---

Problem 1: `model` is a Hydra defaults group, not a key of configs/train.yaml; the fix is to list real keys under params and depend on the group's YAML file (or configs/) in deps
