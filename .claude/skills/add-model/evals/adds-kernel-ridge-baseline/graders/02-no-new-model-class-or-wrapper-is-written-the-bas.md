---
type: llm
focus: trace
weight: 1
---

No new model class or wrapper is written: the baseline is a config option (configs/model/<name>.yaml) with `_target_` set to the upstream class, a `name`, and its constructor arguments as keys
