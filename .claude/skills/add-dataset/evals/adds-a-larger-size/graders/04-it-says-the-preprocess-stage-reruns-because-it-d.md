---
type: llm
focus: trace
weight: 1
---

It says the preprocess stage reruns because it depends on configs/ (or reads the config), and that the result is produced with `dvc repro`, not by running the script by hand
