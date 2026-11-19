---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

The input folder holds the files of my branch, at their paths in the repo; it adds a first training pipeline (configs, two scripts, two DVC stages). Treat them as part of the repo. Sweep the regularisation strength over 0.01, 0.1 and 1.0 and give me a table of the three results that I can put in the docs.
