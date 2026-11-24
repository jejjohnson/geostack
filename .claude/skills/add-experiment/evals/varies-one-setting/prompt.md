---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

The input folder holds the files of my branch, at their paths in the repo; it adds a first training pipeline (configs, two scripts, two DVC stages). Treat them as part of the repo. I want to rerun the training with a wind-speed prior of 5 m/s instead of 3 m/s and compare the two results. Set that up.
