---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

We have per-pixel methane columns from a satellite (already retrieved, with gaps under cloud) and a handful of ground-based column measurements at fixed sites. The goal is an hourly gap-free methane field and, from that, emission rates per source. Lay out the pipeline for me in terms of our stack: the stages in order, what each one turns into what, which package does the computing at each stage, where the data changes carrier, and where the ground sites come in. I don't want code yet. Save it as pipeline.md.
