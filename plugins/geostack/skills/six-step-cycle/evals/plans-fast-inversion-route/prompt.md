---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

Estimating an emission rate from one methane scene with MCMC on the plume model in this library takes far too long for the volume of scenes we get; I want an answer with uncertainty in milliseconds. Lay out the route from the model we have to that fast inference: the stages in order, what each learned piece is trained on, what each stage is checked against before the next one starts, and which of it this library already ships. No code yet. Save it as route.md.
