---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

For a methods section I need each of these placed on our research-to-production cycle: which step it is, what it is trained on if anything, and what it has to be validated against. (a) a Gaussian puff transport model; (b) an ensemble Kalman filter run on a shallow-water model; (c) a Fourier neural operator trained on runs of that shallow-water model; (d) 4D-Var with that neural operator as the forward model; (e) a normalising flow that maps a radiance spectrum to a posterior over the methane column, trained on radiative-transfer simulations; (f) a matched-filter methane retrieval; (g) a U-Net that segments plumes, trained on masks drawn by human annotators; (h) retraining the neural operator after the shallow-water model gets a new drag term. Save it as placement.md.
