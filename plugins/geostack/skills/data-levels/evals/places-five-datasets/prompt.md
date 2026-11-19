---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

I'm writing the data section of a project note and need to say where each of these sits in our stack: (a) per-pixel methane columns retrieved from a shortwave-infrared imaging spectrometer, with gaps where the retrieval failed; (b) daily gridded sea-level-anomaly maps made by optimal interpolation of along-track altimetry, which the data provider's catalogue labels "L4"; (c) temperature and salinity profiles from Argo floats; (d) 2 m air temperature from an atmospheric reanalysis; (e) a map of 20-year return levels of marine heatwave intensity. For each one give its level, the carrier it travels on, and which of our packages reads it, indexes it and transforms it. Save the answer as placement.md.
