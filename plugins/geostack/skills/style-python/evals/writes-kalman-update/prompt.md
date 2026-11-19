---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

Add the Kalman filter analysis step to our filtering package, as kalman.py (it will live at myfilter/_src/kalman.py; the package's base error is MyfilterError in myfilter/_src/errors.py, which you can assume exists). Given a prior mean and covariance, a linear observation operator, the observation noise covariance and an observation vector, it returns the posterior mean and covariance. JAX and Equinox, in our house style. Save it as kalman.py.
