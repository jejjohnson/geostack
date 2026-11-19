---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

docs/guide/kernels.md compares krigax's two kernels in words and numbers. Write the specimen figure for it: the script docs/assets/figures.py, and the edit to the page that embeds the figure. From the library folder the script runs with `PYTHONPATH=src uv run -q --no-project --python 3.12 --with jax --with equinox --with lineax --with einx --with jaxtyping --with matplotlib --with cyclopts --with loguru python docs/assets/figures.py`. Save figures.py, the PNG it writes and the edited kernels.md.
