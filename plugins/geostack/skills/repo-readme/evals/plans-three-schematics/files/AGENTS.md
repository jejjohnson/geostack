# AGENTS.md

krigax is not on PyPI yet; install from GitHub (`pip install "krigax @ git+https://github.com/jejjohnson/krigax"`).

```bash
make install     # uv sync --all-groups
make test        # uv run pytest -v
make lint        # uv run ruff check .
make format      # uv run ruff format .
```

Tests run in float64 (`jax.config.update("jax_enable_x64", True)` in `tests/conftest.py`).
