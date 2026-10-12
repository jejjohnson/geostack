---
applyTo: "src/**/*.py,tests/**/*.py,scripts/**/*.py"
---

# Python Coding Standards

## Modern Python (3.12+)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Built-in generics: `list[int]`, `dict[str, Any]` not `List[int]`, `Dict[str, Any]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- `dataclasses` or `attrs` for data containers
- `Enum` for fixed sets of constants
- Context managers (`with` statements) for resource handling
- Specific exception types (never bare `except:`)
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

## Where code goes

- Reusable logic lives in the package under `src/`, takes plain arguments
  (never a Hydra `DictConfig`) and has a test; `scripts/` read the config,
  call the package and write outputs (see "Boundaries" in `/AGENTS.md`).
- Every setting a run depends on is a key in `configs/`, not a literal.
- Seed once per run from the config; pass a `numpy.random.Generator` down
  rather than drawing from global state.

## Package Preferences

A runtime dependency goes into `pixi.toml` (and `pixi.lock` is
regenerated); never `pip install` / `conda install` by hand.

| Purpose | Preferred Package |
|---------|-------------------|
| Configuration | `hydra-core` / `omegaconf`, `hydra-zen` |
| Data and pipeline versioning | `dvc` |
| Labelled arrays, NetCDF | `xarray`, `netcdf4` |
| Data containers | `dataclasses` (stdlib) or `attrs` |
| Logging | `loguru` |
| Path handling | `pathlib` (stdlib) |
| HTTP | `httpx` |
| Testing | `pytest` |

## Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for all public APIs (Google style)
- Inline comments explaining *why*, not *what*
- Scientific algorithms should include Unicode equations in docstrings (e.g. `# σ² = Σ(xᵢ − μ)² / N`)
- Public classes and functions include a short usage example where it helps a reader
