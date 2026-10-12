---
name: adopt-template
description: Turn a fresh copy of the research project template into your own project — rename the example package (myproject) and the distribution / project names everywhere, replace the example code, configs, pipeline stages and tests, rewrite every passage marked `template-example:` in the agent files and the example building skills, regenerate the capability index, update the release, citation, docs and label metadata, and verify. Use right after creating a repository from the template, or when asked to rename the package or "make this template mine".
---

# Adopt the template

Work on a branch, step by step, and run the checks at the end; commit when
they pass. Pick the names first:

| Name | Example | Where it appears |
|---|---|---|
| Import name | `myproject` | `src/myproject/`, imports, `PKGROOT`, `PACKAGE` in `scripts/capabilities.py`, `dvc.yaml` deps, docs |
| Distribution name | `my-research-project` | `pyproject.toml`, `pixi.toml`, `release-please-config.json`, `pixi.lock` |
| Title | `My Research Project` | `README.md`, `myst.yml`, `CITATION.cff`, `docs/index.md`, the package docstring |
| GitHub `owner/repo` | `your-org/your-repo` | README badges and links, `CITATION.cff`, `docs/tutorials/getting_started.md` |
| Author | `Jane Doe`, `Your Name` | `myst.yml`, `CITATION.cff`, `LICENSE`, `README.md` |

## 1. Find every reference

```bash
grep -rnI --exclude-dir={.git,.venv,.pixi,_build} \
  -e myproject -e my-research-project -e "My Research Project" \
  -e your-org -e your-repo -e "Jane Doe" -e "Doe, Jane" -e "Your Name" .
```

## 2. Rename

```bash
NEW=coastal_sst              # import name
DIST=coastal-sst             # distribution name
TITLE="Coastal SST"

git mv src/myproject "src/$NEW"
files() {
  grep -rlI --exclude-dir={.git,.venv,.pixi,_build} -e "$1" . \
    | grep -v -e '^./CHANGELOG.md$' -e '^./pixi.lock$' -e '^./uv.lock$' -e '^./.claude/skills/adopt-template/'
}
files myproject | xargs sed -i "s/myproject/$NEW/g"
files my-research-project | xargs sed -i "s/my-research-project/$DIST/g"
files "My Research Project" | xargs sed -i "s/My Research Project/$TITLE/g"
```

(On macOS use `sed -i ''`.) Then by hand:

- `your-org/your-repo`, the author and affiliation in `README.md`,
  `CITATION.cff`, `myst.yml`, `LICENSE` and
  `docs/tutorials/getting_started.md`;
- `CHANGELOG.md`: empty it (keep the `# Changelog` heading) and set
  `.release-please-manifest.json` and the versions in `pyproject.toml`,
  `pixi.toml`, `CITATION.cff` and `src/<NEW>/__init__.py` to your starting
  version — release-please manages them from then on;
- `pixi.lock`: regenerate, never edit (`pixi lock`); `uv.lock` is
  gitignored (pixi is the lockfile of record);
- `.devcontainer/devcontainer.json` `name`, the README badges and
  description, `references.bib`.

## 3. Replace the example

The example package, configs, stages and notebooks are placeholders:

- `src/<NEW>/data/loading.py`, `models/baseline.py`,
  `models/transformer.py`, `trainers/trainer.py` and their tests
  (`tests/test_data.py`, `tests/test_models.py`, `tests/test_configs.py`):
  replace them with your code as you write it (the `add-dataset` and
  `add-model` skills); delete what you do not need.
  `utils/reproducibility.py` (`seed_everything`) is generic: keep it.
- `configs/model/*.yaml`, `configs/data/*.yaml`, `configs/train.yaml`:
  your groups and keys.
- `dvc.yaml` and `scripts/`: your stages (the `add-experiment` skill).
  The example pipeline runs as shipped on an empty `data/raw/`
  (`pixi run dvc repro`); keep its conventions as you replace it: no
  git-tracked file inside a stage output, `params` naming real keys, and
  the `.gitignore` exceptions for the `.dvc` pointers and `cache: false`
  metrics DVC expects in git.
- `notebooks/`, `marimo_notebooks/`, `docs/notes/*.md`: your analyses and
  write-up.

## 4. Rewrite the agent files

- List the passages to rewrite:

  ```bash
  grep -rnE '(<!--|#) template-example:' --exclude-dir={.git,.venv,.pixi} .
  ```

  Each marks a passage about the example (`AGENTS.md`: the project map,
  the reuse table, the test table; `CODE_REVIEW.md`: the examples;
  `scripts/capabilities.py`: `PACKAGE`; the building skills). Rewrite each
  from your code, then delete the marker; when the grep prints nothing,
  also delete the "Template note" at the top of `AGENTS.md` and the
  template bullet in `CLAUDE.md` and `.github/copilot-instructions.md`.
- Rewrite the building skills in `.claude/skills/` (`add-experiment`,
  `add-dataset`, `add-model`, `add-notebook`, `reproduce-results`,
  `write-up-results`) for your project's extension points: real files,
  config groups, stages and test files. Add a skill for each new kind of
  thing people will add; delete the ones that do not apply.
- Rename `.claude/skills/myproject-review/` to
  `<your-project>-review/` (and its `name:`), and update the names in the
  "Recipes" table of `AGENTS.md` and in `CLAUDE.md`.
- Point the two subagents in `.claude/agents/` at your domain: add your
  package's shared helpers to `reuse-reviewer`, and your field's pitfalls
  (units, leakage patterns, evaluation protocol) to
  `reproducibility-reviewer`.
- If you build on libraries from your own stack, list them in `UPSTREAM`
  in `scripts/capabilities.py`.
- Delete this skill once the project is adopted.

## 5. Regenerate and verify

```bash
uv sync --all-groups
make capabilities                       # rewrites docs/api/capabilities.md
make test
uv run --group lint ruff check .
uv run --group lint ruff format --check .
make typecheck
pixi lock && pixi install && pixi run test
pixi run dvc dag
make docs
```

…and the grep of step 1 prints nothing outside `CHANGELOG.md` and this
skill.

## 6. GitHub

- `make gh-labels` creates the label taxonomy (`.github/scripts/create-labels.sh`;
  edit its `area:*` labels for your project first).
- Settings → Pages → Source: GitHub Actions (for `pages.yml`); add a DVC
  remote (`pixi run dvc remote add -d <name> <url>`, credentials in
  `.dvc/config.local`) so `dvc push` / `pull` and the reproducibility
  workflow can reach the data.
- Connect Zenodo for a DOI per release, and add it to `CITATION.cff`.
