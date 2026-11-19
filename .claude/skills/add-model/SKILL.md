---
name: add-model
description: Add a model to the research project — found in an upstream library through the capability indexes or added there (geostack has no package of its own), then a Hydra config-group option (configs/model/<name>.yaml) whose keys are the class's constructor arguments, the entry script that builds it from the config, and a test that the option builds. Use when asked to add, port or implement a model, baseline, architecture or estimator, or a new model option in configs/model.
---

# Add a model

Read "Boundaries" and contracts 1 and 2 in `AGENTS.md` first; this is the
step-by-step.

**Where this repo stands.** geostack has no package of its own, so there is
no models folder here and no class is written here. A model class is
reusable code: it lives in the upstream library that owns the concept.
What this repo holds for a model is a config option, the wiring in an
entry script, and a test. Until the first training pipeline lands there is
no `configs/` directory either: look before you assume a file exists
(`ls configs scripts`), and say in your report what you had to create.

## 1. Make sure it does not exist yet

- Search the capability indexes, one per layer: `docs/api/capabilities.md`
  (GeoStack), `capabilities_numerics.md`, `capabilities_geoml.md` (kernellib,
  geonnax, pyrox, filterax, vardax, …) and `capabilities_geomodels.md`. Say
  in your report which file you searched, for what, and what you found.
  kernellib's `KRR` is there, for instance, so a kernel ridge baseline is
  not written.
- A model the stack has is used from the stack, before a third-party class
  that does the same.
- `configs/model/`, if it exists: a model that differs only in its settings
  (a size, a kernel, a lengthscale) is a new option for the same class, or
  a command-line override, not a new class or wrapper.

## 2. The class lives upstream

- **It exists upstream:** nothing to write. Go to step 3.
- **It does not exist:** the class goes into the library that owns the
  concept, in the lowest layer that may hold it (`AGENTS.md`, "The layers"),
  with its tests, its docstring (the model's equation in Unicode and its
  reference) and its entry in that library's capability index. Name the
  library and its layer. That is a change in another repository: describe
  it, or open the issue there if asked. Do not draft the class in this
  repo: not in a script, not in a notebook, not in a new `src/` folder and
  not in a "draft" file.
- Until the class lands upstream and the submodule pin moves, a config
  option that names it cannot build. Say so; do not commit an option whose
  `_target_` does not exist.

## 3. The config (`configs/model/<name>.yaml`)

```yaml
_target_: kernellib.KRR
name: krr
kernel:
  _target_: kernellib.RBF
  lengthscale: 1.0
regularization: 1.0e-3
```

- `_target_` is the upstream class, `name` is the option's name, and every
  other key is a constructor argument, with the class's own default. An
  argument that is itself an object (a kernel) is a nested `_target_`.
- Check the keys against the real class: its signature in the submodule
  (`references/<layer>/<library>`), or its row in the capability index.
  When the submodule is not checked out and the index does not show the
  arguments, say the keys are not checked and mark them as to be confirmed;
  do not guess a spelling (`matern52` or `Matern(nu=2.5)`) and present it
  as right.
- Choose it with `model=<name>`, or make it the default in the `defaults`
  list of the entry config.

## 4. Wire it into training

The entry script builds whatever option is chosen from its `_target_`,
with every key but `name`, so a new option needs no change to the script:

```python
def build_model(model_cfg: DictConfig):
    kwargs = {k: v for k, v in OmegaConf.to_container(model_cfg).items() if k != "name"}
    return instantiate(kwargs)
```

- A constructor argument that is not a config value (the data's input
  size, a PRNG key) comes from the script, passed as a keyword of
  `instantiate`.
- Check that the new model answers the calls the script makes on the
  others (`fit` and `predict`, or a call on an array of a given shape). A
  model on 2-D fields does not drop into a script that feeds flat feature
  vectors: say what has to change.
- An entry point written with hydra-zen builds its config from the class
  (`builds(TheClass, populate_full_signature=True)`).

## 5. Tests

- The model's own tests are upstream, with the class.
- Here, one test builds every option in `configs/model/` (create
  `tests/test_configs.py` with the first option): compose the config with
  `model=<name>` and instantiate it as the script does. It starts with
  `pytest.importorskip("hydra")`, so it runs under pixi, in the environment
  of the layer the model comes from.

## 6. Pipeline and docs

- The training stage depends on `configs/`, so a new option or a changed
  key reruns it. A stage also depends on the upstream code it imports:
  the submodule folder of the library (`references/<layer>/<library>`), or
  `pixi.lock`.
- A library from a layer above GeoStack runs in that layer's pixi
  environment (`pixi run -e geoml …`, `-e numerics`, `-e geomodels`).
- Describe the model in `docs/notes/methods.md` (the `write-up-results`
  skill), with its reference in `references.bib`.

## 7. Verify

```bash
pixi run -e <layer> pytest tests/test_configs.py -v -o addopts=   # every model= option builds
pixi run -e <layer> python scripts/<entry>.py model=<name>        # a short run, by hand, to see it start
pixi run dvc repro <stage>                                        # the run that counts
```

Then the `pre-pr-check` skill.

## 8. Report

Say which index you searched and what it holds, whether the class exists
upstream or has to be added there (and where), which keys you checked
against the class and which are to be confirmed, what you ran and what
you did not run and why, and what the user has to do next.
