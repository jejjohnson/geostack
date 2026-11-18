---
name: role-ladder
description: The role ladder of the GeoStack — structure, primitive, strategy, model, algorithm, step, recipe, with orchestration above — read as grammar (nouns, verbs, adverbs, noun phrases, sentences, paragraphs). Use to explain an algorithm, a method or a library by the roles its pieces play (for explainers, walkthroughs, docs and READMEs), to say which rung a name belongs on, to design or review an API so each piece does one job, or whenever another skill names a rung such as "a strategy" or "a recipe".
---

# The role ladder

Draft 1 · 12 Nov 2026 · from GeoStack's kinds-and-roles design page (`docs/design/kinds_and_roles.md` and `docs/design/pages/kinds-and-roles.html` @ `343d60e`). Where this file and that page disagree, the page wins.

Every library in the stack exposes names that do different jobs.
The ladder names the jobs, and it reads as grammar: structures are the nouns, primitives the verbs that act on them, strategies the adverbs that say how, models the noun phrases, algorithms the sentences, steps a sentence with its own settings, and recipes the paragraphs.

## 1. The ladder

| Rung | Role | Grammar | What it is | Kind | Examples |
|---|---|---|---|---|---|
| ⓪ | structure | noun | what the verbs act on: the data, and the library's own objects | Module, or a third-party array | `Kronecker`, `LowRankUpdate`, `RBF`, `AffineGrid`; `GeoArray`, `DataArray` |
| ① | primitive | verb | one call, dispatched on the structure | function | `solve`, `logdet`, `cholesky`, `to_pixel`, `crop` |
| ② | strategy | adverb | how to compute: settings only, no data | Module | `CGSolver`, `NystromPreconditioner`, `UnscentedIntegrator`, `diffrax.Tsit5` |
| ③ | model | noun phrase | an object with meaning and fitted state | Module | `MultivariateNormal`, `LGSSM`, `BilinearPlan`, `Background` |
| ④ | algorithm | sentence | a method over models, as a function: do this, to that, this way | function | `kalman_filter`, `sparse_conditional`, `etkf_transform`, `matched_filter` |
| ⑤ | step | a sentence with its settings | one transformation, ready to chain, with a config | Operator | `MatchedFilter`, `Reproject`, `JaxModelOp` |
| ⑥ | recipe | paragraph | steps in order, with their settings, making one job | Operator | `Sequential`, `Graph`, `DACycle`, `TrainingLoop`, a `recipe.yaml` |

Two things sit beside the ladder and are not rungs.

- **Setup** builds state from data, once, outside `jit`: a model's state, or a structure such as a low-rank factor (`randomized_svd`, `rp_cholesky`, a k-NN graph).
- **Orchestration** runs recipes over many items, on a schedule or behind an API (DVC, Airflow, FastAPI). It runs a recipe without changing it.

One file per role, in `references/`: `structure.md`, `primitive.md`, `strategy.md`, `model.md`, `algorithm.md`, `step.md`, `recipe.md`, `orchestration.md`.
Read the one you need; each says what the role is, how to tell it apart, how to explain it and how it goes wrong.
`references/kinds.md` has the three kinds and what each may take, and `references/worked-example.md` has one job climbed up the whole ladder.

## 2. Which rung is it?

Ask in this order, and stop at the first yes.

1. Does it run other things on a schedule, over many items or behind an API? It is orchestration, not a rung.
2. Does it compose steps into one job, in code or as data? ⑥ recipe.
3. Is it one transformation with its own config, ready to chain? ⑤ step.
4. Is it a function that takes a model and says what to do with it? ④ algorithm.
5. Is it an object that holds meaning and fitted state? ③ model.
6. Does it hold only settings for how something is computed, and no data? ② strategy.
7. Is it one call on an object, with no state of its own? ① primitive.
8. Is it the thing the calls act on? ⓪ structure.

A name sits on exactly one rung.
When it seems to sit on two, it is two things packaged together, and the explanation gets clearer once they are split: a matched filter that fits its own background is a model (the background) and an algorithm (the filter).

## 3. Explaining with the ladder

Use this whenever asked to explain an algorithm, a method or a library.

1. **Start from the sentence.** State the algorithm in one line: do this, to that, this way.
2. **Name the nouns.** What does it act on? Give each structure its shape and its unit.
3. **Name the verbs.** Which primitives does it call on those nouns? One line each, with the equation.
4. **Name the adverbs.** What can the caller swap without changing the maths? Those are the strategies.
5. **Name the noun phrases.** What is fitted or held between calls? That is the model, and say what its state is.
6. **Return to the sentence.** Now write the algorithm as a function over the model, using only the verbs and adverbs already named.
7. **Go up only if the reader works there.** A step wraps the algorithm with settings; a recipe chains steps. Leave them out for a reader who lives below them.

Lay the pieces out in one table, because it shows what is shared:

| Piece | Rung | Kind | Used by |
|---|---|---|---|
| a PSD operator Σ | ⓪ structure | Module | every rung above |
| `target_signature` | ① primitive | function | the filter, and the background's noise |
| `LedoitWolf` | ② strategy | Module | `Background.fit`; swap in another estimator |
| `Background` | ③ model | Module | the filter (its solve) and the segmentation threshold |
| `matched_filter` | ④ algorithm | function | one step |

`references/worked-example.md` has the whole table for this job.

Three habits keep an explanation honest.

- **Pitch it at the reader's band (§4) and treat the rung below as given.** Say what the floor guarantees in one line, and do not open it.
- **Go one rung down, never two.** If the reader needs more, name the rung below and stop there.
- **Say what is swappable.** Every adverb named is a claim that the maths does not change when it is replaced. Check that claim.

## 4. Who works where

Each kind of user works in a band of the ladder and treats the rung below it as atomic: their floor.

| Who | Works at | Their floor | Speaks in |
|---|---|---|---|
| data engineers | orchestration, ⑥ | recipes, often as data | carriers and files |
| geoscience researchers | ⑤, ⑥ | steps | band names, CRS, fill values |
| ML researchers | ②, ③, ④ | primitives | shapes |
| numerics and CS | ⓪ to ④ | JAX, lineax, diffrax | shapes |

## 5. Rules that follow

- **The role fixes the kind.** ① and ④ are functions; ⓪, ② and ③ are Modules; ⑤ and ⑥ are Operators. Module is not a rung; it is how three rungs are packaged.
- **The role is not the import order.** Inside a library imports mostly follow the roles upward, but verbs dispatch on nouns and some adverbs call verbs. Explain in grammar order anyway: nouns, verbs, adverbs.
- **"Primitive" names one role.** Use "function" for the kind and "floor" for the rung someone treats as given.
- **Operator with a capital O is pipekit's kind.** "Linear operator" (a ⓪ structure) and "observation operator" keep their field meanings.
- **A library need not have every rung.** gaussx stops at ④.

## 6. Checklist

- [ ] Every piece named sits on one rung, and the rung is stated.
- [ ] Nouns come before verbs, verbs before adverbs.
- [ ] Each adverb is something a caller can really swap.
- [ ] The model's state is said in words, with what builds it.
- [ ] The explanation stays in one band and goes at most one rung below it.
- [ ] The piece, rung, kind and used-by table is there when more than three pieces are named.
