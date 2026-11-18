# One job, up the ladder

The job: from an EMIT radiance scene, retrieve a methane enhancement, segment the plume, and estimate its emission rate.
Written as four lifted functions, each step owns everything it needs and nothing is shared.
Rebuilt the way a library is built, the same job climbs the ladder, and most pieces are used by more than one piece above them.

| Piece | Rung | Kind | Used by |
|---|---|---|---|
| `GeoArray`, `SpectralAxis`, a PSD operator | ⓪ structure | Module, carrier | every rung above: the scene and the wind are carriers, Σ is a lineax operator |
| `target_signature` | ① primitive | function | the retrieval step, and the background's noise |
| `pixel_area` | ① primitive | function | `emission_rate`, twice: the plume's mass and its length |
| `large_components` | ① primitive | function | `segment` |
| `air_mass_factor`, `valid_mask`, `interp_at` | ① primitive | function | the retrieval step and the wind step |
| `LedoitWolf`, `SigmaThreshold` | ② strategy | Module | `Background.fit` and `segment`; swap in another estimator or rule |
| `Background` | ③ model | Module | `matched_filter` (its solve) and the noise σ_y, which sets the segmentation threshold |
| `matched_filter`, `segment`, `emission_rate` | ④ algorithm | function | one step each |
| `MethaneRetrieval`, `PlumeSegmentation`, `WindAt`, `EmissionRate` | ⑤ step | Operator | the recipe; each also runs alone |
| `plume_recipe` | ⑥ recipe | Operator, a Graph | orchestration: a DVC stage, an endpoint, a sweep over the threshold |

What the climb shows.

- The matched filter as one object splits into a model (`Background`, the fitted state) and an algorithm (`matched_filter`, the call). The split is what lets segmentation reuse the noise level σ_y.
- The wind speed was a parameter of the last function; as a step input it is wired, so a graph can connect it.
- The recipe is a graph, not a chain, because the enhancement feeds both segmentation and quantification.
