# Notes for whoever writes the README

- Docs site: https://jejjohnson.github.io/krigax (pages: `/quickstart/`, `/reference/`). There is no design page and no tutorials yet.
- Who uses it: plumax imports `GaussianProcess`, `fit` and `predict` (`from krigax import GaussianProcess, fit, predict`) for its background-wind interpolation. pyrox imports `RBF` and `Matern32` only through its `krigax` extra. Nothing else imports it.
- What it builds on: lineax (the operators and the solvers), equinox (the Modules), einx (the contractions), JAX.
- The three README figures have not been drawn in geostack yet.
- No benchmark has been run.
