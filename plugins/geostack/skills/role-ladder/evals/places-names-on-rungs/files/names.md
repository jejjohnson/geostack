# Names to place

| Name | What it is |
|---|---|
| `Kronecker` | a class that holds two matrices A and B and stands for the linear operator A ⊗ B |
| `solve` | a function, `solve(operator, vector)`, that picks its method from the operator's type |
| `CGSolver` | a class with fields `rtol` and `max_steps`, passed as `solve(..., solver=CGSolver(rtol=1e-6))` |
| `randomized_svd` | a function that draws random probes and returns a low-rank factor of a matrix; it is called once, before the jitted loop |
| `Background` | a class with a `fit(radiance)` classmethod that stores the mean and the covariance of a scene |
| `matched_filter` | a function, `matched_filter(radiance, background, target)`, that returns the enhancement |
| `MatchedFilter` | a pipekit Operator built from a config (`target: ch4`, `band_dim: band`); it takes a GeoArray and returns a GeoArray |
| `Sequential` | a pipekit Operator that runs a list of Operators in order, and can be written as a `recipe.yaml` |
| `nightly_plumes` | an Airflow DAG that runs the plume `recipe.yaml` over every scene of the day |
| `AutoMatchedFilter` | a class whose `__call__(radiance, target)` estimates the scene's mean and covariance and then filters, in one go |
| `diffrax.Tsit5` | the Runge-Kutta scheme passed to an ODE solve |
| `UnscentedIntegrator` | a class with fields `alpha`, `beta`, `kappa` that says how a Gaussian expectation is computed |
