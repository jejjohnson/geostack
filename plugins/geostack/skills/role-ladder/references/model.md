# ③ Model: the noun phrase

**What it is.** A noun with meaning and state: an object that holds fields θ and state ψ, where ψ is built once by a setup step, ψ = setup(X, u; θ).

**Kind.** A Module. Setup may do work that cannot be traced; the call afterwards is pure and jittable.

**Examples.** `MultivariateNormal`, `LGSSM`, `GaussianExpFam`, `BilinearPlan`; in the methane example, `Background`, which holds the mean and covariance fitted on a scene.

**How to tell.** It means something ("a Gaussian with this mean") and it remembers something between calls. Two algorithms can share it.

**How to explain it.** Say what it represents, what its state is, and what builds the state. Then list which algorithms use it: sharing is the reason it exists. `Background` gives the matched filter its solve and the segmentation its noise level.

**How it goes wrong.** A model that also runs the method is a model and an algorithm in one object; split them and the second user of the state appears. A model never holds or takes a carrier.
