# Proposal: `obsfit`, a small retrieval library

Five public names.

```python
class RidgeSolver(eqx.Module):
    """How the normal equations are solved."""
    rtol: float
    train_radiance: Float[Array, "sample channel"]   # kept so the preconditioner can be rebuilt

class Smoother(eqx.Module):
    """The spectral smoothing model."""
    width: float
    order: int

class Retrieve(eqx.Module):
    """The retrieval algorithm."""
    background: Background | None = None

    def __call__(self, radiance, target, *, use_cg=False, precondition=False,
                 shrink=True, shrink_target="identity", robust=False):
        if self.background is None:
            background = Background.fit(radiance, shrink=shrink)
        ...

class RetrieveStep(Operator):
    """One retrieval, ready to chain."""
    target: GeoArray        # the target signature, set when the step is built
    band_dim: str = "band"

    def __call__(self, scene: GeoArray) -> GeoArray:
        radiance = scene.data
        radiance = jnp.where(scene.mask, radiance, 0.0)
        return scene.with_data(matched_filter(radiance, self.target.data))

class DailyRetrieval(Operator):
    """The recipe."""
    steps: list[Operator]
    start: str
    end: str

    def __call__(self):
        for day in date_range(self.start, self.end):
            scene = read(day)
            out = self.steps[0](scene)
            out = out.where(out > 3 * out.std())     # keep the strong pixels
            out = self.steps[1](out)
            write(out, day)
```

`Background` and `matched_filter` already exist. Throughout the docs we call `Retrieve` "the primitive of the library", since everything else is built on it.
