# The three kinds, and what each may take

The role fixes the kind, so a contract is written once per kind.

| Kind | Rungs | Form | Contract |
|---|---|---|---|
| function | ①, ④ | y = f(x, u; θ) | pure; arrays in and out; parameters passed on every call; never sees a name, a CRS or a fill value, which is what lets it run under `jit` |
| Module | ⓪, ②, ③ | ψ = setup(X, u; θ), then y = f(x, u; θ, ψ) | fields θ and state ψ; setup runs once and may do work that cannot be traced; the call is pure and jittable; never holds or takes a carrier |
| Operator | ⑤, ⑥ | h[f, θ](x_geo; α) = ι_α(f(π_α(x_geo); θ), x_geo) | unwrap with π_α, compute on plain arrays, rewrap with ι_α; the config α is names and settings only and never reaches f |

Here x is the data, u is what the function needs to know about where the data sits (pixel areas, a transform), θ are parameters, ψ is fitted state, and α is the config.

Three invariants hold across the stack.

1. **Arrays go everywhere.** Every kind takes them.
2. **Carriers stop at GeoStack.** An Operator on carriers unwraps before it calls down; nothing below takes a carrier as input.
3. **Configs stop at Operators.** Hydra and JSON configs build Operators and apps; Modules and functions take plain values.

The data axis has three values: a plain array, a named array (names and coordinates per axis) and a carrier (a named array with a georeference).
