# ① Primitive: the verb

**What it is.** One call on a noun, dispatched on the noun's structure: y = f(x; θ).

**Kind.** A function. Pure, arrays in and out, parameters passed on every call.

**Examples.** `solve`, `logdet`, `cholesky`, `diag`, `trace`; `to_pixel`, `crop`; in the methane example, `target_signature` and `pixel_area`.

**How to tell.** It has no state and knows nothing of the method that calls it. `pixel_area` knows about affine grids and nothing about plumes.

**How to explain it.** One equation and one line: what goes in, what comes out, what it costs. Say which structures give it a fast path.

**How it goes wrong.** A verb that takes a flag to change its numerics has swallowed an adverb; pass a strategy instead. "Primitive" is also not a synonym for "low-level": the rung someone treats as given is their floor.
