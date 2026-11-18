# ⑤ Step: a sentence with its own settings

**What it is.** One transformation, ready to chain: an Operator h[f, θ] that lifts a function onto a carrier with a config α. It unwraps the carrier, calls down the ladder on plain arrays, and rewraps.

**Kind.** An Operator. Its config is names and settings only, so it round-trips through JSON.

**Examples.** `MatchedFilter`, `Reproject`, `RGBRecipe`, `JaxModelOp`; in the methane example, `MethaneRetrieval`, `PlumeSegmentation`, `WindAt` and `EmissionRate`.

**How to tell.** It carries its own settings and can run alone in a notebook or inside a recipe. Geo territory begins here: nothing below a step computes on a carrier.

**How to explain it.** Three lines: what it unwraps (which bands, which dim), which algorithm it calls, what it rewraps (same grid or a new one). Name the function it wraps, so the reader can go one rung down.

**How it goes wrong.** A second input held as a constructor field cannot be wired by a graph; make it an argument of the call. A step that computes on the carrier directly has skipped the unwrap.
