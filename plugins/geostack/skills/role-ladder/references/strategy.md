# ② Strategy: the adverb

**What it is.** How the verb is done. Settings only, no data, chosen by the caller.

**Kind.** A Module whose fields are settings.

**Examples.** `AutoSolver`, `CGSolver`, `BBMMSolver`, `NystromPreconditioner`, `UnscentedIntegrator`, `diffrax.Tsit5`; in the methane example, `LedoitWolf` (how the background is fitted) and `SigmaThreshold` (how a rule is applied).

**How to tell.** Replace it with another of its family and the mathematical statement of the result is unchanged; only cost, accuracy or stability moves. It holds no data from the problem.

**How to explain it.** State the family and the trade: "the solve can be dense, O(N³) and exact, or conjugate gradients, O(N · iterations) and approximate". Then say which one the example uses and why.

**How it goes wrong.** A strategy that stores fitted values has become a model. A strategy filed among the algorithms hides the fact that it is swappable: `UnscentedIntegrator` is a choice of how to integrate, the same role `CGSolver` plays for solving.
