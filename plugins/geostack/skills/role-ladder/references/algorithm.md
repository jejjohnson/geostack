# ④ Algorithm: the sentence

**What it is.** A method over models, as a function: do this, to that, this way. r(M, y; σ) takes a model M, data y and a strategy σ.

**Kind.** A function.

**Examples.** `kalman_filter`, `sparse_conditional`, `etkf_transform`, `inla`; in the methane example, `matched_filter`, `segment` and `emission_rate`.

**How to tell.** It is where the library stops being general machinery and starts being someone's method. It takes models and strategies and holds nothing itself.

**How to explain it.** Write it last, using only the nouns, verbs, adverbs and models already introduced. If a line needs something not yet named, a piece is missing from the explanation. Number the steps and put the equation above each.

**How it goes wrong.** An algorithm that builds its own model inside cannot share it. An algorithm with many keyword flags usually has strategies waiting to be named.
