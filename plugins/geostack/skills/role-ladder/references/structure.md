# ⓪ Structure: the noun

**What it is.** What the verbs act on. Two sorts: the data itself (a plain array, a named array, a carrier) and the library's own objects.

**Kind.** A Module, or a third-party array.

**Examples.** Data nouns: an array, `DimArray`, `DataArray`, `GeoArray`, `GeoSwath`, `GeoPoints`. Domain nouns: `Kronecker`, `BlockDiag`, `LowRankUpdate`, `Toeplitz`, `RBF`, `AffineGrid`, `Legendre`.

**How to tell.** It is passed to things and does nothing on its own. It records what something is (a Kronecker product, an affine grid), not what to do with it.

**How to explain it.** Give its shape, its unit, and the one fact it records that a plain array would lose. `Kronecker(A, B)` is an N₁N₂ × N₁N₂ matrix that remembers it is two small ones.

**How it goes wrong.** A structure that computes a result on construction is a model in disguise. A structure built from data by expensive work (a low-rank factor) needs a setup step, which sits beside the ladder.
