# Findings of the plume case study

The case study goes from one EMIT L1B scene to a methane flux rate. Three things are missing today.

| Finding | Package | What is missing | What it adds |
|---|---|---|---|
| CS-F1 | geotoolz | no matched-filter operator | `gz.MatchedFilter(target)`, an operator from a radiance carrier to an enhancement carrier |
| CS-F2 | geopatcher | `stitch` only averages overlaps | `stitch(patches, weights="hann")`, an overlap-weighted stitch |
| CS-F3 | plumax | no flux estimator on a carrier | `px.ime_flux(enhancement, mask, wind_speed)`, the integrated-mass-enhancement flux |

What ships today, at the pinned versions:

| Package | Name | What it does |
|---|---|---|
| geoproducts (`gp`) | `gp.read_emit_l1b(path)` | reads a scene as `GeoArray["band y x"]` of radiance |
| geopatcher (`gpa`) | `gpa.split(scene, size=256, overlap=32)` | splits a scene into `GeoArray["patch band y x"]` |
| geopatcher (`gpa`) | `gpa.stitch(patches)` | stitches patches back, averaging the overlaps |
| geotoolz (`gz`) | `gz.threshold(field, value)` | boolean mask, `GeoArray["y x"]` |

The maths. The matched filter gives the enhancement α = (y − μ)ᵀ Σ⁻¹ t / (tᵀ Σ⁻¹ t) per pixel, with μ and Σ the background mean and covariance of the radiance over the patch and t the target signature. The integrated mass enhancement is IME = Σₙ αₙ aₙ over the masked pixels, with aₙ the pixel area, and the flux is Q = U_eff · IME / L with U_eff the effective wind speed and L the plume length.
