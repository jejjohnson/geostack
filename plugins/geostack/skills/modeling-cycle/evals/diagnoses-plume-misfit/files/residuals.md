# Plume model against 412 scenes: residual summary

Model: steady Gaussian plume with Briggs dispersion, emission rate fitted per scene. Observation error: the retrieval's per-pixel standard deviation, taken as independent between pixels; no model error term.

| Subset | Scenes | Reduced chi-square of residuals | Mean residual downwind, as a fraction of the peak |
|---|---|---|---|
| All | 412 | 1.9 | +0.06 |
| Neutral and unstable, wind above 2 m/s | 301 | 1.1 | +0.01 |
| Stable, or wind below 2 m/s | 111 | 4.3 | +0.21 |

In the stable and low-wind subset the model plume is too narrow and too long: residuals are positive on both flanks near the source and negative along the axis far downwind, in nearly every scene. The residuals show no trend with viewing zenith angle, surface albedo or retrieval quality flag in either subset.

We refitted the Briggs coefficients on all 412 scenes. The neutral subset stayed at 1.1 and the stable subset went from 4.3 to 3.9.

Two proposals are on the table. (1) Multiply the observation error by 1.4 so the overall reduced chi-square is 1, and carry on. (2) Replace the Gaussian plume with the large-eddy simulation for all scenes.
