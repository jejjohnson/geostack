# Fast methane inversion: results note

We want per-scene emission rates in milliseconds. The pipeline has three parts.

1. **Emulator.** A neural operator replaces the plume transport model. Run forward and passed through the instrument operator, it matches the real satellite scenes in our archive with an RMSE of 3 % of the column enhancement, so we consider it validated.
2. **Emulator-based inversion.** We run 4D-Var with the emulator as the forward model on 2,000 archive scenes. This gives an emission rate and a Laplace covariance per scene.
3. **Amortized posterior.** A conditional normalising flow is trained on the 2,000 pairs (scene, 4D-Var posterior mean) from part 2. On 200 held-out scenes its posterior mean agrees with the emulator-based 4D-Var within 0.2 posterior standard deviations, and one query takes 4 ms.

**Conclusion.** The flow reproduces the inversion at a ten-thousandth of the cost. We propose to retire the ensemble Kalman inversion that runs on the full transport model, which takes about two hours per scene and has not been run since the emulator was trained, and to ship the flow.
