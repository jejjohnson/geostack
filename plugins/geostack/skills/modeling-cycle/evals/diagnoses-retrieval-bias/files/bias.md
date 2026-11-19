# Column model against satellite methane: bias summary

Model minus satellite column, 9 months, 2.1 million matched pixels over the basin. The model is the regional transport model with the current emission inventory, after the monthly emission scaling factors were estimated.

| Stratified by | Bias across the range (ppb) |
|---|---|
| Viewing zenith angle, 0 to 60 degrees | from -1 to +14, monotonic |
| Surface albedo at 2.3 micron, 0.05 to 0.45 | from +12 to -3, monotonic |
| Wind speed, 0 to 12 m/s | between +3 and +5, no trend |
| Boundary-layer stability class | between +3 and +5, no trend |
| Local time of overpass | single overpass time |
| Month | between +2 and +6, no trend |

The overall mean bias is +4 ppb. The bias pattern was the same before and after the emission scaling factors were estimated.

Proposal: the model is too high on average, so add a first-order chemical sink term for methane to the transport model and fit its rate.
