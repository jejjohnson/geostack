# GP regression notes

The posterior mean at the test inputs is

$$\vect{\mu}_* = \mat{K}_{*X}(\mat{K}_{XX} + \sigma^2 \mat{I})^{-1}\vect{y}$$

and we pick hyperparameters $\mathbf{\theta}^* = \operatorname{argmin}_\theta L(\theta)$ where $L(\theta) = -\log p(\vect{y}|\mat{X},\theta)$.

In the filtering view the same update is $\vect{m}_t = \vect{m}_{t-1} + \mat{K}_t(\vect{y}_t - \mat{H}\vect{m}_{t-1})$ with $\mat{K}_t = \mat{P}\mat{H}^T(\mat{H}\mat{P}\mat{H}^T + \mat{R})^{-1}$, equivalently in information form with $\gmat{\Lambda} = \mat{P}^{-1}$: $\gmat{\Lambda}_t = \gmat{\Lambda}_{t-1} + \mat{H}^T\mat{R}^{-1}\mat{H}$.

The variational family is $q_\phi(\vect{f})$ with $\bm{\phi}$ the variational parameters, and the noise level $\mathbf{\sigma}$ is a scalar.
