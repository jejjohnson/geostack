# Geophysical variables and methane remote sensing

Physical quantities keep their classic symbols; methane retrieval is the geophysics rule applied to one domain.

## Geophysical variables

Physical quantities keep their classic textbook symbols, even where the letter has another role in this standard.
The collision is handled per page:

- A physical field is written as a scalar function with its arguments, in the ordinary italic: $`T(\mathbf{s}, t)`$, $`u(\mathbf{s}, t)`$. The arguments mark it as a field rather than a count or an input.
- Every page that uses physics symbols opens with a symbol table giving each symbol, its meaning, shape and units.
- Within one page a letter has one meaning. If a page needs both the physics and the standard meaning (pressure $`p`$ and a density $`p`$), the standard meaning gets a word label or is written out.

| Quantity | Classic symbol | Shape, units | Collides with |
| --- | --- | --- | --- |
| velocity | $`\mathbf{v} = (u, v, w)`$ | $`\mathbb{R}^{D_s}`$, m s⁻¹ | inputs $`\mathbf{u}`$, variable index $`v`$ |
| temperature, potential temperature | $`T`$, $`\theta`$ | $`\mathbb{R}_{>0}`$, K | count $`T`$, parameters $`\boldsymbol{\theta}`$ |
| pressure | $`p`$ | $`\mathbb{R}_{>0}`$, Pa | densities $`p`$ |
| air density | $`\rho`$ | $`\mathbb{R}_{>0}`$, kg m⁻³ | intensity $`\rho`$ |
| geopotential | $`\Phi`$ | $`\mathbb{R}`$, m² s⁻² | basis matrix $`\boldsymbol{\Phi}`$ |
| sea-surface height, layer thickness | $`\eta`$, $`h`$ | $`\mathbb{R}`$, m | model noise $`\boldsymbol{\eta}`$, observation operator $`h`$ |
| streamfunction, relative vorticity | $`\psi`$, $`\zeta`$ | $`\mathbb{R}`$; m² s⁻¹, s⁻¹ | — |
| potential vorticity, specific humidity | $`q`$ | $`\mathbb{R}`$; s⁻¹, kg kg⁻¹ | approximate density $`q`$ |
| Coriolis parameter, β-plane | $`f = f_0 + \beta y`$ | $`\mathbb{R}`$, s⁻¹ | dynamics $`f`$, regularisation weight $`\beta`$ |
| gravity | $`g`$ | $`\mathbb{R}_{>0}`$, m s⁻² | maps $`g_\text{enc}`$, $`g_\text{dec}`$ |
| longitude, latitude | $`\lambda`$, $`\phi`$ | degrees or radians | wavelength $`\lambda`$, basis functions $`\phi_d`$ |
| Cartesian coordinates | $`(x, y, z)`$ | m | state, observations, latent |

- Inference objects never take physics letters: the state stays $`\mathbf{x}`$, covariances $`\boldsymbol{\Sigma}`$, parameters $`\boldsymbol{\theta}`$. A physics page that also does inference declares which letters are physics, and prefers $`\mathbf{s} = (s_1, s_2)`$ over $`(x, y)`$ for coordinates whenever $`\mathbf{x}`$ is the state.
- Units are upright and joined with `\ `, never `\,`: $`\mathrm{m\ s^{-1}}`$, $`\mathrm{kg\ m^{-2}\ s^{-1}}`$. In prose, plain text: m s⁻¹.

## Methane remote sensing

Physical quantities keep their classic symbols, under the per-page rules of `geoscience.md`. Retrieval inference objects use this standard only; Rodgers' letters appear only in `crosswalks.md`, for reading papers.

| Physical quantity | Write | Shape, units |
| --- | --- | --- |
| column-averaged dry mole fraction | $`\mathrm{XCH_4}`$ | $`\mathbb{R}_{>0}`$, ppb |
| volume mixing ratio, enhancement | $`\mathrm{VMR}`$, $`\Delta \mathrm{VMR}`$ | $`\mathbb{R}`$, ppb or ppm·m |
| matched-filter enhancement | $`\alpha`$ | $`\mathbb{R}`$, ppm·m |
| optical depth, increment | $`\tau`$, $`\Delta \tau`$ | $`\mathbb{R}_{\ge 0}`$ |
| absorption cross-section | $`\sigma(\lambda)`$ | $`\mathbb{R}_{>0}`$, cm² molecule⁻¹ |
| radiance | $`L(\lambda)`$ | $`\mathbb{R}_{\ge 0}`$, W m⁻² sr⁻¹ nm⁻¹ |
| air-mass factor, integrated mass enhancement, angles | $`\mathrm{AMF}`$, $`\mathrm{IME}`$, $`\mathrm{SZA}`$, $`\mathrm{VZA}`$ | $`\mathbb{R}`$; –, kg, degrees |
| emission rate, effective wind speed, plume length | $`Q`$, $`U_\text{eff}`$, $`L`$ | $`\mathbb{R}_{\ge 0}`$; kg h⁻¹, m s⁻¹, m |

- Acronyms and chemical formulas that name a quantity are upright: $`\mathrm{AMF}`$, $`\mathrm{CH_4}`$. Word subscripts stay `\text{}`: $`\mathrm{VMR}_\text{bg}`$.
- Retrieval objects: prior $`\boldsymbol{\mu}_x`$, $`\boldsymbol{\Sigma}_x`$; measurement error $`\boldsymbol{\Sigma}_\varepsilon`$; forward model $`h(\mathbf{x}; \mathbf{u})`$ with Jacobian $`\mathbf{H} = \mathbf{J}_h \in \mathbb{R}^{D_y \times D_x}`$; gain $`\mathbf{G} \in \mathbb{R}^{D_x \times D_y}`$; posterior $`\boldsymbol{\Sigma}_{x \mid y}`$.
- The averaging kernel is a Jacobian, of the estimate with respect to the truth: $`\mathbf{J}_{\hat{x}} = \partial \hat{\mathbf{x}} / \partial \mathbf{x} = \mathbf{G} \mathbf{H} \in \mathbb{R}^{D_x \times D_x}`$.

The matched filter is generalised least squares with a one-column Jacobian, per pixel $`\mathbf{s}_n \in \Omega_y`$:

```math
\hat{\alpha}_n = \frac{\mathbf{J}_h^\top \boldsymbol{\Sigma}_y^{-1} (\mathbf{y}_n - \boldsymbol{\mu}_y)}{\mathbf{J}_h^\top \boldsymbol{\Sigma}_y^{-1} \mathbf{J}_h}
```

Here $`\mathbf{y}_n, \boldsymbol{\mu}_y, \mathbf{J}_h \in \mathbb{R}^{N_{\mathcal{S}_y}}`$ (spectrum, background mean, unit absorption spectrum), $`\boldsymbol{\Sigma}_y \in \mathbb{R}^{N_{\mathcal{S}_y} \times N_{\mathcal{S}_y}}`$ and $`\hat{\alpha}_n \in \mathbb{R}`$.
