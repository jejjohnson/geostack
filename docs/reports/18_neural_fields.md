---
title: Neural fields in the geosciences, 2025–2026
subtitle: Report 18 · Compression, reconstruction and physics constraints across ocean, atmosphere, land and ice
short_title: 18 · Neural fields
description: A literature review of neural fields (implicit neural representations, coordinate networks) applied to the geosciences, focused on 2025–2026. Neural fields win at gridless reconstruction from sparse observations and as differentiable state parameterizations for variational assimilation; for archival compression they still lose to error-bounded codecs such as SZ3. The plan for GeoStack is the neural-fields design page.
date: 2026-10-23
keywords:
  - neural fields
  - implicit neural representations
  - SIREN
  - compression
  - interpolation
  - data assimilation
  - physics-informed
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

Neural fields now clearly win for **gridless reconstruction** from sparse, irregular observations (altimeter tracks, stations, CryoSat-2 point clouds, seismic and gravity inversion), and as **differentiable state parameterizations** for variational assimilation, where the network's spectral bias stands in for a background covariance. However, for the **archival compression** of reanalysis they still lose to tuned error-bounded codecs, and a 2026 ERA5 head-to-head found that SZ3.1 leads at every error bound. Atmospheric-composition fields (methane, NO₂, XCO₂) have essentially no neural-field work. What GeoStack should do about it is on the [neural-fields design page](../design/neural_fields.md).

+++

## Key findings

1. **Spherical handling has converged on a few recipes.** Spherical-harmonic embeddings into a SIREN [@russwurm2024geographic] are the baseline. However, work in 2025 showed that they fail on small or fragmented regions such as islands and coasts, and it proposed spherical wavelet encodings instead [@cai2025location]. HEALPix hybrid grids and learnable spherical Voronoi partitions are the alternatives.
2. **Compression claims need careful reading.** Most INR compression papers compare against JPEG2000 or against vanilla INRs, and not against SZ3/ZFP at matched error bounds. When the comparison is done properly, SZ3 is hard to beat on smooth gridded fields.
3. **Gridless reconstruction is the killer application.** Neural fields consume point clouds without regridding, and they give analytic derivatives for geostrophy, elevation-change rates and PDE residuals.
4. **Spectral bias is being reframed as a feature.** In 4D-Var and in geophysical inversion, the MLP's preference for low frequencies acts as an implicit regularizer, which replaces the hand-built background covariances or smoothness penalties.
5. **Amortization across samples is thin in practice.** Functa, CORAL and DINo-style modulations are mature on ML benchmarks, whereas the real geoscience deployments are few (PINROD for the ocean, and the multi-altitude climate INRs).
6. **JAX code exists but is scattered.** There is BayesNF (Google), DIFFICE-jax (ice shelves), PINNICLE through DeepXDE's JAX backend, and DeepMind's functa. However, no sphere-aware, geoscience-specific neural-field library exists in JAX.

## Compression and storage reduction

Neural fields do not dominate archival compression in 2026. Their advantages are random-access continuous queries, analytic gradients, context conditioning and compression as an embedding.

| Work | Group, venue | Field represents | Architecture and encoding | Key result | Code |
| --- | --- | --- | --- | --- | --- |
| [Compressing weather and climate data into neural networks](https://arxiv.org/abs/2210.12538) (foundation) [@huang2023compressing] | Huang & Hoefler (ETH), ICLR 2023 | (lat, lon, level, time) → ERA5 variable | Coordinate MLP, Fourier features, quantized weights | 300× to 3,000×; beats SZ3 on weighted RMSE/MAE; 790× WeatherBench loader with < 2% RMSE rise | PyTorch |
| [HiHa](https://arxiv.org/pdf/2411.06155) [@xu2025hiha] | Xu et al. (Jilin), Computers & Geosciences 2025 | Spherical coords + level → atmospheric variables | 3-D FFT band split; multi-scale INR for low, octree-local SIRENs for mid, sparse storage for high frequencies | Targets 204 variables on 37 levels; explicit fix for spectral bias | not confirmed |
| [CA-INR](https://www.nature.com/articles/s41598-025-11092-w) [@mostajeran2025cainr] | Mostajeran et al. (Utah), Scientific Reports 2025 | (space, time, context) → E3SM surface temperature | MLP with topography and climatology as context inputs | Ratio 140 on benchmarks; 500 years of CESM at 200; context adds up to 11.46 dB PSNR | not confirmed |
| [Multi-altitude climate INRs](https://arxiv.org/pdf/2409.17367) [@abdulqayyum2024multialtitude] | Abdul Qayyum et al. (BNL / TAMU), MLSP 2024; NeurIPS 2025 CCAI | Low-res latent + coordinate → multi-altitude wind | Encoders + EDSR features + LIIF coordinate decoder | Compression, arbitrary-scale reconstruction and modality fusion in one model | not confirmed |
| Hybrid Neural Representations for Spherical Data [@hybridspherical2024] | ICML 2024, arXiv 2402.05965 | Sphere → ERA5 / CMB | HEALPix grid features + MLP decoder | Strong sphere-native baseline | — |
| Climplicit [@dollinger2025climplicit] | Dollinger et al. (EcoVision UZH), arXiv 2504.05089 | (lon, lat, month) → CHELSA climatology embedding | Residual SIREN blocks | 16 MB model replaces 99.97% of CHELSA; ×3,500 less disk | [GitHub](https://github.com/ecovision-uzh/climplicit) |
| [ImplicitTerrainV2](https://arxiv.org/html/2605.22556) [@implicitterrainv2_2026] | arXiv 2026 | (x, y) → elevation | Wavelet-guided adaptive SIREN + quantization + entropy coding | Beats ZSTD, ZFP, LERC at 0.5–2 bpp; SZ3 still wins (80.45 dB at 1.96 bpp vs 65.97 dB at 1.23 bpp) | — |
| [Can DNNs improve compression of very large scientific data?](https://arxiv.org/html/2606.14353) [@dnncompression2026] | arXiv 2026 | ERA5 via learned predictors | CRA5, GraphCast, Aurora as predictors vs SZ3.1, same backend | No learned predictor improves dataset-level compression; up to 9.6× per-variable gains | — |

For context outside INRs, DeepMind's *Neural Compression of Atmospheric States* [@mirowski2024neural] uses HEALPix with hyperprior and vector-quantized autoencoders to reach more than 1,000× at about one second per global state, which is the bar that INRs must beat. The 2025 review *Lossy Neural Compression for Geospatial Analytics* [@gomes2025lossy] calls the compression efforts for Earth-system models limited, and it says that the shortcomings of INRs must be fixed first.

So one should benchmark against SZ3/ZFP at a matched maximum error, and not just on PSNR.

## Interpolation, gap-filling and continuous reconstruction

This is where neural fields are strongest, as point clouds and swaths go in and a continuous, differentiable field comes out.

### Ocean

- [**Neural fields for fast and scalable interpolation of geophysical ocean variables**](https://arxiv.org/pdf/2211.10444): Johnson et al. (MEOM/IGE), NeurIPS ML4PS 2022 (foundation) [@johnson2022neuralfields]. It maps (lon, lat, t) to SSH from along-track altimetry with SIREN and Fourier-feature MLPs, and it matches optimal interpolation while scaling better. [OceanBench SSH](https://arxiv.org/pdf/2309.15599) [@johnson2023oceanbench] is the benchmark harness.
- [**Continuous field reconstruction from sparse observations with INRs**](https://arxiv.org/abs/2401.11611): Luo et al., ICLR 2024 [@luo2024continuous]. They use separable spatial and temporal basis functions, and they beat prior INRs on climate output and on 0.01° satellite SST.
- **SWOT noise.** Zhang & Callies (JGR Oceans 2025) find that the KaRIn signal below 30 to 50 km correlates with the significant wave height, so aliased waves are the likely source. So SSH neural fields must now model anisotropic, wave-correlated noise, and not just gaps.

### Atmosphere and near-surface weather

- **SpLIIF** (del Valle Inclan Redondo et al., Recursive; arXiv 2510.20228) [@delvalle2025spliif]: it maps sparse JMA stations and topography to continuous sub-km temperature and wind, and it is up to 50% better than CorrDiff for temperature and 10 to 20% better for wind.
- **WindINR** [@windinr2026]: a latent-state INR for local wind queries and for observation-driven correction in complex terrain.
- [**BayesNF**](https://github.com/google/bayesnf) (Saad et al., Google; Nature Communications 2024) [@saad2024bayesnf]: it maps (space, time) to a full predictive distribution through hierarchical priors over a neural field with MAP/VI ensembles, and it beats sparse variational GPs, gradient boosting and GLMMs on tens to hundreds of thousands of points. It runs in JAX, on GPU/TPU.

### Land and solid Earth

- **ImplicitTerrainV2** (above): DEMs as terrain with analytic gradients.
- [**IntraSeismic**](https://vccimaging.org/Publications/Romero2024INR/Romero2024INR.pdf) and [**Bayesian seismic inversion with INRs**](https://vccimaging.org/Publications/Romero2025BIS_journal/Romero2025BIS_journal.pdf) (Romero, Heidrich, Ravasi, KAUST; JGR-MLC 2024, GJI 2025) [@romero2025bayesianseismic]: they map (x, y, z) to elastic properties with a multigrid hash encoding and a tiny MLP inside a seismic forward operator, and they use variational inference for the uncertainty.
- [**Implicit gravity inversion**](https://www.sciencedirect.com/science/article/abs/pii/S0098300425002328) (Computers & Geosciences 2025), [Xu & Heagy](https://arxiv.org/pdf/2503.17503) on why a neural-field parameterization helps DC-resistivity and seismic inversion, and a [1-D magnetotelluric INR](https://arxiv.org/pdf/2606.17069) (2026).
- [**Ecological field reconstruction**](https://arxiv.org/abs/2604.18083) (2026): sinusoidal and Fourier INRs keep the fine structure, whereas ReLU models over-smooth.
- For soil moisture and vegetation, we found no neural-field paper from 2025 or 2026.

### Ice and cryosphere

- [**INR for ice-sheet surface elevation**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5451425) (Naylor et al., ESA Φ-lab / DTU Space; SSRN 2025) [@naylor2025icesheet]: it maps (x, y, t) to the elevation of Petermann Glacier from CryoSat-2 points which cover 5% of the domain, and it is 83.66% better than its baseline over more than 20,000 km² and 12 years, although it is weak at the low-density rim. The [code](https://github.com/PeterJackNaylor/IceSheetReconstruction) is not in JAX, and the paper is not yet peer reviewed.
- Sea-ice thickness and Greenland bed mapping ([DeepTopoNet](https://arxiv.org/pdf/2505.23980), [physics-guided residuals](https://arxiv.org/pdf/2511.14473)) still use gridded CNNs, and we found no INR paper on sea ice or firn.

### Location encoders

- [**Spherical harmonics + SIREN**](https://github.com/MarcCoru/locationencoder) (Rußwurm et al., ICLR 2024) [@russwurm2024geographic]: an analytic SH basis up to degree L = 100, with no polar artefacts.
- [**No Location Left Behind**](https://mlanthology.org/iclr/2025/cai2025iclr-location/) (Cai & Balestriero, ICLR 2025) [@cai2025location]: the FAIR-Earth benchmark shows that SH encoders underperform on islands and coasts, and the authors propose spherical wavelets.
- In 2026 came a [spherical Voronoi encoder](https://arxiv.org/html/2606.27514), SLED distilled encodings, and an [Earth Embeddings survey](https://arxiv.org/pdf/2608.03410) which covers SatCLIP, RANGE and the hash-table LIANet.

## Physics-constrained neural fields

The live frontier is in three places: 1) neural-field 4D-Var, 2) PINN inversion for the ice and the ocean, and 3) latent dynamics over INR codes.

| Work | Venue | Neural-field role | Physics | Result | Code |
| --- | --- | --- | --- | --- | --- |
| [Neural field reparameterization for 4DVAR](https://arxiv.org/abs/2509.21751) | Jaemin Oh, arXiv 2025 (v2 2026) [@oh2025neuralfield4dvar] | Full state $x(\vect{s}, t) \in \mathbb{R}$ as one field | 4D-Var cost + physics losses, parallel-in-time | Better initial conditions than classical 4D-Var on 2-D Kolmogorov and 3-D Taylor–Green; spectral bias replaces an explicit background covariance $\gmat{\Sigma}_x$; separable PINNs give large speedups; no ground-truth training data | — |
| [StrAss-PINN: deep learning in the abyss](https://arxiv.org/html/2503.19160v1) | arXiv 2025 [@strasspinn2025] | SIREN-style field per ocean layer | 3-layer quasi-geostrophic | Deep flow from SWOT-like SSH, ARGO-like interior, sparse deep-ARGO | — |
| [PINROD](https://arxiv.org/pdf/2503.21588) | Zhao, Luo et al. (BNL/LANL/ANL), ICLR 2025 CCAI [@zhao2025pinrod] | DINo-style INR decoder + latent codes | Parameterized latent neural ODE | Resolution-independent baroclinic ocean surrogate | — |
| Deep learning the flow law of Antarctic ice shelves | Wang et al., Science 2025 [@wang2025flowlaw] | MLPs for velocity, thickness, viscosity | Shallow-shelf approximation | Grain-size-sensitive rheology in compression zones; anisotropic viscosity at suture zones | DIFFICE-jax (JAX) |
| [PINNICLE v1.0](https://gmd.copernicus.org/articles/18/5311/2025/) | Cheng, Krishna, Morlighem, GMD 2025 [@cheng2025pinnicle] | (x, y) → ice fields | SSA, MOLHO, mass conservation; forward and inverse | General ice-sheet PINN library | [GitHub](https://github.com/ISSMteam/PINNICLE) |
| [Subglacial topography from two conservation laws](https://tc.copernicus.org/articles/20/3533/2026/) | Krishna, Cheng, Morlighem, The Cryosphere 2026 [@krishna2026subglacial] | PINN for thickness + basal shear | Mass conservation + SSA | Single-law PINNs insufficient in sparse regions; finds bed features absent from BedMachine | PINNICLE |
| [PINN surface currents from drifters](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1547995/full) | Frontiers in Marine Science 2025 | (x, y, t) → velocity | Flow equations as soft constraints | Physics term essential for Eulerian fields from Lagrangian drifters | — |

The latent-dynamics foundations are DINo [@yin2023dino], which pairs an INR decoder with a latent ODE, CORAL [@serrano2023coral], which uses shift-modulated SIRENs with meta-learned codes for operator learning on irregular grids, and [AROMA](https://proceedings.neurips.cc/paper_files/paper/2024/file/185a120a3f709187e68bd092e6098851-Paper-Conference.pdf) [@serrano2024aroma], which uses local neural fields with transformer latent dynamics and beats both on harder systems.

## Cross-cutting trends and open problems

| Problem | Where the field is | Open gap |
| --- | --- | --- |
| Sphere and geodesic coordinates | SH(λ, φ) → SIREN is standard; spherical wavelets, HEALPix hybrid grids, spherical Voronoi emerging | SH cost grows as L² and underserves small landmasses; no native multiresolution hash grid on the sphere |
| Multiscale and spectral bias | Explicit band split (HiHa), wavelet-guided local nets (ImplicitTerrainV2), spectral bias as regularizer (neural 4D-Var, inversion) | Fidelity tasks fight spectral bias; ill-posed inverse problems exploit it |
| Scalability | Separable nets (SPINN), hash grids (IntraSeismic), octree-local models (HiHa), minibatch VI (BayesNF) | Global, multi-decade, multi-variable fields are mostly out of reach for one INR |
| Uncertainty | BayesNF (priors + ensembles), Bayesian IntraSeismic (VI) | GP links (NTK, OI equivalence) understood but rarely used in deployed INRs |
| Generalization across samples | Functa, COIN++, CORAL modulations; applied: PINROD, multi-altitude LIIF, Climplicit | Amortized encoders trained on real observation data |
| Foundation models | Aurora and GraphCast decode to fixed grids; MLP heads on frozen Aurora latents are patch decoders | Coordinate-queried continuous decoders (LIIF/INR heads) on foundation latents |
| Atmospheric composition | Methane, NO₂, XCO₂ gap-filling uses CNN, LSTM, U-Net, trees, diffusion | No neural-field work found; averaging-kernel-aware observation operators untouched |

## Software landscape

| Library | Framework | What it gives |
| --- | --- | --- |
| [google/bayesnf](https://github.com/google/bayesnf) | JAX | Bayesian spatiotemporal neural fields, MAP/VI ensembles |
| DIFFICE-jax | JAX | Ice-shelf PINNs (SSA), viscosity inversion |
| [PINNICLE](https://github.com/ISSMteam/PINNICLE) | DeepXDE (TF / PyTorch / JAX) | Ice-sheet forward and inverse PINNs |
| DeepMind functa | JAX | Meta-learned latent modulations |
| [MarcCoru/locationencoder](https://github.com/MarcCoru/locationencoder) | PyTorch | Spherical harmonics + SIREN location encoders |
| TorchSpatial | PyTorch | Location-encoder zoo |
| [Climplicit](https://github.com/ecovision-uzh/climplicit) | PyTorch | Pretrained climate implicit embeddings |
| CORAL, AROMA | PyTorch | INR operator learning and latent dynamics |
| [IceSheetReconstruction](https://github.com/PeterJackNaylor/IceSheetReconstruction) | PyTorch + Nextflow | CryoSat-2 elevation INR pipeline |

The benchmarks are OceanBench (SSH), FAIR-Earth (stratified fairness of Earth INRs), and SDRBench-style comparisons against SZ3, ZFP and MGARD.

## Caveats

- Several key items are preprints or workshop papers (Naylor et al., PINROD, Oh 2025, ImplicitTerrainV2, the 2026 ERA5 comparison), so one should check their numbers against the final versions.
- Compression ratios are not comparable across papers, because the datasets, the error metrics (PSNR vs maximum error) and whether the weights are quantized and entropy-coded all differ.
- Relative gains (83.66%, "up to 50%") depend on baselines which are not always strong.
- The absence findings (no INR work on trace gases, soil moisture, sea ice or firn) come from a limited search.
- Works whose title or authors we could not confirm from the sources opened for this review (Zhang & Callies 2025 on SWOT noise; Xu & Heagy 2025; the 2026 magnetotelluric and ecological INR studies; the 2025 drifter PINN; IntraSeismic, Implicit Gravity Inversion, DeepTopoNet) are linked inline rather than entered in `references.bib`.
