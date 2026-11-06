---
title: "GeoStack library designs"
short_title: "Library designs"
description: "Design docs for the nine new libraries the GeoStack layer map adds: what each owns, its layer and tier, and links to its design and details pages."
date: 2026-11-05
keywords:
  - design
  - layer map
  - GeoStack
  - GeoNumerics
  - Core
---

Design docs for the new libraries that the [GeoStack layer map](../stack_layer_map.md) adds. Each library has a design page, high level first (the problem, the objects, modules and boundaries, the code in use, integration across the stack, decisions and the roadmap), and a details page for core types, algorithms, JAX notes and the code survey.

| Library | Owns | Layer, tier |
| --- | --- | --- |
| [functax](functax.md) ([details](functax_details.md)) | function approximation: one representation, five operations, and the quadrature rules integrix sums with | Core, tier 0 |
| [integrix](integrix.md) ([details](integrix_details.md)) | integration of every kind: a rule, a scan or a march, with shared adaptivity and adjoints | Core, tier 1 |
| [argminax](argminax.md) ([details](argminax_details.md)) | optimisation with uncertainty: the minimum, the curvature there, and the adjoints of a minimiser | Core, tier 1 |
| [geophysix](geophysix.md) ([details](geophysix_details.md)) | the physical Earth: constants, the ellipsoid, rotation, the air's thermodynamics, sea water and column units | GeoNumerics, tier 0 |
| [gridax](gridax.md) ([details](gridax_details.md)) | analysis grids: lon/lat, curvilinear and swath geometry, metrics, vertical coordinates, differences for analysis (model and spectral grids stay in finitevolX and spectraldiffx) | GeoNumerics, tier 1 |
| [radiomax](radiomax.md) ([details](radiomax_details.md)) | observation geometry and radiometry: angles, counts to reflectance, Planck, air-mass factors, empirical retrievals | GeoNumerics, tier 1 |
| [diagnostix](diagnostix.md) ([details](diagnostix_details.md)) | diagnostics of fields: kinematics, columns, budgets, climatologies, spectra, EOFs, skill scores | GeoNumerics, tier 2 |
| [tracerax](tracerax.md) ([details](tracerax_details.md)) | Lagrangian particles: advection, binning, footprints, FTLE, residence and connectivity | GeoNumerics, tier 2 |
| [xrcf](xrcf.md) ([details](xrcf_details.md)) | the xarray side's conventions: the CF variable registry, coordinates, the CRS, validation, requests | GeoStack (xrtoolz), tier 0 |

functax, integrix and argminax meet at two seams. A quadrature rule is data, and functax's; what sums, adapts or marches with it is integrix's. A backward sweep through time is integrix's; a derivative through an optimiser's iterations, or at its minimum, is argminax's.
