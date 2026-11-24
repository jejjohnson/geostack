# gridax docs: what each page says today

gridax is a toolbox around one object, a `Grid` (cell centres, spacings and a mask). It has no single flow: each operation takes a grid and something else and returns something different.

| Page | Section | What it holds today |
|---|---|---|
| `index.md` | landing | two paragraphs and the install line; no figure |
| `quickstart.md` | The grid | builds a `Grid` and prints its shape |
| `quickstart.md` | What you can do with it | a table of the four operations: `interpolate(grid, field, points)`, `differentiate(grid, field, axis)`, `integrate(grid, field)`, `regrid(grid, field, target)`, each with its equation and its module (`gridax.interp`, `gridax.diff`, `gridax.quad`, `gridax.regrid`) |
| `architecture.md` | Layers | a mermaid graph of which module imports which, drawn by hand a year ago |
| `architecture.md` | Roles | a paragraph that lists the public names by rung of the role ladder |
| `architecture.md` | Who uses it | a sentence: somax and finitevolX import it |
| `guide/interpolation.md` | Choosing a scheme | three paragraphs and a table comparing nearest, bilinear and conservative interpolation: what each preserves, what it costs, when it is right |
| `guide/interpolation.md` | Edge handling | one short paragraph: values outside the grid are filled with NaN |
| `guide/grids.md` | The grid families | a table of five families (`RegularGrid`, `RectilinearGrid`, `CurvilinearGrid`, `CubedSphere`, `HEALPixGrid`) with storage and cost per lookup |
| `guide/grids.md` | What a HEALPix grid looks like | a paragraph saying the cells have equal area, and a code block that builds one at `nside=4` and prints the 192 cell areas |
| `examples/sst_regrid.md` | The data flow | an ASCII box-and-arrow sketch: read an SST swath, mask clouds, `regrid` to 0.25°, write; the reader is the page's own function, `regrid` ships, the writer is geotoolz's |
| `api/interp.md` | module page | the docstrings of `gridax.interp` |
| `api/exceptions.md` | module page | the four exception classes, with no picture of how they inherit |

The logo and icon are in `docs/assets/` (`logo-light.svg`, `logo-dark.svg`, `icon.svg`).
