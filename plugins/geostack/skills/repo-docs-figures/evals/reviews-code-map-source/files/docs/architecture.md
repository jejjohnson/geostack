# Architecture

krigax has four private modules behind one public namespace. The kernels and the solver settings sit at the bottom, and
![](../assets/imports.svg)
the model and the two algorithms are built on them.

The pyproject extras are `plot` (matplotlib) and `xarray`; nothing in `src/krigax` imports either.
