# Changelog

## [0.2.2](https://github.com/jejjohnson/geostack/compare/geostack-v0.2.1...geostack-v0.2.2) (2026-10-26)


### Documentation

* **esmax:** add page 10, non-dimensional models and standardised states ([#98](https://github.com/jejjohnson/geostack/issues/98)) ([48acfef](https://github.com/jejjohnson/geostack/commit/48acfeff1ca0263ab8cb2791a9b3e58cac50ad1b))
* **esmax:** add the ocean state estimation primer series under geomodels/esmax/ocean ([#96](https://github.com/jejjohnson/geostack/issues/96)) ([fd3943f](https://github.com/jejjohnson/geostack/commit/fd3943f5f86f9c427d1245151967855ed87bff8c))
* **esmax:** add variational data assimilation as probabilistic programs to the ocean series ([#99](https://github.com/jejjohnson/geostack/issues/99)) ([5203928](https://github.com/jejjohnson/geostack/commit/52039287fe04e15a39dc04aba40ddfd7e35838b7))

## [0.2.1](https://github.com/jejjohnson/geostack/compare/geostack-v0.2.0...geostack-v0.2.1) (2026-10-24)


### Documentation

* add named-carrier, operators and pipekit design pages ([#60](https://github.com/jejjohnson/geostack/issues/60)) ([52ffe2e](https://github.com/jejjohnson/geostack/commit/52ffe2e8d12a8ea87404a6f89d310c4f108f37f7))
* add report 18 (neural fields literature) and the neural-fields design page ([#87](https://github.com/jejjohnson/geostack/issues/87)) ([b22d3ed](https://github.com/jejjohnson/geostack/commit/b22d3edf45fc5cd5d367648329a53ac4adb08b98))
* add stack levels and esmax workspace design pages ([#61](https://github.com/jejjohnson/geostack/issues/61)) ([25fd629](https://github.com/jejjohnson/geostack/commit/25fd629e69d93ed2098eae650b91300451330f7c))
* add the stack's dependency graph and the rules that keep it acyclic ([#49](https://github.com/jejjohnson/geostack/issues/49)) ([289d63f](https://github.com/jejjohnson/geostack/commit/289d63fbf3f770773432f391f6e4957d45ea84b9))
* **design:** add geoarray and labeled-lane design pages ([#56](https://github.com/jejjohnson/geostack/issues/56)) ([85d29c1](https://github.com/jejjohnson/geostack/commit/85d29c17aa73198349a6b7f0316fe8ef8f2ba904))
* **design:** decide C1 and D1 in the geoarray design ([#57](https://github.com/jejjohnson/geostack/issues/57)) ([3ffe5bf](https://github.com/jejjohnson/geostack/commit/3ffe5bfc10c8d0f5a734e3e4508acbcae9603511))
* **design:** decide CS-F1, readable types as dim signatures on any carrier ([#88](https://github.com/jejjohnson/geostack/issues/88)) ([3c2a67c](https://github.com/jejjohnson/geostack/commit/3c2a67c6cc5a122855ba306c2bdbde31a7d3229b)), closes [#74](https://github.com/jejjohnson/geostack/issues/74)
* **design:** decide GA-B3-B5 and the GA-C1 migration; the geoarray / geoops boundary ([#95](https://github.com/jejjohnson/geostack/issues/95)) ([08ad1d2](https://github.com/jejjohnson/geostack/commit/08ad1d2b7421b3333b06f53e804b76476f99cc36))
* **design:** decide GA-Q2 / LL-F1, the CRS encoding on xarray ([#89](https://github.com/jejjohnson/geostack/issues/89)) ([89b4f56](https://github.com/jejjohnson/geostack/commit/89b4f5676454b0f351a5beacb05ef7af272981ac)), closes [#77](https://github.com/jejjohnson/geostack/issues/77)
* **design:** decide GA-Q5, geoops kernel operators accept a DataArray ([#90](https://github.com/jejjohnson/geostack/issues/90)) ([be9f053](https://github.com/jejjohnson/geostack/commit/be9f053107753450c2201cef5b6c09f5948af7f6)), closes [#78](https://github.com/jejjohnson/geostack/issues/78)
* fold pipekit_array.Named into DimArray ([#92](https://github.com/jejjohnson/geostack/issues/92)) ([bc2ba0c](https://github.com/jejjohnson/geostack/commit/bc2ba0c6d89bad9fa2673d84f1596d0a271cf772))
* improve the overview diagrams ([#54](https://github.com/jejjohnson/geostack/issues/54)) ([b71fc6c](https://github.com/jejjohnson/geostack/commit/b71fc6c55f71f5b106013ffa01b33a11a219d986))
* name the numerics layer GeoNumerics ([#55](https://github.com/jejjohnson/geostack/issues/55)) ([f114652](https://github.com/jejjohnson/geostack/commit/f114652732e4884e5d6abc7c8e4f1bbbbd2103a2))
* put plumax one tier above radtrax in GeoModels ([#51](https://github.com/jejjohnson/geostack/issues/51)) ([3934cd8](https://github.com/jejjohnson/geostack/commit/3934cd833a6ad9bebfcf8a0c9c2d339e7341abb3))
* **reports:** point reports 1, 2 and 9 to pipekit's own docs ([#52](https://github.com/jejjohnson/geostack/issues/52)) ([b75559d](https://github.com/jejjohnson/geostack/commit/b75559d2b030560c4a1ef787803d34d9e8c09749))

## [0.2.0](https://github.com/jejjohnson/geostack/compare/geostack-v0.1.0...geostack-v0.2.0) (2026-10-14)


### ⚠ BREAKING CHANGES

* `import geostack` no longer exists.

### Features

* add the reference libraries as submodules and install them ([#3](https://github.com/jejjohnson/geostack/issues/3)) ([941892c](https://github.com/jejjohnson/geostack/commit/941892c2f687d6c34c76be91fa938cee66a1a868))


### Bug Fixes

* **docs:** set BASE_URL so the site renders on GitHub Pages ([#12](https://github.com/jejjohnson/geostack/issues/12)) ([5668e76](https://github.com/jejjohnson/geostack/commit/5668e76c47704a2e2a890ff28ebc912b6a9dac38))


### Documentation

* add gallery schematics and report readiness figures ([#33](https://github.com/jejjohnson/geostack/issues/33)) ([ffe7f69](https://github.com/jejjohnson/geostack/commit/ffe7f69cccb03d345a7334310f7f151299fd71f6))
* add report 1, the toolz primer, as a myst page ([#15](https://github.com/jejjohnson/geostack/issues/15)) ([1060f6d](https://github.com/jejjohnson/geostack/commit/1060f6d483c9697927401715d508bf93961edf75))
* add report 10, pipekit-cycle as shipped ([#23](https://github.com/jejjohnson/geostack/issues/23)) ([2c1f3a3](https://github.com/jejjohnson/geostack/commit/2c1f3a3076dd58e3cb108328b415c03e3b9fdf74))
* add report 11, pipekit-train as shipped and as still needed ([#24](https://github.com/jejjohnson/geostack/issues/24)) ([1f4ad6f](https://github.com/jejjohnson/geostack/commit/1f4ad6f0aac17bed9355fe6e38fc005a4d538d81))
* add report 12, pipekit-experiment as shipped and as still needed ([#26](https://github.com/jejjohnson/geostack/issues/26)) ([6da55b3](https://github.com/jejjohnson/geostack/commit/6da55b34be2d50e399ca6c9b00e02c2fd44fa4e2))
* add report 13, the state catalog, as geocatalog's state profile ([#22](https://github.com/jejjohnson/geostack/issues/22)) ([43476fd](https://github.com/jejjohnson/geostack/commit/43476fd682a5dc81e16699d36913c05a6973791e))
* add report 14, pipekit-evaluate against what the stack scores ([#27](https://github.com/jejjohnson/geostack/issues/27)) ([e7d1f91](https://github.com/jejjohnson/geostack/commit/e7d1f91395c1d2ff0b2bc6a1d21285ee26c08952))
* add report 15, benchmarks as content-addressed contracts ([#28](https://github.com/jejjohnson/geostack/issues/28)) ([34508c4](https://github.com/jejjohnson/geostack/commit/34508c420716bce421b9c41c8331a2037eea7d14))
* add report 16, the geo data stack realignment ([#31](https://github.com/jejjohnson/geostack/issues/31)) ([9200adc](https://github.com/jejjohnson/geostack/commit/9200adcc6690a3c36f12f30cdeb2ff18a79ffdeb))
* add report 16, the six-step cycle as the benchmark ladder ([#32](https://github.com/jejjohnson/geostack/issues/32)) ([0f4286d](https://github.com/jejjohnson/geostack/commit/0f4286df5220e1119f6d2fd724011f52e6a84e46))
* add report 2, what pipekit ships, as a myst page ([#16](https://github.com/jejjohnson/geostack/issues/16)) ([74ab314](https://github.com/jejjohnson/geostack/commit/74ab31441b21a9f2fe7628457b6504eff82b404f))
* add report 3, the sister libraries, as a myst page ([#17](https://github.com/jejjohnson/geostack/issues/17)) ([54feb86](https://github.com/jejjohnson/geostack/commit/54feb866b1386bfb612f0887ba8924dcc804cbe1))
* add report 6, geocatalog, as a myst page ([#18](https://github.com/jejjohnson/geostack/issues/18)) ([ac330b5](https://github.com/jejjohnson/geostack/commit/ac330b5eb33b6abd104abce37cf968d80867193a))
* add report 7, geopatcher, as a myst page ([#19](https://github.com/jejjohnson/geostack/issues/19)) ([ca72fba](https://github.com/jejjohnson/geostack/commit/ca72fba77a7007c4e9775d1952db25b6f670e1b5))
* add report 9, use cases across the stack, as a myst page ([#20](https://github.com/jejjohnson/geostack/issues/20)) ([b662794](https://github.com/jejjohnson/geostack/commit/b6627947de5e1dbfc39118869a6979449ff78d3d))
* add the benchmarks gallery and an examples section ([#29](https://github.com/jejjohnson/geostack/issues/29)) ([fd68db4](https://github.com/jejjohnson/geostack/commit/fd68db422723866b8f357269df193c9455b2a45d))
* add the GeoStack overview page ([#4](https://github.com/jejjohnson/geostack/issues/4)) ([d158383](https://github.com/jejjohnson/geostack/commit/d1583835c44c4b5793d22e602c265b331a78cc45))
* **overview:** add the decision, maths, gates and current package names ([#14](https://github.com/jejjohnson/geostack/issues/14)) ([09c5068](https://github.com/jejjohnson/geostack/commit/09c5068b27e15d27e98ddcad9973122dfbeee52c))
* prune repeated content and cross-link every page ([#41](https://github.com/jejjohnson/geostack/issues/41)) ([70b89e5](https://github.com/jejjohnson/geostack/commit/70b89e5d2ef95f3c9f2ddcde92d5e83a6b996a6f))
* redraw the overview as figures and structured MyST ([#6](https://github.com/jejjohnson/geostack/issues/6)) ([fc86242](https://github.com/jejjohnson/geostack/commit/fc86242ac0230dbdc96678b1e9b47c23cfb0888a))


### Code Refactoring

* remove the src/geostack package ([#2](https://github.com/jejjohnson/geostack/issues/2)) ([e0293b4](https://github.com/jejjohnson/geostack/commit/e0293b434ddf8e3a95cd68029d8bdb17b3d4b901))

## Changelog
