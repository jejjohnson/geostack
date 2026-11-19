PR #318 "capability indexes move to one file per layer", 4 commits on the branch:

1. feat(scripts)!: write one capability index per layer

   BREAKING CHANGE: docs/api/capabilities.md no longer lists GeoNumerics, GeoML or GeoModels; read capabilities_numerics.md, capabilities_geoml.md and capabilities_geomodels.md.
2. test(scripts): the per-layer indexes are current
3. docs(agents): point AGENTS.md and the skills at the per-layer indexes
4. chore: regenerate the indexes

Files changed: scripts/capabilities.py, tests/test_capabilities.py, AGENTS.md, .claude/skills/*/SKILL.md (4 files), docs/api/capabilities*.md (4 files).
