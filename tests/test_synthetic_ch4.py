"""The synthetic methane spectrum shared by the radtrax notebooks.

radtrax · 1 and radtrax · 2 both draw their spectrum from
``notebooks/geomodels/synthetic_ch4.py``; the pages compare numbers across the
two, so the construction must be deterministic for a seed and must hit the
band-averaged cross-sections it is scaled to. Box-shaped band responses stand
in for Sentinel-2's, so the test needs no SRF table.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "synthetic_ch4", ROOT / "notebooks" / "geomodels" / "synthetic_ch4.py"
)
assert _spec is not None and _spec.loader is not None
ch4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch4)


def _boxes() -> tuple[np.ndarray, np.ndarray]:
    # B11-like 1569-1658 nm and B12-like 2113-2286 nm boxes on the shared grid
    f11 = ((ch4.WL >= 1569) & (ch4.WL <= 1658)).astype(float)
    f12 = ((ch4.WL >= 2113) & (ch4.WL <= 2286)).astype(float)
    return f11, f12


def test_band_means_hit_their_targets() -> None:
    f11, f12 = _boxes()
    sigma = ch4.methane_cross_section(np.random.default_rng(0), f11, f12, 0.40, 0.10)
    assert np.isclose(ch4.band_mean(sigma, f12), 0.40, rtol=1e-12)
    assert np.isclose(ch4.band_mean(sigma, f11), 0.04, rtol=1e-12)
    assert np.all(sigma >= 0)


def test_same_seed_same_spectrum_and_same_stream() -> None:
    f11, f12 = _boxes()
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    a = ch4.methane_cross_section(rng_a, f11, f12)
    b = ch4.methane_cross_section(rng_b, f11, f12)
    np.testing.assert_array_equal(a, b)
    # the generator is left in the same state, so a notebook's later draws match
    assert rng_a.random() == rng_b.random()
