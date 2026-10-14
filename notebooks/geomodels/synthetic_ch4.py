"""The synthetic methane spectrum shared by the radtrax notebooks.

HITRAN is not reachable from the environments these notebooks were written
in, so radtrax · 1 and radtrax · 2 use one synthetic spectrum: Lorentz lines
drawn at random in the 1.66 μm and 2.3 μm methane bands, scaled so that the
Sentinel-2 band 11 and band 12 averages of the cross-section are chosen
values (σ̄₁₁ = 0.04, σ̄₁₂ = 0.40 m²/kg by default, a modelling choice).

Both notebooks call :func:`methane_cross_section` with their own generator,
seeded once, so the two pages share the same spectrum draw for draw. This
module is a stand-in for radtrax, which does not exist yet; it moves there
when radtrax does (radtrax · 3, upcoming work). ``tests/test_synthetic_ch4.py``
checks it.
"""

from __future__ import annotations

import numpy as np

#: monochromatic wavelength grid, nm                                    (G,)
WL = np.arange(1500.0, 2500.0, 0.01)
#: Lorentz half-width, nm: broadened to keep the grid coarse
HWHM = 0.1


def lines(rng: np.random.Generator, n: int, lo: float, hi: float, centre: float,
          width: float) -> tuple[np.ndarray, np.ndarray]:
    """Line positions, uniform over [lo, hi], and log-normal strengths.

    Strengths sit under a Gaussian envelope exp(−(λ − centre)² / 2 width²).
    """
    pos = rng.uniform(lo, hi, n)
    return pos, rng.lognormal(0.0, 0.8, n) * np.exp(-0.5 * ((pos - centre) / width) ** 2)


def lorentz_sum(pos: np.ndarray, s: np.ndarray, wl: np.ndarray = WL) -> np.ndarray:
    """σ_mono(λ) = Σ_i s_i (HWHM/π) / ((λ − λ_i)² + HWHM²), cut at ±4 nm.   (G,)"""
    out = np.zeros_like(wl)
    for p, si in zip(pos, s, strict=True):
        sl = slice(*np.searchsorted(wl, [p - 4.0, p + 4.0]))
        out[sl] += si * (HWHM / np.pi) / ((wl[sl] - p) ** 2 + HWHM**2)
    return out


def band_mean(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """⟨x⟩_b = ∫ x f dλ / ∫ f dλ, on the shared grid."""
    return (x * f).sum(-1) / f.sum()


def methane_cross_section(rng: np.random.Generator, f11: np.ndarray, f12: np.ndarray,
                          sig12: float = 0.40, ratio11: float = 0.10) -> np.ndarray:
    """The synthetic CH₄ cross-section σ(λ) on :data:`WL`, in m²/kg.     (G,)

    Draws 3,000 lines in the 2.3 μm band, then 600 in the 1.66 μm band, from
    ``rng`` (in that order, so a caller's later draws are unchanged), and
    scales them so that ⟨σ⟩₁₂ = ``sig12`` and ⟨σ⟩₁₁ = ``ratio11`` · ``sig12``
    under the band responses ``f12`` and ``f11`` sampled on :data:`WL`.
    """
    band_23 = lorentz_sum(*lines(rng, 3000, 2165.0, 2420.0, 2310.0, 70.0))
    band_166 = lorentz_sum(*lines(rng, 600, 1620.0, 1710.0, 1666.0, 15.0))
    sigma = band_23 * sig12 / band_mean(band_23, f12)
    return sigma + band_166 * (ratio11 * sig12 - band_mean(sigma, f11)) / band_mean(
        band_166, f11
    )


def sentinel2_band_responses(satellite: str = "S2A") -> tuple[np.ndarray, np.ndarray]:
    """Sentinel-2 band 11 and 12 responses from georeader, sampled on :data:`WL`."""
    from georeader.readers.S2_SAFE_reader import read_srf

    srf = read_srf(satellite)
    f11 = np.interp(WL, srf.index.values, srf["B11"].values, left=0, right=0)
    f12 = np.interp(WL, srf.index.values, srf["B12"].values, left=0, right=0)
    return f11, f12
