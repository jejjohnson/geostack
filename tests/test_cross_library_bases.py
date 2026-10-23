"""spectraldiffx and geonnax compute the same spherical bases.

Both libraries compute the orthonormal associated Legendre functions

    P̄ₗᵐ(μ) = (-1)ᵐ √((2l+1)/2 · (l-m)!/(l+m)!) Pₗᵐ(μ),   ∫₋₁¹ P̄ₗᵐ P̄ₗ'ᵐ dμ = δₗₗ',

spectraldiffx for its pseudo-spectral transform on a Gauss grid (numerics),
geonnax for its traced bases and dense SHT layer (GeoML). They may not import
each other: spectraldiffx sits below geonnax (docs/api/dependency_graph.md,
rule 1), and geonnax takes no GeoNumerics runtime dependency. So each keeps
its own implementation of the same recurrence, and this is the one place that
holds them to each other: the Legendre tables to 1e-12 at T127 and T255, and
the two spherical harmonic transforms on the same band-limited field
(jejjohnson/geostack#63, epic #62).

Runs in the ``geoml`` pixi environment (which installs numerics + geoml);
skips where either library is absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


sdx = pytest.importorskip("spectraldiffx")
gbasis = pytest.importorskip("geonnax.basis")
glayers = pytest.importorskip("geonnax.layers")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _geonnax_tables(mu: np.ndarray, l_max: int) -> np.ndarray:
    """geonnax's P̄ₗᵐ(μⱼ) rearranged to spectraldiffx's [m, l, j] layout."""
    with jax.enable_x64(True):
        cols = np.asarray(
            gbasis.associated_legendre(jnp.asarray(mu, jnp.float64), l_max)
        )  # (Ny, (L+1)(L+2)/2), columns (l, m)
    tables = np.zeros((l_max + 1, l_max + 1, mu.size))  # (m, l, j)
    for k, (ell, m) in enumerate(gbasis.associated_legendre_indices(l_max)):
        tables[m, ell] = cols[:, k]
    return tables


@pytest.mark.parametrize("n_lat", [128, 256], ids=["T127", "T255"])
def test_associated_legendre_tables_agree(n_lat):
    # Same recurrence, normalisation and Condon-Shortley phase on the same
    # Gauss nodes: measured agreement 1.7e-13 (T127) and 1.2e-12 (T255);
    # 1e-11 leaves margin for platform rounding.
    with jax.enable_x64(True):
        grid = sdx.SphericalGrid2D.from_N_L(Nx=2 * n_lat, Ny=n_lat)
        mu = np.asarray(grid.cos_theta)  # (Ny,), North to South
        sdx_tables = np.asarray(grid.legendre_table)[:n_lat]  # (m <= l_max, l, j)
    np.testing.assert_allclose(
        sdx_tables, _geonnax_tables(mu, n_lat - 1), rtol=0, atol=1e-11
    )


def test_spherical_harmonic_transforms_agree():
    # A random band-limited field in geonnax's real tesseral basis
    #   Yₗ⁰ = P̄ₗ⁰/√(2π),  Yₗᵐ = P̄ₗᵐ cos(mφ)/√π,  Yₗ⁻ᵐ = P̄ₗᵐ sin(mφ)/√π  (m > 0)
    # is synthesised by geonnax's dense layer and analysed by spectraldiffx's
    # FFT + Legendre transform. With an unnormalised FFT over Nx longitudes,
    #   û(l, 0) = Nx/√(2π) · aₗ⁰,   û(l, m) = Nx/(2√π) · (aₗᵐ - i aₗ⁻ᵐ)  (m > 0),
    # so the coefficients must come back. Measured agreement 1.8e-13.
    l_max, n_lat, n_lon = 31, 32, 64
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal((l_max + 1) ** 2)  # geonnax layout: l, m = -l..l
    with jax.enable_x64(True):
        sht = glayers.SphericalHarmonicTransform.init(n_lat, n_lon, l_max)
        field = np.asarray(sht.inverse(jnp.asarray(coeffs)[None]))[0]  # (n_lat, n_lon)
        # geonnax orders latitudes by ascending cos θ; spectraldiffx North to South.
        grid = sdx.SphericalGrid2D.from_N_L(Nx=n_lon, Ny=n_lat)
        u_hat = np.asarray(grid.transform(jnp.asarray(field[::-1])))  # (l, m FFT)

    recovered = np.empty_like(coeffs)
    k = 0
    for ell in range(l_max + 1):
        for m in range(-ell, ell + 1):
            c = u_hat[ell, abs(m)]
            if m == 0:
                recovered[k] = math.sqrt(2 * math.pi) / n_lon * c.real
            elif m > 0:
                recovered[k] = 2 * math.sqrt(math.pi) / n_lon * c.real
            else:
                recovered[k] = -2 * math.sqrt(math.pi) / n_lon * c.imag
            k += 1
    np.testing.assert_allclose(recovered, coeffs, rtol=0, atol=1e-10)
