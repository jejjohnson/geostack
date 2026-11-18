# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Background · the modeling cycle, example 3: persistency
#
# Regenerates every measured number on `docs/design/pages/cycle-persistency.html`
# (stub: `docs/design/cycle_persistency.md`). One facility emits in events for a year;
# two instruments overpass it. The question is how often it emits, answered three
# ways: the detection fraction, a constant-rate point process thinned by each
# instrument's detection curve, and the same with a daily cycle.
#
# Needs plumax with its `population` extra (the `geomodels` pixi environment). It was
# run against plumax at 290f31f, newer than the submodule pin. Every setting is read
# from `configs/background/cycle_persistency.yaml` and recorded in the output. Not a
# stage of the DVC pipeline; run it by hand, after `cycle_quantification.py`:
#
#     pixi run -e geomodels python notebooks/background/cycle_persistency.py
#
# It writes `docs/images/background_cycle_persistency/results.json`. Seeded; timings
# depend on the machine. About four minutes.

# %%
"""Example 3 of the modeling cycle: how often a facility emits, from a catalogue.

The intensities, marks, detection curves, the thinned point-process likelihood and
the persistency metrics are plumax's. This file simulates one facility, composes
them and measures.
"""

from __future__ import annotations

import json
import math
import time

import einx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from numpyro.infer import MCMC, NUTS
from plumax.population.fitting import TMTPPPosterior, log_thinned_rate, tmtpp_iw_model
from plumax.population.importance import EventEvidence
from plumax.population.intensity import INTENSITY_REGISTRY, DiurnalSinusoidalIntensity
from plumax.population.marks import LognormalMark
from plumax.population.persistency import (
    expected_count,
    expected_wait_time,
    occurrence_probability,
    posterior_metric,
)
from plumax.population.pod import InstrumentPOD, calibration_from_draw, log_pod_union
from plumax.population.processes import padded_event_buffer
from plumax.population.totals import total_emission

# %%
# Settings come from configs/background/cycle_persistency.yaml; the recorded output
# goes to docs/images/background_cycle_persistency (set BACKGROUND_OUT to write
# elsewhere).
import os as _os
import pathlib as _pathlib

from omegaconf import OmegaConf

_root = _pathlib.Path.cwd()
while not (_root / "myst.yml").exists() and _root != _root.parent:
    _root = _root.parent
OUT = _pathlib.Path(
    _os.environ.get("BACKGROUND_OUT", _root / "docs/images/background_cycle_persistency")
)
OUT.mkdir(parents=True, exist_ok=True)
CFG = OmegaConf.load(_root / "configs/background/cycle_persistency.yaml")

# %%
# Shapes: E events; P overpasses; K = 2 instruments; S draws per event; D posterior draws
# Units: [t] = days; [λ] = events day⁻¹; [Q] = kg h⁻¹; [D_e] = h

NAMES = ("hyperspectral", "multispectral")
MARK = LognormalMark()


def instrument_pods() -> list[InstrumentPOD]:
    """Each instrument's published detection curve."""
    # today (plumax)
    # P_d(Q) = 1 / (1 + (Q₅₀ / Q)^k)
    return [
        InstrumentPOD(
            name=name,
            model="log_logistic",
            q50=CFG.instruments[name].q50_kg_h,
            q50_std=CFG.instruments[name].q50_log_std,
            slope=CFG.instruments[name].slope,
        )
        for name in NAMES
    ]


# === facility ===
def simulate_events(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The truth: event start times [days] and emission rates [kg h⁻¹]."""
    truth = CFG.truth.intensity
    # today (plumax)
    # λ(t) = max(0, λ₀ + A sin(2πt − φ))
    intensity = DiurnalSinusoidalIntensity(
        lambda_0=truth.lambda_0, amplitude=truth.amplitude, phase_hours=truth.phase_hours
    )
    # thinning: candidates at the rate λ_max, each kept with probability λ(t) / λ_max
    ceiling = float(intensity.max_intensity())
    n_candidates = rng.poisson(ceiling * CFG.window_days)
    candidates = np.sort(rng.uniform(0.0, CFG.window_days, n_candidates))
    keep = rng.uniform(size=n_candidates) < np.asarray(intensity(candidates)) / ceiling
    times = candidates[keep]
    # Q = exp(μ + σ ε)
    log_median = math.log(CFG.truth.mark.median_kg_h)
    rates = np.exp(log_median + CFG.truth.mark.sigma * rng.standard_normal(times.size))
    return times, rates


def simulate_overpasses(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Clear overpass times [days] and which instrument made each, no two overlapping."""
    visible = CFG.truth.event_duration_h / 24.0
    times, instruments = [], []
    for index, name in enumerate(NAMES):
        settings = CFG.instruments[name]
        days = np.arange(0.5 * settings.revisit_days, CFG.window_days, settings.revisit_days)
        days = np.floor(days[rng.uniform(size=days.size) < settings.clear_fraction])
        hours = rng.uniform(settings.local_hours[0], settings.local_hours[1], days.size)
        times.append(days + hours / 24.0)
        instruments.append(np.full(days.size, index))
    times, instruments = np.concatenate(times), np.concatenate(instruments)
    order = np.argsort(times)
    times, instruments = times[order], instruments[order]
    # an overpass sees the events that started in the hours before it; drop one whose
    # window would overlap the previous overpass's
    keep = np.concatenate([[True], np.diff(times) > visible])
    return times[keep], instruments[keep]


def detect(
    event_times: np.ndarray,
    event_rates: np.ndarray,
    pass_times: np.ndarray,
    pass_instruments: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Which events an overpass saw, and what it reported for each."""
    visible = CFG.truth.event_duration_h / 24.0
    pods = instrument_pods()
    # the overpass whose window (τ − D_e, τ] holds each event, if any
    following = np.searchsorted(pass_times, event_times)
    following = np.clip(following, 0, pass_times.size - 1)
    in_view = (pass_times[following] - event_times < visible) & (
        pass_times[following] >= event_times
    )
    coverage = np.zeros((event_times.size, len(NAMES)), bool)
    coverage[np.arange(event_times.size), pass_instruments[following]] = True
    # today (plumax)
    # log P_d^∪(Q), here of the one instrument that was looking
    calibration = calibration_from_draw(
        pods,
        {f"pod.{pod.name}.{field}": getattr(pod, field) for pod in pods for field in ("q50", "slope")},
    )
    log_pod = log_pod_union(pods, coverage=jnp.asarray(coverage), calibration=calibration)
    probability = np.exp(np.asarray(log_pod(jnp.asarray(event_rates))))
    detected = in_view & (rng.uniform(size=event_times.size) < probability)
    return {
        "in_view": in_view,
        "detected": detected,
        "pass_index": following,
        "coverage": coverage,
        "probability": probability,
    }


def reported_rates(
    rates: np.ndarray, log_std: float, rng: np.random.Generator
) -> Float[np.ndarray, "E S"]:
    """What example 2 hands over for each detection: draws of Q around its estimate."""
    # log Q̂ = log Q + s ε
    estimate = np.log(rates) + log_std * rng.standard_normal(rates.size)
    # log Q⁽ˢ⁾ = log Q̂ + s ε_s
    noise = rng.standard_normal((rates.size, CFG.evidence_draws))
    draws = einx.add("e, e s -> e s", estimate, log_std * noise)
    return np.exp(draws)


# === fraction ===
def detection_fraction(pass_selection: np.ndarray, pass_has_detection: np.ndarray) -> dict:
    """Model 1: the share of overpasses with a detection."""
    looks = int(pass_selection.sum())
    hits = int((pass_selection & pass_has_detection).sum())
    return {"overpasses": looks, "with_detection": hits, "fraction": round(hits / looks, 4)}


# === process ===
def coverage_intervals(
    pass_times: np.ndarray, pass_instruments: np.ndarray
) -> list[tuple[float, float, np.ndarray]]:
    """Who was looking when: one window of length D_e before each overpass."""
    visible = CFG.truth.event_duration_h / 24.0
    one_hot = np.eye(len(NAMES), dtype=bool)
    return [
        (float(time - visible), float(time), one_hot[instrument])
        for time, instrument in zip(pass_times, pass_instruments, strict=True)
    ]


def fit_process(
    intensity: str,
    detection_times: np.ndarray,
    rate_draws: Float[np.ndarray, "E S"],
    detection_coverage: np.ndarray,
    intervals: list,
    key: Array,
) -> TMTPPPosterior:
    """Models 2 and 3: a marked point process thinned by who looked and what they see."""
    window = (0.0, CFG.window_days)
    order = np.argsort(detection_times)
    # today (plumax)
    event_times, mask = padded_event_buffer(detection_times[order], window)
    # today (plumax)
    # flat per-event prior: every draw weighs the same
    evidence = EventEvidence(
        log_weights=jnp.zeros(rate_draws.shape), q_samples=jnp.asarray(rate_draws[order])
    )
    # today (plumax)
    # log L = Σᵢ [log λ(tᵢ) + mᵢ(θ)] − Σ_w Λ_w p̄_w(θ)
    sampler = MCMC(
        NUTS(tmtpp_iw_model),
        num_warmup=CFG.sampler.warmup,
        num_samples=CFG.sampler.samples,
        progress_bar=False,
    )
    sampler.run(
        key,
        event_times,
        mask,
        evidence,
        mark=MARK,
        intensity=intensity,
        window=window,
        pods=instrument_pods(),
        coverage=jnp.asarray(detection_coverage[order]),
        coverage_intervals=intervals,
        support=tuple(CFG.support_kg_h),
        n_nodes=CFG.quadrature_nodes,
    )
    return TMTPPPosterior(intensity, MARK, dict(sampler.get_samples()))


# === compare ===
def expected_detections(
    posterior: TMTPPPosterior, pass_times: np.ndarray, pass_instruments: np.ndarray
) -> Float[np.ndarray, "D P"]:
    """Λ_w p̄_w for every overpass and a thinned set of posterior draws."""
    visible = CFG.truth.event_duration_h / 24.0
    pods = instrument_pods()
    step = max(1, posterior.num_samples // CFG.predictive_draws)
    # nine nodes across each window, for the trapezoid rule
    offsets = jnp.linspace(-visible, 0.0, 9)
    nodes = einx.add("p, n -> p n", jnp.asarray(pass_times), offsets)

    def one_draw(draw: dict[str, Array]) -> Float[Array, " P"]:
        kernel = posterior.kernel(draw)
        theta = posterior.theta(draw)
        # Λ_w = ∫_w λ(t) dt
        rates = jax.vmap(kernel)(nodes)
        expected_events = jnp.trapezoid(rates, dx=visible / 8)
        calibration = calibration_from_draw(pods, draw)
        shares = []
        for index in range(len(NAMES)):
            covering = jnp.arange(len(NAMES)) == index
            log_pod = log_pod_union(pods, coverage=covering, calibration=calibration)
            # today (plumax)
            # p̄ = ∫ f(Q | θ) P_d(Q) dQ
            log_share = log_thinned_rate(
                MARK, theta, log_pod, tuple(CFG.support_kg_h), n_nodes=CFG.quadrature_nodes
            )
            shares.append(jnp.exp(log_share))
        share = jnp.stack(shares)[jnp.asarray(pass_instruments)]
        return expected_events * share

    draws = {name: values[::step] for name, values in posterior.samples.items()}
    return np.asarray(jax.lax.map(one_draw, draws))


def group_check(expected: Float[np.ndarray, "D P"], observed: np.ndarray, groups: dict) -> dict:
    """Observed and expected detections per group, and a χ² over the groups."""
    table, chi_squared = {}, 0.0
    for name, selection in groups.items():
        predicted = expected[:, selection].sum(1)
        mean = float(predicted.mean())
        seen = int(observed[selection].sum())
        table[name] = {"observed": seen, "expected": round(mean, 1)}
        # Poisson: variance = mean
        chi_squared += (seen - mean) ** 2 / mean
    table["chi_squared"] = round(chi_squared, 1)
    return table


def quantiles(samples: Array) -> list[float]:
    """The 5 %, 50 % and 95 % quantiles of a set of draws."""
    return [round(float(value), 4) for value in np.quantile(np.asarray(samples), [0.05, 0.5, 0.95])]


def persistency_metrics(posterior: TMTPPPosterior) -> dict:
    """The four operational numbers, each with its posterior quantiles."""
    window = CFG.metrics.maintenance_window_h / 24.0
    # today (plumax)
    # P(N(t, t + h) ≥ 1) = 1 − exp(−Λ(t, t + h))
    morning = posterior_metric(
        occurrence_probability, posterior, CFG.metrics.morning_start_h / 24.0, window
    )
    night = posterior_metric(
        occurrence_probability, posterior, CFG.metrics.night_start_h / 24.0, window
    )
    # today (plumax)
    # 𝔼[Δt | t₀] = ∫₀^∞ exp(−Λ(t₀, t₀ + s)) ds
    wait = posterior_metric(expected_wait_time, posterior, CFG.metrics.wait_from_h / 24.0)
    # today (plumax)
    # 𝔼[N(0, T)] = Λ(T)
    count = posterior_metric(
        expected_count,
        posterior,
        0.0,
        CFG.window_days,
        nodes_per_day=CFG.metrics.nodes_per_day,
    )
    return {
        "occurrence_morning": quantiles(morning["samples"]),
        "occurrence_night": quantiles(night["samples"]),
        "wait_hours": quantiles(24.0 * wait["samples"]),
        "events_per_year": quantiles(count["samples"]),
    }


def true_metrics() -> dict:
    """The same four numbers from the truth's own intensity."""
    truth = CFG.truth.intensity
    intensity = DiurnalSinusoidalIntensity(
        lambda_0=truth.lambda_0, amplitude=truth.amplitude, phase_hours=truth.phase_hours
    )
    window = CFG.metrics.maintenance_window_h / 24.0
    return {
        "occurrence_morning": round(
            float(occurrence_probability(intensity, CFG.metrics.morning_start_h / 24.0, window)), 4
        ),
        "occurrence_night": round(
            float(occurrence_probability(intensity, CFG.metrics.night_start_h / 24.0, window)), 4
        ),
        "wait_hours": round(
            24.0 * float(expected_wait_time(intensity, CFG.metrics.wait_from_h / 24.0)), 4
        ),
        "events_per_year": round(float(expected_count(intensity, 0.0, CFG.window_days)), 1),
    }


def site_summary(posterior: TMTPPPosterior) -> dict:
    """Posterior quantiles of every sampled site."""
    return {name: quantiles(values) for name, values in posterior.samples.items()}


# === run ===
if __name__ == "__main__":
    results: dict = {"config": OmegaConf.to_container(CFG)}
    rng = np.random.default_rng(CFG.seed)
    key_constant, key_diurnal = jax.random.split(jax.random.key(CFG.seed))
    quantification = json.loads((_root / CFG.input).read_text())
    log_std = quantification["event"]["log_rate_std"]
    results["event_log_std"] = log_std
    duration = CFG.truth.event_duration_h

    # --- the facility's year, and what the overpasses caught of it ---
    event_times, event_rates = simulate_events(rng)
    pass_times, pass_instruments = simulate_overpasses(rng)
    seen = detect(event_times, event_rates, pass_times, pass_instruments, rng)
    detected = seen["detected"]
    rate_draws = reported_rates(event_rates[detected], log_std, rng)
    pass_detections = np.bincount(seen["pass_index"][detected], minlength=pass_times.size)
    hours = 24.0 * (pass_times % 1.0)
    results["truth"] = {
        "events": int(event_times.size),
        "events_in_view": int(seen["in_view"].sum()),
        "detections": int(detected.sum()),
        "overpasses": {name: int((pass_instruments == i).sum()) for i, name in enumerate(NAMES)},
        "mean_rate_kg_h": round(float(event_rates.mean()), 1),
        "mean_detected_rate_kg_h": round(float(event_rates[detected].mean()), 1),
        "total_mass_t": round(float(event_rates.sum()) * duration / 1000.0, 1),
        "mass_of_detections_t": round(float(event_rates[detected].sum()) * duration / 1000.0, 1),
        "overpasses_with_two_detections": int((pass_detections > 1).sum()),
        "metrics": true_metrics(),
    }

    # --- model 1: the detection fraction, overall and by who looked when ---
    groups = {
        f"{name}_{half}": (pass_instruments == index) & selection
        for index, name in enumerate(NAMES)
        for half, selection in (
            ("morning", hours < CFG.metrics.noon_h),
            ("afternoon", hours >= CFG.metrics.noon_h),
        )
    }
    groups = {name: selection for name, selection in groups.items() if selection.any()}
    everything = np.ones(pass_times.size, bool)
    has_detection = pass_detections > 0
    overall = detection_fraction(everything, has_detection)
    results["fraction"] = {
        "all": overall,
        **{name: detection_fraction(selection, has_detection) for name, selection in groups.items()},
    }
    # its prediction for each group: the same fraction of that group's overpasses
    fraction_expected = np.full((1, pass_times.size), overall["fraction"])
    results["fraction"]["check"] = group_check(fraction_expected, has_detection, groups)

    # --- models 2 and 3: the thinned point process, constant and with a daily cycle ---
    intervals = coverage_intervals(pass_times, pass_instruments)
    posteriors = {}
    for intensity, key in (("constant", key_constant), ("diurnal", key_diurnal)):
        start = time.perf_counter()
        posterior = fit_process(
            intensity, event_times[detected], rate_draws, seen["coverage"][detected], intervals, key
        )
        seconds = time.perf_counter() - start
        expected = expected_detections(posterior, pass_times, pass_instruments)
        posteriors[intensity] = posterior
        results[intensity] = {
            "seconds": round(seconds, 1),
            "sites": site_summary(posterior),
            "check": group_check(expected, pass_detections, groups),
            "metrics": persistency_metrics(posterior),
        }
        # today (plumax)
        # M_total = Λ 𝔼_f[Q] D_e,  M_missing = Λ 𝔼_f[Q (1 − P_d(Q))] D_e
        total = total_emission(
            posterior,
            horizon=(0.0, CFG.window_days),
            event_duration=duration,
            log_pod=None,
            q_min=CFG.support_kg_h[0],
            q_max=CFG.support_kg_h[1],
            pods=instrument_pods(),
            nodes_per_day=CFG.metrics.nodes_per_day,
        )
        results[intensity]["total_mass_t"] = quantiles(total.samples_kg / 1000.0)
    results["fraction"]["implied_events_per_year"] = round(
        overall["fraction"] / (duration / 24.0) * CFG.window_days, 1
    )

    with open(__file__) as source_file:
        results["lines"] = len(source_file.read().splitlines())
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "config"}, indent=1))
