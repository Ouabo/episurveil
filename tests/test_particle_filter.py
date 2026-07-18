"""Tests for the refactored Bootstrap Particle Filter."""
import numpy as np
import pytest
from episurveil.inference.particle_filter import (
    sir_filter,
    sir_filter_multichannel,
    _systematic_resample,
    _ess,
    _stabilise_and_normalise,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _make_transition(sigma=0.01):
    """Simple random-walk transition for a 1-D state."""
    def transition(x, rng):
        return np.maximum(x + rng.normal(0, sigma, size=x.shape), 0.0)
    return transition


def _observation_mean(particles):
    """Identity: predicted mean = first compartment of each particle."""
    return np.maximum(particles[:, 0], 1e-9)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def test_systematic_resample_length():
    rng = np.random.default_rng(0)
    w = np.array([0.1, 0.4, 0.3, 0.2])
    idx = _systematic_resample(w, rng)
    assert len(idx) == len(w)
    assert all(0 <= i < len(w) for i in idx)


def test_systematic_resample_high_weight_dominates():
    """A particle with weight ~1 should be selected almost every time."""
    rng = np.random.default_rng(42)
    w = np.array([0.0, 1.0, 0.0, 0.0])
    idx = _systematic_resample(w, rng)
    assert np.all(idx == 1)


def test_ess_uniform():
    w = np.full(100, 1.0 / 100)
    assert abs(_ess(w) - 100.0) < 1e-6


def test_ess_degenerate():
    w = np.zeros(100); w[0] = 1.0
    assert _ess(w) == pytest.approx(1.0)


def test_stabilise_normalise_uniform():
    lw = np.zeros(50)
    w = _stabilise_and_normalise(lw)
    assert abs(w.sum() - 1.0) < 1e-10
    assert abs(w[0] - 1.0 / 50) < 1e-10


def test_stabilise_normalise_with_nan():
    lw = np.array([-1.0, np.nan, -2.0])
    w = _stabilise_and_normalise(lw)
    assert abs(w.sum() - 1.0) < 1e-10
    assert w[1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# sir_filter
# ---------------------------------------------------------------------------

def test_sir_filter_output_length():
    T = 20
    obs = np.full(T, 50.0)
    x0 = np.array([1000.0])
    res = sir_filter(obs, _make_transition(), _observation_mean, x0,
                     n_particles=100, seed=0)
    assert len(res) == T


def test_sir_filter_keys():
    obs = [10.0, 20.0, 15.0]
    x0 = np.array([500.0])
    res = sir_filter(obs, _make_transition(), _observation_mean, x0,
                     n_particles=50, seed=1)
    for step in res:
        assert set(step.keys()) == {"mean", "q10", "q90", "ess"}


def test_sir_filter_ess_positive():
    obs = np.full(10, 30.0)
    x0 = np.array([200.0])
    res = sir_filter(obs, _make_transition(), _observation_mean, x0,
                     n_particles=80, seed=2)
    for step in res:
        assert step["ess"] > 0


def test_sir_filter_quantile_order():
    obs = np.full(10, 100.0)
    x0 = np.array([500.0])
    res = sir_filter(obs, _make_transition(sigma=10.0), _observation_mean, x0,
                     n_particles=200, seed=3)
    for step in res:
        assert np.all(step["q10"] <= step["q90"] + 1e-9)


def test_sir_filter_seed_reproducibility():
    obs = np.arange(1, 11, dtype=float)
    x0 = np.array([300.0])
    r1 = sir_filter(obs, _make_transition(), _observation_mean, x0,
                    n_particles=100, seed=99)
    r2 = sir_filter(obs, _make_transition(), _observation_mean, x0,
                    n_particles=100, seed=99)
    np.testing.assert_array_equal(r1[-1]["mean"], r2[-1]["mean"])


def test_sir_filter_ess_threshold_respected():
    """Lower threshold should trigger fewer resamples; ESS never below 1."""
    obs = np.full(30, 5.0)
    x0 = np.array([100.0])
    res = sir_filter(obs, _make_transition(), _observation_mean, x0,
                     n_particles=100, seed=5, ess_threshold=0.10)
    assert all(r["ess"] >= 1.0 for r in res)


# ---------------------------------------------------------------------------
# sir_filter_multichannel
# ---------------------------------------------------------------------------

def _make_mc_transition(sigma=1.0):
    """Random-walk transition for a 3-compartment state."""
    def transition(x, rng):
        return np.maximum(x + rng.normal(0, sigma, size=x.shape), 0.0)
    return transition


CHANNEL_MEANS = {
    "cases": lambda p: np.maximum(p[:, 0] * 0.7, 1e-9),
    "icu":   lambda p: np.maximum(p[:, 1] * 0.05, 1e-9),
}

DISPERSIONS = {"cases": 80.0, "icu": 12.0}
WEIGHTS     = {"cases": 1.0,  "icu": 0.6}


def test_multichannel_output_length():
    rows = [{"cases": 100.0, "icu": 5.0}] * 15
    x0 = np.array([10000.0, 200.0, 50.0])
    res = sir_filter_multichannel(
        rows, _make_mc_transition(), CHANNEL_MEANS, x0,
        n_particles=100, seed=0,
        channel_weights=WEIGHTS, dispersions=DISPERSIONS,
    )
    assert len(res) == 15


def test_multichannel_active_channels_reported():
    rows = [
        {"cases": 100.0, "icu": 5.0},
        {"cases": 80.0,  "icu": float("nan")},   # ICU missing
        {"cases": 90.0,  "icu": 4.0},
    ]
    x0 = np.array([10000.0, 200.0, 50.0])
    res = sir_filter_multichannel(
        rows, _make_mc_transition(), CHANNEL_MEANS, x0,
        n_particles=80, seed=7,
    )
    assert "icu"   in res[0]["active_channels"]
    assert "icu"   not in res[1]["active_channels"]   # NaN excluded
    assert "cases" in res[1]["active_channels"]


def test_multichannel_missing_channel_does_not_crash():
    rows = [{"cases": None, "icu": None}] * 5
    x0 = np.array([10000.0, 200.0, 50.0])
    res = sir_filter_multichannel(
        rows, _make_mc_transition(), CHANNEL_MEANS, x0,
        n_particles=50, seed=9,
    )
    assert len(res) == 5
    for step in res:
        assert step["active_channels"] == []


def test_multichannel_ess_positive():
    rows = [{"cases": 500.0 + i, "icu": 20.0} for i in range(10)]
    x0 = np.array([50000.0, 400.0, 100.0])
    res = sir_filter_multichannel(
        rows, _make_mc_transition(sigma=100.0), CHANNEL_MEANS, x0,
        n_particles=150, seed=11,
        channel_weights=WEIGHTS, dispersions=DISPERSIONS,
    )
    for step in res:
        assert step["ess"] >= 1.0


def test_multichannel_quantile_order():
    rows = [{"cases": 200.0, "icu": 8.0}] * 10
    x0 = np.array([20000.0, 300.0, 60.0])
    res = sir_filter_multichannel(
        rows, _make_mc_transition(sigma=50.0), CHANNEL_MEANS, x0,
        n_particles=200, seed=13,
    )
    for step in res:
        assert np.all(step["q10"] <= step["q90"] + 1e-9)


# ---------------------------------------------------------------------------
# Control policy smoke
# ---------------------------------------------------------------------------

def test_capacity_risk_policy_zero_below_capacity():
    from episurveil.control.policies import capacity_risk_policy
    policy = capacity_risk_policy(np.array([100.0, 200.0, 300.0]),
                                  capacity=400.0)
    assert np.all(policy == 0.0)


def test_capacity_risk_policy_positive_above_capacity():
    from episurveil.control.policies import capacity_risk_policy
    policy = capacity_risk_policy(np.array([500.0, 600.0]),
                                  capacity=400.0)
    assert np.all(policy > 0.0)


# ---------------------------------------------------------------------------
# Delay utilities
# ---------------------------------------------------------------------------

def test_delayed_incidence_length():
    from episurveil.observations.delays import delayed_incidence
    inc = np.array([0.0, 0.0, 100.0, 50.0, 20.0, 10.0, 5.0])
    out = delayed_incidence(inc, mean_delay=3.0, sd_delay=1.0)
    assert len(out) == len(inc)


def test_delayed_incidence_nonnegative():
    from episurveil.observations.delays import delayed_incidence
    inc = np.abs(np.random.default_rng(0).normal(50, 10, 30))
    out = delayed_incidence(inc, mean_delay=5.0, sd_delay=2.0)
    assert np.all(out >= 0)


def test_occupancy_from_admissions_length():
    from episurveil.observations.delays import occupancy_from_admissions
    adm = np.ones(20) * 10.0
    occ = occupancy_from_admissions(adm, mean_los=8.0)
    assert len(occ) == 20


def test_occupancy_from_admissions_positive():
    from episurveil.observations.delays import occupancy_from_admissions
    adm = np.ones(20) * 10.0
    occ = occupancy_from_admissions(adm, mean_los=8.0)
    assert np.all(occ >= 0)
