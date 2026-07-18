"""Bootstrap Particle Filter (BPF) with systematic resampling.

Two public entry points
-----------------------
sir_filter              – single scalar observation channel
sir_filter_multichannel – aligned dict observations, tempered multi-channel likelihood

Design notes
------------
* ESS threshold defaults to 0.45 N (matching the SVEIAHR paper).
* Posterior quantiles are computed from *weighted* samples, not unweighted
  particle arrays, so they reflect true posterior uncertainty.
* Log-weights are carried between steps with log-sum-exp stabilisation and
  reset to uniform after every resample event.
* The multichannel variant exposes per-channel tempering weights (alpha_k)
  and dispersions (phi_k) to allow generalised-Bayesian down-weighting of
  mis-specified severity channels.
"""
from __future__ import annotations

import numpy as np

from ..observations.likelihoods import loglik
from ..observations.multi_signal import joint_loglik  # noqa: F401 (re-exported)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """O(N) systematic resampling with a single uniform draw."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(weights), positions)


def _ess(weights: np.ndarray) -> float:
    """Effective sample size  1 / sum(w_i^2)."""
    return float(1.0 / np.sum(np.square(weights)))


def _stabilise_and_normalise(log_weights: np.ndarray) -> np.ndarray:
    """Numerically stable log-sum-exp normalisation; returns proper weights."""
    lw = np.nan_to_num(
        np.asarray(log_weights, dtype=float),
        nan=-1e12, neginf=-1e12, posinf=0.0,
    )
    lw -= lw.max()                                 # shift for numerical safety
    w = np.exp(np.clip(lw, -700.0, 0.0))
    total = w.sum()
    if total > 0.0 and np.isfinite(total):
        return w / total
    return np.full(len(lw), 1.0 / len(lw))


def _weighted_quantile_draws(
    particles: np.ndarray,
    weights: np.ndarray,
    quantiles: tuple[float, ...],
    n_draw: int,
    rng: np.random.Generator,
) -> dict[float, np.ndarray]:
    """Sample *n_draw* indices proportional to *weights*, return per-compartment quantiles."""
    n = len(weights)
    idx = rng.choice(n, size=min(n_draw, n), replace=True, p=weights)
    draws = particles[idx]
    return {q: np.quantile(draws, q, axis=0) for q in quantiles}


# ---------------------------------------------------------------------------
# Public: single-channel filter
# ---------------------------------------------------------------------------

def sir_filter(
    observations,
    transition,
    observation_mean,
    x0,
    n_particles: int = 500,
    seed: int = 0,
    family: str = "negative_binomial",
    dispersion: float = 20.0,
    ess_threshold: float = 0.45,
    n_quantile_draws: int = 2000,
) -> list[dict]:
    """Bootstrap particle filter for a single observation channel.

    Parameters
    ----------
    observations : sequence of float
        Observed values, one per time step.
    transition : callable (x, rng) -> x_new
        Stochastic state transition for a single particle vector.
    observation_mean : callable (particles: ndarray shape (N,d)) -> ndarray (N,)
        Predicted mean for each particle.
    x0 : array-like
        Initial state vector; every particle starts here.
    n_particles : int
        Number of particles N.
    seed : int
        Random seed for full reproducibility.
    family : str
        ``"negative_binomial"`` or ``"poisson"``.
    dispersion : float
        Negative-binomial dispersion phi (ignored for Poisson).
    ess_threshold : float
        Resample when ESS < ess_threshold * N.  Default 0.45.
    n_quantile_draws : int
        Weighted posterior draws used for predictive quantile estimation.

    Returns
    -------
    list[dict]  – one dict per time step with keys:
        ``mean``  weighted posterior mean (shape d),
        ``q10``   10th weighted posterior quantile (shape d),
        ``q90``   90th weighted posterior quantile (shape d),
        ``ess``   effective sample size (float).
    """
    rng = np.random.default_rng(seed)
    x0_arr = np.asarray(x0, dtype=float)
    if x0_arr.ndim == 2:
        particles = x0_arr.copy()
        n_particles = len(particles)
    else:
        particles = np.tile(x0_arr, (n_particles, 1))
    log_weights = np.zeros(n_particles)
    threshold = ess_threshold * n_particles
    results: list[dict] = []

    for y in observations:
        # 1. Propagate ---------------------------------------------------------
        particles = np.array([transition(p, rng) for p in particles], dtype=float)

        # 2. Weight update -----------------------------------------------------
        means = np.asarray(observation_mean(particles), dtype=float)
        ll = np.asarray(
            loglik(float(y), means, family=family, dispersion=dispersion),
            dtype=float,
        )
        log_weights += ll
        weights = _stabilise_and_normalise(log_weights)
        log_weights = np.log(np.maximum(weights, 1e-300))

        # 3. ESS check and resampling ------------------------------------------
        ess = _ess(weights)
        if ess < threshold:
            idx = _systematic_resample(weights, rng)
            particles = particles[idx]
            log_weights = np.zeros(n_particles)
            weights = np.full(n_particles, 1.0 / n_particles)

        # 4. Posterior summaries -----------------------------------------------
        q = _weighted_quantile_draws(particles, weights, (0.1, 0.9), n_quantile_draws, rng)
        results.append({
            "mean": weights @ particles,
            "q10":  q[0.1],
            "q90":  q[0.9],
            "ess":  ess,
        })

    return results


# ---------------------------------------------------------------------------
# Public: multi-channel filter
# ---------------------------------------------------------------------------

def sir_filter_multichannel(
    rows,
    transition,
    observation_means: dict,
    x0,
    n_particles: int = 500,
    seed: int = 0,
    channel_weights: dict | None = None,
    dispersions: dict | None = None,
    ess_threshold: float = 0.45,
    n_quantile_draws: int = 2000,
) -> list[dict]:
    """Bootstrap particle filter for aligned multi-channel surveillance data.

    Parameters
    ----------
    rows : sequence of dict
        One dict per time step; keys are channel names, values are scalar
        observations.  Channels with ``None`` or ``nan`` values are excluded
        from the weight update at that step (missing-data convention).
    transition : callable (x, rng[, row]) -> x_new
        Stochastic state transition.  If it accepts three arguments the
        current observation ``row`` is passed (for exogenous inputs such as
        daily vaccination counts).
    observation_means : dict[str, callable]
        Maps each channel name to a function
        ``(particles: ndarray (N,d)) -> ndarray (N,)``.
    x0 : array-like
        Initial state vector.
    n_particles : int
        Number of particles N.
    seed : int
        Random seed.
    channel_weights : dict[str, float] or None
        Per-channel tempering weights alpha_k (generalised-Bayesian
        down-weighting).  All channels default to 1.0.
    dispersions : dict[str, float] or None
        Per-channel negative-binomial dispersion phi_k.  Default 20.0.
    ess_threshold : float
        Resample when ESS < ess_threshold * N.  Default 0.45.
    n_quantile_draws : int
        Weighted posterior draws used for predictive quantile estimation.

    Returns
    -------
    list[dict]  – one dict per time step with keys:
        ``mean``             weighted posterior mean (shape d),
        ``q10``              10th weighted posterior quantile (shape d),
        ``q90``              90th weighted posterior quantile (shape d),
        ``ess``              effective sample size (float),
        ``active_channels``  list of channel names used at this step.
    """
    rng = np.random.default_rng(seed)
    x0_arr = np.asarray(x0, dtype=float)
    if x0_arr.ndim == 2:
        particles = x0_arr.copy()
        n_particles = len(particles)
    else:
        particles = np.tile(x0_arr, (n_particles, 1))
    log_weights = np.zeros(n_particles)
    threshold = ess_threshold * n_particles
    channel_weights = channel_weights or {}
    dispersions = dispersions or {}
    results: list[dict] = []

    for row in rows:
        # 1. Propagate ---------------------------------------------------------
        try:
            particles = np.array(
                [transition(p, rng, row) for p in particles], dtype=float
            )
        except TypeError:
            particles = np.array(
                [transition(p, rng) for p in particles], dtype=float
            )

        # 2. Compute predicted means for every registered channel ---------------
        means = {
            k: np.asarray(fn(particles), dtype=float)
            for k, fn in observation_means.items()
        }

        # 3. Update log-weights from available channels ------------------------
        active: list[str] = []
        for k, y in row.items():
            if k not in means:
                continue
            try:
                yf = float(y)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(yf):
                continue
            alpha = channel_weights.get(k, 1.0)
            phi   = dispersions.get(k, 20.0)
            ll = np.asarray(
                loglik(yf, means[k], family="negative_binomial", dispersion=phi),
                dtype=float,
            )
            log_weights += alpha * ll
            active.append(k)

        weights = _stabilise_and_normalise(log_weights)
        log_weights = np.log(np.maximum(weights, 1e-300))

        # 4. ESS check and resampling ------------------------------------------
        ess = _ess(weights)
        if ess < threshold:
            idx = _systematic_resample(weights, rng)
            particles = particles[idx]
            log_weights = np.zeros(n_particles)
            weights = np.full(n_particles, 1.0 / n_particles)

        # 5. Posterior summaries -----------------------------------------------
        q = _weighted_quantile_draws(particles, weights, (0.1, 0.9), n_quantile_draws, rng)
        results.append({
            "mean":            weights @ particles,
            "q10":             q[0.1],
            "q90":             q[0.9],
            "ess":             ess,
            "active_channels": active,
        })

    return results
