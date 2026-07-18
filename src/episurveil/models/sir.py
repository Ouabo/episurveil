"""
SIR model with time-varying transmission rate beta_t.

States      : S, I, R
Log-RW param: beta_t
Fixed params: gamma (recovery rate), Q_C (case detection probability)

ODE
---
  lambda_t = beta_t * I / N
  dS/dt = -lambda_t * S
  dI/dt =  lambda_t * S - gamma * I
  dR/dt =  gamma * I

Observations
------------
  cases  ~ NegBin(Q_C * beta_t * S * I / N, phi_cases)

Derived output
--------------
  R_eff(t) = beta_t * S_t / (N * gamma)
"""
from __future__ import annotations

import numpy as np
from episurveil.models.base import EpiModel


class SIRModel(EpiModel):
    """
    Parameters
    ----------
    N          : total population
    gamma      : recovery rate (day^-1), default 0.10 (10-day infectious period)
    Q_C        : case detection probability (fixed), default 0.50
    beta_init  : initial guess for beta, default 0.20
    sigma_beta : log-RW noise scale for beta, default 0.04
    beta_min/max : hard bounds on beta
    phi_cases  : NegBin dispersion for case channel
    """

    def __init__(
        self,
        N:          float = 1e6,
        gamma:      float = 0.10,
        Q_C:        float = 0.50,
        beta_init:  float = 0.20,
        sigma_beta: float = 0.04,
        beta_min:   float = 0.01,
        beta_max:   float = 2.00,
        phi_cases:  float = 50.0,
        I0_frac:    float = 1e-4,
    ):
        self.N          = float(N)
        self.gamma      = gamma
        self.Q_C        = Q_C
        self.beta_init  = beta_init
        self._sigma     = np.array([sigma_beta])
        self._log_min   = np.array([np.log(beta_min)])
        self._log_max   = np.array([np.log(beta_max)])
        self.phi_cases  = phi_cases
        self._rng       = np.random.default_rng(42)
        self._I0_frac   = I0_frac

    @property
    def state_names(self):
        return ["S", "I", "R"]

    @property
    def param_names(self):
        return ["beta"]

    @property
    def obs_channels(self):
        return ["cases"]

    def init_particles(self, N: int, rng: np.random.Generator) -> np.ndarray:
        self._rng = rng
        lo = max(self._I0_frac * 0.5, 1e-7)
        hi = min(self._I0_frac * 2.0, 0.10)
        I0 = self.N * rng.uniform(lo, hi, N)
        S0 = np.maximum(self.N - I0, 0.0)
        R0 = np.zeros(N)
        log_beta0 = np.log(self.beta_init) + rng.standard_normal(N) * 0.30
        log_beta0 = np.clip(log_beta0, self._log_min[0], self._log_max[0])
        return np.column_stack([S0, I0, R0, log_beta0])

    def step(self, particles: np.ndarray, t: int, exog=None) -> np.ndarray:
        p = particles.copy()
        S, I, R = p[:, 0], p[:, 1], p[:, 2]
        log_beta = p[:, 3:4]

        # log-RW update
        log_beta = self._log_rw_step(log_beta, self._sigma, self._log_min, self._log_max, self._rng)
        beta = np.exp(log_beta[:, 0])

        # ODE — Euler
        N_live = np.maximum(S + I + R, 1.0)
        lam  = beta * I / N_live
        wane = getattr(self, "omega_r", 0.0) * R   # R → S waning flow
        dS = -lam * S + wane
        dI =  lam * S - self.gamma * I
        dR =  self.gamma * I - wane

        p[:, 0] = np.maximum(S + dS, 0.0)
        p[:, 1] = np.maximum(I + dI, 0.0)
        p[:, 2] = np.maximum(R + dR, 0.0)
        p[:, 3] = log_beta[:, 0]
        return p

    def observation_mean(self, particles: np.ndarray) -> dict[str, np.ndarray]:
        S, I = particles[:, 0], particles[:, 1]
        beta = np.exp(particles[:, 3])
        N_live = np.maximum(particles[:, :3].sum(axis=1), 1.0)
        incidence = beta * S * I / N_live
        return {"cases": np.maximum(self.Q_C * incidence, 1e-6)}

    def R_eff(self, particles: np.ndarray) -> np.ndarray:
        S = particles[:, 0]
        beta = np.exp(particles[:, 3])
        N_live = np.maximum(particles[:, :3].sum(axis=1), 1.0)
        return beta * S / (N_live * self.gamma)
