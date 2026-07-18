"""
SEIR model with time-varying transmission rate beta_t.

States      : S, E, I, R
Log-RW param: beta_t
Fixed params: sigma (1/incubation), gamma (1/infectious), Q_C (detection)

ODE
---
  lambda_t = beta_t * I / N
  dS/dt = -lambda_t * S
  dE/dt =  lambda_t * S - sigma * E
  dI/dt =  sigma * E - gamma * I
  dR/dt =  gamma * I

Observations
------------
  cases  ~ NegBin(Q_C * sigma * E,  phi_cases)
           (flow from E→I per day, fraction Q_C detected)

Derived output
--------------
  R_eff(t) = beta_t * S_t / (N_living * gamma)
"""
from __future__ import annotations

import numpy as np
from episurveil.models.base import EpiModel


class SEIRModel(EpiModel):
    """
    Parameters
    ----------
    N          : total population
    sigma      : rate of E→I transition (1/incubation period), default 1/5
    gamma      : recovery rate (day^-1), default 1/10
    Q_C        : case detection probability (fixed), default 0.50
    beta_init  : initial beta guess
    sigma_beta : log-RW noise scale
    phi_cases  : NegBin dispersion for case channel
    """

    def __init__(
        self,
        N:          float = 1e6,
        sigma:      float = 1 / 5,
        gamma:      float = 1 / 10,
        Q_C:        float = 0.50,
        beta_init:  float = 0.25,
        sigma_beta: float = 0.04,
        beta_min:   float = 0.01,
        beta_max:   float = 2.00,
        phi_cases:  float = 50.0,
        omega_r:    float = 0.0,
        I0_frac:    float = 1e-4,
    ):
        self.N          = float(N)
        self.sigma      = sigma
        self.gamma      = gamma
        self.Q_C        = Q_C
        self.omega_r    = omega_r
        self.beta_init  = beta_init
        self._sigma_rw  = np.array([sigma_beta])
        self._log_min   = np.array([np.log(beta_min)])
        self._log_max   = np.array([np.log(beta_max)])
        self.phi_cases  = phi_cases
        self._rng       = np.random.default_rng(42)
        self._I0_frac   = I0_frac

    @property
    def state_names(self):
        return ["S", "E", "I", "R"]

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
        E0 = self.N * rng.uniform(lo, hi, N)
        I0 = self.N * rng.uniform(lo, hi, N)
        S0 = np.maximum(self.N - E0 - I0, 0.0)
        R0 = np.zeros(N)
        log_b = np.log(self.beta_init) + rng.standard_normal(N) * 0.30
        log_b = np.clip(log_b, self._log_min[0], self._log_max[0])
        return np.column_stack([S0, E0, I0, R0, log_b])

    def step(self, particles: np.ndarray, t: int, exog=None) -> np.ndarray:
        p = particles.copy()
        S, E, I, R = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        log_b = p[:, 4:5]

        log_b = self._log_rw_step(log_b, self._sigma_rw, self._log_min, self._log_max, self._rng)
        beta = np.exp(log_b[:, 0])

        N_live = np.maximum(S + E + I + R, 1.0)
        lam  = beta * I / N_live
        wane = getattr(self, "omega_r", 0.0) * R   # R → S waning flow
        dS = -lam * S + wane
        dE =  lam * S - self.sigma * E
        dI =  self.sigma * E - self.gamma * I
        dR =  self.gamma * I - wane

        p[:, 0] = np.maximum(S + dS, 0.0)
        p[:, 1] = np.maximum(E + dE, 0.0)
        p[:, 2] = np.maximum(I + dI, 0.0)
        p[:, 3] = np.maximum(R + dR, 0.0)
        p[:, 4] = log_b[:, 0]
        return p

    def observation_mean(self, particles: np.ndarray) -> dict[str, np.ndarray]:
        E = particles[:, 1]
        incidence = self.sigma * E
        return {"cases": np.maximum(self.Q_C * incidence, 1e-6)}

    def R_eff(self, particles: np.ndarray) -> np.ndarray:
        S = particles[:, 0]
        beta = np.exp(particles[:, 4])
        N_live = np.maximum(particles[:, :4].sum(axis=1), 1.0)
        return beta * S / (N_live * self.gamma)
