"""
SEIRD model: SEIR + disease-induced deaths with time-varying IFR delta_t.

States       : S, E, I, R, D
Log-RW params: beta_t, delta_t (infection fatality rate)
Fixed params : sigma, gamma, Q_C

ODE
---
  lambda_t = beta_t * I / N
  dS/dt = -lambda_t * S
  dE/dt =  lambda_t * S - sigma * E
  dI/dt =  sigma * E - (gamma + delta_t) * I
  dR/dt =  gamma * I
  dD/dt =  delta_t * I

Observations
------------
  cases  ~ NegBin(Q_C * sigma * E,  phi_cases)
  deaths ~ NegBin(delta_t * I,       phi_deaths)

Derived output
--------------
  R_eff(t) = beta_t * S_t / (N_living * (gamma + delta_t))
  IFR_t    = delta_t / (gamma + delta_t)   [infection fatality ratio]
"""
from __future__ import annotations

import numpy as np
from episurveil.models.base import EpiModel


class SEIRDModel(EpiModel):

    def __init__(
        self,
        N:           float = 1e6,
        sigma:       float = 1 / 5,
        gamma:       float = 1 / 10,
        Q_C:         float = 0.50,
        beta_init:   float = 0.25,
        delta_init:  float = 0.005,
        sigma_beta:  float = 0.04,
        sigma_delta: float = 0.03,
        beta_min:    float = 0.01,  beta_max:  float = 2.00,
        delta_min:   float = 1e-5,  delta_max: float = 0.10,
        phi_cases:   float = 50.0,
        phi_deaths:  float = 10.0,
        I0_frac:     float = 1e-4,
    ):
        self.N = float(N)
        self.sigma = sigma
        self.gamma = gamma
        self.Q_C   = Q_C
        self._sigma_rw = np.array([sigma_beta, sigma_delta])
        self._log_min  = np.array([np.log(beta_min),  np.log(delta_min)])
        self._log_max  = np.array([np.log(beta_max),  np.log(delta_max)])
        self._init_log = np.array([np.log(beta_init), np.log(delta_init)])
        self.phi_cases  = phi_cases
        self.phi_deaths = phi_deaths
        self._rng = np.random.default_rng(42)
        self._I0_frac = I0_frac

    @property
    def state_names(self):
        return ["S", "E", "I", "R", "D"]

    @property
    def param_names(self):
        return ["beta", "delta"]

    @property
    def obs_channels(self):
        return ["cases", "deaths"]

    def init_particles(self, N: int, rng: np.random.Generator) -> np.ndarray:
        self._rng = rng
        lo = max(self._I0_frac * 0.5, 1e-7)
        hi = min(self._I0_frac * 2.0, 0.10)
        E0 = self.N * rng.uniform(lo, hi, N)
        I0 = self.N * rng.uniform(lo, hi, N)
        S0 = np.maximum(self.N - E0 - I0, 0.0)
        R0 = np.zeros(N)
        D0 = np.zeros(N)
        log_p = self._init_log + rng.standard_normal((N, 2)) * [0.30, 0.30]
        log_p = np.clip(log_p, self._log_min, self._log_max)
        return np.column_stack([S0, E0, I0, R0, D0, log_p])

    def step(self, particles: np.ndarray, t: int, exog=None) -> np.ndarray:
        p = particles.copy()
        S, E, I, R, D = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]
        log_p = p[:, 5:7]

        log_p = self._log_rw_step(log_p, self._sigma_rw, self._log_min, self._log_max, self._rng)
        beta, delta = np.exp(log_p[:, 0]), np.exp(log_p[:, 1])

        N_live = np.maximum(S + E + I + R, 1.0)
        lam = beta * I / N_live
        dS = -lam * S
        dE =  lam * S - self.sigma * E
        dI =  self.sigma * E - (self.gamma + delta) * I
        dR =  self.gamma * I
        dD =  delta * I

        p[:, 0] = np.maximum(S + dS, 0.0)
        p[:, 1] = np.maximum(E + dE, 0.0)
        p[:, 2] = np.maximum(I + dI, 0.0)
        p[:, 3] = np.maximum(R + dR, 0.0)
        p[:, 4] = D + np.maximum(dD, 0.0)
        p[:, 5:7] = log_p
        return p

    def observation_mean(self, particles: np.ndarray) -> dict[str, np.ndarray]:
        E, I = particles[:, 1], particles[:, 2]
        delta = np.exp(particles[:, 6])
        return {
            "cases":  np.maximum(self.Q_C * self.sigma * E, 1e-6),
            "deaths": np.maximum(delta * I, 1e-6),
        }

    def R_eff(self, particles: np.ndarray) -> np.ndarray:
        S = particles[:, 0]
        beta  = np.exp(particles[:, 5])
        delta = np.exp(particles[:, 6])
        N_live = np.maximum(particles[:, :4].sum(axis=1), 1.0)
        return beta * S / (N_live * (self.gamma + delta))
