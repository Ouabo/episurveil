"""
SEIRV model: SEIR + vaccination compartment with waning immunity.

States       : S, E, I, R, V
Log-RW params: beta_t
Exogenous    : nu_t  (daily vaccination rate, fraction of S vaccinated)
Fixed params : sigma, gamma, epsilon (vaccine efficacy), omega_v (waning)

ODE
---
  lambda_t = beta_t * I / N_living
  dS/dt = -lambda_t * S - nu*S + omega_v*V
  dV/dt =  nu*S - (1-eps)*lambda_t*V - omega_v*V
  dE/dt =  lambda_t*S + (1-eps)*lambda_t*V - sigma*E
  dI/dt =  sigma*E - gamma*I
  dR/dt =  gamma*I

Exogenous input
---------------
Pass a DataFrame column 'nu' (daily rate) via the `exog` dict in `step()`,
or leave None (defaults to 0 = no vaccination).

Observations
------------
  cases  ~ NegBin(Q_C * sigma * E, phi_cases)

Derived output
--------------
  R_eff(t) = beta_t * [S_t + (1-eps)*V_t] / (N_living * gamma)
  herd_threshold = 1 - 1/R0_t     (herd immunity threshold at each t)
"""
from __future__ import annotations

import numpy as np
from episurveil.models.base import EpiModel


class SEIRVModel(EpiModel):

    def __init__(
        self,
        N:           float = 1e6,
        sigma:       float = 1 / 5,
        gamma:       float = 1 / 10,
        epsilon:     float = 0.85,
        omega_v:     float = 1 / 180,
        Q_C:         float = 0.50,
        beta_init:   float = 0.25,
        sigma_beta:  float = 0.04,
        beta_min:    float = 0.01,
        beta_max:    float = 2.00,
        phi_cases:   float = 50.0,
        I0_frac:     float = 1e-4,
    ):
        self.N       = float(N)
        self.sigma   = sigma
        self.gamma   = gamma
        self.epsilon = epsilon
        self.omega_v = omega_v
        self.Q_C     = Q_C
        self._sigma_rw = np.array([sigma_beta])
        self._log_min  = np.array([np.log(beta_min)])
        self._log_max  = np.array([np.log(beta_max)])
        self._beta_init = beta_init
        self.phi_cases  = phi_cases
        self._rng = np.random.default_rng(42)
        self._I0_frac   = I0_frac

    @property
    def state_names(self):
        return ["S", "E", "I", "R", "V"]

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
        V0 = np.zeros(N)
        S0 = np.maximum(self.N - E0 - I0, 0.0)
        R0 = np.zeros(N)
        log_b = np.log(self._beta_init) + rng.standard_normal(N) * 0.30
        log_b = np.clip(log_b, self._log_min[0], self._log_max[0])
        return np.column_stack([S0, E0, I0, R0, V0, log_b])

    def step(self, particles: np.ndarray, t: int, exog=None) -> np.ndarray:
        p   = particles.copy()
        S, E, I, R, V = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]
        log_b = p[:, 5:6]

        log_b = self._log_rw_step(log_b, self._sigma_rw, self._log_min, self._log_max, self._rng)
        beta = np.exp(log_b[:, 0])
        nu   = float(exog.get("nu", 0.0)) if exog else 0.0

        N_live = np.maximum(S + E + I + R + V, 1.0)
        lam  = beta * I / N_live
        dS   = -lam * S - nu * S + self.omega_v * V
        dV   =  nu * S - (1 - self.epsilon) * lam * V - self.omega_v * V
        dE   =  lam * S + (1 - self.epsilon) * lam * V - self.sigma * E
        dI   =  self.sigma * E - self.gamma * I
        dR   =  self.gamma * I

        p[:, 0] = np.maximum(S + dS, 0.0)
        p[:, 1] = np.maximum(E + dE, 0.0)
        p[:, 2] = np.maximum(I + dI, 0.0)
        p[:, 3] = np.maximum(R + dR, 0.0)
        p[:, 4] = np.maximum(V + dV, 0.0)
        p[:, 5] = log_b[:, 0]
        return p

    def observation_mean(self, particles: np.ndarray) -> dict[str, np.ndarray]:
        E = particles[:, 1]
        return {"cases": np.maximum(self.Q_C * self.sigma * E, 1e-6)}

    def R_eff(self, particles: np.ndarray) -> np.ndarray:
        S, V = particles[:, 0], particles[:, 4]
        beta = np.exp(particles[:, 5])
        N_live = np.maximum(particles[:, :5].sum(axis=1), 1.0)
        eff_susc = S + (1 - self.epsilon) * V
        return beta * eff_susc / (N_live * self.gamma)
