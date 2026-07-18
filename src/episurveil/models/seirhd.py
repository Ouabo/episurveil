"""
SEIRHD model: SEIR + Hospital ward + Deaths.
Suitable for fitting to cases + hospitalisations + deaths simultaneously.

States       : S, E, I, H, R, D
Log-RW params: beta_t, tau_i_t (hosp admission rate), delta_h_t (in-hosp CFR)
Fixed params : sigma, gamma_i, gamma_h, Q_C, LOS (length of stay)

ODE
---
  lambda_t = beta_t * I / N_living
  dS/dt = -lambda_t * S
  dE/dt =  lambda_t * S - sigma * E
  dI/dt =  sigma * E - (gamma_i + tau_i_t) * I
  dH/dt =  tau_i_t * I - (gamma_h + delta_h_t) * H
  dR/dt =  gamma_i * I + gamma_h * H
  dD/dt =  delta_h_t * H

Observations
------------
  cases  ~ NegBin(Q_C * sigma * E,         phi_cases)
  hosp   ~ NegBin(tau_i_t * I * LOS,       phi_hosp)   [occupancy proxy]
  deaths ~ NegBin(delta_h_t * H,            phi_deaths)

Derived output
--------------
  R_eff(t)    = beta_t * S_t / (N_living * (gamma_i + tau_i_t))
  hosp_rate_t = tau_i_t / (gamma_i + tau_i_t)   [fraction of I hospitalised]
  CFR_t       = delta_h_t / (gamma_h + delta_h_t)
"""
from __future__ import annotations

import numpy as np
from episurveil.models.base import EpiModel


class SEIRHDModel(EpiModel):

    def __init__(
        self,
        N:            float = 1e6,
        sigma:        float = 1 / 5,
        gamma_i:      float = 0.10,
        gamma_h:      float = 1 / 12,
        Q_C:          float = 0.50,
        LOS:          float = 8.0,
        beta_init:    float = 0.25,
        tau_i_init:   float = 0.005,
        delta_h_init: float = 0.02,
        sigma_beta:   float = 0.04,
        sigma_tau:    float = 0.02,
        sigma_delta:  float = 0.02,
        beta_min:  float = 0.01, beta_max:    float = 2.00,
        tau_min:   float = 1e-4, tau_max:     float = 0.10,
        delta_min: float = 1e-4, delta_max:   float = 0.15,
        phi_cases:    float = 50.0,
        phi_hosp:     float = 15.0,
        phi_deaths:   float = 10.0,
        omega_r:      float = 0.0,
        I0_frac:      float = 1e-4,
    ):
        self.N       = float(N)
        self.sigma   = sigma
        self.gamma_i = gamma_i
        self.gamma_h = gamma_h
        self.Q_C     = Q_C
        self.LOS     = LOS
        self.omega_r = omega_r
        self._sigma_rw = np.array([sigma_beta, sigma_tau, sigma_delta])
        self._log_min  = np.array([np.log(beta_min),  np.log(tau_min),  np.log(delta_min)])
        self._log_max  = np.array([np.log(beta_max),  np.log(tau_max),  np.log(delta_max)])
        self._init_log = np.array([np.log(beta_init), np.log(tau_i_init), np.log(delta_h_init)])
        self.phi_cases  = phi_cases
        self.phi_hosp   = phi_hosp
        self.phi_deaths = phi_deaths
        self._rng = np.random.default_rng(42)
        self._I0_frac = I0_frac

    @property
    def state_names(self):
        return ["S", "E", "I", "H", "R", "D"]

    @property
    def param_names(self):
        return ["beta", "tau_i", "delta_h"]

    @property
    def obs_channels(self):
        return ["cases", "hosp", "deaths"]

    def init_particles(self, N: int, rng: np.random.Generator) -> np.ndarray:
        self._rng = rng
        lo = max(self._I0_frac * 0.5, 1e-7)
        hi = min(self._I0_frac * 2.0, 0.10)
        E0 = self.N * rng.uniform(lo, hi, N)
        I0 = self.N * rng.uniform(lo, hi, N)
        H0 = self.N * rng.uniform(lo * 0.1, hi * 0.1, N)
        D0 = np.zeros(N)
        S0 = np.maximum(self.N - E0 - I0 - H0, 0.0)
        R0 = np.zeros(N)
        log_p = self._init_log + rng.standard_normal((N, 3)) * [0.30, 0.30, 0.30]
        log_p = np.clip(log_p, self._log_min, self._log_max)
        return np.column_stack([S0, E0, I0, H0, R0, D0, log_p])

    def step(self, particles: np.ndarray, t: int, exog=None) -> np.ndarray:
        p = particles.copy()
        S, E, I, H, R, D = (p[:, i] for i in range(6))
        log_p = p[:, 6:9]

        log_p   = self._log_rw_step(log_p, self._sigma_rw, self._log_min, self._log_max, self._rng)
        beta    = np.exp(log_p[:, 0])
        tau_i   = np.exp(log_p[:, 1])
        delta_h = np.exp(log_p[:, 2])

        N_live = np.maximum(S + E + I + H + R, 1.0)
        lam  = beta * I / N_live
        wane = getattr(self, "omega_r", 0.0) * R   # R → S waning flow
        dS  = -lam * S + wane
        dE  =  lam * S - self.sigma * E
        dI  =  self.sigma * E - (self.gamma_i + tau_i) * I
        dH  =  tau_i * I - (self.gamma_h + delta_h) * H
        dR  =  self.gamma_i * I + self.gamma_h * H - wane
        dD  =  delta_h * H

        p[:, 0] = np.maximum(S + dS, 0.0)
        p[:, 1] = np.maximum(E + dE, 0.0)
        p[:, 2] = np.maximum(I + dI, 0.0)
        p[:, 3] = np.maximum(H + dH, 0.0)
        p[:, 4] = np.maximum(R + dR, 0.0)
        p[:, 5] = D + np.maximum(dD, 0.0)
        p[:, 6:9] = log_p
        return p

    def observation_mean(self, particles: np.ndarray) -> dict[str, np.ndarray]:
        E, I, H = particles[:, 1], particles[:, 2], particles[:, 3]
        tau_i   = np.exp(particles[:, 7])
        delta_h = np.exp(particles[:, 8])
        return {
            "cases":  np.maximum(self.Q_C * self.sigma * E, 1e-6),
            "hosp":   np.maximum(tau_i * I * self.LOS,      1e-6),
            "deaths": np.maximum(delta_h * H,               1e-6),
        }

    def R_eff(self, particles: np.ndarray) -> np.ndarray:
        S     = particles[:, 0]
        beta  = np.exp(particles[:, 6])
        tau_i = np.exp(particles[:, 7])
        N_live = np.maximum(particles[:, :5].sum(axis=1), 1.0)
        return beta * S / (N_live * (self.gamma_i + tau_i))
