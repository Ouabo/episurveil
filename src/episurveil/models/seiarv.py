"""
SEIARV model: SEIR + Asymptomatic compartment + Vaccination + dynamic detection.

States       : S, E, A, I, R, V
Log-RW params: beta_t, Q_C_t  (transmission + detection jointly estimated)
Exogenous    : nu_t (vaccination rate)
Fixed params : sigma, kappa (symptom prob), gamma_a, gamma_i,
               eta_a (relative asymptomatic transmissibility),
               epsilon (vaccine efficacy), omega_v (waning)

Force of infection
------------------
  lambda_t = beta_t * [eta_a*A + (1 - Q_C_t)*I] / N_living
  (detected symptomatic I self-isolate; A never detected)

ODE
---
  dS/dt = -lambda_t*S - nu*S + omega_v*V
  dV/dt =  nu*S - (1-eps)*lambda_t*V - omega_v*V
  dE/dt =  (lambda_t*S + (1-eps)*lambda_t*V) - sigma*E
  dA/dt =  (1-kappa)*sigma*E - gamma_a*A
  dI/dt =  kappa*sigma*E - gamma_i*I
  dR/dt =  gamma_a*A + gamma_i*I

Observations
------------
  cases  ~ NegBin(Q_C_t * kappa * sigma * E, phi_cases)
           (detected symptomatic new infections)
  asym_screen ~ NegBin(Q_A * gamma_a * A, phi_asym)  [optional]

Derived output
--------------
  R_eff(t) = beta_t * [S_t+(1-eps)*V_t]/N * [eta_a/gamma_a + (1-Q_C_t)/gamma_i]
  detection_rate_t = Q_C_t
"""
from __future__ import annotations

import numpy as np
from episurveil.models.base import EpiModel


class SEIARVModel(EpiModel):

    def __init__(
        self,
        N:            float = 1e6,
        sigma:        float = 1 / 5,
        kappa:        float = 0.60,
        gamma_a:      float = 0.125,
        gamma_i:      float = 0.10,
        eta_a:        float = 0.50,
        epsilon:      float = 0.85,
        omega_v:      float = 1 / 180,
        beta_init:    float = 0.25,
        q_c_init:     float = 0.40,
        sigma_beta:   float = 0.04,
        sigma_qc:     float = 0.02,
        beta_min:     float = 0.01,   beta_max:  float = 2.00,
        q_c_min:      float = 0.05,   q_c_max:   float = 0.99,
        phi_cases:    float = 50.0,
        I0_frac:      float = 1e-4,
    ):
        self.N       = float(N)
        self.sigma   = sigma
        self.kappa   = kappa
        self.gamma_a = gamma_a
        self.gamma_i = gamma_i
        self.eta_a   = eta_a
        self.epsilon = epsilon
        self.omega_v = omega_v
        self._sigma_rw = np.array([sigma_beta, sigma_qc])
        self._log_min  = np.array([np.log(beta_min), np.log(q_c_min)])
        self._log_max  = np.array([np.log(beta_max), np.log(q_c_max)])
        self._init_log = np.array([np.log(beta_init), np.log(q_c_init)])
        self.phi_cases = phi_cases
        self._rng = np.random.default_rng(42)
        self._I0_frac  = I0_frac

    @property
    def state_names(self):
        return ["S", "E", "A", "I", "R", "V"]

    @property
    def param_names(self):
        return ["beta", "Q_C"]

    @property
    def obs_channels(self):
        return ["cases"]

    def init_particles(self, N: int, rng: np.random.Generator) -> np.ndarray:
        self._rng = rng
        lo = max(self._I0_frac * 0.5, 1e-7)
        hi = min(self._I0_frac * 2.0, 0.10)
        E0 = self.N * rng.uniform(lo, hi, N)
        A0 = self.N * rng.uniform(lo, hi, N)
        I0 = self.N * rng.uniform(lo, hi, N)
        V0 = np.zeros(N)
        S0 = np.maximum(self.N - E0 - A0 - I0, 0.0)
        R0 = np.zeros(N)
        log_p = self._init_log + rng.standard_normal((N, 2)) * [0.30, 0.20]
        log_p = np.clip(log_p, self._log_min, self._log_max)
        return np.column_stack([S0, E0, A0, I0, R0, V0, log_p])

    def step(self, particles: np.ndarray, t: int, exog=None) -> np.ndarray:
        p = particles.copy()
        S, E, A, I, R, V = (p[:, i] for i in range(6))
        log_p = p[:, 6:8]

        log_p = self._log_rw_step(log_p, self._sigma_rw, self._log_min, self._log_max, self._rng)
        beta = np.exp(log_p[:, 0])
        q_c  = np.exp(log_p[:, 1])
        nu   = float(exog.get("nu", 0.0)) if exog else 0.0

        N_live = np.maximum(S + E + A + I + R + V, 1.0)
        lam  = beta * (self.eta_a * A + (1 - q_c) * I) / N_live
        dS   = -lam * S - nu * S + self.omega_v * V
        dV   =  nu * S - (1 - self.epsilon) * lam * V - self.omega_v * V
        dE   =  lam * S + (1 - self.epsilon) * lam * V - self.sigma * E
        dA   =  (1 - self.kappa) * self.sigma * E - self.gamma_a * A
        dI   =  self.kappa * self.sigma * E - self.gamma_i * I
        dR   =  self.gamma_a * A + self.gamma_i * I

        p[:, 0] = np.maximum(S + dS, 0.0)
        p[:, 1] = np.maximum(E + dE, 0.0)
        p[:, 2] = np.maximum(A + dA, 0.0)
        p[:, 3] = np.maximum(I + dI, 0.0)
        p[:, 4] = np.maximum(R + dR, 0.0)
        p[:, 5] = np.maximum(V + dV, 0.0)
        p[:, 6:8] = log_p
        return p

    def observation_mean(self, particles: np.ndarray) -> dict[str, np.ndarray]:
        E   = particles[:, 1]
        q_c = np.exp(particles[:, 7])
        incidence = self.kappa * self.sigma * E
        return {"cases": np.maximum(q_c * incidence, 1e-6)}

    def R_eff(self, particles: np.ndarray) -> np.ndarray:
        S, V = particles[:, 0], particles[:, 5]
        beta = np.exp(particles[:, 6])
        q_c  = np.exp(particles[:, 7])
        N_live = np.maximum(particles[:, :6].sum(axis=1), 1.0)
        eff_susc = (S + (1 - self.epsilon) * V) / N_live
        # Weighted sum of infectious periods
        r_a = self.eta_a / self.gamma_a
        r_i = (1 - q_c) / self.gamma_i
        return beta * eff_susc * ((1 - self.kappa) * r_a + self.kappa * r_i)
