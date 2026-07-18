"""Two-control optimal NPI + testing strategy for the SVEAIHCRD model.

Controls
--------
u_t in [0, 1]  — NPI intensity
    beta_t = beta_base * exp(-alpha_npi * u_t)
    u=0 : baseline transmission; u=1 : full lockdown (~78% reduction for alpha=1.5)

v_t in [0, 1]  — testing / detection intensity
    Q_C_t = Q_C_base + (0.99 - Q_C_base) * v_t
    Detected symptomatic cases (I) self-isolate -> only (1 - Q_C_t) fraction of I transmits.
    Asymptomatic cases (A) are never detected; they always transmit at reduced rate eta_A.
    v=0 : passive detection at baseline rate Q_C_base (from BPF posterior)
    v=1 : maximum mass testing -> Q_C -> 0.99, nearly all symptomatic I's detected & isolated

Modified force of infection
---------------------------
    lambda_t = beta_t * ((1 - Q_C_t)*I + eta_A*A) / N
    Testing only reduces I's contribution; A's contribution is unaffected.

Objective
---------
J = w_H * sum(H_t)            hospitalization burden (bed-days)
  + w_C * sum(C_t)            ICU burden (ICU-days)
  + w_D * (D_T - D_0)         cumulative deaths
  + w_u * sum(u_t^2)          NPI economic cost  (typically w_u >> w_v)
  + w_v * sum(v_t^2)          testing cost       (mass PCR / rapid tests)
  + w_du * sum((Du_t)^2)      NPI smoothness penalty
  + w_dv * sum((Dv_t)^2)      testing-ramp smoothness penalty
  + cap_pen * capacity violations (H, C)

Solver
------
L-BFGS-B on 2T variables [u_0...u_{T-1}, v_0...v_{T-1}], converges in ~5-30 s.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import minimize

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from episurveil.models.sveaihcrd import Parameters, NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA_NPI    = 1.50     # full lockdown reduces beta by exp(-1.5) ~ 78%
H_MAX_DE     = 20_000   # Germany hospital capacity (COVID beds, conservative)
C_MAX_DE     =  5_000   # Germany ICU capacity (COVID beds, near peak)
CAP_PENALTY  = 1e5      # soft capacity penalty weight
Q_C_MAX      = 0.99     # maximum achievable detection rate


# ---------------------------------------------------------------------------
# Weight dataclass
# ---------------------------------------------------------------------------
@dataclass
class ControlWeights:
    """Costs in 'death-equivalent' units.

    Defaults:
      w_H  = 1e-3  : 1,000 hospitalization-days ~ 1 death
      w_C  = 5e-3  : 200 ICU-days ~ 1 death
      w_D  = 1.0   : 1 death = 1 unit (reference)
      w_u  = 50.0  : 1 day full lockdown ~ 50 death-equivalents (GDP loss)
      w_v  = 2.0   : 1 day mass testing  ~ 2 death-equivalents  (test cost only)
      w_du = 2.0   : smoothness — penalises rapid NPI oscillation
      w_dv = 0.5   : smoothness — penalises rapid testing-level oscillation
    """
    w_H:  float = 1e-3
    w_C:  float = 5e-3
    w_D:  float = 1.0
    w_u:  float = 50.0    # NPI economic cost — typically >> w_v
    w_v:  float = 2.0     # testing cost      — cheaper than NPI
    w_du: float = 2.0
    w_dv: float = 0.5


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ControlResult:
    u_opt:         np.ndarray   # (T,) optimal NPI trajectory
    v_opt:         np.ndarray   # (T,) optimal testing trajectory
    q_c_traj:      np.ndarray   # (T,) Q_C_t = Q_C_base + (0.99-Q_C_base)*v_t
    beta_traj:     np.ndarray   # (T,) effective beta_t = beta_base*exp(-alpha*u_t)
    traj_opt:      dict         # {comp: array(T+1)} under (u*, v*)
    traj_baseline: dict         # {comp: array(T+1)} no controls (u=0, v=0)
    dates:         list

    cost_total:    float
    cost_health:   float
    cost_npi:      float
    cost_testing:  float
    cost_smooth:   float
    cost_capacity: float

    deaths_averted:      int
    hosp_days_averted:   int
    icu_days_averted:    int

    # Fraction of R_eff reduction attributable to testing vs. NPI
    npi_contribution_pct:     float  # % of total beta_eff reduction from NPI
    testing_contribution_pct: float  # % from testing (detected isolation)

    converged: bool
    message:   str

    pareto: Optional[list] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Core ODE simulation (two-control version)
# ---------------------------------------------------------------------------
def _simulate_two_ctrl(
    x0:         np.ndarray,
    u_seq:      np.ndarray,
    v_seq:      np.ndarray,
    beta_base:  float,
    tau_i:      float,
    delta_h:    float,
    q_c_base:   float,
    alpha_npi:  float = ALPHA_NPI,
) -> dict:
    """Forward simulate SVEAIHCRD with NPI control u_t and testing control v_t.

    Modified force of infection:
        lambda_t = beta_t * ((1 - Q_C_t)*I + eta_A*A) / N
    where beta_t = beta_base * exp(-alpha_npi * u_t)
    and   Q_C_t  = q_c_base + (Q_C_MAX - q_c_base) * v_t

    Uses sub-daily Euler (dt=0.25) for numerical stability.
    Returns {compartment: array(T+1)}.
    """
    T   = len(u_seq)
    p   = Parameters(tau_i=float(tau_i), delta_h=float(delta_h), nu=0.0)
    traj = {n: np.zeros(T + 1) for n in NAMES}
    x    = np.asarray(x0, dtype=float).copy()
    for i, n in enumerate(NAMES):
        traj[n][0] = x[i]

    dt_sub = 0.25
    n_sub  = 4

    for t in range(T):
        u    = float(np.clip(u_seq[t], 0.0, 1.0))
        v    = float(np.clip(v_seq[t], 0.0, 1.0))
        beta_t = float(beta_base) * np.exp(-alpha_npi * u)
        q_c_t  = q_c_base + (Q_C_MAX - q_c_base) * v

        for _ in range(n_sub):
            S, V, E, A, I, H, C, R, D = x
            living = max(S + V + E + A + I + H + C + R, 1e-12)
            # Only undetected I transmit freely
            lam   = beta_t * ((1.0 - q_c_t) * I + p.eta_a * A) / living
            inf_v = (1.0 - p.vaccine_efficacy) * lam * V

            dx = np.array([
                -lam*S - p.nu*S + p.omega_v*V + p.omega_r*R,
                p.nu*(S+E+A+R) - inf_v - p.omega_v*V,
                lam*S + inf_v - p.sigma*E - p.nu*E,
                (1.0-p.kappa)*p.sigma*E - p.gamma_a*A - p.nu*A,
                p.kappa*p.sigma*E - (p.gamma_i + p.tau_i)*I,
                p.tau_i*I - (p.gamma_h + p.tau_h + p.delta_h)*H,
                p.tau_h*H - (p.gamma_c + p.delta_c)*C,
                p.gamma_a*A + p.gamma_i*I + p.gamma_h*H + p.gamma_c*C - p.omega_r*R - p.nu*R,
                p.delta_h*H + p.delta_c*C,
            ])
            x = np.maximum(x + dt_sub * dx, 0.0)
            x[8] = max(x[8], traj["D"][t])  # D is cumulative — never decreases

        for i, n in enumerate(NAMES):
            traj[n][t + 1] = x[i]

    return traj


# ---------------------------------------------------------------------------
# Objective (flat 2T-vector: first T = u, next T = v)
# ---------------------------------------------------------------------------
def _objective_2ctrl(
    uv:         np.ndarray,
    x0:         np.ndarray,
    beta_base:  float,
    tau_i:      float,
    delta_h:    float,
    q_c_base:   float,
    weights:    ControlWeights,
    h_max:      float,
    c_max:      float,
    alpha_npi:  float,
) -> float:
    T   = len(uv) // 2
    u   = uv[:T]
    v   = uv[T:]
    traj = _simulate_two_ctrl(x0, u, v, beta_base, tau_i, delta_h, q_c_base, alpha_npi)

    H        = traj["H"][1:]
    C        = traj["C"][1:]
    D_delta  = traj["D"][-1] - traj["D"][0]
    du       = np.diff(u, prepend=u[0])
    dv       = np.diff(v, prepend=v[0])

    cost  = weights.w_H  * float(np.sum(H))
    cost += weights.w_C  * float(np.sum(C))
    cost += weights.w_D  * float(D_delta)
    cost += weights.w_u  * float(np.sum(u ** 2))
    cost += weights.w_v  * float(np.sum(v ** 2))
    cost += weights.w_du * float(np.sum(du ** 2))
    cost += weights.w_dv * float(np.sum(dv ** 2))
    cost += CAP_PENALTY  * float(
        np.sum(np.maximum(0.0, H - h_max) ** 2) +
        np.sum(np.maximum(0.0, C - c_max) ** 2)
    )
    return cost


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_optimal_control(
    x0:         np.ndarray,
    beta_base:  float,
    tau_i:      float,
    delta_h:    float,
    q_c_base:   float        = 0.40,
    T:          int          = 90,
    weights:    ControlWeights = None,
    h_max:      float        = H_MAX_DE,
    c_max:      float        = C_MAX_DE,
    alpha_npi:  float        = ALPHA_NPI,
    start_date: str          = "2020-03-10",
) -> ControlResult:
    """Find optimal (u*(t), v*(t)) over T days starting from BPF state x0.

    Parameters
    ----------
    q_c_base : baseline detection rate from BPF posterior at start_date
               (v=0 keeps Q_C at this level; v=1 pushes it to 0.99)
    """
    if weights is None:
        weights = ControlWeights()

    import pandas as pd
    dates = [str((pd.Timestamp(start_date) + pd.Timedelta(days=i)).date())
             for i in range(T + 1)]

    # Baseline: no controls
    traj_baseline = _simulate_two_ctrl(
        x0, np.zeros(T), np.zeros(T),
        beta_base, tau_i, delta_h, q_c_base, alpha_npi,
    )

    # Initial guess: moderate NPI, moderate testing
    uv0 = np.concatenate([np.full(T, 0.25), np.full(T, 0.30)])
    bounds = [(0.0, 1.0)] * (2 * T)

    res = minimize(
        _objective_2ctrl,
        uv0,
        args=(x0, beta_base, tau_i, delta_h, q_c_base,
              weights, h_max, c_max, alpha_npi),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-9, "gtol": 1e-6, "maxls": 40},
    )

    u_opt = np.clip(res.x[:T], 0.0, 1.0)
    v_opt = np.clip(res.x[T:], 0.0, 1.0)
    traj_opt = _simulate_two_ctrl(
        x0, u_opt, v_opt, beta_base, tau_i, delta_h, q_c_base, alpha_npi,
    )

    # Derived trajectories
    beta_traj = beta_base * np.exp(-alpha_npi * u_opt)
    q_c_traj  = q_c_base + (Q_C_MAX - q_c_base) * v_opt

    # Cost breakdown
    H_opt   = traj_opt["H"][1:]
    C_opt   = traj_opt["C"][1:]
    D_opt   = traj_opt["D"][-1] - traj_opt["D"][0]
    D_base  = traj_baseline["D"][-1] - traj_baseline["D"][0]
    H_base  = float(np.sum(traj_baseline["H"][1:]))
    C_base  = float(np.sum(traj_baseline["C"][1:]))
    du_opt  = np.diff(u_opt, prepend=u_opt[0])
    dv_opt  = np.diff(v_opt, prepend=v_opt[0])

    cost_health   = (weights.w_H * float(np.sum(H_opt))
                     + weights.w_C * float(np.sum(C_opt))
                     + weights.w_D * float(D_opt))
    cost_npi      = weights.w_u  * float(np.sum(u_opt ** 2))
    cost_testing  = weights.w_v  * float(np.sum(v_opt ** 2))
    cost_smooth   = (weights.w_du * float(np.sum(du_opt ** 2))
                     + weights.w_dv * float(np.sum(dv_opt ** 2)))
    cost_cap      = CAP_PENALTY * float(
        np.sum(np.maximum(0.0, H_opt - h_max) ** 2) +
        np.sum(np.maximum(0.0, C_opt - c_max) ** 2)
    )

    # Decompose R_eff reduction:
    # beta_eff = beta_base * exp(-alpha*u) * (1-Q_C)*I_frac  (approx)
    # NPI reduction factor: exp(-alpha*mean_u)
    # Testing reduction factor: (1 - mean_Q_C) / (1 - q_c_base)
    mean_u    = float(np.mean(u_opt))
    mean_q_c  = float(np.mean(q_c_traj))
    npi_factor   = 1.0 - np.exp(-alpha_npi * mean_u)       # fraction reduced by NPI
    test_factor  = (mean_q_c - q_c_base) / max(1.0 - q_c_base, 1e-9)  # fraction isolated by testing
    total        = max(npi_factor + test_factor * (1.0 - npi_factor), 1e-9)
    npi_pct      = 100.0 * npi_factor / total
    test_pct     = 100.0 - npi_pct

    return ControlResult(
        u_opt=u_opt, v_opt=v_opt,
        q_c_traj=q_c_traj, beta_traj=beta_traj,
        traj_opt=traj_opt, traj_baseline=traj_baseline,
        dates=dates,
        cost_total=float(res.fun),
        cost_health=cost_health, cost_npi=cost_npi,
        cost_testing=cost_testing, cost_smooth=cost_smooth,
        cost_capacity=cost_cap,
        deaths_averted=int(round(D_base - D_opt)),
        hosp_days_averted=int(round(H_base - float(np.sum(H_opt)))),
        icu_days_averted=int(round(C_base - float(np.sum(C_opt)))),
        npi_contribution_pct=float(npi_pct),
        testing_contribution_pct=float(test_pct),
        converged=res.success,
        message=res.message if hasattr(res, "message") else "",
    )


def run_pareto_sweep(
    x0:        np.ndarray,
    beta_base: float,
    tau_i:     float,
    delta_h:   float,
    q_c_base:  float  = 0.40,
    T:         int    = 90,
    h_max:     float  = H_MAX_DE,
    c_max:     float  = C_MAX_DE,
    alpha_npi: float  = ALPHA_NPI,
    n_points:  int    = 8,
    start_date: str   = "2020-03-10",
) -> list[dict]:
    """Sweep w_u/w_v ratio to trace the Pareto frontier.

    Fixes w_v=2 and sweeps w_u from 0.1 to 1000.
    Returns list of dicts summarising each optimal solution.
    """
    w_u_values = np.logspace(-1, 3, n_points)
    points = []
    for w_u in w_u_values:
        w = ControlWeights(w_u=float(w_u))
        r = run_optimal_control(
            x0, beta_base, tau_i, delta_h, q_c_base,
            T, w, h_max, c_max, alpha_npi, start_date,
        )
        points.append({
            "w_u":                   float(w_u),
            "mean_u":                float(np.mean(r.u_opt)),
            "mean_v":                float(np.mean(r.v_opt)),
            "mean_q_c":              float(np.mean(r.q_c_traj)),
            "npi_cost_raw":          float(np.sum(r.u_opt ** 2)),
            "testing_cost_raw":      float(np.sum(r.v_opt ** 2)),
            "deaths_averted":        r.deaths_averted,
            "hosp_days_averted":     r.hosp_days_averted,
            "icu_days_averted":      r.icu_days_averted,
            "npi_contribution_pct":  r.npi_contribution_pct,
            "testing_contribution_pct": r.testing_contribution_pct,
            "result":                r,
        })
    return points
