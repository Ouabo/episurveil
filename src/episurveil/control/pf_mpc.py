"""
PF-MPC: Particle-Filter-guided Model Predictive Control (Receding Horizon).

Framework
---------
The Bootstrap Particle Filter runs offline and produces a filtered posterior
mean (x̄_t, θ̄_t) for every historical day t. The receding-horizon controller
uses this sequence of state estimates as if they arrived online:

    for t = t0, t0+1, ..., t0+T_sim-1:
        1. READ  posterior mean at t  →  x̄_t, β̄_t, τ̄_I,t, δ̄_H,t, Q̄_C,t
        2. SOLVE H-day optimal control from (x̄_t, θ̄_t)   → u*[0..H-1], v*[0..H-1]
        3. APPLY first action only: u_mpc[t] = u*[0],  v_mpc[t] = v*[0]
        4. WARM-START next solve: shift u*[1..H-1] and append last value

This is the standard PF-MPC loop.  The "stochastic" component lives entirely
in the particle filter: it handles the nonlinear, non-Gaussian dynamics and
produces a principled state estimate that is passed to the MPC at each step.
Running a fully stochastic MPC (particles propagated inside the optimizer)
is O(K) times more expensive per function evaluation and is computationally
intractable for most real problems.

Counterfactual trajectory
-------------------------
After the MPC loop, we simulate what the epidemic WOULD have looked like under
the MPC policy.  To make this realistic, we use the BPF-estimated β_t as the
background transmission rate (it captures variant emergence, vaccination, and
seasonal behaviour that are independent of NPI).  The MPC then modulates β_t
via  β_eff = β_t × exp(-α u_t).

This answers:  "Given everything that actually changed (Omicron emergence, waning
immunity, ...), what would the epidemic have looked like if the MPC policy had
been applied instead of the actual interventions?"

Key parameters
--------------
H     : lookahead horizon (days).  H=14 (two weeks) gives a good speed/quality
        trade-off and is clinically meaningful.
T_sim : retrospective simulation length (days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import minimize

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from episurveil.models.sveaihcrd import Parameters, NAMES
from episurveil.control.optimal_control import (
    ControlWeights, _objective_2ctrl, _simulate_two_ctrl,
    H_MAX_DE, C_MAX_DE, ALPHA_NPI, Q_C_MAX,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class PFMPCResult:
    """Output of run_pf_mpc."""
    dates:        list           # length T_sim + 1

    # Recommended control trajectories (length T_sim)
    u_mpc:        np.ndarray
    v_mpc:        np.ndarray
    q_c_traj:     np.ndarray    # Q_C_t = f(v_mpc_t)
    beta_eff:     np.ndarray    # β_t * exp(-α u_mpc)  — effective transmission

    # BPF-tracked background parameters used at each step
    beta_bpf:     np.ndarray    # β̄_t from filter (uncontrolled)
    q_c_base_bpf: np.ndarray    # Q̄_C,t from filter (baseline detection)

    # Counterfactual epidemic trajectories (length T_sim + 1 each)
    traj_mpc:      dict         # {comp: array} under MPC controls
    traj_baseline: dict         # {comp: array} under no controls

    # Per-step convergence
    converged:    np.ndarray    # bool array, length T_sim
    n_iters:      np.ndarray    # number of L-BFGS-B iterations per step

    # Summary metrics
    deaths_averted:    int
    hosp_days_averted: int
    icu_days_averted:  int

    H:     int    # lookahead horizon used
    T_sim: int    # simulation length


# ---------------------------------------------------------------------------
# Adaptive forward simulation (time-varying β from BPF)
# ---------------------------------------------------------------------------
def _simulate_mpc_adaptive(
    x0:           np.ndarray,
    u_seq:        np.ndarray,
    v_seq:        np.ndarray,
    beta_seq:     np.ndarray,    # BPF β̄_t at each day — captures variants/seasons
    tau_i:        float,
    delta_h:      float,
    q_c_base_seq: np.ndarray,   # BPF Q̄_C,t at each day
    alpha_npi:    float = ALPHA_NPI,
) -> dict:
    """Forward simulate with time-varying β and Q_C_base from BPF.

    At each day t:
        β_eff = β_bpf[t] × exp(-alpha_npi × u[t])
        Q_C_t = q_c_base[t] + (Q_C_MAX - q_c_base[t]) × v[t]

    This gives a realistic counterfactual: variant emergence and vaccination
    are preserved in β_bpf; only NPI policy changes through u[t].
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
        u      = float(np.clip(u_seq[t], 0.0, 1.0))
        v      = float(np.clip(v_seq[t], 0.0, 1.0))
        beta_t = float(beta_seq[t]) * np.exp(-alpha_npi * u)
        q_c_t  = float(q_c_base_seq[t]) + (Q_C_MAX - float(q_c_base_seq[t])) * v

        for _ in range(n_sub):
            S, V, E, A, I, H, C, R, D = x
            living = max(S + V + E + A + I + H + C + R, 1e-12)
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
                p.gamma_a*A + p.gamma_i*I + p.gamma_h*H + p.gamma_c*C
                - p.omega_r*R - p.nu*R,
                p.delta_h*H + p.delta_c*C,
            ])
            x = np.maximum(x + dt_sub * dx, 0.0)
            x[8] = max(x[8], traj["D"][t])

        for i, n in enumerate(NAMES):
            traj[n][t + 1] = x[i]

    return traj


# ---------------------------------------------------------------------------
# Main PF-MPC loop
# ---------------------------------------------------------------------------
def run_pf_mpc(
    df_filter:  pd.DataFrame,
    t0:         str,
    T_sim:      int            = 60,
    H:          int            = 14,
    weights:    ControlWeights = None,
    h_max:      float          = H_MAX_DE,
    c_max:      float          = C_MAX_DE,
    alpha_npi:  float          = ALPHA_NPI,
    progress_cb = None,        # optional callback(step, T_sim) for UI progress
) -> PFMPCResult:
    """Receding-horizon PF-MPC over a historical period.

    Parameters
    ----------
    df_filter : BPF output DataFrame (must have date + *_mean columns)
    t0        : start date (string 'YYYY-MM-DD')
    T_sim     : number of days to simulate (total horizon)
    H         : lookahead window per MPC step (days)
    weights   : cost weights (uses ControlWeights defaults if None)
    progress_cb : callable(step, T_sim) invoked after each step — use for
                  Streamlit st.progress() updates
    """
    if weights is None:
        weights = ControlWeights()

    comp_names = ["S", "V", "E", "A", "I", "H", "C", "R", "D"]
    dates_all  = pd.to_datetime(df_filter["date"])
    t0_ts      = pd.Timestamp(t0)

    # Pre-extract BPF columns for speed
    def _get(row, key, default):
        v = row.get(key, default)
        return float(v) if v is not None else float(default)

    # Storage
    u_mpc        = np.zeros(T_sim)
    v_mpc        = np.zeros(T_sim)
    beta_bpf     = np.zeros(T_sim)
    q_c_base_bpf = np.zeros(T_sim)
    tau_i_bpf    = np.zeros(T_sim)
    delta_h_bpf  = np.zeros(T_sim)
    converged    = np.zeros(T_sim, dtype=bool)
    n_iters_arr  = np.zeros(T_sim, dtype=int)

    # Warm-start initialisation (moderate NPI, moderate testing)
    u_warm = np.full(H, 0.25)
    v_warm = np.full(H, 0.30)

    actual_T = 0
    for step in range(T_sim):
        t_curr = t0_ts + pd.Timedelta(days=step)
        mask   = dates_all == t_curr
        if not mask.any():
            break
        row = df_filter[mask].iloc[0]

        # BPF posterior mean at this step
        x0_step    = np.array([_get(row, f"{c}_mean", 0.0) for c in comp_names])
        beta_s     = _get(row, "beta_mean",    0.44)
        tau_i_s    = _get(row, "tau_i_mean",   0.0021)
        delta_h_s  = _get(row, "delta_h_mean", 0.019)
        q_c_base_s = _get(row, "rho_c_mean",   0.35)

        beta_bpf[step]     = beta_s
        q_c_base_bpf[step] = q_c_base_s
        tau_i_bpf[step]    = tau_i_s
        delta_h_bpf[step]  = delta_h_s

        # Auto-floor capacity to initial state (avoids infeasible constraint)
        h_max_eff = max(h_max, x0_step[5] * 1.05)
        c_max_eff = max(c_max, x0_step[6] * 1.05)

        # Solve H-day control from current BPF posterior mean
        uv0    = np.concatenate([u_warm, v_warm])
        bounds = [(0.0, 1.0)] * (2 * H)

        res = minimize(
            _objective_2ctrl,
            uv0,
            args=(x0_step, beta_s, tau_i_s, delta_h_s, q_c_base_s,
                  weights, h_max_eff, c_max_eff, alpha_npi),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-8, "gtol": 1e-5},
        )

        u_H = np.clip(res.x[:H], 0.0, 1.0)
        v_H = np.clip(res.x[H:], 0.0, 1.0)

        # Apply only first action (receding horizon)
        u_mpc[step] = u_H[0]
        v_mpc[step] = v_H[0]
        converged[step]   = bool(res.success)
        n_iters_arr[step] = int(res.nit)

        # Warm-start next step: shift solution, repeat last value
        u_warm = np.append(u_H[1:], u_H[-1])
        v_warm = np.append(v_H[1:], v_H[-1])

        actual_T = step + 1
        if progress_cb is not None:
            progress_cb(step + 1, T_sim)

    T_actual = actual_T
    u_mpc        = u_mpc[:T_actual]
    v_mpc        = v_mpc[:T_actual]
    beta_bpf     = beta_bpf[:T_actual]
    q_c_base_bpf = q_c_base_bpf[:T_actual]
    tau_i_bpf    = tau_i_bpf[:T_actual]
    delta_h_bpf  = delta_h_bpf[:T_actual]

    # Dates
    dates = [str((t0_ts + pd.Timedelta(days=i)).date()) for i in range(T_actual + 1)]

    # Initial state (day t0)
    row0    = df_filter[dates_all == t0_ts].iloc[0]
    x0_full = np.array([_get(row0, f"{c}_mean", 0.0) for c in comp_names])
    tau_i0  = _get(row0, "tau_i_mean",   0.0021)
    delta_h0= _get(row0, "delta_h_mean", 0.019)

    # Counterfactual: epidemic under MPC controls with time-varying β from BPF
    traj_mpc = _simulate_mpc_adaptive(
        x0_full, u_mpc, v_mpc,
        beta_bpf, tau_i0, delta_h0, q_c_base_bpf, alpha_npi,
    )
    # Baseline: no controls, same time-varying β
    traj_baseline = _simulate_mpc_adaptive(
        x0_full,
        np.zeros(T_actual), np.zeros(T_actual),
        beta_bpf, tau_i0, delta_h0, q_c_base_bpf, alpha_npi,
    )

    # Derived trajectories
    q_c_traj = q_c_base_bpf + (Q_C_MAX - q_c_base_bpf) * v_mpc
    beta_eff  = beta_bpf * np.exp(-alpha_npi * u_mpc)

    # Summary
    D_base = traj_baseline["D"][-1] - traj_baseline["D"][0]
    D_mpc  = traj_mpc["D"][-1]     - traj_mpc["D"][0]
    H_base = float(np.sum(traj_baseline["H"][1:]))
    H_mpc  = float(np.sum(traj_mpc["H"][1:]))
    C_base = float(np.sum(traj_baseline["C"][1:]))
    C_mpc  = float(np.sum(traj_mpc["C"][1:]))

    return PFMPCResult(
        dates=dates,
        u_mpc=u_mpc, v_mpc=v_mpc,
        q_c_traj=q_c_traj, beta_eff=beta_eff,
        beta_bpf=beta_bpf, q_c_base_bpf=q_c_base_bpf,
        traj_mpc=traj_mpc, traj_baseline=traj_baseline,
        converged=converged, n_iters=n_iters_arr,
        deaths_averted=int(round(D_base - D_mpc)),
        hosp_days_averted=int(round(H_base - H_mpc)),
        icu_days_averted=int(round(C_base - C_mpc)),
        H=H, T_sim=T_actual,
    )
