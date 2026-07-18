"""SVEAIHCRD five-parameter multichannel particle-filter validation on Germany data.

Key calibration notes (Germany, March 10 2020 start)
------------------------------------------------------
* Pre-lockdown R0 ≈ 3.5  →  beta_mean = R0 × (gamma_I + tau_I) ≈ 3.5 × 0.127 = 0.44
* Q_C (detection/reporting rate) is now the 5th log-random-walk parameter.
  Prior mean 0.20 reflects limited PCR capacity in early March 2020 (serological
  studies later showed ~4-10x under-count in early waves).  The filter identifies
  the true Q_C trajectory from the case signal.
* Each particle is initialised INDEPENDENTLY with log-normal compartment scatter
  and wide log-normal parameter priors, creating immediate weight diversity.

Usage
-----
    python scripts/run_sveaihcrd_validation.py

Outputs (data/processed/)
--------------------------
    sveaihcrd_filter_output.csv    -- daily filtered means, 80 % intervals, ESS,
                                      all compartments, q_c trajectory
    sveaihcrd_filter_metrics.json  -- RMSE, MAE, 80 % coverage per channel
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json
import numpy as np
import pandas as pd

from episurveil.models.sveaihcrd import Parameters, rhs
from episurveil.inference.particle_filter import sir_filter_multichannel
from episurveil.observations.delays import apply_case_delay

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_PARTICLES   = 2000
SEED          = 42
ESS_THRESHOLD = 0.45
BURN_IN       = 30
N_POPULATION  = 83_200_000
# Germany's first COVID vaccination: Dec 27 2020 (BNT162b2, nursing homes).
# Enforced as a hard cutoff so nu(t)=0 for all t before this date regardless
# of any data artefact in the vaccinated_pct column.
VACC_START    = pd.Timestamp("2020-12-27")

# Process-noise scales (log-space random-walk innovations)
SIGMA_BETA    = 0.040
SIGMA_TAU_I   = 0.018
SIGMA_DELTA_H = 0.018
SIGMA_RHO_C   = 0.018
SIGMA_Q_C     = 0.015   # testing policy drifts slowly; occasional structural breaks
SIGMA_STATE   = 0.010   # multiplicative compartment noise

# Log-space bounds (hard clipping after each random-walk step)
BOUNDS = {
    "beta":    (np.log(0.05),  np.log(6.0)),
    "tau_i":   (np.log(1e-4),  np.log(0.08)),
    "delta_h": (np.log(1e-4),  np.log(0.10)),
    "rho_c":   (np.log(0.02),  np.log(0.80)),
    # Q_C lower bound 0.20: prevents the degenerate "1% detection → 50× epidemic"
    # solution that depletes S in ~300 days.  Serological studies put early-2020
    # German detection at 10-30%; 0.20 is a conservative floor.
    "q_c":     (np.log(0.20),  np.log(0.99)),
}

# Tempered channel weights (generalised-Bayesian down-weighting)
CHANNEL_WEIGHTS = {"cases": 1.0, "icu": 0.60, "deaths": 0.40, "hosp": 0.10}

# NegBin dispersions per channel
DISPERSIONS = {"cases": 80.0, "icu": 12.0, "deaths": 10.0, "hosp": 15.0}

# Fixed observation constants
KAPPA = 0.62   # symptomatic fraction (same as ODE model)
LOS   = 8.0    # mean hospital length of stay (days)

# ---------------------------------------------------------------------------
# Prior means for dynamic parameters
# ---------------------------------------------------------------------------
# Pre-lockdown R0 ≈ 3.5, gamma_I + tau_I ≈ 1/8 + 0.0021 = 0.127
BETA_MEAN    = 0.44
TAU_I_MEAN   = 0.0021
DELTA_H_MEAN = 0.019
RHO_C_MEAN   = 0.23
# Q_C prior: ~40% detection in early Mar 2020.
# Self-consistent check: Q_C * kappa * sigma * E0 = 0.40 * 0.62 * (1/5.5) * 36k ≈ 1620/day
# which matches the ~1600 confirmed cases on March 10, 2020.
Q_C_MEAN     = 0.40

# ---------------------------------------------------------------------------
# Initial compartment means (Germany March 10, 2020)
# Calibrated at Q_C_MEAN=0.40:
#   observed ~1600/day → true symptomatic ≈ 1600 / (0.40 × 0.62) ≈ 6450/day
#   consistency check: Q_C * kappa * sigma * E0 = 0.40 * 0.62 * (1/5.5) * 36k ≈ 1620 ✓
#   I = kappa * sigma * E / (gamma_i + tau_i) = 0.62 * 0.182 * 36k / 0.127 ≈ 31k
#   A = (1-kappa)/kappa * I ≈ 0.61 * 31k ≈ 19k
# H, C, R, D from RKI records on that date.
# ---------------------------------------------------------------------------
I0_MEAN  =  31_000
A0_MEAN  =  19_000
E0_MEAN  =  36_000
H0_MEAN  =    800
C0_MEAN  =    150
R0_MEAN  =  8_000
D0_MEAN  =     30
V0_MEAN  =      0   # vaccination started Dec 2020
S0_MEAN  = N_POPULATION - (I0_MEAN + A0_MEAN + E0_MEAN + H0_MEAN
                            + C0_MEAN + R0_MEAN + D0_MEAN + V0_MEAN)

# ---------------------------------------------------------------------------
# State layout:  [S, V, E, A, I, H, C, R, D,
#                 log_beta, log_tau_i, log_delta_h, log_rho_c, log_q_c]
#                 (14 elements)
# ---------------------------------------------------------------------------
IDX = {s: i for i, s in enumerate(
    ["S", "V", "E", "A", "I", "H", "C", "R", "D",
     "log_beta", "log_tau_i", "log_delta_h", "log_rho_c", "log_q_c"]
)}
D_STATE = len(IDX)  # 14


# ---------------------------------------------------------------------------
# Per-particle initialisation
# ---------------------------------------------------------------------------
def make_x0(rng: np.random.Generator) -> np.ndarray:
    """Draw one initial particle with independent log-normal scatter."""
    x = np.zeros(D_STATE)

    def _ln(mean, sig):
        return max(mean * np.exp(rng.normal(0.0, sig)), 1.0)

    x[IDX["E"]] = _ln(E0_MEAN,  0.50)
    x[IDX["A"]] = _ln(A0_MEAN,  0.50)
    x[IDX["I"]] = _ln(I0_MEAN,  0.50)
    x[IDX["H"]] = _ln(H0_MEAN,  0.60)
    x[IDX["C"]] = _ln(C0_MEAN,  0.60)
    x[IDX["R"]] = _ln(R0_MEAN,  0.40)
    x[IDX["D"]] = _ln(D0_MEAN,  0.40)
    x[IDX["V"]] = V0_MEAN
    x[IDX["S"]] = max(
        N_POPULATION - sum(x[IDX[k]] for k in ["V","E","A","I","H","C","R","D"]),
        1.0,
    )

    # Log-RW parameters: wide prior spans plausible range
    x[IDX["log_beta"]]    = rng.normal(np.log(BETA_MEAN),    0.50)
    x[IDX["log_tau_i"]]   = rng.normal(np.log(TAU_I_MEAN),   0.60)
    x[IDX["log_delta_h"]] = rng.normal(np.log(DELTA_H_MEAN), 0.50)
    x[IDX["log_rho_c"]]   = rng.normal(np.log(RHO_C_MEAN),   0.50)
    x[IDX["log_q_c"]]     = rng.normal(np.log(Q_C_MEAN),     0.80)
    return x


# ---------------------------------------------------------------------------
# Stochastic state transition
# ---------------------------------------------------------------------------
def _base_params(x: np.ndarray) -> Parameters:
    return Parameters(
        beta    = float(np.exp(np.clip(x[IDX["log_beta"]],    *BOUNDS["beta"]))),
        tau_i   = float(np.exp(np.clip(x[IDX["log_tau_i"]],   *BOUNDS["tau_i"]))),
        delta_h = float(np.exp(np.clip(x[IDX["log_delta_h"]], *BOUNDS["delta_h"]))),
        nu      = 0.0,
    )


def transition(x: np.ndarray, rng: np.random.Generator, row: dict) -> np.ndarray:
    """One-day stochastic transition for the 14-element augmented state."""
    x = np.asarray(x, dtype=float).copy()

    # 1. Evolve log-random-walk parameters (clipped to hard bounds)
    x[IDX["log_beta"]]    = np.clip(
        x[IDX["log_beta"]]    + SIGMA_BETA    * rng.standard_normal(), *BOUNDS["beta"])
    x[IDX["log_tau_i"]]   = np.clip(
        x[IDX["log_tau_i"]]   + SIGMA_TAU_I   * rng.standard_normal(), *BOUNDS["tau_i"])
    x[IDX["log_delta_h"]] = np.clip(
        x[IDX["log_delta_h"]] + SIGMA_DELTA_H * rng.standard_normal(), *BOUNDS["delta_h"])
    x[IDX["log_rho_c"]]   = np.clip(
        x[IDX["log_rho_c"]]   + SIGMA_RHO_C   * rng.standard_normal(), *BOUNDS["rho_c"])
    x[IDX["log_q_c"]]     = np.clip(
        x[IDX["log_q_c"]]     + SIGMA_Q_C     * rng.standard_normal(), *BOUNDS["q_c"])

    # 2. ODE step (Euler, dt=1 day)
    p = _base_params(x)
    comp = x[:9].copy()
    vacc_first  = float(row.get("d_vacc_first", 0.0) or 0.0)
    vacc_frac_t = float(row.get("vacc_frac",    0.0) or 0.0)
    omega_v = Parameters().omega_v   # 1/180

    # nu_eff(t) = new first-dose rate  +  booster-compensation rate
    #
    #   new first doses : vacc_first / N                       [new people entering V]
    #   compensation    : omega_v * vacc_frac(t)               [replaces those who waned]
    #
    # At steady state this keeps V ≈ vaccinated_pct × N, i.e. V represents
    # "all individuals who received at least one dose" (your suggestion).
    # The compensation term implicitly models booster doses without needing
    # a separate booster data series.
    #
    # Before VACC_START both terms are 0 (vacc_first=0 and vacc_frac=0).
    nu_t = vacc_first / N_POPULATION + omega_v * vacc_frac_t
    p_vacc = Parameters(
        beta=p.beta, tau_i=p.tau_i, delta_h=p.delta_h,
        nu=nu_t,
    )
    comp_new = np.maximum(comp + rhs(0.0, comp, p_vacc), 0.0)

    # 3. Multiplicative compartment process noise
    comp_new[:9] = np.maximum(
        comp_new[:9] * np.exp(rng.normal(0.0, SIGMA_STATE, size=9)), 0.0
    )

    # 4. Rescale living compartments to preserve mass balance
    living_det = max(
        comp[:8].sum() - float(np.exp(x[IDX["log_delta_h"]])) * comp[IDX["H"]],
        1.0,
    )
    living_now = max(comp_new[:8].sum(), 1e-9)
    comp_new[:8] *= living_det / living_now

    x[:9] = comp_new
    return x


# ---------------------------------------------------------------------------
# Observation mean functions  (Q_C now read per-particle from state)
# ---------------------------------------------------------------------------
def _obs_cases(particles: np.ndarray) -> np.ndarray:
    E   = np.maximum(particles[:, IDX["E"]], 0.0)
    q_c = np.exp(np.clip(particles[:, IDX["log_q_c"]], *BOUNDS["q_c"]))
    sigma = Parameters().sigma
    return np.maximum(q_c * KAPPA * sigma * E, 1e-9)


def _obs_icu(particles: np.ndarray) -> np.ndarray:
    H   = np.maximum(particles[:, IDX["H"]], 0.0)
    rho = np.exp(np.clip(particles[:, IDX["log_rho_c"]], *BOUNDS["rho_c"]))
    return np.maximum(rho * H, 1e-9)


def _obs_deaths(particles: np.ndarray) -> np.ndarray:
    H     = np.maximum(particles[:, IDX["H"]], 0.0)
    delta = np.exp(np.clip(particles[:, IDX["log_delta_h"]], *BOUNDS["delta_h"]))
    return np.maximum(delta * H, 1e-9)


def _obs_hosp(particles: np.ndarray) -> np.ndarray:
    I   = np.maximum(particles[:, IDX["I"]], 0.0)
    tau = np.exp(np.clip(particles[:, IDX["log_tau_i"]], *BOUNDS["tau_i"]))
    return np.maximum(tau * I * LOS, 1e-9)


OBSERVATION_MEANS = {
    "cases":  _obs_cases,
    "icu":    _obs_icu,
    "deaths": _obs_deaths,
    "hosp":   _obs_hosp,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng_init = np.random.default_rng(SEED)

    # -- Load and prepare data -----------------------------------------------
    panel = pd.read_csv(
        ROOT / "data/processed/germany_integrated_panel.csv",
        parse_dates=["date"],
    )

    # Use 0 for the delay convolution input (NaN = no reports = 0), but keep NaN
    # in the output wherever there were no reported cases so the filter skips
    # those days rather than treating missing data as confirmed-zero observations.
    reported = panel["reported_cases"].to_numpy(dtype=float)
    reported_for_delay = np.where(np.isfinite(reported), reported, 0.0)
    delay_adj = apply_case_delay(reported_for_delay, mean_delay=5.0, sd_delay=2.0)
    panel["cases"] = np.where(np.isfinite(reported), delay_adj, np.nan)

    panel["icu"]    = panel["icu_occupancy"]          # NaN → filter skips
    panel["deaths"] = panel["deaths"]                 # NaN → filter skips
    panel["hosp"]   = panel["hospitalization_proxy"]  # NaN → filter skips

    vp  = panel["vaccinated_pct"].fillna(0.0).to_numpy()
    vfp = panel["fully_vaccinated_pct"].fillna(0.0).to_numpy()
    panel["d_vacc_first"] = (
        np.maximum(np.diff(vp, prepend=vp[0]), 0.0) * N_POPULATION / 100.0
    )
    panel["d_vacc_second"] = (
        np.maximum(np.diff(vfp, prepend=vfp[0]), 0.0) * N_POPULATION / 100.0
    )
    # Hard cutoff: nu(t) = 0 for all t < VACC_START regardless of data artefacts.
    pre_vacc = panel["date"] < VACC_START
    panel.loc[pre_vacc, "d_vacc_first"]  = 0.0
    panel.loc[pre_vacc, "d_vacc_second"] = 0.0
    panel.loc[pre_vacc, "vaccinated_pct"] = 0.0
    n_pre = pre_vacc.sum()
    n_zeroed = int((panel.loc[pre_vacc, "d_vacc_first"] > 0).sum())
    print(f"  Vaccination: nu=0 enforced for {n_pre} days before {VACC_START.date()} "
          f"({n_zeroed} artefact rows zeroed)")

    # vacc_frac(t) = observed fraction of population with >= 1 dose.
    # Used to compute the booster-compensation term in nu_eff(t).
    panel["vacc_frac"] = panel["vaccinated_pct"].fillna(0.0) / 100.0

    rows = panel[
        ["cases", "icu", "deaths", "hosp", "d_vacc_first", "d_vacc_second", "vacc_frac"]
    ].to_dict("records")

    # -- Build DIVERSE initial particle cloud --------------------------------
    print(f"Initialising {N_PARTICLES} particles independently ...")
    all_x0 = np.array([make_x0(rng_init) for _ in range(N_PARTICLES)])
    print(f"  log_beta spread: [{all_x0[:, IDX['log_beta']].min():.2f}, "
          f"{all_x0[:, IDX['log_beta']].max():.2f}]  "
          f"(beta in [{np.exp(all_x0[:, IDX['log_beta']].min()):.2f}, "
          f"{np.exp(all_x0[:, IDX['log_beta']].max()):.2f}])")
    print(f"  log_q_c spread: [{all_x0[:, IDX['log_q_c']].min():.2f}, "
          f"{all_x0[:, IDX['log_q_c']].max():.2f}]  "
          f"(q_c in [{np.exp(all_x0[:, IDX['log_q_c']].min()):.3f}, "
          f"{np.exp(all_x0[:, IDX['log_q_c']].max()):.3f}])")
    print(f"  E spread: [{all_x0[:, IDX['E']].min():.0f}, "
          f"{all_x0[:, IDX['E']].max():.0f}]")

    # -- Run filter ----------------------------------------------------------
    print(f"\nRunning SVEAIHCRD filter  N={N_PARTICLES}  T={len(rows)} days ...")
    results = sir_filter_multichannel(
        rows,
        transition,
        OBSERVATION_MEANS,
        all_x0,
        n_particles    = N_PARTICLES,
        seed           = SEED,
        channel_weights= CHANNEL_WEIGHTS,
        dispersions    = DISPERSIONS,
        ess_threshold  = ESS_THRESHOLD,
    )

    # -- Assemble output CSV -------------------------------------------------
    records = []
    for i, (row, res) in enumerate(zip(rows, results)):
        m, q10, q90 = res["mean"], res["q10"], res["q90"]

        m2   = m[None, :]
        q10_ = q10[None, :]
        q90_ = q90[None, :]

        records.append({
            "date":               panel["date"].iloc[i].strftime("%Y-%m-%d"),
            # observation predictions
            "pred_cases_mean":    float(_obs_cases(m2)[0]),
            "pred_cases_q10":     float(_obs_cases(q10_)[0]),
            "pred_cases_q90":     float(_obs_cases(q90_)[0]),
            "pred_icu_mean":      float(_obs_icu(m2)[0]),
            "pred_icu_q10":       float(_obs_icu(q10_)[0]),
            "pred_icu_q90":       float(_obs_icu(q90_)[0]),
            "pred_deaths_mean":   float(_obs_deaths(m2)[0]),
            "pred_deaths_q10":    float(_obs_deaths(q10_)[0]),
            "pred_deaths_q90":    float(_obs_deaths(q90_)[0]),
            "pred_hosp_mean":     float(_obs_hosp(m2)[0]),
            "pred_hosp_q10":      float(_obs_hosp(q10_)[0]),
            "pred_hosp_q90":      float(_obs_hosp(q90_)[0]),
            # dynamic parameters (5 particle-tracked + 1 data-driven)
            "beta_mean":          float(np.exp(m[IDX["log_beta"]])),
            "tau_i_mean":         float(np.exp(m[IDX["log_tau_i"]])),
            "delta_h_mean":       float(np.exp(m[IDX["log_delta_h"]])),
            "rho_c_mean":         float(np.exp(m[IDX["log_rho_c"]])),
            "q_c_mean":           float(np.exp(m[IDX["log_q_c"]])),
            # nu_eff(t) is particle-independent (data-driven), saved for dashboard
            "nu_eff":             float(row.get("d_vacc_first", 0.0) / N_POPULATION
                                        + Parameters().omega_v * row.get("vacc_frac", 0.0)),
            # compartments — posterior mean and 80% CI
            **{f"{c}_mean": float(m[IDX[c]])    for c in ["S","V","E","A","I","H","C","R","D"]},
            **{f"{c}_q10":  float(q10[IDX[c]])  for c in ["S","V","E","A","I","H","C","R","D"]},
            **{f"{c}_q90":  float(q90[IDX[c]])  for c in ["S","V","E","A","I","H","C","R","D"]},
            # diagnostics
            "ess":                res["ess"],
            # raw observations
            "obs_cases":          row["cases"],
            "obs_icu":            row["icu"],
            "obs_deaths":         row["deaths"],
            "obs_hosp":           row["hosp"],
            # un-delay-adjusted case count for display
            "raw_cases":          float(panel["reported_cases"].iloc[i] or 0.0),
        })

    out_df = pd.DataFrame(records)
    out_path = ROOT / "data/processed/sveaihcrd_filter_output.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    # -- Metrics (post burn-in) ----------------------------------------------
    eval_df = out_df.iloc[BURN_IN:]

    def _metrics(obs_col, pred_col, q10_col, q90_col):
        obs  = eval_df[obs_col].to_numpy()
        pred = eval_df[pred_col].to_numpy()
        q10  = eval_df[q10_col].to_numpy()
        q90  = eval_df[q90_col].to_numpy()
        mask = np.isfinite(obs) & (obs > 0)
        obs, pred, q10, q90 = obs[mask], pred[mask], q10[mask], q90[mask]
        return {
            "rmse":           float(np.sqrt(np.mean((obs - pred) ** 2))),
            "mae":            float(np.mean(np.abs(obs - pred))),
            "coverage_80pct": float(np.mean((obs >= q10) & (obs <= q90))),
            "mean_obs":       float(np.mean(obs)),
            "n":              int(mask.sum()),
        }

    metrics = {
        "cases":        _metrics("obs_cases",  "pred_cases_mean",  "pred_cases_q10",  "pred_cases_q90"),
        "icu":          _metrics("obs_icu",     "pred_icu_mean",    "pred_icu_q10",    "pred_icu_q90"),
        "deaths":       _metrics("obs_deaths",  "pred_deaths_mean", "pred_deaths_q10", "pred_deaths_q90"),
        "hosp":         _metrics("obs_hosp",    "pred_hosp_mean",   "pred_hosp_q10",   "pred_hosp_q90"),
        "mean_ess":     float(eval_df["ess"].mean()),
        "n_particles":  N_PARTICLES,
        "burn_in_days": BURN_IN,
    }

    met_path = ROOT / "data/processed/sveaihcrd_filter_metrics.json"
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {met_path}")

    print("\n=== Validation metrics (post burn-in) ===")
    for ch in ["cases", "icu", "deaths", "hosp"]:
        m_ch = metrics[ch]
        print(f"  {ch:8s}  RMSE={m_ch['rmse']:10.1f}  MAE={m_ch['mae']:9.1f}"
              f"  80%-cov={m_ch['coverage_80pct']:.3f}  mean_obs={m_ch['mean_obs']:.1f}")
    print(f"  mean ESS = {metrics['mean_ess']:.0f} / {N_PARTICLES}")

    # Q_C trajectory summary
    q_c_series = out_df["q_c_mean"].to_numpy()
    print(f"\n  Q_C(t) range: [{q_c_series.min():.3f}, {q_c_series.max():.3f}]")
    print(f"  Q_C start (day 1): {q_c_series[0]:.3f}   Q_C end: {q_c_series[-1]:.3f}")


if __name__ == "__main__":
    main()
