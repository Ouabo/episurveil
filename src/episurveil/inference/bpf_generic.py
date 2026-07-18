"""
Generic Bootstrap Particle Filter compatible with any EpiModel subclass.

Usage
-----
    from episurveil.models.seir import SEIRModel
    from episurveil.inference.bpf_generic import run_bpf

    model = SEIRModel(N=83_200_000, sigma=1/5, gamma=1/10, Q_C=0.5)
    result = run_bpf(model, obs_df, N=2000)
    print(result[["date", "beta_mean", "R_eff_mean"]])

obs_df
------
A pandas DataFrame with:
  - 'date'              : datetime column
  - one column per observation channel matching model.obs_channels
    (use NaN for missing values — those days are skipped in the likelihood)
  - optional columns for exogenous inputs (e.g. 'nu' for vaccination rate)

channel_weights (tempering)
---------------------------
Dict mapping channel name → alpha in (0, 1].
Default: all channels receive weight 1.0 (no tempering).
Reduce alpha for noisy or conflicting channels (e.g. 0.4 for deaths).

phi (NegBin dispersions)
------------------------
Dict mapping channel name → dispersion.
Default: read from model.phi_{channel} attribute if present, else 50.

Returns
-------
pd.DataFrame with columns:
  date
  {state}_mean, {state}_q10, {state}_q90    for each state
  {param}_mean, {param}_q10, {param}_q90    for each log-RW parameter
  R_eff_mean, R_eff_q10, R_eff_q90
  ess
  obs_{channel}                              original observations
  pred_{channel}_mean                        posterior predictive mean
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import gammaln

from episurveil.models.base import EpiModel


# ── NegBin log-likelihood ───────────────────────────────────────────────────
def _negbin_loglik(y: float, mu: np.ndarray, phi: float) -> np.ndarray:
    """Log-likelihood of NegBin(mu, phi) for scalar observation y."""
    mu  = np.maximum(mu, 1e-9)
    phi = max(phi, 1e-3)
    # NB parametrisation: mean=mu, var=mu + mu^2/phi
    ll = (
        gammaln(y + phi) - gammaln(phi) - gammaln(y + 1)
        + phi * np.log(phi / (phi + mu))
        + y   * np.log(mu  / (phi + mu))
    )
    return ll


@dataclass
class FilterMetrics:
    channel: str
    rmse:     float
    mae:      float
    cov80:    float
    mean_obs: float


# ── main function ───────────────────────────────────────────────────────────
def run_bpf(
    model:            EpiModel,
    obs_df:           pd.DataFrame,
    N:                int   = 2000,
    ess_threshold:    float = 0.45,
    channel_weights:  dict  = None,
    phi:              dict  = None,
    seed:             int   = 42,
    exog_df:          pd.DataFrame = None,
    burn_in:          int   = 30,
    progress:         bool  = True,
    return_particles: bool  = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, np.ndarray]":
    """
    Run the Bootstrap Particle Filter for `model` against `obs_df`.

    Parameters
    ----------
    model            : any EpiModel subclass
    obs_df           : DataFrame with 'date' + channel columns
    N                : particle count
    ess_threshold    : resample when ESS < threshold * N
    channel_weights  : tempering weights per channel (default all 1.0)
    phi              : NegBin dispersions per channel
    seed             : RNG seed
    exog_df          : optional DataFrame with 'date' + exogenous columns
    burn_in          : skip first n days from metric computation
    progress         : print progress every 100 steps

    Returns
    -------
    pd.DataFrame (one row per date)
    """
    rng = np.random.default_rng(seed)
    obs_df = obs_df.copy()
    obs_df["date"] = pd.to_datetime(obs_df["date"])
    T = len(obs_df)

    # defaults
    if channel_weights is None:
        channel_weights = {ch: 1.0 for ch in model.obs_channels}
    if phi is None:
        phi = {}
        for ch in model.obs_channels:
            attr = f"phi_{ch}"
            phi[ch] = getattr(model, attr, 50.0)

    # ── Adaptive initial-condition scaling ────────────────────────────────
    # Estimate I0_frac from the first valid observation so the particles are
    # initialised at the right epidemic scale regardless of model defaults.
    # Works by inverting the observation model: obs ≈ Q_C × rate × E or I.
    if hasattr(model, "_I0_frac") and hasattr(model, "N") and model.N > 0:
        _q_c    = getattr(model, "Q_C", 0.5)
        _sigma  = getattr(model, "sigma", None)
        _gamma  = getattr(model, "gamma", getattr(model, "gamma_i", 0.10))
        for _t in range(min(10, T)):
            _val = obs_df.iloc[_t].get("cases", np.nan)
            if pd.isna(_val) or float(_val) <= 0:
                continue
            if _sigma is not None:
                # SEIR-like: obs = Q_C * sigma * E  → E = obs / (Q_C * sigma)
                _E_est = float(_val) / max(_q_c * _sigma, 1e-9)
            else:
                # SIR-like: obs = Q_C * gamma * I  → I = obs / (Q_C * gamma)
                _E_est = float(_val) / max(_q_c * _gamma, 1e-9)
            model._I0_frac = float(np.clip(_E_est / model.N, 1e-7, 0.05))
            break

    # initialise particles
    particles = model.init_particles(N, rng)          # (N, aug_dim)
    log_w     = np.zeros(N)

    # output storage
    s_names = model.state_names
    p_names = model.param_names
    n_s     = model.n_states
    n_p     = model.n_params

    rows = []

    for t in range(T):
        row_obs = obs_df.iloc[t]
        exog_t  = (
            exog_df[exog_df["date"] == row_obs["date"]].iloc[0].to_dict()
            if exog_df is not None else None
        )

        # ── propagate ──────────────────────────────────────────────────────
        particles = model.step(particles, t, exog=exog_t)

        # ── weight update ──────────────────────────────────────────────────
        obs_means = model.observation_mean(particles)
        for ch in model.obs_channels:
            y = row_obs.get(ch, np.nan)
            if pd.isna(y):
                continue
            alpha = channel_weights.get(ch, 1.0)
            phi_ch = phi.get(ch, 50.0)
            log_w += alpha * _negbin_loglik(float(y), obs_means[ch], phi_ch)

        # stabilise and normalise
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()

        # ── ESS and systematic resampling ──────────────────────────────────
        ess = 1.0 / float(np.sum(w ** 2))
        if ess < ess_threshold * N:
            # systematic resample
            positions = (rng.random() + np.arange(N)) / N
            cumsum = np.cumsum(w)
            idx = np.searchsorted(cumsum, positions)
            idx = np.clip(idx, 0, N - 1)
            particles = particles[idx]
            log_w = np.zeros(N)
            w     = np.full(N, 1.0 / N)

        # ── posterior summaries ────────────────────────────────────────────
        r_eff = model.R_eff(particles)
        idx_w = rng.choice(N, size=N, replace=True, p=w)
        samp  = particles[idx_w]

        row = {"date": row_obs["date"], "ess": ess}

        # states
        for i, name in enumerate(s_names):
            vals = samp[:, i]
            row[f"{name}_mean"] = float(np.sum(w * particles[:, i]))
            row[f"{name}_q10"]  = float(np.quantile(vals, 0.10))
            row[f"{name}_q90"]  = float(np.quantile(vals, 0.90))

        # log-RW parameters (stored as log; exponentiate for output)
        for j, name in enumerate(p_names):
            log_vals = particles[:, n_s + j]
            vals     = np.exp(samp[:, n_s + j])
            row[f"{name}_mean"] = float(np.exp(np.sum(w * log_vals)))
            row[f"{name}_q10"]  = float(np.quantile(vals, 0.10))
            row[f"{name}_q90"]  = float(np.quantile(vals, 0.90))

        # R_eff + P(R_eff > 1) — key surveillance signal
        r_samp = r_eff[idx_w]
        row["R_eff_mean"] = float(np.sum(w * r_eff))
        row["R_eff_q10"]  = float(np.quantile(r_samp, 0.10))
        row["R_eff_q90"]  = float(np.quantile(r_samp, 0.90))
        row["P_growing"]  = float(np.sum(w * (r_eff > 1.0)))

        # observation echo + predictive mean + 80% credible interval
        samp_obs = model.observation_mean(samp)
        for ch in model.obs_channels:
            row[f"obs_{ch}"]        = row_obs.get(ch, np.nan)
            row[f"pred_{ch}_mean"]  = float(np.sum(w * obs_means[ch]))
            row[f"pred_{ch}_q10"]   = float(np.quantile(samp_obs[ch], 0.10))
            row[f"pred_{ch}_q90"]   = float(np.quantile(samp_obs[ch], 0.90))

        rows.append(row)

        if progress and (t + 1) % 100 == 0:
            print(f"  step {t+1}/{T}  ESS={ess:.0f}  "
                  f"R_eff={row['R_eff_mean']:.2f}", flush=True)

    df_out = pd.DataFrame(rows)
    if return_particles:
        return df_out, particles   # caller uses particles for forecast

    # ── validation metrics (post burn-in) ──────────────────────────────────
    df_eval = df_out.iloc[burn_in:]
    print("\n=== Validation metrics (post burn-in) ===")
    for ch in model.obs_channels:
        obs_col  = f"obs_{ch}"
        pred_col = f"pred_{ch}_mean"
        q10_col  = f"pred_{ch}_q10" if f"pred_{ch}_q10" in df_eval.columns else None
        q90_col  = f"pred_{ch}_q90" if f"pred_{ch}_q90" in df_eval.columns else None

        mask = df_eval[obs_col].notna()
        if not mask.any():
            continue
        y_obs  = df_eval.loc[mask, obs_col].values
        y_pred = df_eval.loc[mask, pred_col].values
        rmse   = float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
        mae    = float(np.mean(np.abs(y_obs - y_pred)))
        print(f"  {ch:<12} RMSE={rmse:>10.1f}  MAE={mae:>8.1f}  "
              f"mean_obs={y_obs.mean():.1f}")

    print(f"  mean ESS = {df_out['ess'].mean():.0f} / {N}")
    return df_out


# ── short-term forecast ─────────────────────────────────────────────────────

def forecast_bpf(
    model:            EpiModel,
    final_particles:  np.ndarray,
    horizon:          int   = 14,
    start_date:       "pd.Timestamp | None" = None,
    seed:             int   = 1,
    beta_multiplier:  float = 1.0,
) -> pd.DataFrame:
    """
    Forward-propagate the final particle set from run_bpf to produce a forecast.

    Parameters
    ----------
    model            : same EpiModel used in run_bpf (parameters are baked in)
    final_particles  : particle array returned by run_bpf(return_particles=True)
    horizon          : number of days to forecast
    start_date       : last date of the observed series; forecast dates start at +1
    seed             : RNG seed
    beta_multiplier  : scenario intervention — scales every particle's beta by this
                       factor at the start of the forecast (e.g. 0.70 = 30% reduction)

    Returns
    -------
    pd.DataFrame with columns:
      date, {state}_mean/q10/q90, {param}_mean/q10/q90,
      R_eff_mean/q10/q90, P_growing, pred_{ch}_mean/q10/q90
    No obs_ columns (nothing to compare against).
    """
    rng       = np.random.default_rng(seed)
    particles = final_particles.copy()
    N_p       = len(particles)

    s_names = model.state_names
    p_names = model.param_names
    n_s     = model.n_states
    n_p     = model.n_params

    # Apply beta multiplier once at the start (instantaneous intervention)
    if beta_multiplier != 1.0 and n_p > 0:
        log_shift = np.log(float(beta_multiplier))
        particles[:, n_s] = np.clip(
            particles[:, n_s] + log_shift,
            model._log_min[0], model._log_max[0],
        )

    rows = []
    for h in range(horizon):
        particles = model.step(particles, h, exog=None)
        obs_means = model.observation_mean(particles)
        r_eff     = model.R_eff(particles)

        # Equal weights — no likelihood update in the forecast horizon
        idx_w = rng.choice(N_p, size=N_p, replace=True)
        samp  = particles[idx_w]

        row = {
            "date": (
                start_date + pd.Timedelta(days=h + 1)
                if start_date is not None
                else h + 1
            )
        }

        for i, name in enumerate(s_names):
            vals = particles[:, i]
            row[f"{name}_mean"] = float(np.mean(vals))
            row[f"{name}_q10"]  = float(np.quantile(vals, 0.10))
            row[f"{name}_q90"]  = float(np.quantile(vals, 0.90))

        for j, name in enumerate(p_names):
            log_vals = particles[:, n_s + j]
            vals     = np.exp(samp[:, n_s + j])
            row[f"{name}_mean"] = float(np.exp(np.mean(log_vals)))
            row[f"{name}_q10"]  = float(np.quantile(vals, 0.10))
            row[f"{name}_q90"]  = float(np.quantile(vals, 0.90))

        r_samp = r_eff[idx_w]
        row["R_eff_mean"] = float(np.mean(r_eff))
        row["R_eff_q10"]  = float(np.quantile(r_samp, 0.10))
        row["R_eff_q90"]  = float(np.quantile(r_samp, 0.90))
        row["P_growing"]  = float(np.mean(r_eff > 1.0))

        samp_obs = model.observation_mean(samp)
        for ch in model.obs_channels:
            row[f"pred_{ch}_mean"] = float(np.mean(obs_means[ch]))
            row[f"pred_{ch}_q10"]  = float(np.quantile(samp_obs[ch], 0.10))
            row[f"pred_{ch}_q90"]  = float(np.quantile(samp_obs[ch], 0.90))

        rows.append(row)

    return pd.DataFrame(rows)
