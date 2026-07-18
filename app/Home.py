"""EpiSurveil Control Platform -- interactive dashboard.

Displays SVEAIHCRD bootstrap-particle-filter results for German COVID-19
surveillance (March 2020 - March 2023).

Tabs: Filter output | Compartments | Dynamic parameters | Metrics | Fixed parameters | ESS
"""
from __future__ import annotations
import json
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT         = Path(__file__).resolve().parents[1]
import sys, importlib
sys.path.insert(0, str(ROOT / "src"))
# Force fresh import so Streamlit hot-reload picks up changes to the control module
for _key in [k for k in sys.modules if k.startswith("episurveil.control")]:
    del sys.modules[_key]
from episurveil.control.optimal_control import (
    run_optimal_control, run_pareto_sweep,
    ControlWeights, H_MAX_DE, C_MAX_DE,
)
from episurveil.control.pf_mpc import run_pf_mpc, PFMPCResult
FILTER_CSV   = ROOT / "data/processed/sveaihcrd_filter_output.csv"
METRICS_JSON = ROOT / "data/processed/sveaihcrd_filter_metrics.json"

# ── Alert log ────────────────────────────────────────────────────────────────
_ALERT_LOG_PATH = ROOT / "data" / "alert_log.csv"
_ALERT_LOG_COLS = [
    "timestamp", "source", "country", "last_obs_date",
    "R_eff_mean", "R_eff_q10", "R_eff_q90", "P_growing",
    "status", "model", "N_particles",
]

def _append_alert_log(rows: list) -> None:
    """Append a list of dicts to the persistent alert log CSV."""
    import csv
    _ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _ALERT_LOG_PATH.exists()
    with open(_ALERT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_ALERT_LOG_COLS,
                           extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)

def _read_alert_log() -> "pd.DataFrame":
    """Return the full alert log as a DataFrame, or empty if none exists."""
    if _ALERT_LOG_PATH.exists():
        return pd.read_csv(_ALERT_LOG_PATH,
                           parse_dates=["timestamp", "last_obs_date"])
    return pd.DataFrame(columns=_ALERT_LOG_COLS)
N_POPULATION = 83_200_000
BURN_IN      = 30

C = {
    "cases":"#1E88E5","icu":"#E53935","deaths":"#8E24AA","hosp":"#F4511E",
    "S":"#1565C0","V":"#00897B","E":"#FDD835","A":"#FB8C00",
    "I":"#E53935","H":"#8E24AA","C":"#D81B60","R":"#43A047","D":"#546E7A",
    "beta":"#1565C0","tau_i":"#2E7D32","delta_h":"#B71C1C","rho_c":"#E65100",
    "nu_eff":"#00838F",
}
CI_OPACITY = 0.15
OBS_OPACITY = 0.50
LINE_W = 1.6

st.set_page_config(page_title="EpiSurveil - SVEAIHCRD Germany",
                   layout="wide", page_icon="\U0001f9a0")

@st.cache_data(ttl=30)
def load_data():
    df = pd.read_csv(FILTER_CSV, parse_dates=["date"])
    with open(METRICS_JSON) as f:
        metrics = json.load(f)
    return df, metrics

if not FILTER_CSV.exists():
    st.error("Filter output not found. Run `python scripts/run_sveaihcrd_validation.py` first.")
    st.stop()

df_full, metrics = load_data()

# ---------------------------------------------------------------------------
# Sidebar — date-range selector
# ---------------------------------------------------------------------------
st.sidebar.title("EpiSurveil")
st.sidebar.markdown("**SVEAIHCRD · Germany COVID-19**")
st.sidebar.markdown("---")
st.sidebar.markdown("### Date range")

date_min = df_full["date"].min().date()
date_max = df_full["date"].max().date()

sel = st.sidebar.slider(
    "Select display period",
    min_value=date_min,
    max_value=date_max,
    value=(date_min, date_max),
    format="MMM YYYY",
)
start_date, end_date = pd.Timestamp(sel[0]), pd.Timestamp(sel[1])

# Apply date filter (always keep burn-in excluded from metrics, but show full range in plots)
df = df_full[(df_full["date"] >= start_date) & (df_full["date"] <= end_date)].copy()
df_eval = df[df["date"] >= (df_full["date"].iloc[BURN_IN])].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"Showing **{len(df):,}** days  \n"
    f"{start_date:%d %b %Y} – {end_date:%d %b %Y}"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Filter config**")
st.sidebar.markdown("N = 2,000 · ESS = 0.45 · seed = 42")
st.sidebar.markdown("---")
if st.sidebar.button("Refresh data", help="Reload CSV after re-running the filter"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.markdown("---")

# ── Sidebar feedback link (always visible) ───────────────────────────────
st.sidebar.markdown("**💬 Feedback**")
st.sidebar.markdown(
    "Found a bug or have a suggestion?  \n"
    "[Open a GitHub Issue]"
    "(https://github.com/YOUR-USERNAME/episurveil/issues/new"
    "?labels=feedback&title=[Feedback]&body=**What+I+tried**%3A%0A%0A"
    "**What+happened**%3A%0A%0A**Suggestions**%3A%0A)"
    "  ·  "
    "[View all issues]"
    "(https://github.com/YOUR-USERNAME/episurveil/issues)"
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: RKI, DIVI, OWID  \n"
    "Daily reporting ended **April 2023**; panel extended to **Oct 2024** via SARI "
    "sentinel hospitalization (RKI COVID-SARI). Cases/deaths/ICU are NaN after April 2023.  \n"
    "To extend further: download fresh OWID/DIVI data and re-run `extend_panel.py`."
)

# ---------------------------------------------------------------------------
# Header KPIs
# ---------------------------------------------------------------------------
st.title("EpiSurveil Control Platform")
st.caption(
    "Bootstrap Particle Filter · SVEAIHCRD · 5 time-varying parameters · "
    "Germany COVID-19 · March 2020 – October 2024"
)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Cases 80% cov.",  f"{metrics['cases']['coverage_80pct']:.1%}")
c2.metric("ICU 80% cov.",    f"{metrics['icu']['coverage_80pct']:.1%}")
c3.metric("Deaths 80% cov.", f"{metrics['deaths']['coverage_80pct']:.1%}")
c4.metric("Hosp 80% cov.",   f"{metrics['hosp']['coverage_80pct']:.1%}")
mean_ess = metrics.get("mean_ess", 0)
c5.metric("Mean ESS", f"{mean_ess:.0f} / 2,000")
st.markdown("---")

burn_date = df_full["date"].iloc[BURN_IN].strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ci_band(fig, row, col, x, q10, q90, color):
    x_a = list(x) + list(x)[::-1]
    y_a = list(q90) + list(q10)[::-1]
    fig.add_trace(go.Scatter(x=x_a, y=y_a, fill="toself", fillcolor=color,
                             opacity=CI_OPACITY, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=row, col=col)

def _line(fig, row, col, x, y, color, name, dash="solid"):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                             line=dict(color=color, width=LINE_W, dash=dash)),
                  row=row, col=col)

def _scatter(fig, row, col, x, y, name):
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=name,
                             marker=dict(color="gray", size=2.5, opacity=OBS_OPACITY)),
                  row=row, col=col)

def _fmt(v):
    if v >= 1e6: return f"{v/1e6:.2f}M"
    if v >= 1e3: return f"{v/1e3:.0f}k"
    return f"{v:.1f}"


def _build_epidemic_report(
    res: pd.DataFrame,
    model_key: str,
    obs_channels: list,
    param_names: list,
    N_particles: int,
    N_pop: int,
    gamma: float,
    beta_max: float,
    disease_label: str,
    country_label: str,
    burn: int = 30,
    forecast_df: "pd.DataFrame | None" = None,
    scenario_df: "pd.DataFrame | None" = None,
    alert_threshold: float = 0.80,
    beta_reduction_pct: int = 0,
) -> str:
    """Return a markdown epidemic surveillance report from BPF output."""
    import datetime as _dt

    df = res.iloc[burn:].copy() if len(res) > burn else res.copy()
    date_start = df["date"].iloc[0]
    date_end   = df["date"].iloc[-1]
    n_days     = len(df)
    today      = _dt.date.today().isoformat()

    L = []   # lines

    L += [f"# Epidemic Surveillance Report",
          f"",
          f"**Generated**: {today}  ",
          f"**Model**: {model_key}  ",
          f"**Disease / variant**: {disease_label}  ",
          f"**Country / region**: {country_label}  ",
          f"**Analysis period**: {date_start.date()} to {date_end.date()} ({n_days} days)  ",
          f"**Population (N)**: {N_pop:,}  ",
          f"**Particles**: {N_particles}  ",
          "", "---", ""]

    # ── 0. Alert status ──────────────────────────────────────────────────────
    if "P_growing" in df.columns:
        L += ["## 0. Current Epidemic Status", ""]
        _pg_now   = float(df["P_growing"].iloc[-1])
        _pg_peak  = float(df["P_growing"].max())
        _pg_peak_date = df["date"].iloc[int(df["P_growing"].argmax())].date()
        _days_alert   = int((df["P_growing"] >= alert_threshold).sum())

        if _pg_now >= alert_threshold:
            L.append(f"> ### 🔴 ALERT — Epidemic is GROWING")
            L.append(f"> P(R_eff > 1) = **{_pg_now:.0%}** on {date_end.date()} "
                     f"(threshold: {alert_threshold:.0%})")
        elif _pg_now >= 0.50:
            L.append(f"> ### 🟡 CAUTION — Growth uncertain")
            L.append(f"> P(R_eff > 1) = **{_pg_now:.0%}** on {date_end.date()}")
        else:
            L.append(f"> ### 🟢 Under control")
            L.append(f"> P(R_eff > 1) = **{_pg_now:.0%}** on {date_end.date()}")
        L.append("")

        L += ["| Metric | Value |", "|---|---|",
              f"| P(R_eff > 1) at end of period | {_pg_now:.0%} |",
              f"| Peak P(R_eff > 1) | {_pg_peak:.0%} on {_pg_peak_date} |",
              f"| Days above alert threshold ({alert_threshold:.0%}) | {_days_alert} / {n_days} |",
              ""]

        if forecast_df is not None and "P_growing" in forecast_df.columns:
            _pg_7  = float(forecast_df["P_growing"].iloc[min(6,  len(forecast_df)-1)])
            _pg_14 = float(forecast_df["P_growing"].iloc[min(13, len(forecast_df)-1)])
            L += [f"| P(growing) in 7 days  | {_pg_7:.0%} |",
                  f"| P(growing) in 14 days | {_pg_14:.0%} |", ""]

        L += ["---", ""]

    # ── 1. Fit quality ───────────────────────────────────────────────────────
    def _grade(mape):
        if mape < 15:  return "✅ Excellent"
        if mape < 30:  return "🟡 Good"
        if mape < 50:  return "🟠 Fair"
        return "🔴 Poor"

    L += ["## 1. Fit Quality", "",
          "| Channel | RMSE | MAE | MAPE | Grade |",
          "|---|---|---|---|---|"]

    grades, ch_stats = [], {}
    for ch in obs_channels:
        obs_col, pred_col = f"obs_{ch}", f"pred_{ch}_mean"
        if obs_col not in df.columns or pred_col not in df.columns:
            continue
        mask = df[obs_col].notna() & (df[obs_col] >= 0)
        if mask.sum() < 5:
            continue
        y_obs  = df.loc[mask, obs_col].values
        y_pred = df.loc[mask, pred_col].values
        rmse = float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
        mae  = float(np.mean(np.abs(y_obs - y_pred)))
        mape = float(np.mean(np.abs((y_obs - y_pred) / np.maximum(y_obs, 1)))) * 100
        g = _grade(mape)
        grades.append(mape)
        ch_stats[ch] = dict(rmse=rmse, mae=mae, mape=mape)
        L.append(f"| {ch.capitalize()} | {rmse:,.0f} | {mae:,.0f} | {mape:.1f}% | {g} |")

    overall_mape = float(np.mean(grades)) if grades else None
    L += [""]
    if overall_mape is not None:
        L.append(f"**Overall fit grade**: {_grade(overall_mape)} (mean MAPE = {overall_mape:.1f}%)")
    L += ["", "---", ""]

    # ── 2. R_eff and epidemic phases ─────────────────────────────────────────
    L += ["## 2. Epidemic Trajectory", ""]
    r        = df["R_eff_mean"]
    r_smooth = r.rolling(7, center=True, min_periods=1).mean()
    r_mean   = float(r.mean())
    r_max    = float(r.max())
    r_min    = float(r.min())
    r_max_date = df["date"].iloc[int(r.argmax())].date()
    r_min_date = df["date"].iloc[int(r.argmin())].date()
    pct_above  = float((r > 1.0).mean()) * 100

    L += [f"| Statistic | Value |", "|---|---|",
          f"| Mean R_eff | {r_mean:.2f} |",
          f"| Peak R_eff | {r_max:.2f} on {r_max_date} |",
          f"| Minimum R_eff | {r_min:.2f} on {r_min_date} |",
          f"| Time with R_eff > 1 | {pct_above:.0f}% of period |",
          ""]

    above = (r_smooth > 1.0).astype(int)
    n_up   = int((above.diff() == 1).sum())
    n_waves = max(n_up, 1 if above.iloc[0] == 1 else 0)
    L.append(f"**Waves detected**: {n_waves}")
    L.append("")

    # Phase blocks (≥ 7 days)
    phase_arr = np.where(r_smooth > 1.05, "Growth",
                np.where(r_smooth < 0.95, "Decline", "Transition"))
    phase_rows = []
    cur, st_i = phase_arr[0], 0
    last_i = len(phase_arr) - 1
    for i in range(1, len(phase_arr)):
        changed = phase_arr[i] != cur
        is_last = i == last_i
        if changed or is_last:
            # end index: include i only if it's the last element AND same phase
            en_i = i if (is_last and not changed) else i - 1
            dur  = en_i - st_i + 1
            if dur >= 7:
                phase_rows.append(dict(
                    phase=cur,
                    start=df["date"].iloc[st_i].date(),
                    end=df["date"].iloc[en_i].date(),
                    days=dur,
                    r_mean=float(r.iloc[st_i:en_i+1].mean()),
                ))
            cur, st_i = phase_arr[i], i

    if phase_rows:
        L += ["**Epidemic phases** (≥ 7 consecutive days):", "",
              "| Phase | Start | End | Duration | Mean R_eff |",
              "|---|---|---|---|---|"]
        icons = {"Growth": "📈", "Decline": "📉", "Transition": "➡️"}
        for pr in phase_rows:
            ic = icons.get(pr["phase"], "")
            L.append(f"| {ic} {pr['phase']} | {pr['start']} | {pr['end']} "
                     f"| {pr['days']} days | {pr['r_mean']:.2f} |")
    L += ["", "---", ""]

    # ── 3. Transmission β(t) ─────────────────────────────────────────────────
    ceiling_hits = 0
    if "beta_mean" in df.columns:
        L += ["## 3. Transmission Rate β(t)", ""]
        b = df["beta_mean"]
        b_mean  = float(b.mean())
        b_max_v = float(b.max())
        b_min_v = float(b.min())
        b_std   = float(b.std())
        b_max_date = df["date"].iloc[int(b.argmax())].date()
        ceiling_hits = int((b > 0.95 * beta_max).sum())
        ceiling_pct  = 100 * ceiling_hits / max(len(b), 1)
        L += [f"| Statistic | Value |", "|---|---|",
              f"| Mean β | {b_mean:.3f} |",
              f"| Min β | {b_min_v:.3f} |",
              f"| Max β | {b_max_v:.3f} on {b_max_date} |",
              f"| Std β | {b_std:.3f} |",
              f"| β ceiling (β_max = {beta_max:.2f}) | {ceiling_hits} days ({ceiling_pct:.0f}%) |",
              f"| Implied R₀ range | [{b_min_v/gamma:.1f}, {b_max_v/gamma:.1f}] |",
              ""]
        if ceiling_hits > 10:
            L.append(f"> ⚠️ β hit the ceiling {ceiling_hits} days ({ceiling_pct:.0f}% of period). "
                     f"Raise β_max or enable waning immunity (ω_R > 0).")
            L.append("")
        L += ["---", ""]

    # ── 4. Filter health (ESS) ───────────────────────────────────────────────
    ess_ratio, n_resamp, resamp_pct = None, 0, 0.0
    if "ess" in df.columns:
        L += ["## 4. Particle Filter Health (ESS)", ""]
        ess = df["ess"]
        ess_mean = float(ess.mean())
        ess_min  = float(ess.min())
        ess_min_date = df["date"].iloc[int(ess.argmin())].date()
        thr = 0.45 * N_particles
        n_resamp    = int((ess < thr).sum())
        resamp_pct  = 100 * n_resamp / max(len(ess), 1)
        ess_ratio   = ess_mean / N_particles

        def _ess_grade(ratio):
            if ratio > 0.70: return "✅ Healthy"
            if ratio > 0.45: return "🟡 Acceptable"
            return "🔴 Poor — collapse suspected"

        L += [f"| Statistic | Value |", "|---|---|",
              f"| Mean ESS | {ess_mean:.0f} / {N_particles} "
              f"({100*ess_ratio:.0f}%) — {_ess_grade(ess_ratio)} |",
              f"| Minimum ESS | {ess_min:.0f} / {N_particles} on {ess_min_date} |",
              f"| Resample threshold | {thr:.0f} (0.45 × N) |",
              f"| Resampling events | {n_resamp} / {len(ess)} days ({resamp_pct:.0f}%) |",
              ""]
        if ess_ratio > 0.90:
            L.append("> 🔴 **ESS ≈ N** — all particles carry equal weight. "
                     "This is a sign of filter collapse (susceptible depletion or wrong N_pop).")
            L.append("")
        elif resamp_pct > 50:
            L.append(f"> 🟡 Frequent resampling ({resamp_pct:.0f}% of days). "
                     "Consider reducing σ_β or increasing N particles.")
            L.append("")
        L += ["---", ""]

    # ── 5. Compartment summary ───────────────────────────────────────────────
    comp_rows = []
    for sn in ["S", "E", "I", "H", "R", "D"]:
        col = f"{sn}_mean"
        if col in df.columns:
            vals = df[col]
            comp_rows.append(dict(
                name=sn,
                peak=float(vals.max()),
                peak_date=df["date"].iloc[int(vals.argmax())].date(),
                final=float(vals.iloc[-1]),
            ))
    if comp_rows:
        L += ["## 5. Compartment Summary", "",
              "| Compartment | Peak value | Peak date | Final value |",
              "|---|---|---|---|"]
        for cr in comp_rows:
            L.append(f"| {cr['name']} | {cr['peak']:,.0f} | {cr['peak_date']} | {cr['final']:,.0f} |")
        L += ["", "---", ""]

    # ── 6. Interpretation (narrative) ────────────────────────────────────────
    L += ["## 6. Interpretation", ""]
    pts = []

    # Fit
    if overall_mape is not None:
        if overall_mape < 15:
            pts.append(f"The filter achieved an **excellent fit** (mean MAPE {overall_mape:.1f}%), "
                       "closely tracking the observed epidemic curve.")
        elif overall_mape < 30:
            pts.append(f"The filter achieved a **good fit** (mean MAPE {overall_mape:.1f}%). "
                       "Some deviations are expected given model structural simplifications.")
        else:
            pts.append(f"Fit quality is **fair to poor** (mean MAPE {overall_mape:.1f}%). "
                       "Consider revising population size, Q_C, or β_max.")

    # Waves
    if n_waves == 1:
        pts.append(f"A **single epidemic wave** was identified "
                   f"({date_start.date()} → {date_end.date()}), "
                   f"peaking at R_eff = {r_max:.2f} on {r_max_date}.")
    else:
        pts.append(f"**{n_waves} epidemic waves** were identified over the period. "
                   f"R_eff reached a maximum of {r_max:.2f} on {r_max_date}. "
                   f"The epidemic was in a growth phase {pct_above:.0f}% of the time.")

    # β
    if "beta_mean" in df.columns:
        if ceiling_hits > 10:
            pts.append(f"Transmission rate β saturated at its ceiling (β_max = {beta_max:.2f}) "
                       f"on {ceiling_hits} days — the model may be underestimating peak transmission. "
                       f"Raise β_max to R₀_max × γ for the dominant variant.")
        else:
            pts.append(f"β ranged from {b_min_v:.3f} to {b_max_v:.3f} "
                       f"(R₀ equivalent: [{b_min_v/gamma:.1f}, {b_max_v/gamma:.1f}]). "
                       "No ceiling saturation detected.")

    # ESS
    if ess_ratio is not None:
        if ess_ratio > 0.90:
            pts.append("ESS remained near N throughout — a hallmark of **filter collapse** "
                       "(susceptible depletion or incorrect N_pop). "
                       "Enable waning immunity or check the population size.")
        elif ess_ratio > 0.70:
            pts.append(f"The particle filter maintained **healthy diversity** "
                       f"(mean ESS {ess_mean:.0f}/{N_particles}, {100*ess_ratio:.0f}%), "
                       f"with {n_resamp} resampling events.")
        else:
            pts.append(f"ESS dropped frequently (mean {ess_mean:.0f}/{N_particles}). "
                       f"Increasing N or reducing σ_β may improve stability.")

    for pt in pts:
        L.append(f"- {pt}")
    L += ["", "---", ""]

    # ── 7. Suggested adjustments ────────────────────────────────────────────
    L += ["## 7. Suggested Adjustments", ""]
    sugg = []
    if overall_mape is not None and overall_mape > 30:
        sugg.append("**Poor fit**: verify population size matches the region "
                    "(e.g. Germany = 83,200,000). Check Q_C against known detection rates.")
    if ceiling_hits > 10:
        sugg.append(f"**β ceiling** ({ceiling_hits} days): raise β_max to at least "
                    f"{beta_max * 1.5:.2f} (= R₀_max × γ for the dominant variant).")
    if ess_ratio is not None and ess_ratio > 0.88:
        sugg.append("**Filter collapse (ESS ≈ N)**: enable waning immunity (ω_R ≈ 1/180) "
                    "and confirm N_pop is the true regional population.")
    if resamp_pct > 50:
        sugg.append(f"**Frequent resampling ({resamp_pct:.0f}%)**: reduce σ_β (try 0.03–0.04) "
                    "or increase particles (N ≥ 1000).")

    if sugg:
        for s in sugg:
            L.append(f"- {s}")
    else:
        L.append("No critical issues detected. Results appear epidemiologically consistent.")

    L += ["", "---", ""]

    # ── 8. Forecast summary ──────────────────────────────────────────────────
    if forecast_df is not None and len(forecast_df) > 0:
        L += ["## 8. Short-Term Forecast", "",
              f"Horizon: **{len(forecast_df)} days** "
              f"({forecast_df['date'].iloc[0].date()} to "
              f"{forecast_df['date'].iloc[-1].date()})",
              ""]

        # Per-channel forecast table
        for ch in obs_channels:
            mean_col = f"pred_{ch}_mean"
            q10_col  = f"pred_{ch}_q10"
            q90_col  = f"pred_{ch}_q90"
            if mean_col not in forecast_df.columns:
                continue
            L += [f"**{ch.capitalize()} forecast** (80% CI):", "",
                  "| Date | Predicted | Lower 10% | Upper 90% |",
                  "|---|---|---|---|"]
            for _, frow in forecast_df.iterrows():
                q10 = f"{frow[q10_col]:,.1f}" if q10_col in frow else "—"
                q90 = f"{frow[q90_col]:,.1f}" if q90_col in frow else "—"
                L.append(f"| {frow['date'].date()} "
                         f"| {frow[mean_col]:,.1f} | {q10} | {q90} |")
            L.append("")

        # R_eff and P_growing forecast
        if "R_eff_mean" in forecast_df.columns:
            L += ["**R_eff and epidemic growth probability forecast:**", "",
                  "| Date | R_eff (mean) | R_eff 80% CI | P(growing) |",
                  "|---|---|---|---|"]
            for _, frow in forecast_df.iterrows():
                r10 = f"{frow['R_eff_q10']:.2f}" if "R_eff_q10" in frow else "—"
                r90 = f"{frow['R_eff_q90']:.2f}" if "R_eff_q90" in frow else "—"
                pg  = f"{frow['P_growing']:.0%}" if "P_growing" in frow else "—"
                L.append(f"| {frow['date'].date()} "
                         f"| {frow['R_eff_mean']:.2f} | [{r10}, {r90}] | {pg} |")
            L.append("")

        L += ["---", ""]

    # ── 9. Scenario comparison ────────────────────────────────────────────────
    if scenario_df is not None and forecast_df is not None and len(scenario_df) > 0:
        L += [f"## 9. Intervention Scenario (−{beta_reduction_pct}% transmission)", "",
              f"Counterfactual: β reduced by **{beta_reduction_pct}%** at forecast start "
              f"(e.g. non-pharmaceutical intervention, behaviour change).", ""]

        for ch in obs_channels:
            mean_col = f"pred_{ch}_mean"
            if mean_col not in scenario_df.columns or mean_col not in forecast_df.columns:
                continue
            L += [f"**{ch.capitalize()} — baseline vs scenario:**", "",
                  "| Date | Baseline | Scenario | Reduction |",
                  "|---|---|---|---|"]
            for i, (_, srow) in enumerate(scenario_df.iterrows()):
                if i >= len(forecast_df):
                    break
                base = forecast_df.iloc[i][mean_col]
                scen = srow[mean_col]
                reduc = (base - scen) / max(base, 1e-9) * 100
                L.append(f"| {srow['date'].date()} "
                         f"| {base:,.1f} | {scen:,.1f} | {reduc:.0f}% |")
            L.append("")

        if "R_eff_mean" in scenario_df.columns and "R_eff_mean" in forecast_df.columns:
            L += ["**R_eff — baseline vs scenario:**", "",
                  "| Date | R_eff baseline | R_eff scenario | P(growing) baseline | P(growing) scenario |",
                  "|---|---|---|---|---|"]
            for i, (_, srow) in enumerate(scenario_df.iterrows()):
                if i >= len(forecast_df):
                    break
                brow = forecast_df.iloc[i]
                pg_b = f"{brow['P_growing']:.0%}" if "P_growing" in brow else "—"
                pg_s = f"{srow['P_growing']:.0%}" if "P_growing" in srow else "—"
                L.append(f"| {srow['date'].date()} "
                         f"| {brow['R_eff_mean']:.2f} | {srow['R_eff_mean']:.2f} "
                         f"| {pg_b} | {pg_s} |")
            L.append("")

        L += ["---", ""]

    L += ["",
          "*Report generated by **EpiSurveil Control Platform** — Bootstrap Particle Filter.*  ",
          f"*Model: {model_key} | Disease: {disease_label} | Generated: {today}*"]
    return "\n".join(L)


def _report_to_html(md_text: str, title: str = "Epidemic Surveillance Report") -> str:
    """Convert the markdown report to a standalone styled HTML document."""
    import re

    css = """
    body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;
         padding:0 20px;color:#222;line-height:1.6}
    h1{color:#1a237e;border-bottom:3px solid #1a237e;padding-bottom:8px}
    h2{color:#283593;border-bottom:1px solid #c5cae9;padding-bottom:4px;margin-top:2em}
    h3{color:#3949ab}
    table{border-collapse:collapse;width:100%;margin:1em 0}
    th{background:#e8eaf6;color:#1a237e;padding:8px 12px;text-align:left;
       border:1px solid #c5cae9}
    td{padding:7px 12px;border:1px solid #e0e0e0}
    tr:nth-child(even) td{background:#f5f5f5}
    blockquote{background:#fff8e1;border-left:4px solid #ffc107;
               margin:1em 0;padding:12px 16px;border-radius:0 4px 4px 0}
    blockquote h3{margin:0 0 4px;font-size:1.1em}
    blockquote p{margin:0}
    code{background:#f5f5f5;padding:2px 5px;border-radius:3px;font-size:0.9em}
    hr{border:none;border-top:1px solid #e0e0e0;margin:2em 0}
    em{color:#555}
    .alert-red{background:#ffebee;border-left:4px solid #e53935;
               padding:12px 16px;margin:1em 0;border-radius:0 4px 4px 0}
    .alert-green{background:#e8f5e9;border-left:4px solid #43a047;
                 padding:12px 16px;margin:1em 0}
    .alert-yellow{background:#fff8e1;border-left:4px solid #fdd835;
                  padding:12px 16px;margin:1em 0}
    """

    def _md_to_html(text):
        lines = text.split("\n")
        out, in_table, in_block = [], False, False
        for line in lines:
            # blockquote
            if line.startswith("> "):
                if not in_block:
                    out.append("<blockquote>"); in_block = True
                inner = line[2:]
                inner = re.sub(r"^### (.+)$", r"<h3>\1</h3>", inner)
                inner = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", inner)
                out.append(f"<p>{inner}</p>")
                continue
            else:
                if in_block:
                    out.append("</blockquote>"); in_block = False
            # headings
            m = re.match(r"^(#{1,3}) (.+)$", line)
            if m:
                lvl = len(m.group(1)); txt = m.group(2)
                txt = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", txt)
                out.append(f"<h{lvl}>{txt}</h{lvl}>"); continue
            # HR
            if line.strip() in ("---", "***"):
                out.append("<hr>"); continue
            # table header
            if re.match(r"^\|.+\|$", line) and "|---|" not in line:
                if not in_table:
                    out.append("<table><thead>"); in_table = True
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    out.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
                    out.append("</thead><tbody>")
                else:
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    row_html = ""
                    for c in cells:
                        c = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", c)
                        row_html += f"<td>{c}</td>"
                    out.append(f"<tr>{row_html}</tr>")
                continue
            if "|---|" in line:
                continue
            # end of table
            if in_table:
                out.append("</tbody></table>"); in_table = False
            # empty line
            if not line.strip():
                out.append(""); continue
            # list
            if line.startswith("- "):
                txt = line[2:]
                txt = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", txt)
                out.append(f"<li>{txt}</li>"); continue
            # normal paragraph
            txt = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            txt = re.sub(r"\*(.+?)\*", r"<em>\1</em>", txt)
            out.append(f"<p>{txt}</p>")
        if in_table:
            out.append("</tbody></table>")
        if in_block:
            out.append("</blockquote>")
        return "\n".join(out)

    body = _md_to_html(md_text)
    return (
        f"<!DOCTYPE html><html lang='en'><head>"
        f"<meta charset='UTF-8'>"
        f"<title>{title}</title>"
        f"<style>{css}</style></head>"
        f"<body>{body}</body></html>"
    )


# ===========================================================================
# TABS
# ===========================================================================
tab_filt, tab_comp, tab_dyn, tab_met, tab_prm, tab_mod, tab_ess, tab_ctrl, tab_cmp, tab_exp, tab_live, tab_geo, tab_log, tab_help = st.tabs(
    ["Filter output","Compartments","Parameters","Metrics","Fixed parameters","Model description","ESS","Optimal control","Model comparison","Model explorer","Live data","Country comparison","Alert history","Methodology"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Filter output
# Scale ranges (observed max): cases 230k | hosp 26k | icu 5.7k | deaths 964
# Ratios: 9x, 4.5x, 6x → all separate panels
# ---------------------------------------------------------------------------
REPORTING_END = "2023-04-07"   # Germany stopped daily COVID reporting

with tab_filt:
    st.subheader("Filtered state vs. observations — 80% posterior interval")
    st.caption(
        "Gray dots = raw observed. Solid line = filtered posterior mean (white background) "
        "or **model projection** (gray background). Shaded band = 80% CI.  \n"
        "**Vertical dashed line** = end of daily reporting (7 Apr 2023).  \n"
        "**Gray background** = no observations available: cases/ICU/deaths channels are NaN; "
        "the filter free-runs on ODE dynamics — the solid line is a projection, not a filtered estimate.  \n"
        "Hospital proxy (top-right) remains partially constrained by weekly SARI sentinel data (tempering α=0.10)."
    )

    ch_cfg = [
        ("cases","pred_cases_mean","pred_cases_q10","pred_cases_q90",
         "raw_cases","Reported cases  [projection after Apr 2023]", True),
        ("hosp","pred_hosp_mean","pred_hosp_q10","pred_hosp_q90",
         "obs_hosp","Hospital proxy  [SARI sentinel after Apr 2023]", False),
        ("icu","pred_icu_mean","pred_icu_q10","pred_icu_q90",
         "obs_icu","ICU occupancy (DIVI)  [projection after Apr 2023]", True),
        ("deaths","pred_deaths_mean","pred_deaths_q10","pred_deaths_q90",
         "obs_deaths","Daily deaths  [projection after Apr 2023]", True),
    ]

    fig1 = make_subplots(rows=2, cols=2,
                         subplot_titles=[c[5] for c in ch_cfg],
                         vertical_spacing=0.14, horizontal_spacing=0.08)

    subplot_idx = {(1,1): 1, (1,2): 2, (2,1): 3, (2,2): 4}
    x_end = df_eval["date"].max().strftime("%Y-%m-%d")

    for idx, ((r,cc), (ch,mn,q10,q90,obs_col,_,no_data_after_cutoff)) in enumerate(
            zip([(1,1),(1,2),(2,1),(2,2)], ch_cfg), start=1):
        color = C[ch]
        x = df_eval["date"]
        _ci_band(fig1, r, cc, x, df_eval[q10], df_eval[q90], color)
        _line(fig1, r, cc, x, df_eval[mn], color, "Filtered mean")
        obs_data = df_eval[obs_col] if obs_col in df_eval else df_eval["obs_"+ch]
        _scatter(fig1, r, cc, x, obs_data, "Observed")

        # Vertical line at reporting cutoff
        fig1.add_shape(
            type="line", x0=REPORTING_END, x1=REPORTING_END,
            y0=0, y1=1, yref="y domain",
            line=dict(dash="dash", color="#546E7A", width=1.0), opacity=0.7,
            row=r, col=cc,
        )

        # Gray "no data" shaded region after cutoff (only for channels with no obs)
        if no_data_after_cutoff and REPORTING_END < x_end:
            fig1.add_shape(
                type="rect",
                x0=REPORTING_END, x1=x_end,
                y0=0, y1=1, yref="y domain",
                fillcolor="rgba(180,180,180,0.12)",
                line=dict(width=0),
                row=r, col=cc,
            )

    # Single annotation for the cutoff line (top-left panel, x1 axis)
    fig1.add_annotation(
        x=REPORTING_END, y=0.97,
        xref="x", yref="y domain",
        text="Reporting ended",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=8, color="#546E7A"),
    )

    fig1.update_layout(height=620, showlegend=False, hovermode="x unified",
                       margin=dict(t=50,b=30,l=40,r=20), template="plotly_white")
    fig1.update_xaxes(showgrid=False,
                      range=[df_eval["date"].min(), df_eval["date"].max()])
    fig1.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
    st.plotly_chart(fig1, use_container_width=True)

    st.info(
        "**Note on the 80% band in the gray region (post Apr 2023).**  \n"
        "Inside the observed period, the band is a filtered posterior interval: "
        "at each step the particle weights are updated by the likelihood of the observations, "
        "which continuously pulls the ensemble toward plausible states and keeps the band narrow.  \n"
        "After daily reporting ended, cases, ICU, and deaths contribute **no weight updates**. "
        "The band is therefore driven purely by the spread of 2,000 particles "
        "undergoing unconstrained log-random walks on all five parameters. "
        "Because β_t diffuses at σ=0.04 day⁻¹, after T steps the standard deviation of "
        "log β grows as 0.04√T — roughly doubling the plausible range every ~600 days. "
        "This is **particle ensemble spread**, not a calibrated forecast uncertainty: "
        "it widens indefinitely and should not be interpreted as a prediction interval. "
        "Only the hospital proxy panel retains a weak observational anchor "
        "via weekly SARI sentinel data (tempering α=0.10)."
    )

# ---------------------------------------------------------------------------
# TAB 2 — Compartments
# Grouping by scale:
#   S,V  → millions  → same panel
#   E,A,I → 5k-300k  → same panel
#   H,C   → 0-20k    → same panel (ratio ~4x)
#   R     → cumulative recovered (millions) → own panel
#   D     → cumulative deaths (200k)       → own panel
# ---------------------------------------------------------------------------
with tab_comp:
    st.subheader("Filtered compartment trajectories — posterior mean + 80% CI")
    st.caption(
        "Solid line = posterior mean. Shaded band = 80% CI (10th–90th percentile across particles). "
        "**Note on D (cumulative deaths):** "
        "The model accumulates deaths from both H→D (constrained by the daily-deaths "
        "observation) **and** C→D (fixed rate δ_C=0.01 day⁻¹, unconstrained by any "
        "observation channel). This structural double-source inflates D above the "
        "official RKI tally. "
        "Dashed line = cumulative observed deaths (RKI)."
    )

    comp_cols = ["S_mean","V_mean","E_mean","A_mean","I_mean",
                 "H_mean","C_mean","R_mean","D_mean"]
    has_ci = all(f"{c}_q10" in df.columns for c in ["S","V","E","A","I","H","C","R","D"])

    if not all(c in df.columns for c in comp_cols):
        st.warning("Compartment columns not found — re-run `run_sveaihcrd_validation.py`.")
    else:
        cfig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "S (Susceptible) and V (Vaccinated)  [persons]",
                "E, A, I — Exposed / Asymptomatic / Infectious  [persons]",
                "H (Hospitalised) and C (Critical/ICU)  [persons]",
                "R — Recovered (cumulative)  [persons]",
                "D — Dead (cumulative)  [persons]",
                "",
            ],
            vertical_spacing=0.12, horizontal_spacing=0.08,
        )

        def _comp_traces(fig, r, cc, entries):
            """Draw CI band + mean line for each (name, colour) in entries."""
            for name, color in entries:
                if has_ci:
                    _ci_band(fig, r, cc,
                             df["date"], df[f"{name}_q10"], df[f"{name}_q90"], color)
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df[f"{name}_mean"], mode="lines",
                    name=name, line=dict(color=color, width=LINE_W)),
                    row=r, col=cc)

        # A — S and V
        _comp_traces(cfig, 1, 1, [("S", C["S"]), ("V", C["V"])])

        # B — E, A, I
        _comp_traces(cfig, 1, 2, [("E", C["E"]), ("A", C["A"]), ("I", C["I"])])

        # C — H and C
        _comp_traces(cfig, 2, 1, [("H", C["H"]), ("C", C["C"])])

        # D — R alone
        _comp_traces(cfig, 2, 2, [("R", C["R"])])

        # E — D alone with true reference line
        _comp_traces(cfig, 3, 1, [("D", C["D"])])
        # True cumulative deaths (observed sum as proxy)
        cfig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df_full[df_full["date"] <= df["date"].max()]["obs_deaths"].fillna(0).cumsum(),
                mode="lines", name="D (obs. cumsum)",
                line=dict(color="#B0BEC5", width=1.2, dash="dot"),
            ),
            row=3, col=1,
        )
        cfig.add_annotation(
            x=0.01, y=0.05, xref="paper", yref="paper",
            text="Dashed = cumulative obs. deaths (data); solid = model D",
            showarrow=False, font=dict(size=9, color="#607D8B"), xanchor="left",
        )

        # Vertical "reporting ended" line on each compartment subplot
        for r_c, c_c in [(1,1),(1,2),(2,1),(2,2),(3,1)]:
            cfig.add_shape(
                type="line", x0=REPORTING_END, x1=REPORTING_END,
                y0=0, y1=1, yref="y domain",
                line=dict(dash="dash", color="#546E7A", width=0.8), opacity=0.5,
                row=r_c, col=c_c,
            )

        cfig.update_layout(height=820, hovermode="x unified", template="plotly_white",
                           margin=dict(t=50,b=30,l=40,r=20))
        cfig.update_xaxes(showgrid=False,
                          range=[df["date"].min(), df["date"].max()])
        cfig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
        st.plotly_chart(cfig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — Dynamic parameters (5 parameters: 3×2 layout, cell [3,2] empty)
# ---------------------------------------------------------------------------
with tab_dyn:
    st.subheader("Time-varying parameter estimates — posterior means")
    st.caption(
        "Five log-random-walk parameters tracked by the BPF. "
        "Q_C(t) is the case-detection rate — new in this version. "
        "Dashed vertical line = burn-in end."
    )

    par_cfg = [
        ("beta_mean",   "beta_t — transmission rate",             C["beta"],       (1,1)),
        ("tau_i_mean",  "tau_I,t — hospitalisation rate I→H",     C["tau_i"],      (1,2)),
        ("delta_h_mean","delta_H,t — in-hospital death rate H→D", C["delta_h"],    (2,1)),
        ("rho_c_mean",  "rho_C,t — ICU fraction of hospitalised", C["rho_c"],      (2,2)),
        ("q_c_mean",    "Q_C,t — case detection / reporting rate","#6A1B9A",       (3,1)),
        ("nu_eff",      "nu_eff(t) — effective vaccination rate", C["nu_eff"],     (3,2)),
    ]

    pfig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[p[1] for p in par_cfg],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    for col_name, title, color, (r, cc) in par_cfg:
        if col_name not in df.columns:
            pfig.add_annotation(
                text="Re-run filter to populate this panel",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color="gray"),
            )
            continue
        pfig.add_trace(go.Scatter(x=df["date"], y=df[col_name], mode="lines",
                                  name=title, line=dict(color=color, width=LINE_W)),
                       row=r, col=cc)
        pfig.add_shape(type="line", x0=burn_date, x1=burn_date,
                       y0=0, y1=1, yref="y domain",
                       line=dict(dash="dash", color="black", width=0.8), opacity=0.5,
                       row=r, col=cc)
        # Reporting-end line (gray dashed)
        pfig.add_shape(type="line", x0=REPORTING_END, x1=REPORTING_END,
                       y0=0, y1=1, yref="y domain",
                       line=dict(dash="dash", color="#546E7A", width=0.8), opacity=0.5,
                       row=r, col=cc)

    has_q_c = "q_c_mean" in df.columns

    if has_q_c:
        # add_shape targets a specific subplot; add_hline bleeds into all panels
        q_c_ymax = float(df["q_c_mean"].max()) if "q_c_mean" in df.columns else 1.0
        pfig.add_shape(
            type="line",
            x0=df["date"].iloc[0], x1=df["date"].iloc[-1],
            y0=1.0, y1=1.0,
            line=dict(dash="dot", color="gray", width=0.8),
            row=3, col=1,
        )
        pfig.add_annotation(
            x=df["date"].iloc[-1], y=1.0, xref="x5", yref="y5",
            text="Q_C=1 (perfect detection)",
            showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(size=9, color="gray"),
        )

    pfig.update_layout(height=750, hovermode="x unified", template="plotly_white",
                       showlegend=False, margin=dict(t=50, b=30, l=40, r=20))
    pfig.update_xaxes(showgrid=False,
                      range=[df["date"].min(), df["date"].max()])
    pfig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
    st.plotly_chart(pfig, use_container_width=True)

    missing = [c for c, *_ in par_cfg if c not in df.columns]
    if missing:
        st.info(f"Columns not yet in output CSV: {missing}. Re-run `python scripts/run_sveaihcrd_validation.py`.")

# ---------------------------------------------------------------------------
# TAB 4 — Metrics
# ---------------------------------------------------------------------------
with tab_met:
    st.subheader("In-sample validation metrics — post 30-day burn-in")

    rows_tbl = []
    for ch, label in [("cases","Reported cases"),("icu","ICU occupancy"),
                      ("deaths","Daily deaths"),("hosp","Hospital proxy")]:
        m = metrics[ch]
        rows_tbl.append({"Channel":label, "Mean obs.":_fmt(m["mean_obs"]),
                         "RMSE":_fmt(m["rmse"]), "MAE":_fmt(m["mae"]),
                         "80% coverage":f"{m['coverage_80pct']:.1%}", "n days":m["n"]})

    st.dataframe(pd.DataFrame(rows_tbl), hide_index=True, use_container_width=True)
    st.markdown(
        "**Under-coverage for deaths/hosp** is a known artefact of aggressive resampling: "
        "particles homogenise after each resample event, narrowing the 80% CI beyond "
        "the true posterior spread.  A regularised or jittered resampling step would widen "
        "the intervals.  Cases coverage (74.8%) is close to the 80% nominal level."
    )

    ca, cb = st.columns(2)
    with ca:
        st.subheader("Channel configuration")
        st.dataframe(pd.DataFrame([
            {"Channel":"Cases",  "phi":80, "alpha":1.00},
            {"Channel":"ICU",    "phi":12, "alpha":0.60},
            {"Channel":"Deaths", "phi":10, "alpha":0.40},
            {"Channel":"Hosp",   "phi":15, "alpha":0.10},
        ]), hide_index=True, use_container_width=True)
    with cb:
        st.subheader("ESS summary")
        n_res = int((df_full["ess"] < 900).sum())
        st.dataframe(pd.DataFrame([
            {"Stat":"Mean ESS", "Value":f"{mean_ess:.0f}"},
            {"Stat":"Min ESS",  "Value":f"{df_full['ess'].min():.0f}"},
            {"Stat":"Resample events","Value":f"{n_res} / {len(df_full)}"},
            {"Stat":"Resample rate","Value":f"{100*n_res/len(df_full):.1f}%"},
        ]), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 5 — Fixed parameters table
# ---------------------------------------------------------------------------
with tab_prm:
    st.subheader("All fixed model parameters")
    st.caption(
        "Dynamic parameters (beta, tau_i, delta_h, rho_c) are tracked as "
        "log-random walks; their prior means and process-noise scales are listed below. "
        "All other parameters are fixed for every particle throughout the filter run."
    )

    st.markdown("#### Structural / epidemiological constants")
    st.dataframe(pd.DataFrame([
        {"Symbol":"N",           "Value":"83,200,000",   "Unit":"persons",   "Description":"German population (Destatis 2020)"},
        {"Symbol":"sigma",       "Value":"1/5.5 = 0.182","Unit":"day⁻¹",    "Description":"E→AI rate (incubation period 5.5 d)"},
        {"Symbol":"kappa",       "Value":"0.62",          "Unit":"—",         "Description":"Symptomatic fraction"},
        {"Symbol":"eta_A",       "Value":"0.50",          "Unit":"—",         "Description":"Relative infectivity of asymptomatic"},
        {"Symbol":"gamma_A",     "Value":"1/7 = 0.143",  "Unit":"day⁻¹",    "Description":"A→R recovery rate"},
        {"Symbol":"gamma_I",     "Value":"1/8 = 0.125",  "Unit":"day⁻¹",    "Description":"I→R recovery rate"},
        {"Symbol":"gamma_H",     "Value":"1/12 = 0.083", "Unit":"day⁻¹",    "Description":"H→R recovery rate"},
        {"Symbol":"gamma_C",     "Value":"1/10 = 0.100", "Unit":"day⁻¹",    "Description":"C→R recovery rate"},
        {"Symbol":"tau_H",       "Value":"0.030",         "Unit":"day⁻¹",    "Description":"H→C (ICU admission) rate (fixed)"},
        {"Symbol":"delta_C",     "Value":"0.010",         "Unit":"day⁻¹",    "Description":"C→D death rate  [⚠ NOT constrained by data — inflates D]"},
        {"Symbol":"omega_V",     "Value":"1/180",              "Unit":"day⁻¹",    "Description":"V→S waning rate (6-month immunity)"},
        {"Symbol":"omega_R",     "Value":"1/365",              "Unit":"day⁻¹",    "Description":"R→S waning rate (12-month immunity)"},
        {"Symbol":"vacc. eff.",  "Value":"0.75",               "Unit":"—",         "Description":"Vaccine efficacy against infection"},
        {"Symbol":"nu_eff(t)",   "Value":"TIME-VARYING / DATA","Unit":"day⁻¹",    "Description":"nu_eff(t) = d_vacc_first(t)/N + omega_V * vaccinated_pct(t).  First term: new first doses.  Second term: booster compensation — replaces waned individuals so V tracks 'all who received >= 1 dose'.  Applied uniformly to S, E, A, R.  Hard-zeroed before 2020-12-27."},
    ]), hide_index=True, use_container_width=True)

    st.markdown("#### Observation model")
    st.dataframe(pd.DataFrame([
        {"Symbol":"Q_C,t",        "Value":"TIME-VARYING","Unit":"—","Description":"Case detection rate — 5th dynamic parameter (prior mean 0.40 in Mar 2020; bounds [0.20, 0.99])"},
        {"Symbol":"kappa",        "Value":"0.620","Unit":"—",         "Description":"Symptomatic fraction (same as ODE model)"},
        {"Symbol":"LOS",          "Value":"8.0",  "Unit":"days",      "Description":"Mean hospital length of stay (hosp proxy)"},
        {"Symbol":"delay mean",   "Value":"5.0",  "Unit":"days",      "Description":"Gamma kernel mean for case reporting delay"},
        {"Symbol":"delay SD",     "Value":"2.0",  "Unit":"days",      "Description":"Gamma kernel SD for case reporting delay"},
        {"Symbol":"phi_cases",    "Value":"80",   "Unit":"—",         "Description":"NegBin dispersion — cases"},
        {"Symbol":"phi_ICU",      "Value":"12",   "Unit":"—",         "Description":"NegBin dispersion — ICU"},
        {"Symbol":"phi_deaths",   "Value":"10",   "Unit":"—",         "Description":"NegBin dispersion — deaths"},
        {"Symbol":"phi_hosp",     "Value":"15",   "Unit":"—",         "Description":"NegBin dispersion — hospital proxy"},
        {"Symbol":"alpha_cases",  "Value":"1.00", "Unit":"—",         "Description":"Tempering weight — cases"},
        {"Symbol":"alpha_ICU",    "Value":"0.60", "Unit":"—",         "Description":"Tempering weight — ICU"},
        {"Symbol":"alpha_deaths", "Value":"0.40", "Unit":"—",         "Description":"Tempering weight — deaths"},
        {"Symbol":"alpha_hosp",   "Value":"0.10", "Unit":"—",         "Description":"Tempering weight — hospital proxy"},
    ]), hide_index=True, use_container_width=True)

    st.markdown("#### Dynamic parameter priors (initial particle draw, March 10 2020)")
    st.dataframe(pd.DataFrame([
        {"Parameter":"beta_t",    "Prior mean":"0.44",   "Prior SD (log)":"0.50",
         "RW noise sigma":"0.040","Log bounds":"[ln 0.05, ln 6.0]",
         "Note":"R0 = beta / (gamma_I + tau_I) = 0.44 / 0.127 = 3.5 (pre-lockdown)"},
        {"Parameter":"tau_I,t",   "Prior mean":"0.0021", "Prior SD (log)":"0.60",
         "RW noise sigma":"0.018","Log bounds":"[ln 1e-4, ln 0.08]","Note":""},
        {"Parameter":"delta_H,t", "Prior mean":"0.019",  "Prior SD (log)":"0.50",
         "RW noise sigma":"0.018","Log bounds":"[ln 1e-4, ln 0.10]","Note":""},
        {"Parameter":"rho_C,t",   "Prior mean":"0.23",   "Prior SD (log)":"0.50",
         "RW noise sigma":"0.018","Log bounds":"[ln 0.02, ln 0.80]","Note":""},
        {"Parameter":"Q_C,t",     "Prior mean":"0.20",   "Prior SD (log)":"0.80",
         "RW noise sigma":"0.025","Log bounds":"[ln 0.01, ln 0.99]",
         "Note":"Testing rate ~20% in early Mar 2020 (limited PCR capacity); "
                "filter learns true trajectory from case observations"},
    ]), hide_index=True, use_container_width=True)

    st.markdown("#### Initial compartment prior means (March 10, 2020)")
    st.dataframe(pd.DataFrame([
        {"Compartment":"S","Prior mean":"~83,129,790","Log scatter (sigma)":"0.02",
         "Derivation":"N minus all other compartments"},
        {"Compartment":"V","Prior mean":"0","Log scatter (sigma)":"—",
         "Derivation":"Vaccination started Dec 2020"},
        {"Compartment":"E","Prior mean":"36,000","Log scatter (sigma)":"0.50",
         "Derivation":"~2 x I (exposed > infectious)"},
        {"Compartment":"A","Prior mean":"11,000","Log scatter (sigma)":"0.50",
         "Derivation":"I x (1-kappa)/kappa"},
        {"Compartment":"I","Prior mean":"18,000","Log scatter (sigma)":"0.50",
         "Derivation":"~1600 detected/day / Q_C x LOS_I"},
        {"Compartment":"H","Prior mean":"800",   "Log scatter (sigma)":"0.60","Derivation":""},
        {"Compartment":"C","Prior mean":"150",   "Log scatter (sigma)":"0.60","Derivation":"ICU"},
        {"Compartment":"R","Prior mean":"8,000", "Log scatter (sigma)":"0.40","Derivation":""},
        {"Compartment":"D","Prior mean":"30",    "Log scatter (sigma)":"0.40","Derivation":"~30 deaths in Germany by 10 Mar 2020"},
    ]), hide_index=True, use_container_width=True)

    st.markdown("#### Filter settings")
    st.dataframe(pd.DataFrame([
        {"Setting":"N particles",         "Value":"2,000"},
        {"Setting":"ESS threshold",       "Value":"0.45 x N = 900"},
        {"Setting":"Resampling method",   "Value":"Systematic (O(N))"},
        {"Setting":"Random seed",         "Value":"42"},
        {"Setting":"Burn-in (excluded)",  "Value":"30 days"},
        {"Setting":"State noise sigma",   "Value":"0.010 (multiplicative, all compartments)"},
        {"Setting":"Quantile draws",      "Value":"2,000 weighted samples per step"},
    ]), hide_index=True, use_container_width=True)

    st.info(
        "**Why D exceeds RKI count (~168k):** "
        "The model accumulates deaths from H→D (delta_h x H, constrained by obs_deaths) "
        "AND from C→D (delta_c=0.01 x C, **not** constrained by any channel). "
        "The unconstrained C→D path contributes roughly "
        "delta_c x mean(C) x T ≈ 0.01 x 500 x 1096 ≈ **5,500 extra deaths**. "
        "The remaining ~40k gap is driven by delta_h being slightly overestimated "
        "during the Omicron wave when deaths were low relative to hospitalisation. "
        "Fix: add C deaths to the observation model: obs_deaths ~ NegBin(delta_h x H + delta_c x C, phi_D)."
    )

    st.warning(
        "**Data coverage after April 2023:** Germany ceased mandatory daily COVID-19 reporting "
        "on 7 April 2023. The panel is extended to **October 2024** using weekly "
        "RKI COVID-SARI sentinel hospitalisation incidence (all-age, 00+). "
        "Cases, deaths, and ICU are NaN after April 2023 — the filter free-runs on those channels. "
        "Only the hospitalisation channel (tempering alpha=0.10) remains active post-April 2023. "
        "To extend to the present: download fresh OWID and DIVI data, then run "
        "`python scripts/extend_panel.py` and `python scripts/run_sveaihcrd_validation.py`."
    )

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# TAB 6 — Model description
# ---------------------------------------------------------------------------
with tab_mod:
    st.subheader("SVEAIHCRD model — equations, data & control")
    st.caption(
        "Full reference for the 9-compartment flagship model: ODEs, observation channels, "
        "Germany data coverage, optimal control and PF-MPC formulation. "
        "For the general BPF algorithm and all other models see the **Methodology** tab."
    )

    st.markdown("### Compartment model (ODE system)")
    st.markdown(
        "The SVEAIHCRD model tracks nine population compartments across continuous time:"
    )
    st.dataframe(
        pd.DataFrame([
            ("S", "Susceptible — no immunity"),
            ("V", "Vaccinated — reduced susceptibility"),
            ("E", "Exposed — incubating, not yet infectious"),
            ("A", "Asymptomatic — infectious, no symptoms"),
            ("I", "Infectious — symptomatic"),
            ("H", "Hospitalised"),
            ("C", "Critical — ICU"),
            ("R", "Recovered — temporary immunity"),
            ("D", "Dead — cumulative"),
        ], columns=["Compartment", "Meaning"]),
        hide_index=True, use_container_width=True,
    )

    st.markdown("#### Ordinary differential equations")
    for eq in [
        r"$$\dot{S} = -\lambda S - \nu(t) S + \omega_V V + \omega_R R$$",
        r"$$\dot{V} = \nu(t)(S+E+A+R) - (1-\varepsilon)\lambda V - \omega_V V$$",
        r"$$\dot{E} = \lambda S + (1-\varepsilon)\lambda V - \sigma E - \nu(t) E$$",
        r"$$\dot{A} = (1-\kappa)\sigma E - \gamma_A A - \nu(t) A$$",
        r"$$\dot{I} = \kappa\,\sigma E - (\gamma_I + \tau_I)\,I$$",
        r"$$\dot{H} = \tau_I I - (\gamma_H + \tau_H + \delta_H) H$$",
        r"$$\dot{C} = \tau_H H - (\gamma_C + \delta_C) C$$",
        r"$$\dot{R} = \gamma_A A + \gamma_I I + \gamma_H H + \gamma_C C - \omega_R R - \nu(t) R$$",
        r"$$\dot{D} = \delta_H H + \delta_C C$$",
    ]:
        st.markdown(eq)

    st.markdown(
        r"The **force of infection** is $\lambda(t) = \beta_t\,(I + \eta_A A)\,/\,N$. "
        r"Five parameters evolve as log-random walks: $\beta_t$, $\tau_{I,t}$, $\delta_{H,t}$, $\rho_{C,t}$, $Q_{C,t}$. "
        "All others are fixed (see Fixed parameters tab)."
    )

    st.info(
        "For the BPF algorithm, NegBin likelihood formula, and ESS diagnostics see the **Methodology** tab.",
        icon="ℹ️",
    )

    st.markdown("---")
    st.markdown("### Observation equations")
    st.markdown(
        r"Each observed channel $y_t$ is linked to a modelled mean $\mu_t$ "
        r"via a Negative Binomial likelihood with dispersion $\phi$ and tempering weight $\alpha$:"
    )
    st.markdown(
        "| Channel | Modelled mean $\\mu_t$ | $\\phi$ | $\\alpha$ | Source |\n"
        "|---------|----------------------|--------|-----------|--------|\n"
        "| Reported cases | $Q_{C,t}\\,\\kappa\\,\\sigma\\,E_t$ (delay-convolved) | 80 | 1.00 | Daily (RKI) |\n"
        "| ICU occupancy | $C_t$ | 12 | 0.60 | Daily (DIVI) |\n"
        "| Daily deaths | $\\delta_H H_t + \\delta_C C_t$ | 10 | 0.40 | Daily (RKI) |\n"
        "| Hospital proxy | $\\tau_I\\,I_t \\times \\mathrm{LOS}$ | 15 | 0.10 | Daily / SARI weekly |\n"
    )
    st.markdown(
        "**Case reporting delay**: a Gamma convolution kernel (mean 5 d, SD 2 d) is applied to simulated "
        "incidence before comparing to reported cases.  \n"
        r"**$Q_{C,t}$ — 5th dynamic parameter**: log-random-walk bounded $[0.20,\,0.99]$, prior mean 0.40 in "
        "March 2020. The filter learns the true detection trajectory from the case signal."
    )

    st.markdown("---")
    st.markdown("### Data coverage and post-reporting extension")
    st.markdown(
        "Germany stopped mandatory daily COVID-19 reporting on **7 April 2023**. "
        "From that date the panel is extended using weekly **RKI COVID-SARI** sentinel hospitalization "
        "incidence (all ages, 00+), calibrated to the existing proxy scale (factor ≈ 3.85). "
        "Cases, deaths, and ICU are treated as NaN and skipped by the filter — only the hospital "
        "proxy channel (α = 0.10) remains active. "
        "The panel currently extends to **October 2024** (last local SARI update).  \n"
        "In the post-April 2023 region the 80% CI band reflects **unconstrained particle spread** "
        "(random-walk diffusion), not a calibrated forecast interval."
    )

    st.markdown("---")
    st.markdown("### Optimal Control Problem")

    st.markdown(
        "The **Optimal Control tab** computes the best joint intervention strategy over a user-defined "
        "horizon $T$ (typically 90 days), starting from the BPF posterior state on the chosen date."
    )

    st.markdown("#### Two control variables")
    st.markdown(
        "| Control | Range | What it represents | Effect on the model |\n"
        "|---------|-------|--------------------|---------------------|\n"
        r"| $u_t$ — NPI intensity | [0, 1] | Restriction level: 0 = no measures, 1 = full lockdown | Reduces contact rate: $\beta_t = \hat\beta_0\,e^{-\alpha\,u_t}$, with $\alpha=1.5$ giving ~78% reduction at $u=1$ |"
        "\n"
        r"| $v_t$ — Testing intensity | [0, 1] | Testing programme scale: 0 = routine surveillance, 1 = mass testing | Raises detection: $Q_{C,t} = Q_{C,0} + (0.99 - Q_{C,0})\,v_t$; detected symptomatic cases self-isolate |"
    )

    st.markdown("#### Modified force of infection")
    st.markdown(
        r"Testing reduces the infectious contribution of symptomatic cases I "
        r"(detected individuals self-isolate), while asymptomatic A always transmit freely:"
    )
    st.markdown(
        r"$$\lambda_t = \beta_t \,\frac{(1 - Q_{C,t})\,I_t + \eta_A\,A_t}{N}$$"
    )
    st.markdown(
        r"NPI and testing act on **different mechanisms**: NPI cuts the contact rate $\beta_t$; "
        r"testing removes detected infectors from the pool. They complement rather than duplicate each other."
    )

    st.markdown("#### Objective function")
    st.markdown(
        "The optimiser minimises a weighted sum of health burden and policy cost "
        "over the horizon, expressed in **death-equivalent units**:"
    )
    st.markdown(
        r"$$J = \underbrace{w_H \sum_{t=1}^T H_t + w_C \sum_{t=1}^T C_t + w_D\,\Delta D_T}_{\text{health burden}}"
        r"\;+\; \underbrace{w_u \sum_{t=1}^T u_t^2 + w_v \sum_{t=1}^T v_t^2}_{\text{policy cost}}"
        r"\;+\; \underbrace{w_{\Delta u}\sum(\Delta u_t)^2 + w_{\Delta v}\sum(\Delta v_t)^2}_{\text{smoothness}}"
        r"\;+\; \underbrace{\text{cap. penalty}}_{\text{soft constraint}}$$"
    )
    st.markdown(
        "| Term | Meaning | Default weight |\n"
        "|------|---------|----------------|\n"
        r"| $w_H \sum H_t$ | Bed-days in hospital | $w_H = 10^{-3}$ (1,000 bed-days ≈ 1 death) |"
        "\n"
        r"| $w_C \sum C_t$ | ICU-days | $w_C = 5\times10^{-3}$ (200 ICU-days ≈ 1 death) |"
        "\n"
        r"| $w_D\,\Delta D$ | Cumulative deaths | $w_D = 1$ (reference unit) |"
        "\n"
        r"| $w_u \sum u_t^2$ | NPI economic cost (GDP loss) | $w_u = 50$ (1 day full lockdown ≈ 50 death-equiv.) |"
        "\n"
        r"| $w_v \sum v_t^2$ | Testing programme cost | $w_v = 2$ (mass testing ≈ 25× cheaper than lockdown) |"
        "\n"
        r"| Smoothness | Penalises abrupt policy changes | $w_{\Delta u}=2,\;w_{\Delta v}=0.5$ |"
        "\n"
        "| Capacity penalty | Soft constraint: $10^5 \cdot \sum \max(0, H_t - H_{\max})^2$ | Discourages exceeding hospital capacity |"
    )

    st.markdown("#### How it is solved")
    st.markdown(
        "**Direct transcription** converts the continuous control problem into a finite-dimensional "
        "non-linear optimisation over $2T$ variables — one $u_t$ and one $v_t$ per day:"
    )
    st.markdown(
        "1. **Discretise** the ODE with a sub-daily Euler scheme (4 steps per day, dt = 0.25) "
        "for numerical stability at the control timescale.\n"
        "2. **Optimise** using **L-BFGS-B** (a gradient-based algorithm that respects box "
        "constraints $u_t, v_t \\in [0,1]$ exactly). Gradients are computed by finite differences "
        "inside `scipy.optimize.minimize`.\n"
        "3. **Initialise** from a moderate guess ($u_t = 0.25$, $v_t = 0.30$) to guide the solver "
        "toward the interior of the feasible region.\n"
        "4. **Convergence** is typically achieved in 5–15 seconds for $T=90$ days "
        "(180 variables, up to 400 iterations)."
    )
    st.markdown(
        "**Pareto frontier**: by sweeping $w_u$ over a log-grid from 0.1 to 1000 (8 points) "
        "while holding $w_v = 2$ fixed, the model traces the trade-off between policy cost "
        "and health outcomes. The resulting curve shows how aggressively relaxing NPI forces "
        "the optimiser to compensate with more testing — revealing the substitutability "
        "between the two levers."
    )
    st.caption(
        "The control ODE is initialised from the **BPF filtered posterior mean** at the chosen date: "
        "compartment sizes ($H_0, C_0, \\ldots$) and time-varying parameters "
        "($\\hat\\beta_0, \\hat\\tau_I, \\hat\\delta_H, \\hat Q_C$) are all taken from the particle-weighted mean. "
        "This is the statistically principled starting point — the posterior mean minimises mean-squared error "
        "among point estimates. "
        "The remaining limitation is that the **posterior uncertainty** (the spread of the 2,000-particle cloud) "
        "is not propagated forward: the projection treats the posterior mean as exact. "
        "A more complete analysis would run the control under a sample of plausible parameter draws "
        "to obtain confidence bands on the optimal policy itself."
    )

    st.markdown("---")
    st.markdown("### Adaptive Control: PF-MPC (Receding Horizon)")

    st.markdown(
        "The single-horizon optimal control (above) computes a **fixed plan** from one starting date. "
        "**PF-MPC** replaces this with a real-time feedback loop: the particle filter continuously "
        "updates the state estimate as new data arrives, and the controller re-solves the optimisation "
        "every day using the latest posterior. Only today's action is applied; tomorrow it re-solves again."
    )

    st.markdown("#### The feedback loop")
    st.markdown(
        r"$$\underbrace{\text{Observe } y_t}_{\text{surveillance}} "
        r"\;\longrightarrow\; "
        r"\underbrace{\text{BPF update: } p(x_t \mid y_{1:t})}_{\text{state estimation}} "
        r"\;\longrightarrow\; "
        r"\underbrace{\text{Solve } H\text{-day control from } \bar x_t,\,\bar\theta_t}_{\text{MPC}} "
        r"\;\longrightarrow\; "
        r"\underbrace{\text{Apply } u^*_t,\, v^*_t}_{\text{policy}} "
        r"\;\longrightarrow\; t+1$$"
    )
    st.markdown(
        "At each step the MPC receives the BPF **posterior mean** $(\\bar x_t, \\bar\\theta_t)$ "
        "as its state estimate and solves the same two-control objective over the next $H$ days. "
        "Only the **first action** $(u^*_t, v^*_t)$ is actually applied — the rest of the plan "
        "is discarded and recomputed the next day with fresh information. "
        "This is the **receding horizon** principle."
    )

    st.markdown("#### Why this matters")
    st.markdown(
        "| Scenario | Fixed plan (single-horizon) | PF-MPC (adaptive) |\n"
        "|----------|-----------------------------|--------------------|\n"
        "| New variant raises β mid-period | Plan stays unchanged — under-reacts | Detects β̂_t spike next BPF update → tightens u* immediately |\n"
        "| Wave fades faster than expected | Plan may over-restrict unnecessarily | BPF sees β̂_t falling → relaxes u*, v* early |\n"
        "| Vaccine rollout lowers susceptibility | Fixed IFR assumptions persist | ν_eff(t) and δ̂_H,t update in posterior → softer controls recommended |\n"
        "| Parameter estimate improves with data | Starting estimate frozen | Each day's solve uses the most recent posterior mean |"
    )

    st.markdown("#### Algorithm")
    st.markdown(
        "**Inputs:** BPF filter output CSV (posterior means at every historical day), "
        "simulation start $t_0$, simulation length $T_{\\mathrm{sim}}$, lookahead $H$.  \n"
        "**Loop:** for $t = t_0, \\ldots, t_0 + T_{\\mathrm{sim}} - 1$:"
    )
    st.markdown(
        "1. **Read** BPF posterior mean: compartments $(\\bar S_t, \\ldots, \\bar D_t)$ "
        "and parameters $(\\hat\\beta_t, \\hat\\tau_{I,t}, \\hat\\delta_{H,t}, \\hat Q_{C,t})$.\n"
        "2. **Solve** the $H$-day two-control problem from $(\\bar x_t, \\hat\\theta_t)$ "
        "using L-BFGS-B (warm-started from the previous shifted solution).\n"
        "3. **Apply** only the first action: $u_{\\mathrm{MPC}}(t) = u^*(0)$, "
        "$v_{\\mathrm{MPC}}(t) = v^*(0)$.\n"
        "4. **Shift** warm-start: $u_{\\mathrm{warm}} \\leftarrow [u^*(1), \\ldots, u^*(H-1), u^*(H-1)]$.\n"
        "5. **Advance** to $t+1$."
    )
    st.markdown(
        r"**Counterfactual trajectory:** uses $\hat\beta_t$ from the BPF at every day (not fixed at $t_0$), "
        r"so variant emergence, vaccination, and seasonal effects are preserved. "
        r"Only the NPI modulation changes: $\beta_{\mathrm{eff},t} = \hat\beta_t\,e^{-\alpha\,u_{\mathrm{MPC}}(t)}$."
    )

    st.markdown("#### Computational cost")
    st.markdown(
        "Each step solves a $2H$-variable optimisation. With warm-starting, convergence is typically "
        "reached in fewer than 15 L-BFGS-B iterations (~0.1–0.3 s per step). "
        "A 60-day simulation with $H=14$ runs in approximately **15–20 seconds** in the dashboard."
    )

    st.caption(
        "Note on 'stochastic' vs. 'deterministic' MPC: running a fully stochastic MPC — where the "
        "particle cloud is propagated *inside* the optimiser at every function evaluation — would cost "
        "O(K) times more per step (K = number of particles) and is computationally intractable for "
        "most real problems. The approach here is the standard practical solution: the BPF provides "
        "the nonlinear, non-Gaussian state estimate; the controller solves a deterministic problem "
        "conditioned on that estimate. The feedback loop through the BPF is what makes this genuinely "
        "adaptive, not merely open-loop."
    )

    st.markdown("---")
    st.markdown("### Known model limitations")
    st.dataframe(
        pd.DataFrame([
            ("~~Fixed Q_C~~",
             "~~Detection rate frozen at 0.72 — over-estimated early 2020~~",
             "FIXED: Q_C,t is now the 5th log-RW parameter, bounded [0.20, 0.99]"),
            ("Unconstrained C→D path",
             "delta_C=0.01 x C adds ~5,500 deaths/year with no observational constraint; "
             "model D exceeds RKI tally by ~46k over the study period",
             "Add delta_C x C to the deaths observation equation: "
             "obs_deaths ~ NegBin(delta_H x H + delta_C x C, phi_D)"),
            ("No age structure",
             "Cannot reproduce age-differential IFR, vaccine prioritisation, or wave-specific severity shifts",
             "Multi-age-group SVEAIHCRD with age-stratified beta and tau_I"),
            ("Fixed incubation period",
             "sigma = 1/5.5 day^-1 throughout; Omicron incubation ~3 d, Delta ~4 d",
             "Time-varying sigma_t as 6th log-RW parameter, or variant-flag switching"),
            ("Deaths 80% coverage = 27.7%",
             "Particle homogenisation after frequent resampling narrows the posterior CI "
             "below the nominal 80% level",
             "MCMC rejuvenation kernel (Liu-West) or jittered systematic resampling"),
            ("Hosp 80% coverage = 63%",
             "SARI sentinel (alpha=0.10) provides only a weak signal post-April 2023; "
             "the CI widens as particles drift under unconstrained random walks",
             "Download fresh DIVI ICU + OWID data; re-run extend_panel.py"),
            ("Panel ends October 2024",
             "XBB/JN.1/KP waves partially covered via SARI only; "
             "no cases, deaths, or ICU signal after April 2023",
             "Fresh OWID/DIVI download and re-run extend_panel.py to reach present"),
            ("Single-patch model",
             "Spatial heterogeneity (federal states, urban/rural) not captured",
             "Multi-patch extension already scaffolded in src/episurveil/models/multipatch.py"),
        ], columns=["Limitation", "Effect", "Potential fix"]),
        hide_index=True, use_container_width=True,
    )

# TAB 7 — ESS
# ---------------------------------------------------------------------------
with tab_ess:
    st.subheader("Effective Sample Size over time")
    st.caption(
        "ESS = 1/sum(w_i^2); N = 2,000. "
        "Resampling when ESS < 0.45 x N = 900 (red). "
        "Drops = observations distinguish particles (filter learning)."
    )

    ess_fig = go.Figure()
    ess_fig.add_trace(go.Scatter(
        x=df["date"], y=df["ess"], mode="lines", name="ESS",
        line=dict(color="#37474F", width=1.0),
        fill="tozeroy", fillcolor="rgba(55,71,79,0.08)",
    ))
    ess_fig.add_hline(y=900, line_dash="dash", line_color="#E53935", line_width=1.2)
    ess_fig.add_annotation(x=0.99, y=900, xref="paper", yref="y",
                           text="Resample threshold (0.45 x N = 900)",
                           showarrow=False, xanchor="right", yanchor="bottom",
                           font=dict(size=10, color="#E53935"))
    ess_fig.add_shape(type="line", x0=burn_date, x1=burn_date, y0=0, y1=2100,
                      line=dict(dash="dash", color="black", width=0.8), opacity=0.5)
    ess_fig.add_annotation(x=burn_date, y=2050, xref="x", yref="y",
                           text=f"Burn-in end (day {BURN_IN})",
                           showarrow=False, xanchor="left",
                           font=dict(size=9, color="black"))

    ess_fig.update_layout(
        height=320,
        yaxis=dict(range=[0, 2100], title="ESS"),
        margin=dict(t=20, b=30, l=50, r=20),
        hovermode="x unified", template="plotly_white", showlegend=False,
    )
    ess_fig.update_xaxes(showgrid=False,
                         range=[df["date"].min(), df["date"].max()])
    ess_fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
    st.plotly_chart(ess_fig, use_container_width=True)

    n_res = int((df["ess"] < 900).sum())
    st.info(
        f"Resampling events in selected period: **{n_res}** / {len(df)} days "
        f"({100*n_res/max(len(df),1):.1f}%).  "
        f"Mean ESS = **{df['ess'].mean():.0f}**.  "
        f"Min ESS = **{df['ess'].min():.0f}**."
    )

# ---------------------------------------------------------------------------
# TAB 8 — Optimal NPI control
# ---------------------------------------------------------------------------
with tab_ctrl:
    st.subheader("Optimal NPI + Testing Control Explorer")
    st.caption(
        "Jointly optimises two complementary controls starting from the BPF posterior state at the chosen date: "
        "u*(t) — NPI intensity (reduces contact rate) and v*(t) — testing intensity (detected positives self-isolate)."
    )

    st.markdown(
        r"**Controls:** "
        r"$\beta_t = \hat\beta_0 e^{-\alpha u_t}$ (NPI);  "
        r"$Q_{C,t} = Q_{C,0} + (0.99-Q_{C,0})\,v_t$ (testing).  "
        r"**Transmission:** $\lambda_t = \beta_t[(1-Q_{C,t})I_t + \eta_A A_t]/N$  "
        r"— detected symptomatic cases (I) self-isolate; asymptomatic A always transmit.  "
        r"**Objective:** $J = w_H\!\sum H_t + w_C\!\sum C_t + w_D D_T "
        r"+ w_u\!\sum u_t^2 + w_v\!\sum v_t^2 "
        r"+ w_{\Delta u}\!\sum(\Delta u_t)^2 + w_{\Delta v}\!\sum(\Delta v_t)^2$"
    )
    st.markdown("---")

    # ---- Left column: inputs ----
    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown("#### Setup")

        # Start date — select from filtered dates
        avail_dates = df_full["date"].dt.date.tolist()
        default_idx = min(200, len(avail_dates) - 1)
        ctrl_start = st.selectbox(
            "Start date (BPF initial condition)",
            options=avail_dates,
            index=default_idx,
            help="The BPF posterior mean at this date is used as x0 for the ODE simulation.",
        )
        ctrl_horizon = st.slider("Horizon T (days)", 30, 180, 90, step=15)

        st.markdown("#### Weights")
        st.caption("Expressed in 'death equivalents' — adjust to reflect policy priorities.")

        w_u  = st.slider("Economic cost of lockdown (w_u)  —  1 day full lockdown = w_u deaths-equiv.",
                         1.0, 200.0, 50.0, step=1.0)
        w_H  = st.slider("Hospital burden weight (w_H × 1000 per bed-day)",
                         0.1, 5.0, 1.0, step=0.1) * 1e-3
        w_C  = st.slider("ICU burden weight (w_C × 1000 per ICU-day)",
                         0.5, 20.0, 5.0, step=0.5) * 1e-3
        smooth = st.checkbox("Penalise rapid NPI changes (w_Δu = 2)", value=True)
        w_du = 2.0 if smooth else 0.0

        st.markdown("#### Capacity thresholds")
        h_max_ctrl = st.number_input("Hospital capacity H_max (beds)", 5000, 30000,
                                     int(H_MAX_DE), step=1000)
        c_max_ctrl = st.number_input("ICU capacity C_max (beds)", 1000, 10000,
                                     int(C_MAX_DE), step=500)

        st.markdown("#### NPI efficacy")
        alpha_npi = st.slider("α — full lockdown reduces β by exp(−α)",
                              0.5, 3.0, 1.5, step=0.1,
                              help="α=1.5 → 78% reduction.  α=1.0 → 63%.  α=2.0 → 86%.")

        st.markdown("#### Testing cost")
        w_v  = st.slider(
            "Testing cost (w_v)  —  1 day mass testing = w_v deaths-equiv.",
            0.1, 30.0, 2.0, step=0.1,
            help="Typical PCR mass-testing programme costs ~60x less than full lockdown GDP loss. "
                 "With w_u=50 and w_v=2 the optimiser naturally prefers testing over strict NPI.",
        )
        smooth_v = st.checkbox("Penalise rapid testing-level changes (w_Δv = 0.5)", value=True)
        w_dv = 0.5 if smooth_v else 0.0

        run_opt    = st.button("Run optimisation", type="primary")
        run_pareto = st.button("Generate Pareto frontier (slow ~90 s)")

    with col_out:
        # Look up BPF state at selected date
        row_ctrl = df_full[df_full["date"].dt.date == ctrl_start]
        if row_ctrl.empty:
            st.warning("Selected date not found in filter output.")
            st.stop()

        row_ctrl = row_ctrl.iloc[0]
        comp_names = ["S","V","E","A","I","H","C","R","D"]
        x0_ctrl = np.array([float(row_ctrl.get(f"{c}_mean", 0.0)) for c in comp_names])
        beta_base_ctrl  = float(row_ctrl.get("beta_mean",    0.44))
        tau_i_ctrl      = float(row_ctrl.get("tau_i_mean",   0.0021))
        delta_h_ctrl    = float(row_ctrl.get("delta_h_mean", 0.019))
        q_c_base_ctrl   = float(row_ctrl.get("rho_c_mean",   0.35))

        st.markdown(
            f"**Initial state on {ctrl_start}:**  "
            f"H={x0_ctrl[5]:,.0f}  |  C={x0_ctrl[6]:,.0f}  |  "
            f"β_base={beta_base_ctrl:.3f}  |  τ_I={tau_i_ctrl:.4f}  |  "
            f"δ_H={delta_h_ctrl:.4f}  |  Q_C_base={q_c_base_ctrl:.3f}"
        )
        st.caption(
            r"$Q_{C,t}(v_t) = Q_{C,\mathrm{base}} + (0.99 - Q_{C,\mathrm{base}})\,v_t$  — "
            r"detected symptomatic I self-isolate; A always transmit:  $\lambda_t = \beta_t\,[(1-Q_{C,t})\,I_t + \eta_A A_t]\,/\,N$"
        )

        weights_ctrl = ControlWeights(
            w_H=w_H, w_C=w_C, w_D=1.0,
            w_u=w_u, w_v=w_v,
            w_du=w_du, w_dv=w_dv,
        )

        # If initial H or C already exceeds the user-set threshold, silently raise
        # the threshold to the initial value — the optimizer cannot empty wards instantly.
        H0_ctrl = x0_ctrl[5]
        C0_ctrl = x0_ctrl[6]
        h_max_eff = max(float(h_max_ctrl), H0_ctrl * 1.05)
        c_max_eff = max(float(c_max_ctrl), C0_ctrl * 1.05)
        if h_max_eff > h_max_ctrl or c_max_eff > c_max_ctrl:
            st.caption(
                f"ℹ️ Initial occupancy (H={H0_ctrl:,.0f}, C={C0_ctrl:,.0f}) "
                f"already meets or exceeds the capacity thresholds you set. "
                f"Effective thresholds auto-raised to H_max={h_max_eff:,.0f}, "
                f"C_max={c_max_eff:,.0f} — hospitalised patients can't be discharged "
                f"instantly; recovery takes ~12 days."
            )

        # ---- Run optimisation ----
        if run_opt:
            with st.spinner(f"Optimising {ctrl_horizon}-day NPI + testing trajectory..."):
                result = run_optimal_control(
                    x0=x0_ctrl, beta_base=beta_base_ctrl,
                    tau_i=tau_i_ctrl, delta_h=delta_h_ctrl,
                    q_c_base=q_c_base_ctrl,
                    T=ctrl_horizon, weights=weights_ctrl,
                    h_max=h_max_eff, c_max=c_max_eff,
                    alpha_npi=alpha_npi,
                    start_date=str(ctrl_start),
                )
            st.session_state["ctrl_result"] = result

        # ---- Run Pareto sweep ----
        if run_pareto:
            with st.spinner("Running Pareto sweep (8 optimisations)..."):
                pareto_pts = run_pareto_sweep(
                    x0=x0_ctrl, beta_base=beta_base_ctrl,
                    tau_i=tau_i_ctrl, delta_h=delta_h_ctrl,
                    q_c_base=q_c_base_ctrl,
                    T=ctrl_horizon, h_max=h_max_eff, c_max=c_max_eff,
                    alpha_npi=alpha_npi, n_points=8,
                    start_date=str(ctrl_start),
                )
            st.session_state["pareto_pts"] = pareto_pts

        # ---- Display results ----
        if "ctrl_result" in st.session_state:
            r = st.session_state["ctrl_result"]

            # --- KPI row 1: health outcomes ---
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Deaths averted",    f"{r.deaths_averted:,}")
            k2.metric("Hosp-days averted", f"{r.hosp_days_averted:,}")
            k3.metric("ICU-days averted",  f"{r.icu_days_averted:,}")
            k4.metric("Converged",         "Yes" if r.converged else "No")

            # --- KPI row 2: cost breakdown ---
            k5, k6, k7, k8 = st.columns(4)
            k5.metric("Health cost (death-equiv.)",   f"{r.cost_health:,.0f}")
            k6.metric("NPI cost (death-equiv.)",      f"{r.cost_npi:,.0f}")
            k7.metric("Testing cost (death-equiv.)",  f"{r.cost_testing:,.0f}")
            k8.metric("Total J",                      f"{r.cost_total:,.0f}")

            # --- Policy decomposition ---
            st.markdown(
                f"**Policy mix:** "
                f"NPI accounts for **{r.npi_contribution_pct:.0f}%** of transmission reduction, "
                f"detection-isolation for **{r.testing_contribution_pct:.0f}%**.  "
                f"Mean u\\* = {float(np.mean(r.u_opt)):.2f}  |  "
                f"Mean v\\* = {float(np.mean(r.v_opt)):.2f}  |  "
                f"Mean Q_C\\* = {float(np.mean(r.q_c_traj)):.3f}"
            )

            if r.cost_capacity > 1.0:
                st.warning(
                    f"Capacity soft-constraint exceeded (penalty term = {r.cost_capacity:,.0f}).  "
                    "This typically means the epidemic was already large on the chosen start date "
                    "and hospitalisations continue rising for several days even under maximum controls "
                    "(existing patients take ~12 days to recover — controls affect new admissions, "
                    "not those already in hospital).  "
                    "Try choosing an earlier start date, or raise h_max / c_max to match "
                    "the actual peak you see in the trajectory plot below."
                )

            # ---- Figure 1: three-panel control trajectories ----
            dates_ctrl = pd.to_datetime(r.dates)
            dates_T    = dates_ctrl[:-1]   # length T

            fig_u = make_subplots(
                rows=3, cols=1,
                subplot_titles=[
                    "NPI intensity  u*(t)  [0 = no restrictions, 1 = full lockdown]",
                    "Testing intensity  v*(t)  and detection rate  Q_C(t)",
                    "Effective transmission rate  βₜ  =  β₀ · exp(−α u*)",
                ],
                vertical_spacing=0.12,
            )

            # Row 1 — NPI u*(t)
            fig_u.add_trace(go.Scatter(
                x=dates_T, y=r.u_opt,
                fill="tozeroy", fillcolor="rgba(30,136,229,0.15)",
                mode="lines", name="u*(t)  NPI",
                line=dict(color="#1E88E5", width=2)),
                row=1, col=1)

            # Row 2 — testing v*(t) and Q_C*(t)
            fig_u.add_trace(go.Scatter(
                x=dates_T, y=r.v_opt,
                fill="tozeroy", fillcolor="rgba(0,131,143,0.12)",
                mode="lines", name="v*(t)  testing intensity",
                line=dict(color="#00838F", width=2)),
                row=2, col=1)
            fig_u.add_trace(go.Scatter(
                x=dates_T, y=r.q_c_traj,
                mode="lines", name="Q_C*(t)  detection rate",
                line=dict(color="#00695C", width=1.8, dash="dot")),
                row=2, col=1)
            fig_u.add_shape(
                type="line",
                x0=dates_T[0], x1=dates_T[-1],
                y0=q_c_base_ctrl, y1=q_c_base_ctrl,
                line=dict(color="#E53935", dash="dash", width=1.0),
                row=2, col=1)

            # Row 3 — effective beta (NPI effect only)
            beta_t_opt  = beta_base_ctrl * np.exp(-alpha_npi * r.u_opt)
            beta_t_base = np.full(len(dates_T), beta_base_ctrl)
            fig_u.add_trace(go.Scatter(
                x=dates_T, y=beta_t_opt,
                mode="lines", name="β*(t)  under optimal NPI",
                line=dict(color="#1E88E5", width=2)),
                row=3, col=1)
            fig_u.add_trace(go.Scatter(
                x=dates_T, y=beta_t_base,
                mode="lines", name="β₀  no NPI",
                line=dict(color="#E53935", width=1.5, dash="dash")),
                row=3, col=1)

            fig_u.update_yaxes(range=[0, 1.05], title_text="u*(t)", row=1, col=1)
            fig_u.update_yaxes(range=[0, 1.05], title_text="rate", row=2, col=1)
            fig_u.update_yaxes(title_text="βₜ", row=3, col=1)
            fig_u.update_layout(
                height=580, template="plotly_white",
                hovermode="x unified", showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0),
                margin=dict(t=50, b=70, l=55, r=20),
            )
            fig_u.update_xaxes(showgrid=False, range=[dates_ctrl[0], dates_ctrl[-1]])
            st.plotly_chart(fig_u, use_container_width=True)

            st.caption(
                "Blue = NPI control u*(t).  Teal = testing intensity v*(t).  "
                "Dark green dotted = resulting detection rate Q_C*(t) — "
                "detected symptomatic I self-isolate — asymptomatic A transmit regardless of testing level.  "
                "Red dashed = Q_C baseline (no additional testing programme).  "
                "Row 3 shows how NPI alone reduces the contact-rate component of beta."
            )

            # ---- Figure 2: H, C, D under optimal vs. baseline ----
            fig_hcd = make_subplots(
                rows=1, cols=3,
                subplot_titles=["Hospitalised H (beds)", "ICU C (beds)", "Cumulative deaths D"],
                horizontal_spacing=0.10,
            )
            for cc, (comp, cap_val) in enumerate([
                ("H", h_max_ctrl),
                ("C", c_max_ctrl),
                ("D", None),
            ], start=1):
                y_opt  = r.traj_opt[comp]
                y_base = r.traj_baseline[comp]
                # Y-axis: tight to actual data range (not the capacity ceiling)
                y_max  = max(float(np.max(y_opt)), float(np.max(y_base))) * 1.08
                fig_hcd.add_trace(go.Scatter(
                    x=dates_ctrl, y=y_opt,
                    mode="lines", name=f"{comp} optimal (NPI+test)",
                    line=dict(color="#1E88E5", width=2)),
                    row=1, col=cc)
                fig_hcd.add_trace(go.Scatter(
                    x=dates_ctrl, y=y_base,
                    mode="lines", name=f"{comp} no controls",
                    line=dict(color="#E53935", width=2, dash="dash")),
                    row=1, col=cc)
                if cap_val is not None and cap_val < y_max * 1.5:
                    # Only draw capacity line if it's within a sensible range of the data
                    fig_hcd.add_shape(
                        type="line", x0=dates_ctrl[0], x1=dates_ctrl[-1],
                        y0=cap_val, y1=cap_val,
                        line=dict(color="#FF6F00", dash="dot", width=1.2),
                        row=1, col=cc)
                fig_hcd.update_yaxes(range=[0, y_max], row=1, col=cc)

            fig_hcd.update_layout(
                height=320, template="plotly_white",
                hovermode="x unified", showlegend=True,
                margin=dict(t=40, b=20, l=40, r=20),
            )
            fig_hcd.update_xaxes(showgrid=False, range=[dates_ctrl[0], dates_ctrl[-1]])
            fig_hcd.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
            st.plotly_chart(fig_hcd, use_container_width=True)

            # ---- Interpretation note ----
            st.markdown("---")
            st.markdown("#### How to read these results")
            npi_pct  = r.npi_contribution_pct
            test_pct = r.testing_contribution_pct
            mean_u   = float(np.mean(r.u_opt))
            mean_v   = float(np.mean(r.v_opt))
            mean_qc  = float(np.mean(r.q_c_traj))

            if mean_u < 0.10:
                npi_label = "almost no restrictions"
            elif mean_u < 0.30:
                npi_label = "light restrictions (e.g. gathering limits, masks)"
            elif mean_u < 0.55:
                npi_label = "moderate restrictions (e.g. partial closures)"
            else:
                npi_label = "strict lockdown"

            if mean_v < 0.15:
                test_label = "no extra testing beyond routine surveillance"
            elif mean_v < 0.40:
                test_label = "moderate testing scale-up"
            else:
                test_label = "mass testing programme"

            if test_pct > npi_pct:
                why_text = (
                    "testing is far cheaper per unit of transmission reduction than lockdown, "
                    "so the model prefers detection-and-isolation over restrictions."
                )
            else:
                why_text = (
                    "NPI is the dominant lever because the baseline detection rate is already high "
                    "or testing cost outweighs its marginal benefit at the given w_v."
                )
            net_benefit = r.cost_health - r.cost_npi - r.cost_testing
            st.info(
                f"**Recommended strategy:** {npi_label} (average NPI level {mean_u:.0%}) "
                f"combined with {test_label} (average testing intensity {mean_v:.0%}, "
                f"raising detection to {mean_qc:.0%} of symptomatic cases).  \n\n"
                f"**Why this mix?** With the given cost weights, {why_text}  \n\n"
                f"**Impact vs. doing nothing:** "
                f"{r.deaths_averted:,} fewer deaths, "
                f"{r.hosp_days_averted:,} fewer hospital-bed-days, "
                f"{r.icu_days_averted:,} fewer ICU-days over the {ctrl_horizon}-day horizon.  \n\n"
                f"**Cost trade-off:** Health burden saved = {r.cost_health:,.0f} death-equivalents. "
                f"NPI cost = {r.cost_npi:,.0f}. Testing cost = {r.cost_testing:,.0f}. "
                f"Net benefit = {net_benefit:,.0f} death-equivalents."
            )
            st.caption(
                "Note: this is a scenario projection from the ODE model, not a probabilistic forecast. "
                "Results are sensitive to the cost weights w_u and w_v — try adjusting them to explore "
                "different policy priorities."
            )

        # ---- Pareto frontier ----
        if "pareto_pts" in st.session_state:
            pts = st.session_state["pareto_pts"]
            st.markdown("---")
            st.markdown("#### Pareto frontier — health burden vs. NPI cost  (testing fixed at w_v=2)")
            st.caption(
                "Each point is a fully-optimised (u*, v*) trajectory for a different NPI cost weight w_u "
                "(testing cost w_v is held fixed at 2). "
                "Left: aggressive NPI, less testing.  Right: minimal NPI, optimiser shifts to mass testing.  "
                "Colour = mean optimal testing intensity v*.  The knee is the pragmatic policy."
            )

            npi_costs  = [p["npi_cost_raw"]    for p in pts]
            deaths_av  = [p["deaths_averted"]   for p in pts]
            mean_us    = [p["mean_u"]            for p in pts]
            mean_vs    = [p["mean_v"]            for p in pts]
            npi_pcts   = [p["npi_contribution_pct"]     for p in pts]
            test_pcts  = [p["testing_contribution_pct"] for p in pts]
            w_u_vals   = [p["w_u"]               for p in pts]

            pfig = go.Figure()
            pfig.add_trace(go.Scatter(
                x=npi_costs, y=deaths_av,
                mode="lines+markers+text",
                text=[f"w_u={v:.1f}<br>NPI {n:.0f}% / test {t:.0f}%"
                      for v, n, t in zip(w_u_vals, npi_pcts, test_pcts)],
                textposition="top center",
                marker=dict(
                    size=10,
                    color=mean_vs,
                    colorscale="Teal",
                    showscale=True,
                    colorbar=dict(title="Mean v*", thickness=12),
                ),
                line=dict(color="#1565C0", width=2),
                name="Pareto frontier",
            ))
            pfig.update_layout(
                height=400,
                xaxis_title="NPI cost  (sum u_t²  — larger = more restrictive lockdown)",
                yaxis_title="Deaths averted vs. no-control baseline",
                template="plotly_white",
                hovermode="closest",
                margin=dict(t=30, b=50, l=60, r=20),
            )
            pfig.update_xaxes(showgrid=True, gridcolor="#EEEEEE")
            pfig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
            st.plotly_chart(pfig, use_container_width=True)

            st.dataframe(pd.DataFrame([{
                "w_u":                f"{p['w_u']:.1f}",
                "Mean NPI u*":        f"{p['mean_u']:.2f}",
                "Mean testing v*":    f"{p['mean_v']:.2f}",
                "Mean Q_C*":          f"{p['mean_q_c']:.3f}",
                "NPI contrib. %":     f"{p['npi_contribution_pct']:.0f}%",
                "Testing contrib. %": f"{p['testing_contribution_pct']:.0f}%",
                "Deaths averted":     f"{p['deaths_averted']:,}",
                "Hosp-days averted":  f"{p['hosp_days_averted']:,}",
                "ICU-days averted":   f"{p['icu_days_averted']:,}",
            } for p in pts]), hide_index=True, use_container_width=True)

        if "ctrl_result" not in st.session_state and "pareto_pts" not in st.session_state:
            st.info("Configure the parameters on the left and click **Run optimisation** to start.")

    # -----------------------------------------------------------------------
    # PF-MPC section — Adaptive Receding-Horizon Control
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## Adaptive Control: PF-MPC (Receding Horizon)")
    st.markdown(
        "**Single-horizon control** (above) asks: *from one date, what is the best fixed-plan policy?*  \n"
        "**PF-MPC** asks: *what would an adaptive real-time policy have looked like over a full period?*  \n\n"
        "At each day $t$ the BPF posterior mean $(\\bar x_t, \\bar\\theta_t)$ is fed to the controller, "
        "a $H$-day optimal control problem is solved, and **only the first action** $(u^*_t, v^*_t)$ is applied. "
        "The process repeats with fresh BPF information the next day.  \n"
        "The counterfactual trajectory uses the BPF-filtered $\\hat\\beta_t$ as the background "
        "transmission rate — preserving variant emergence and vaccination dynamics — with the MPC "
        "policy modulating it via $\\beta_{\\mathrm{eff},t} = \\hat\\beta_t\\,e^{-\\alpha u_t}$."
    )

    mpc_col_in, mpc_col_out = st.columns([1, 2])

    with mpc_col_in:
        st.markdown("#### PF-MPC setup")

        avail_dates_mpc = df_full["date"].dt.date.tolist()
        mpc_start = st.selectbox(
            "Simulation start date",
            options=avail_dates_mpc,
            index=min(200, len(avail_dates_mpc) - 1),
            key="mpc_start",
            help="First day where BPF posterior mean is used as x₀.",
        )
        mpc_T = st.slider("Simulation length T_sim (days)", 30, 180, 60, step=15,
                          key="mpc_T",
                          help="Total days to simulate. Each day runs one control solve.")
        mpc_H = st.slider("Lookahead horizon H (days)", 7, 30, 14, step=7,
                          key="mpc_H",
                          help="Control optimises over the next H days but applies only today's action.")

        st.caption(
            f"Runtime estimate: {mpc_T} steps × ~1 s/step ≈ {mpc_T} s. "
            "Warm-starting from shifted previous solution accelerates each solve."
        )

        st.markdown("#### Cost weights (shared with single-horizon)")
        mpc_w_u = st.slider("NPI cost w_u", 1.0, 200.0, 50.0, step=1.0, key="mpc_wu")
        mpc_w_v = st.slider("Testing cost w_v", 0.1, 30.0, 2.0, step=0.1, key="mpc_wv")

        run_mpc = st.button("Run PF-MPC", type="primary", key="run_mpc_btn")

    with mpc_col_out:
        if run_mpc:
            mpc_weights = ControlWeights(w_u=mpc_w_u, w_v=mpc_w_v)

            progress_bar = st.progress(0, text="Initialising PF-MPC…")
            status_text  = st.empty()

            def _mpc_progress(step, total):
                pct = int(100 * step / total)
                progress_bar.progress(pct,
                    text=f"Step {step}/{total} — solving {mpc_H}-day horizon…")
                status_text.caption(
                    f"Step {step}/{total} complete.  Estimated remaining: "
                    f"{(total - step):.0f} s"
                )

            mpc_result = run_pf_mpc(
                df_filter=df_full,
                t0=str(mpc_start),
                T_sim=mpc_T,
                H=mpc_H,
                weights=mpc_weights,
                h_max=H_MAX_DE,
                c_max=C_MAX_DE,
                progress_cb=_mpc_progress,
            )
            progress_bar.progress(100, text="PF-MPC complete.")
            status_text.empty()
            st.session_state["mpc_result"] = mpc_result

        if "mpc_result" in st.session_state:
            r = st.session_state["mpc_result"]
            dates_mpc = pd.to_datetime(r.dates)
            dates_T   = dates_mpc[:-1]   # length T_sim

            # --- KPIs ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Deaths averted",     f"{r.deaths_averted:,}")
            m2.metric("Hosp-days averted",  f"{r.hosp_days_averted:,}")
            m3.metric("ICU-days averted",   f"{r.icu_days_averted:,}")
            m4.metric("Converged steps",
                      f"{int(r.converged.sum())}/{r.T_sim}")

            m5, m6, m7 = st.columns(3)
            m5.metric("Mean NPI u*",      f"{float(np.mean(r.u_mpc)):.2f}")
            m6.metric("Mean testing v*",  f"{float(np.mean(r.v_mpc)):.2f}")
            m7.metric("Lookahead H",      f"{r.H} days")

            # --- Figure 1: adaptive controls + background β ---
            fig_mpc = make_subplots(
                rows=3, cols=1,
                subplot_titles=[
                    "Adaptive NPI intensity  u*(t)  — re-optimised every day from BPF state",
                    "Adaptive testing intensity  v*(t)  and detection rate  Q_C(t)",
                    "Background β̂_t (BPF)  vs.  effective β after NPI",
                ],
                vertical_spacing=0.12,
            )

            # Row 1: u*(t)
            fig_mpc.add_trace(go.Scatter(
                x=dates_T, y=r.u_mpc,
                fill="tozeroy", fillcolor="rgba(30,136,229,0.15)",
                mode="lines", name="u*(t)  NPI",
                line=dict(color="#1E88E5", width=1.8)),
                row=1, col=1)

            # Row 2: v*(t) and Q_C*(t)
            fig_mpc.add_trace(go.Scatter(
                x=dates_T, y=r.v_mpc,
                fill="tozeroy", fillcolor="rgba(0,131,143,0.12)",
                mode="lines", name="v*(t)  testing",
                line=dict(color="#00838F", width=1.8)),
                row=2, col=1)
            fig_mpc.add_trace(go.Scatter(
                x=dates_T, y=r.q_c_traj,
                mode="lines", name="Q_C*(t)  detection",
                line=dict(color="#00695C", width=1.5, dash="dot")),
                row=2, col=1)
            fig_mpc.add_trace(go.Scatter(
                x=dates_T, y=r.q_c_base_bpf,
                mode="lines", name="Q_C baseline (BPF)",
                line=dict(color="#E53935", width=1.0, dash="dash")),
                row=2, col=1)

            # Row 3: β_bpf vs β_eff
            fig_mpc.add_trace(go.Scatter(
                x=dates_T, y=r.beta_bpf,
                mode="lines", name="β̂_t  uncontrolled (BPF)",
                line=dict(color="#E53935", width=1.5, dash="dash")),
                row=3, col=1)
            fig_mpc.add_trace(go.Scatter(
                x=dates_T, y=r.beta_eff,
                fill="tonexty", fillcolor="rgba(30,136,229,0.08)",
                mode="lines", name="β_eff = β̂_t · exp(−αu*)",
                line=dict(color="#1E88E5", width=2)),
                row=3, col=1)

            fig_mpc.update_yaxes(range=[0, 1.05], title_text="u*(t)", row=1, col=1)
            fig_mpc.update_yaxes(range=[0, 1.05], title_text="rate",  row=2, col=1)
            fig_mpc.update_yaxes(title_text="β",                      row=3, col=1)
            fig_mpc.update_layout(
                height=600, template="plotly_white",
                hovermode="x unified", showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0),
                margin=dict(t=50, b=70, l=55, r=20),
            )
            fig_mpc.update_xaxes(showgrid=False, range=[dates_mpc[0], dates_mpc[-1]])
            st.plotly_chart(fig_mpc, use_container_width=True)

            st.caption(
                "The controller re-solves every day using the BPF posterior mean — so when "
                "the filter detects a rising β (new wave or variant), NPI intensity u*(t) "
                "automatically tightens.  As β falls, controls relax.  "
                "This proactive adaptation is the key advantage of PF-MPC over a fixed plan."
            )

            # --- Figure 2: counterfactual epidemic trajectories ---
            fig_mpc2 = make_subplots(
                rows=1, cols=3,
                subplot_titles=["Hospitalised H", "ICU C", "Cumulative deaths D"],
                horizontal_spacing=0.10,
            )
            for cc, (comp, cap_val) in enumerate(
                [("H", H_MAX_DE), ("C", C_MAX_DE), ("D", None)], start=1
            ):
                y_opt  = r.traj_mpc[comp]
                y_base = r.traj_baseline[comp]
                y_max  = max(float(np.max(y_opt)), float(np.max(y_base))) * 1.08
                fig_mpc2.add_trace(go.Scatter(
                    x=dates_mpc, y=y_opt,
                    mode="lines", name=f"{comp} MPC",
                    line=dict(color="#1E88E5", width=2)),
                    row=1, col=cc)
                fig_mpc2.add_trace(go.Scatter(
                    x=dates_mpc, y=y_base,
                    mode="lines", name=f"{comp} no control",
                    line=dict(color="#E53935", width=2, dash="dash")),
                    row=1, col=cc)
                if cap_val is not None and cap_val < y_max * 1.5:
                    fig_mpc2.add_shape(
                        type="line", x0=dates_mpc[0], x1=dates_mpc[-1],
                        y0=cap_val, y1=cap_val,
                        line=dict(color="#FF6F00", dash="dot", width=1.2),
                        row=1, col=cc)
                fig_mpc2.update_yaxes(range=[0, y_max], row=1, col=cc)

            fig_mpc2.update_layout(
                height=320, template="plotly_white",
                hovermode="x unified", showlegend=True,
                margin=dict(t=40, b=20, l=40, r=20),
            )
            fig_mpc2.update_xaxes(showgrid=False, range=[dates_mpc[0], dates_mpc[-1]])
            fig_mpc2.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
            st.plotly_chart(fig_mpc2, use_container_width=True)

            st.caption(
                "Counterfactual: what would the epidemic have looked like under the MPC adaptive "
                "policy vs. no intervention at all, holding all other drivers (variants, vaccination, "
                "seasonality) fixed at their BPF-estimated values."
            )

            # --- Interpretation note ---
            st.markdown("---")
            st.markdown("#### How to read the PF-MPC result")

            mean_u   = float(np.mean(r.u_mpc))
            mean_v   = float(np.mean(r.v_mpc))
            mean_qc  = float(np.mean(r.q_c_traj))
            max_u    = float(np.max(r.u_mpc))
            max_u_date = dates_T[int(np.argmax(r.u_mpc))].strftime("%d %b %Y")

            if mean_u < 0.10:
                npi_desc = "almost no restrictions were needed"
            elif mean_u < 0.30:
                npi_desc = "light restrictions (gathering limits, masks) were sufficient"
            elif mean_u < 0.55:
                npi_desc = "moderate restrictions (partial closures) were recommended"
            else:
                npi_desc = "strict lockdown-level measures were recommended"

            if mean_v < 0.20:
                test_desc = "routine surveillance — no major testing scale-up needed"
            elif mean_v < 0.60:
                test_desc = "a moderate testing programme (contact tracing, targeted PCR)"
            else:
                test_desc = "mass testing at near-maximum intensity"

            st.info(
                f"**What the adaptive policy recommends over this {r.T_sim}-day period:**  \n\n"
                f"On average, {npi_desc} (mean NPI level {mean_u:.0%}), "
                f"combined with {test_desc} "
                f"(mean testing intensity {mean_v:.0%}, keeping detection at {mean_qc:.0%} of symptomatic cases).  \n\n"
                f"**Strongest response:** The controller recommended its highest NPI intensity "
                f"({max_u:.0%}) on **{max_u_date}** — this is when the BPF detected the largest "
                f"transmission rate β̂_t, signalling a wave or variant emergence. "
                f"After that peak the policy relaxes as β̂_t falls.  \n\n"
                f"**Advantage over a fixed plan:** The single-horizon control (above) computes "
                f"one optimal policy from the starting date and holds it regardless of what "
                f"happens next. The PF-MPC re-evaluates every day: if a new variant raises β "
                f"mid-period, the controller tightens immediately; if the epidemic fades faster "
                f"than expected, it eases early. This responsiveness is the core benefit of "
                f"closing the feedback loop between surveillance (BPF) and decision-making (MPC).  \n\n"
                f"**Counterfactual impact:** Applying this adaptive policy rather than no "
                f"intervention would have averted approximately "
                f"**{r.deaths_averted:,} deaths**, "
                f"**{r.hosp_days_averted:,} hospital-bed-days**, and "
                f"**{r.icu_days_averted:,} ICU-days** over the {r.T_sim}-day window."
            )
            st.caption(
                "Limitation: the counterfactual assumes that all drivers other than NPI policy "
                "(variant emergence, vaccination uptake, seasonal behaviour) remain exactly as "
                "estimated by the BPF. In practice, different policies would have changed "
                "population behaviour, vaccination timing, and potentially variant selection — "
                "effects not captured by this ODE model."
            )

            # --- Convergence diagnostics ---
            with st.expander("Convergence diagnostics"):
                fig_conv = make_subplots(rows=1, cols=2,
                    subplot_titles=["L-BFGS-B iterations per step",
                                    "Fraction of steps converged"])
                fig_conv.add_trace(go.Scatter(
                    x=dates_T, y=r.n_iters,
                    mode="lines", line=dict(color="#546E7A", width=1.0),
                    name="iterations"),
                    row=1, col=1)
                cum_conv = np.cumsum(r.converged) / (np.arange(len(r.converged)) + 1)
                fig_conv.add_trace(go.Scatter(
                    x=dates_T, y=cum_conv * 100,
                    mode="lines", line=dict(color="#2E7D32", width=1.5),
                    name="% converged (cumulative)"),
                    row=1, col=2)
                fig_conv.update_layout(height=250, template="plotly_white",
                                       showlegend=False,
                                       margin=dict(t=40, b=20, l=40, r=20))
                fig_conv.update_xaxes(showgrid=False)
                st.plotly_chart(fig_conv, use_container_width=True)
                st.caption(
                    f"Total steps: {r.T_sim}.  "
                    f"Converged: {int(r.converged.sum())} ({100*r.converged.mean():.0f}%).  "
                    f"Mean iterations: {r.n_iters.mean():.0f}."
                )
        else:
            st.info(
                "Configure the PF-MPC settings on the left and click **Run PF-MPC**.  \n"
                "This retrospective simulation re-solves the control problem at every day "
                "using the BPF posterior — the recommended policy adapts automatically to "
                "detected waves, variant emergence, and waning immunity."
            )

# ===========================================================================
# TAB 9 — Model comparison (SEIR vs SEIRHD vs SVEAIHCRD on German data)
# ===========================================================================
with tab_cmp:
    st.subheader("Model comparison — nested models on German COVID-19 data")
    st.caption(
        "Fits three nested models to the same German data using the generic BPF. "
        "Each model uses only the channels it is designed for. "
        "The R_eff(t) trajectories are then compared side-by-side."
    )

    st.info(
        "**Why nested?**  SEIR uses cases only.  "
        "SEIRHD uses cases + hospitalisation + deaths.  "
        "SVEAIHCRD uses all four channels.  "
        "Simpler models are not wrong — they are designed for settings where fewer channels are available."
    )

    from episurveil.models.seir   import SEIRModel
    from episurveil.models.seirhd import SEIRHDModel
    from episurveil.inference.bpf_generic import run_bpf as _run_bpf

    cmp_col_l, cmp_col_r = st.columns([1, 3])

    with cmp_col_l:
        st.markdown("#### Settings")
        cmp_N = st.slider("Particles N", 200, 1000, 300, step=100, key="cmp_N",
                          help="Lower N for speed. 300 is enough for comparison plots.")
        cmp_models = st.multiselect(
            "Models to run",
            ["SEIR (cases only)", "SEIRHD (cases+hosp+deaths)", "SVEAIHCRD (all channels, pre-computed)"],
            default=["SEIR (cases only)", "SEIRHD (cases+hosp+deaths)", "SVEAIHCRD (all channels, pre-computed)"],
            key="cmp_models",
        )

        st.markdown("#### Date range")
        cmp_date_start = st.date_input(
            "Start date", value=pd.Timestamp("2020-03-10").date(),
            key="cmp_date_start",
        )
        cmp_date_end = st.date_input(
            "End date", value=pd.Timestamp("2021-06-30").date(),
            key="cmp_date_end",
            help="SEIR/SEIRHD have no vaccination compartment — keep to ≤18 months "
                 "or use waning below to reduce β_t inflation on longer series.",
        )

        st.markdown("#### Waning immunity")
        cmp_omega_r = st.slider(
            "ω_R — waning rate (day⁻¹)",
            min_value=0.0, max_value=1/60, value=1/90,
            step=1/360, format="%.4f",
            key="cmp_omega_r",
            help="Flow R→S per day (immunity duration ≈ 1/ω_R days). "
                 "0 = permanent immunity (classic SEIR). "
                 "1/90 ≈ 3-month waning — appropriate for Omicron-era reinfection dynamics.",
        )
        if cmp_omega_r > 0:
            st.caption(
                f"Immunity duration ≈ **{1/cmp_omega_r:.0f} days** "
                f"({1/cmp_omega_r/30:.1f} months)"
            )

        run_cmp = st.button("Run comparison", type="primary", key="run_cmp_btn")

    with cmp_col_r:
        if run_cmp:
            _panel_path = ROOT / "data" / "processed" / "germany_integrated_panel.csv"
            if _panel_path.exists():
                df_panel_raw = pd.read_csv(_panel_path, parse_dates=["date"])
                # normalise column names to what the BPF models expect
                df_panel_raw = df_panel_raw.rename(columns={
                    "reported_cases":        "cases",
                    "hospitalization_proxy": "hosp",
                    "icu_occupancy":         "icu",
                })
            else:
                df_panel_raw = None

            results_cmp = {}

            # SVEAIHCRD: use pre-computed filter output (already in df_full)
            if "SVEAIHCRD (all channels, pre-computed)" in cmp_models:
                _sv_want = (
                    ["date", "R_eff_mean"]
                    + [c for c in ["beta_mean","beta_q10","beta_q90",
                                   "E_mean","E_q10","E_q90",
                                   "obs_cases","obs_icu","obs_hosp","obs_deaths",
                                   "pred_cases_mean","pred_icu_mean",
                                   "pred_hosp_mean","pred_deaths_mean"]
                       if c in df_full.columns]
                )
                _sv_df = df_full[_sv_want].copy() if "R_eff_mean" in df_full.columns else None
                # clip to selected date range for visual alignment
                if _sv_df is not None:
                    _sv_df = _sv_df[
                        (_sv_df["date"] >= pd.Timestamp(cmp_date_start)) &
                        (_sv_df["date"] <= pd.Timestamp(cmp_date_end))
                    ]
                results_cmp["SVEAIHCRD"] = _sv_df if (_sv_df is not None and not _sv_df.empty) else None

                if results_cmp["SVEAIHCRD"] is None:
                    # compute R_eff from SVEAIHCRD filter output, clipped to date range
                    gamma_i  = 0.10
                    eps      = 0.85
                    _df_sv   = df_full[
                        (df_full["date"] >= pd.Timestamp(cmp_date_start)) &
                        (df_full["date"] <= pd.Timestamp(cmp_date_end))
                    ].copy()
                    if not _df_sv.empty:
                        living = _df_sv[["S_mean","V_mean","E_mean","A_mean",
                                         "I_mean","H_mean","C_mean","R_mean"]].sum(axis=1)
                        eff_s  = (_df_sv["S_mean"] + (1 - eps) * _df_sv["V_mean"]) / living
                        r_sv   = _df_sv[["date"]].copy()
                        r_sv["R_eff_mean"] = _df_sv["beta_mean"] * eff_s / (gamma_i + _df_sv["tau_i_mean"])
                        for _ec in ["beta_mean","beta_q10","beta_q90",
                                    "E_mean","E_q10","E_q90",
                                    "obs_cases","obs_icu","obs_hosp","obs_deaths",
                                    "pred_cases_mean","pred_icu_mean",
                                    "pred_hosp_mean","pred_deaths_mean"]:
                            if _ec in _df_sv.columns:
                                r_sv[_ec] = _df_sv[_ec].values
                        results_cmp["SVEAIHCRD"] = r_sv

            if df_panel_raw is not None:
                # apply date range filter
                _d0 = pd.Timestamp(cmp_date_start)
                _d1 = pd.Timestamp(cmp_date_end)
                df_panel_filtered = df_panel_raw[
                    (df_panel_raw["date"] >= _d0) & (df_panel_raw["date"] <= _d1)
                ].copy()
                if df_panel_filtered.empty:
                    st.error("Date range produced an empty panel — adjust start/end dates.")
                    df_panel_filtered = None

                # SEIR — cases only
                if "SEIR (cases only)" in cmp_models and df_panel_filtered is not None:
                    with st.spinner("Running SEIR BPF (cases only) …"):
                        obs_seir = df_panel_filtered[["date", "cases"]].copy()
                        mdl = SEIRModel(N=N_POPULATION, sigma=1/5, gamma=1/10, Q_C=0.50)
                        mdl.omega_r = float(cmp_omega_r)   # waning — set after construction (cache-safe)
                        r_seir = _run_bpf(mdl, obs_seir, N=cmp_N, burn_in=30, progress=False)
                    results_cmp["SEIR"] = r_seir

                # SEIRHD — cases + hosp + deaths
                if "SEIRHD (cases+hosp+deaths)" in cmp_models and df_panel_filtered is not None:
                    with st.spinner("Running SEIRHD BPF (cases+hosp+deaths) …"):
                        obs_seirhd = df_panel_filtered[["date","cases","hosp","deaths"]].copy() \
                            if all(c in df_panel_filtered.columns for c in ["cases","hosp","deaths"]) \
                            else df_panel_filtered[["date","cases"]].assign(hosp=np.nan, deaths=np.nan)
                        mdl = SEIRHDModel(N=N_POPULATION, LOS=8.0)
                        mdl.omega_r = float(cmp_omega_r)   # waning — set after construction (cache-safe)
                        r_seirhd = _run_bpf(mdl, obs_seirhd, N=cmp_N,
                                            channel_weights={"cases":1.0,"hosp":0.1,"deaths":0.4},
                                            burn_in=30, progress=False)
                    results_cmp["SEIRHD"] = r_seirhd
            else:
                st.warning(
                    "Raw German panel CSV not found at "
                    "`data/processed/germany_integrated_panel.csv`. "
                    "Showing SVEAIHCRD pre-computed R_eff only. "
                    "Run `scripts/run_real_data_filter.py` to rebuild the panel."
                )

            # ── R_eff comparison plot ──────────────────────────────────────
            if results_cmp:
                MODEL_COLORS = {
                    "SEIR":      "#1E88E5",
                    "SEIRHD":    "#E53935",
                    "SVEAIHCRD": "#43A047",
                }
                fig_cmp = go.Figure()
                for mname, df_r in results_cmp.items():
                    if df_r is None:
                        continue
                    fig_cmp.add_trace(go.Scatter(
                        x=df_r["date"], y=df_r["R_eff_mean"],
                        mode="lines",
                        name=mname,
                        line=dict(color=MODEL_COLORS.get(mname, "#888"), width=1.6),
                    ))
                fig_cmp.add_hline(y=1.0, line_dash="dot", line_color="#888",
                                  annotation_text="R_eff = 1", annotation_position="bottom right")
                fig_cmp.update_layout(
                    title="R_eff(t) — nested model comparison on German COVID-19 data",
                    xaxis_title="Date", yaxis_title="R_eff(t)",
                    height=420, template="plotly_white",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    margin=dict(t=60, b=30, l=50, r=20),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

                st.caption(
                    "**SEIR** (blue): estimated from case counts alone — β_t and S_t determine R_eff. "
                    "**SEIRHD** (red): adds hospital and death channels, giving tighter severity constraints. "
                    "**SVEAIHCRD** (green): all four channels + asymptomatic compartment + ICU — "
                    "captures the most complexity. Differences between curves reflect model-structure "
                    "uncertainty in addition to parameter uncertainty."
                )

                # ── How R_eff is computed ──────────────────────────────────
                with st.expander("How is R_eff(t) computed for each model?"):
                    st.markdown("""
R_eff(t) is the **effective reproduction number at time t** — the average number of secondary
infections produced by one infectious individual given the current population state.
It is derived from the **next-generation matrix** of each ODE and re-evaluated
at every particle at every BPF step, then summarised as a weighted posterior mean.

---

**SEIR**

> R_eff(t) = β_t · S_t / ( N_live · γ )

| Symbol | Meaning |
|---|---|
| β_t | Time-varying transmission rate (log-RW, tracked by BPF) |
| S_t | Current susceptible pool |
| N_live | S + E + I + R |
| γ | Recovery rate — fixed at 0.10 day⁻¹ (10-day infectious period) |

With no vaccination compartment S_t shrinks monotonically across waves.
When S_t → 0 the filter raises β_t to keep predicted cases on track — this is the
root cause of β inflation seen on long SEIR/SEIRHD runs.

---

**SEIRHD**

> R_eff(t) = β_t · S_t / ( N_live · (γ_I + τ_I,t) )

Added term: **τ_I,t** (time-varying hospitalisation rate, log-RW).
An infectious individual leaves I at rate γ_I + τ_I,t (recovery OR hospitalisation),
so the mean infectious period shortens when hospital pressure is high.
When τ_I,t → 0 the formula collapses to the SEIR case.

---

**SVEAIHCRD**

> R_eff(t) = β_t · [ S_t + (1 − ε) · V_t ] / ( N_live · (γ_I + τ_I,t) )

Added term: **(1 − ε) · V_t** — residual susceptibility of vaccinated individuals
(ε = 0.85 vaccine efficacy).
This keeps the effective susceptible pool alive even after natural infection has
depleted S_t, which is why SVEAIHCRD β_t stays in the 0.2–0.6 range across the
full 1 673-day panel while SEIR/SEIRHD inflate beyond 1.

---

**Why do the three curves diverge?**

| Source | Effect on R_eff |
|---|---|
| No V compartment (SEIR/SEIRHD) | Effective susceptible pool under-counted → β inflates |
| Additional channels (SEIRHD) | Hosp + deaths constrain τ_I,t → R_eff denominator tightens |
| Asymptomatic A compartment (SVEAIHCRD) | Part of transmission is hidden; β is lower for the same observed cases |
| Waning ω_R | R → S flow; larger ω_R replenishes S and suppresses β inflation |

> **Rule of thumb:** When the three curves agree, R_eff is robust to model choice.
> When they diverge, the gap is *model-structure uncertainty* — not measurement noise.
""")

                # ── beta comparison ────────────────────────────────────────
                beta_available = {m: df_r for m, df_r in results_cmp.items()
                                  if df_r is not None and "beta_mean" in df_r.columns}
                if beta_available:
                    with st.expander("Show β_t comparison"):
                        fig_b = go.Figure()
                        for mname, df_r in beta_available.items():
                            color = MODEL_COLORS.get(mname, "#888")
                            if "beta_q90" in df_r.columns:
                                fig_b.add_trace(go.Scatter(
                                    x=df_r["date"], y=df_r["beta_q90"],
                                    fill=None, mode="lines", line=dict(width=0),
                                    showlegend=False))
                                fig_b.add_trace(go.Scatter(
                                    x=df_r["date"], y=df_r["beta_q10"],
                                    fill="tonexty", mode="lines", line=dict(width=0),
                                    fillcolor=f"rgba(128,128,128,0.10)",
                                    showlegend=False))
                            fig_b.add_trace(go.Scatter(
                                x=df_r["date"], y=df_r["beta_mean"],
                                name=mname,
                                line=dict(color=color, width=1.6),
                            ))
                        fig_b.update_layout(
                            height=320, template="plotly_white",
                            yaxis_title="β_t", xaxis_title="Date",
                            title="Transmission rate β_t by model",
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                            margin=dict(t=60, b=20, l=50, r=20))
                        st.plotly_chart(fig_b, use_container_width=True)

                # ── Fixed parameters (transparency) ───────────────────────
                with st.expander("Fixed parameters used in this comparison"):
                    param_table = []
                    _wane_str = (f"{cmp_omega_r:.4f} day⁻¹  "
                                 f"(≈ {1/cmp_omega_r:.0f} d immunity)"
                                 if cmp_omega_r > 0 else "0  (permanent immunity)")
                    if "SEIR" in results_cmp:
                        param_table.append({
                            "Model":           "SEIR",
                            "Population N":    f"{N_POPULATION:,}",
                            "σ (E→I rate)":    "0.200 day⁻¹  (incubation 5 d)",
                            "γ (recovery)":    "0.100 day⁻¹  (infectious 10 d)",
                            "ω_R (waning)":    _wane_str,
                            "Q_C (detection)": "0.50  (fixed)",
                            "LOS":             "—",
                            "Channel weights": "cases × 1.0",
                            "Particles N":     str(cmp_N),
                            "Date range":      f"{cmp_date_start} → {cmp_date_end}",
                            "Source":          "BPF fit on germany_integrated_panel.csv",
                        })
                    if "SEIRHD" in results_cmp:
                        param_table.append({
                            "Model":           "SEIRHD",
                            "Population N":    f"{N_POPULATION:,}",
                            "σ (E→I rate)":    "0.200 day⁻¹  (incubation 5 d)",
                            "γ (recovery)":    "0.100 day⁻¹  (infectious 10 d)",
                            "ω_R (waning)":    _wane_str,
                            "Q_C (detection)": "0.50  (fixed)",
                            "LOS":             "8.0 days",
                            "Channel weights": "cases × 1.0 · hosp × 0.1 · deaths × 0.4",
                            "Particles N":     str(cmp_N),
                            "Date range":      f"{cmp_date_start} → {cmp_date_end}",
                            "Source":          "BPF fit on germany_integrated_panel.csv",
                        })
                    if "SVEAIHCRD" in results_cmp:
                        param_table.append({
                            "Model":           "SVEAIHCRD",
                            "Population N":    f"{N_POPULATION:,}",
                            "σ (E→I rate)":    "0.286 day⁻¹  (latent 3.5 d)",
                            "γ (recovery)":    "0.100 day⁻¹  (infectious 10 d)",
                            "Q_C (detection)": "dynamic (log-RW estimated)",
                            "LOS":             "—",
                            "Channel weights": "cases × 1.0 · icu × 0.3 · hosp × 0.1 · deaths × 0.4",
                            "Particles N":     "2 000  (pre-computed)",
                            "Source":          "sveaihcrd_filter_output.csv  (pre-computed)",
                        })
                    if param_table:
                        st.dataframe(
                            pd.DataFrame(param_table).set_index("Model"),
                            use_container_width=True,
                        )
                    st.caption(
                        "Time-varying parameters (β_t, τ_I_t, δ_H_t, …) are estimated "
                        "jointly by the BPF for each model. Only the fixed structural "
                        "parameters listed above are held constant across the run."
                    )

                # ── Observed vs predicted per model ───────────────────────
                st.markdown("---")
                st.markdown("### Observed vs predicted")
                _CH_COLORS_CMP = {
                    "cases":  "#1E88E5",
                    "icu":    "#00897B",
                    "hosp":   "#8E24AA",
                    "deaths": "#E53935",
                }
                _MODEL_CHANNELS = {
                    "SEIR":      ["cases"],
                    "SEIRHD":    ["cases", "hosp", "deaths"],
                    "SVEAIHCRD": ["cases", "icu", "hosp", "deaths"],
                }
                _BURN = 30   # skip first 30 rows (BPF settling period)
                for mname, df_r in results_cmp.items():
                    if df_r is None:
                        continue
                    channels = [ch for ch in _MODEL_CHANNELS.get(mname, [])
                                if f"obs_{ch}" in df_r.columns
                                and f"pred_{ch}_mean" in df_r.columns]
                    if not channels:
                        continue
                    # skip burn-in rows to avoid initialization spikes
                    df_plot = df_r.iloc[_BURN:].copy() if len(df_r) > _BURN else df_r
                    st.markdown(f"**{mname}**")
                    ncols_fit = min(len(channels), 2)
                    fit_cols  = st.columns(ncols_fit)
                    for ci, ch in enumerate(channels):
                        col_fit = fit_cols[ci % ncols_fit]
                        mask    = df_plot[f"obs_{ch}"].notna()
                        color   = _CH_COLORS_CMP.get(ch, "#555")
                        fig_fit = go.Figure()
                        fig_fit.add_trace(go.Scatter(
                            x=df_plot.loc[mask, "date"],
                            y=df_plot.loc[mask, f"obs_{ch}"],
                            mode="markers",
                            name="Observed",
                            marker=dict(size=3, color="#333", opacity=0.55),
                        ))
                        fig_fit.add_trace(go.Scatter(
                            x=df_plot["date"],
                            y=df_plot[f"pred_{ch}_mean"],
                            mode="lines",
                            name="Predicted",
                            line=dict(color=color, width=2.0),
                        ))
                        fig_fit.update_layout(
                            title=f"{mname} — {ch}",
                            height=240,
                            template="plotly_white",
                            hovermode="x unified",
                            xaxis_title="Date",
                            yaxis_title=ch,
                            legend=dict(orientation="h", y=-0.28, font=dict(size=10)),
                            margin=dict(t=35, b=45, l=45, r=10),
                        )
                        col_fit.plotly_chart(fig_fit, use_container_width=True)

                # ── Exposed compartment E_t comparison ────────────────────
                _e_traces = {}
                for mname, df_r in results_cmp.items():
                    if df_r is not None and "E_mean" in df_r.columns:
                        _e_traces[mname] = df_r

                if _e_traces:
                    st.markdown("---")
                    st.markdown("### Exposed compartment E_t")
                    fig_exp = go.Figure()
                    for mname, df_r in _e_traces.items():
                        color = MODEL_COLORS.get(mname, "#888")
                        if "E_q90" in df_r.columns:
                            fig_exp.add_trace(go.Scatter(
                                x=df_r["date"], y=df_r["E_q90"],
                                fill=None, mode="lines", line=dict(width=0),
                                showlegend=False))
                            fig_exp.add_trace(go.Scatter(
                                x=df_r["date"], y=df_r["E_q10"],
                                fill="tonexty", mode="lines", line=dict(width=0),
                                fillcolor="rgba(128,128,128,0.10)",
                                showlegend=False))
                        fig_exp.add_trace(go.Scatter(
                            x=df_r["date"], y=df_r["E_mean"],
                            mode="lines", name=mname,
                            line=dict(color=color, width=1.8),
                        ))
                    fig_exp.update_layout(
                        title="Exposed population E_t — model comparison",
                        height=340,
                        template="plotly_white",
                        hovermode="x unified",
                        xaxis_title="Date",
                        yaxis_title="Exposed E_t (persons)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                        margin=dict(t=60, b=30, l=60, r=20),
                    )
                    st.plotly_chart(fig_exp, use_container_width=True)
                    st.caption(
                        "E_t represents the latent (incubating) population estimated by each model. "
                        "Differences reflect how each model allocates unobserved burden "
                        "between E, I, and asymptomatic compartments."
                    )
        else:
            st.info(
                "Select which models to compare and click **Run comparison**.  \n\n"
                "Each model is fit to the German COVID-19 data using only "
                "the channels it is designed for, then R_eff(t) is plotted "
                "side by side. Computation takes ~30–90 s depending on N."
            )
            st.markdown("""
| Model | Channels used | Log-RW params | R_eff formula |
|---|---|---|---|
| SEIR | cases | β_t | β_t · S_t / (N · γ) |
| SEIRHD | cases, hosp, deaths | β_t, τ_I_t, δ_H_t | β_t · S_t / (N · (γ_I + τ_I_t)) |
| SVEAIHCRD | cases, ICU, hosp, deaths | β_t, τ_I_t, δ_H_t, ρ_C_t, Q_C_t | β_t · [S_t+(1-ε)V_t] / (N · (γ_I + τ_I_t)) |
""")

# ===========================================================================
# TAB 10 — Model explorer (user uploads own data)
# ===========================================================================
with tab_exp:
    st.subheader("Model explorer — fit any epidemic model to your own data")
    st.caption(
        "Choose a built-in example dataset or upload your own CSV. "
        "Select a model, tune parameters, and run the Bootstrap Particle Filter."
    )

    from episurveil.models.sir    import SIRModel
    from episurveil.models.seir   import SEIRModel
    from episurveil.models.seird  import SEIRDModel
    from episurveil.models.seirv  import SEIRVModel
    from episurveil.models.seiarv import SEIARVModel
    from episurveil.models.seirhd import SEIRHDModel
    from episurveil.inference.bpf_generic import (
        run_bpf as _run_bpf,
        forecast_bpf as _forecast_bpf,
    )

    # ── Built-in example datasets ─────────────────────────────────────────
    _EXAMPLES_DIR      = ROOT / "data" / "examples"
    _EXAMPLES_REAL_DIR = ROOT / "data" / "examples" / "real"
    _BUILTIN_EXAMPLES = {
        "(none — I'll upload my own CSV)": None,
        "🦠  Measles outbreak — West Africa 2019  [SEIR · 200 days]": {
            "file":           "measles_outbreak_west_africa_2019.csv",
            "model_label":    "SEIR   (4 states — cases only)",
            "disease_preset": "Measles",
            "country_preset": "Custom",
            "N_pop":          500_000,
            "channels":       ["cases"],
            "description": (
                "Synthetic measles outbreak in a sub-regional population of **500,000** "
                "with **85% prior vaccination coverage** (15% susceptible). "
                "R₀ = 15, σ = 1/12.5 day⁻¹, γ = 0.125 day⁻¹. "
                "Explosive single wave terminated by herd immunity. "
                "**Recommended preset**: Measles · Q_C = 0.70 · β_max = 2.40."
            ),
        },
        "🤧  Influenza season — Europe 2019–20  [SEIR · 182 days]": {
            "file":           "influenza_season_europe_2019_20.csv",
            "model_label":    "SEIR   (4 states — cases only)",
            "disease_preset": "Influenza (seasonal)",
            "country_preset": "Custom",
            "N_pop":          8_000_000,
            "channels":       ["cases"],
            "description": (
                "Synthetic influenza season for an **8 million** population "
                "(Oct 2019 – Mar 2020). "
                "R₀ = 1.4, σ = 0.50 day⁻¹ (2-day incubation), γ = 0.25 day⁻¹ "
                "(4-day infectious period), Q_C = 0.10 (10% detection). "
                "Classic bell-shaped seasonal wave. "
                "**Recommended preset**: Influenza (seasonal)."
            ),
        },
        "🤧  Influenza multi-wave — Europe 2018–20  [SEIR + waning · 548 days]": {
            "file":           "influenza_multiwav_europe_2018_20.csv",
            "model_label":    "SEIR   (4 states — cases only)",
            "disease_preset": "Influenza (seasonal)",
            "country_preset": "Custom",
            "N_pop":          20_000_000,
            "channels":       ["cases"],
            "description": (
                "Two influenza seasons (Oct 2018 – Mar 2020) for **20 million** population "
                "with ~9-month waning immunity (ω_R = 1/270). "
                "Season 2 is slightly stronger due to antigenic drift (β = 0.40 vs 0.38). "
                "Summer trough visible between waves (β drops to 0.12). "
                "**Key setting**: enable ω_R ≈ 1/270 in Advanced panel to capture the second wave."
            ),
        },
        "🔴  Ebola outbreak — DRC 2018–19  [SEIR · 365 days]": {
            "file":           "ebola_outbreak_drc_2018_19.csv",
            "model_label":    "SEIR   (4 states — cases only)",
            "disease_preset": "Ebola",
            "country_preset": "DR Congo",
            "N_pop":          200_000,
            "channels":       ["cases"],
            "description": (
                "Synthetic Ebola outbreak in a **200,000** population (DRC region). "
                "R_eff starts at 1.9, declining to 0.7 as outbreak response strengthens. "
                "σ = 0.088 day⁻¹ (11.4-day incubation), γ = 0.10 day⁻¹. "
                "Peak ~30 cases/day — the BPF must track a rapidly changing β. "
                "**Recommended preset**: Ebola · Q_C = 0.70 · β_max = 0.22 · N particles ≥ 1000."
            ),
        },
        "🏥  Influenza + hospitalisations — Europe 2019–20  [SEIRHD · 182 days]": {
            "file":           "influenza_seirhd_europe_2019_20.csv",
            "model_label":    "SEIRHD (6 states — cases + hosp + deaths)",
            "disease_preset": "Influenza (seasonal)",
            "country_preset": "Custom",
            "N_pop":          10_000_000,
            "channels":       ["cases", "hosp", "deaths"],
            "description": (
                "Influenza season for **10 million** population with three observation channels: "
                "daily reported cases, hospital occupancy proxy, and daily deaths. "
                "τ_I = 0.002 (0.2% hospitalisation rate), δ_H = 0.003 (2% in-hospital CFR). "
                "Peak hosp ≈ 8,000 · Peak deaths ≈ 50/day. "
                "**Use SEIRHD model** to jointly estimate β_t, τ_I(t), and δ_H(t)."
            ),
        },
        # ── Real datasets (downloaded from public sources) ────────────────
        "🔴  [REAL] Ebola — Guinea 2014–15  [SEIRD · 289 days · cases+deaths]": {
            "file":               "ebola_guinea_2014_16.csv",
            "real":               True,
            "model_label":        "SEIRD  (5 states — cases + deaths)",
            "disease_preset":     "Ebola",
            "country_preset":     "Custom",
            "N_pop":              2_000_000,
            "I0_frac":            1e-5,
            "channels":           ["cases", "deaths"],
            # Wide NegBin dispersions: WHO reports are sparse/irregular;
            # zero days (reporting gaps) are set to NaN in the CSV.
            "bpf_phi":             {"cases": 8, "deaths": 5},
            # Temper deaths channel — higher noise than cases
            "bpf_channel_weights": {"cases": 1.0, "deaths": 0.40},
            # delta_init=0.06 ≈ 37% CFR (Ebola Guinea 2014 was ~40-60%)
            "bpf_model_kwargs":    {"delta_init": 0.06, "phi_cases": 8, "phi_deaths": 5},
            "description": (
                "**Real data** — WHO/cmrivers dataset. Guinea 2014–15 Ebola epidemic. "
                "Daily new cases and deaths linearly interpolated from WHO cumulative reports; "
                "zero-count days (reporting gaps) set to NaN and skipped in the BPF likelihood. "
                "N = 2,000,000 (affected outbreak region, not national population). "
                "Total cases ≈ 2,986 | peak ≈ 59/day. "
                "**Suggested settings**: Ebola preset · beta_max = 0.40 · N particles ≥ 1,000. "
                "**Source**: github.com/cmrivers/ebola (CC0)."
            ),
        },
        "🔴  [REAL] Ebola — Sierra Leone 2014–15  [SEIRD · 289 days · cases+deaths]": {
            "file":               "ebola_sierraleone_2014_16.csv",
            "real":               True,
            "model_label":        "SEIRD  (5 states — cases + deaths)",
            "disease_preset":     "Ebola",
            "country_preset":     "Custom",
            "N_pop":              3_000_000,
            "I0_frac":            1e-5,
            "channels":           ["cases", "deaths"],
            "bpf_phi":             {"cases": 8, "deaths": 5},
            "bpf_channel_weights": {"cases": 1.0, "deaths": 0.40},
            "bpf_model_kwargs":    {"delta_init": 0.06, "phi_cases": 8, "phi_deaths": 5},
            "description": (
                "**Real data** — WHO/cmrivers dataset. Sierra Leone 2014–15 Ebola epidemic. "
                "Largest of the three most-affected countries. "
                "N = 3,000,000 (Western Area + outbreak districts). "
                "Daily new cases and deaths interpolated from WHO cumulative situation reports; "
                "zero-count days set to NaN. "
                "**Suggested settings**: Ebola preset · beta_max = 0.40 · N particles ≥ 1,000. "
                "**Source**: github.com/cmrivers/ebola (CC0)."
            ),
        },
        "🤧  [REAL] Influenza — France 2023–24  [SEIR · 176 days · ILI proxy]": {
            "file":           "influenza_france_2023_24_real.csv",
            "real":           True,
            "model_label":    "SEIR   (4 states — cases only)",
            "disease_preset": "Influenza (seasonal)",
            "country_preset": "France",
            "N_pop":          68_000_000,
            "I0_frac":        5e-3,
            "channels":       ["cases"],
            "description": (
                "**Real data** — ECDC European respiratory virus surveillance. "
                "Weekly ILI (influenza-like illness) consultation rate for France, "
                "converted to daily case-equivalent counts (rate/100k × N / 7) and "
                "forward-filled to daily. Peak ILI rate ≈ 329/100k (peak cases/day ≈ 31,900). "
                "Note: 'cases' = ILI consultations, not confirmed flu — Q_C already absorbed. "
                "**Recommended preset**: Influenza (seasonal) · Q_C = 0.10 · β_max = 0.35. "
                "**Source**: github.com/EU-ECDC/Respiratory_viruses_weekly_data (ECDC open data)."
            ),
        },
        "🤧  [REAL] Influenza — Netherlands 2022–23  [SEIR · 176 days · ILI proxy]": {
            "file":           "influenza_netherlands_2022_23_real.csv",
            "real":           True,
            "model_label":    "SEIR   (4 states — cases only)",
            "disease_preset": "Influenza (seasonal)",
            "country_preset": "Custom",
            "N_pop":          17_900_000,
            "I0_frac":        1e-3,
            "channels":       ["cases"],
            "description": (
                "**Real data** — ECDC European respiratory virus surveillance. "
                "Weekly ILI consultation rate for the Netherlands 2022–23 season, "
                "forward-filled to daily. Peak ILI rate ≈ 99/100k (peak cases/day ≈ 2,500). "
                "Compact, clean single-wave signal — good introductory real dataset. "
                "**Source**: github.com/EU-ECDC/Respiratory_viruses_weekly_data (ECDC open data)."
            ),
        },
        "🤧  [REAL] Influenza — Spain 2023–24  [SEIR · 176 days · ILI proxy]": {
            "file":           "influenza_spain_2023_24_real.csv",
            "real":           True,
            "model_label":    "SEIR   (4 states — cases only)",
            "disease_preset": "Influenza (seasonal)",
            "country_preset": "Spain",
            "N_pop":          47_000_000,
            "I0_frac":        1e-3,
            "channels":       ["cases"],
            "description": (
                "**Real data** — ECDC European respiratory virus surveillance. "
                "Weekly ILI consultation rate for Spain 2023–24, forward-filled to daily. "
                "Peak ILI rate ≈ 141/100k (peak cases/day ≈ 9,400). "
                "**Source**: github.com/EU-ECDC/Respiratory_viruses_weekly_data (ECDC open data)."
            ),
        },
    }

    # ── Preset dicts (hoisted — used by built-in selector AND the widgets) ──
    _COUNTRY_PRESETS = {
        "Custom":                          None,
        "Germany":                    83_200_000,
        "France":                     68_000_000,
        "United Kingdom":             67_000_000,
        "Italy":                      60_000_000,
        "Spain":                      47_000_000,
        "United States":             331_000_000,
        "Brazil":                    215_000_000,
        "India":                   1_400_000_000,
        "South Africa":               60_000_000,
        "DR Congo":                  100_000_000,
    }
    _DISEASE_PRESETS = {
        "Custom": None,
        "COVID-19 — Original 2020": {
            "exp_gamma":      0.10,
            "exp_sigma":      0.20,
            "exp_qc":         0.12,
            "exp_beta_max":   0.40,
            "exp_sigma_beta": 0.05,
            "exp_omega_r":    0.0,
        },
        "COVID-19 — Alpha / Beta (2020–21)": {
            "exp_gamma":      0.10,
            "exp_sigma":      0.20,
            "exp_qc":         0.18,
            "exp_beta_max":   0.55,
            "exp_sigma_beta": 0.05,
            "exp_omega_r":    round(1/180, 4),
        },
        "COVID-19 — Delta (2021)": {
            "exp_gamma":      0.10,
            "exp_sigma":      0.25,
            "exp_qc":         0.20,
            "exp_beta_max":   0.70,
            "exp_sigma_beta": 0.05,
            "exp_omega_r":    round(1/180, 4),
        },
        "COVID-19 — Omicron (2022)": {
            "exp_gamma":      0.13,
            "exp_sigma":      0.29,
            "exp_qc":         0.25,
            "exp_beta_max":   1.95,
            "exp_sigma_beta": 0.05,
            "exp_omega_r":    round(1/90, 4),
        },
        "COVID-19 — Multi-wave 2020–22": {
            "exp_gamma":      0.10,
            "exp_sigma":      0.20,
            "exp_qc":         0.18,
            "exp_beta_max":   0.80,
            "exp_sigma_beta": 0.05,
            "exp_omega_r":    round(1/180, 4),
        },
        "Influenza (seasonal)": {
            "exp_gamma":      0.25,
            "exp_sigma":      0.50,
            "exp_qc":         0.10,
            "exp_beta_max":   0.35,
            "exp_sigma_beta": 0.04,
            "exp_omega_r":    round(1/365, 4),
        },
        "Measles": {
            "exp_gamma":      0.125,
            "exp_sigma":      0.08,
            "exp_qc":         0.70,
            "exp_beta_max":   2.40,
            "exp_sigma_beta": 0.03,
            "exp_omega_r":    0.0,
        },
        "Ebola": {
            "exp_gamma":      0.10,
            "exp_sigma":      0.125,   # 8-day incubation period
            "exp_qc":         0.70,
            "exp_beta_max":   0.40,    # allow R0 up to ~4; Guinea R0 was 1.5-2.5
            "exp_sigma_beta": 0.08,    # faster beta evolution to track multi-wave dynamics
            "exp_omega_r":    0.0,
        },
    }

    _MODEL_REGISTRY = {
        "SIR    (3 states — cases only)":                          "SIR",
        "SEIR   (4 states — cases only)":                         "SEIR",
        "SEIRD  (5 states — cases + deaths)":                     "SEIRD",
        "SEIRV  (5 states — cases + vaccination input)":          "SEIRV",
        "SEIARV (6 states — cases + asymptomatic + detection)":   "SEIARV",
        "SEIRHD (6 states — cases + hosp + deaths)":              "SEIRHD",
    }
    _CHANNEL_MAP = {
        "SIR":    ["cases"],
        "SEIR":   ["cases"],
        "SEIRD":  ["cases", "deaths"],
        "SEIRV":  ["cases"],
        "SEIARV": ["cases"],
        "SEIRHD": ["cases", "hosp", "deaths"],
    }

    # ── Built-in example selector (full-width, above columns) ────────────────
    st.markdown("#### Built-in example datasets")
    _ex_label = st.selectbox(
        "Select a built-in dataset to explore — or choose '(none)' to upload your own",
        list(_BUILTIN_EXAMPLES.keys()),
        key="exp_builtin_select",
    )
    _ex_meta = _BUILTIN_EXAMPLES[_ex_label]

    if _ex_meta is not None:
        # Show description card
        st.info(f"**{_ex_label.split('  [')[0].strip()}**\n\n{_ex_meta['description']}")

        # Apply presets to session state before sliders render
        _ex_dir  = _EXAMPLES_REAL_DIR if _ex_meta.get("real") else _EXAMPLES_DIR
        _ex_path = _ex_dir / _ex_meta["file"]
        if not _ex_path.exists():
            _gen_cmd = (
                "python scripts/download_real_datasets.py"
                if _ex_meta.get("real")
                else "python scripts/generate_example_datasets.py"
            )
            st.error(
                f"Example file not found: `{_ex_meta['file']}`. "
                f"Run `{_gen_cmd}` to generate it."
            )
            st.stop()

        # Propagate model, disease preset, country, N_pop
        _last_ex = st.session_state.get("_exp_builtin_applied", None)
        if _ex_label != _last_ex:
            st.session_state["exp_model"]          = _ex_meta["model_label"]
            st.session_state["exp_disease_preset"] = _ex_meta["disease_preset"]
            st.session_state["exp_country_preset"] = _ex_meta["country_preset"]
            st.session_state["exp_pop"]            = _ex_meta["N_pop"]
            if "I0_frac" in _ex_meta:
                st.session_state["exp_I0_frac"] = _ex_meta["I0_frac"]
            # Apply disease parameters
            _dp = _DISEASE_PRESETS.get(_ex_meta["disease_preset"])
            if _dp:
                for _k, _v in _dp.items():
                    st.session_state[_k] = _v
            # Also reset applied trackers so the preset widgets re-fire
            st.session_state["_exp_disease_applied"] = _ex_meta["disease_preset"]
            st.session_state["_exp_country_applied"] = _ex_meta["country_preset"]
            st.session_state["_exp_builtin_applied"] = _ex_label

        # Load data
        df_user   = pd.read_csv(_ex_path, parse_dates=["date"])
        date_col  = "date"
        df_user   = df_user.sort_values("date")
        num_cols  = [c for c in df_user.columns if c != "date"]
        ch_map    = {ch: ch for ch in _ex_meta["channels"] if ch in df_user.columns}

        st.success(
            f"Loaded **{len(df_user)}** rows · columns: {', '.join(df_user.columns.tolist())} · "
            f"period: {df_user['date'].iloc[0].date()} → {df_user['date'].iloc[-1].date()}"
        )
        st.markdown("---")

    else:
        df_user  = None
        date_col = None
        ch_map   = {}
        num_cols = []

    # Safe defaults — overridden inside the left-column widget block when data is loaded
    run_forecast_btn  = False
    run_scenario_btn  = False
    exp_forecast_days = 14
    exp_beta_reduction = 30
    alert_threshold   = 0.80

    exp_col_l, exp_col_r = st.columns([1, 3])

    with exp_col_l:
        if _ex_meta is None:
            st.markdown("#### 1 — Upload your own data")
        else:
            st.markdown("#### 1 — Data (built-in)")
            st.caption(f"Using: **{_ex_meta['file']}**")

        _MAX_ROWS = 3_650   # ~10 years daily — avoids blocking the server

        if _ex_meta is None:
            uploaded = st.file_uploader(
                "CSV file (date + observation columns)", type=["csv"],
                key="exp_upload",
                help="Must have a date column (any name) and at least one numeric column.",
            )
            if uploaded is not None:
                try:
                    df_user = pd.read_csv(uploaded)
                    df_user.columns = df_user.columns.str.strip()
                    if len(df_user) > _MAX_ROWS:
                        st.error(
                            f"File has {len(df_user):,} rows — maximum is {_MAX_ROWS:,} "
                            f"(≈ 10 years of daily data). Trim the date range and re-upload."
                        )
                        df_user = None
                    else:
                        st.success(f"Loaded {len(df_user)} rows, {len(df_user.columns)} columns.")
                    if df_user is None:
                        st.stop()
                except Exception as _e:
                    st.error(f"Could not read CSV: {str(_e)[:200]}")
                    df_user = None

        # ── Section 2: column mapping (upload only) ───────────────────────
        if df_user is not None and _ex_meta is None:
            try:
                st.markdown("#### 2 — Map columns")
                date_col = st.selectbox("Date column", df_user.columns.tolist(), key="exp_date")
                df_user[date_col] = pd.to_datetime(df_user[date_col], errors="coerce")
                df_user = df_user.dropna(subset=[date_col]).sort_values(date_col)
                num_cols = [c for c in df_user.columns if c != date_col and
                            pd.api.types.is_numeric_dtype(df_user[c])]
            except Exception as _e2:
                st.error(f"Column mapping error: {str(_e2)[:200]}")
                df_user = None

        # ── Sections 3+: model, presets, parameters, run (both paths) ─────
        if df_user is not None:
            try:
                st.markdown("#### 3 — Select model")
                model_label = st.selectbox("Model", list(_MODEL_REGISTRY.keys()), key="exp_model")
                model_key   = _MODEL_REGISTRY[model_label]
                needed_chs  = _CHANNEL_MAP[model_key]

                if _ex_meta is not None:
                    # Built-in: show channel mapping as read-only info
                    st.markdown(f"**Channels:** {', '.join(_ex_meta['channels'])}")
                    for ch in needed_chs:
                        if ch not in ch_map:
                            ch_map[ch] = ch if ch in df_user.columns else None
                else:
                    st.markdown(f"**Required channels:** {', '.join(needed_chs)}")
                    ch_map = {}
                    for ch in needed_chs:
                        best = next((c for c in num_cols if ch in c.lower()),
                                    num_cols[0] if num_cols else None)
                        sel  = st.selectbox(f"Column → '{ch}'", ["(skip)"] + num_cols,
                                            index=num_cols.index(best)+1 if best in num_cols else 0,
                                            key=f"exp_ch_{ch}")
                        ch_map[ch] = sel if sel != "(skip)" else None

                # ── Disease / country presets ────────────────────────────
                st.markdown("#### 3b — Calibration presets")
                _col_ctr, _col_dis = st.columns(2)
                with _col_ctr:
                    country_label = st.selectbox(
                        "Country (sets population N)",
                        list(_COUNTRY_PRESETS.keys()), key="exp_country_preset",
                        help="Sets the N_total parameter below to the country's population."
                    )
                with _col_dis:
                    disease_label = st.selectbox(
                        "Disease / variant",
                        list(_DISEASE_PRESETS.keys()), key="exp_disease_preset",
                        help="Pre-fills σ, γ, Q_C, β_max, σ_β, ω_R with literature values."
                    )

                # Apply country preset once per selection change
                _last_ctr = st.session_state.get("_exp_country_applied", "Custom")
                if country_label != "Custom" and country_label != _last_ctr:
                    st.session_state["exp_pop"] = _COUNTRY_PRESETS[country_label]
                    st.session_state["_exp_country_applied"] = country_label

                # Apply disease preset once per selection change
                _last_dis = st.session_state.get("_exp_disease_applied", "Custom")
                if disease_label != "Custom" and disease_label != _last_dis:
                    for _pk, _pv in _DISEASE_PRESETS[disease_label].items():
                        st.session_state[_pk] = _pv
                    st.session_state["_exp_disease_applied"] = disease_label

                # Feedback caption
                _preset_notes = []
                if country_label != "Custom":
                    _n_fmt = f"{_COUNTRY_PRESETS[country_label]:,}"
                    _preset_notes.append(f"N = {_n_fmt} ({country_label})")
                if disease_label != "Custom":
                    _dp = _DISEASE_PRESETS[disease_label]
                    _or_str = (f"ω_R = 1/{round(1/_dp['exp_omega_r'])}"
                               if _dp.get("exp_omega_r", 0) > 0 else "ω_R = 0")
                    _preset_notes.append(
                        f"γ={_dp['exp_gamma']:.3f}, σ={_dp['exp_sigma']:.3f}, "
                        f"Q_C={_dp['exp_qc']:.2f}, β_max={_dp['exp_beta_max']:.2f}, {_or_str}"
                    )
                if _preset_notes:
                    st.info("**Preset applied** — " + " · ".join(_preset_notes)
                            + "  \nYou can still adjust any parameter below.")

                st.markdown("#### 4 — Parameters")
                exp_N     = st.slider("Particles N",    100, 2000, 500,  step=100, key="exp_N")
                exp_pop   = st.number_input("Population N_total", value=1_000_000,
                                            step=100_000, key="exp_pop")
                exp_gamma = st.slider("Recovery rate γ (day⁻¹)", 0.02, 0.25, 0.10,
                                      step=0.01, key="exp_gamma")
                exp_sigma = st.slider("E→I rate σ (day⁻¹)",     0.05, 0.50, 0.20,
                                      step=0.01, key="exp_sigma",
                                      help="Only used by models with an E compartment.")
                exp_qc    = st.slider("Detection prob Q_C (fixed)", 0.05, 0.99, 0.50,
                                      step=0.01, key="exp_qc",
                                      help="Fraction of symptomatic cases reported. "
                                           "SEIARV estimates this dynamically instead.")
                exp_seed  = st.number_input("Random seed", value=42, step=1, key="exp_seed")

                st.markdown("#### Advanced")
                exp_beta_max = st.slider(
                    "β_max (upper bound on transmission)",
                    0.20, 2.50, 0.80, step=0.05, key="exp_beta_max",
                    help="Hard upper bound on β_t. "
                         "For COVID with γ=0.10: β_max=0.80 → R_eff ≤ 8. "
                         "Raise if β hits the ceiling and predicted cases drop to 0 "
                         "while observed cases are still rising (Delta/Omicron waves).",
                )
                exp_sigma_beta = st.slider(
                    "σ_β (log-RW noise on β_t)",
                    0.005, 0.10, 0.05, step=0.005, key="exp_sigma_beta",
                    help="Daily volatility of log β_t. "
                         "Smaller = smoother β trajectory. "
                         "Larger = β pivots faster after summer troughs and variant waves. "
                         "0.04–0.06 is recommended for multi-wave data.",
                )
                exp_omega_r = st.slider(
                    "ω_R — waning rate (day⁻¹)",
                    0.0, 1/60, 1/180, step=1/360, format="%.4f",
                    key="exp_omega_r",
                    help="Waning immunity: fraction of R that returns to S each day. "
                         "0 = permanent immunity (classic SEIR — one wave only). "
                         "1/180 ≈ 6-month waning (recommended for multi-wave data). "
                         "1/90 ≈ 3-month waning (Omicron reinfection).",
                )
                if exp_omega_r > 0:
                    st.caption(
                        f"Immunity duration ≈ {1/exp_omega_r:.0f} days "
                        f"({1/exp_omega_r/30:.1f} months)"
                    )

                # I0_frac is estimated automatically in run_bpf from the first
                # non-zero observation — no manual slider needed.
                exp_I0_frac = 1e-4   # default; overridden by auto-init at BPF run time

                # Vaccination column (SEIRV only)
                if model_key == "SEIRV":
                    st.markdown("**Vaccination input (exogenous)**")
                    nu_col = st.selectbox("Column → 'nu' (daily vax rate)", ["(none)"] + num_cols,
                                         key="exp_nu")
                else:
                    nu_col = None

                with st.expander("Parameter plausibility guide", expanded=False):
                    st.markdown("""
> **Remark — always use epidemiologically plausible values.**
> Results are only meaningful when the fixed parameters reflect
> what is actually known about the disease and the population studied.

---

**Population size N**

Use the *actual* population of the region in your data.

| Region | N |
|---|---|
| Germany | 83 200 000 |
| France | 68 000 000 |
| United Kingdom | 67 000 000 |
| United States | 331 000 000 |
| Generic city (1 M) | 1 000 000 |

---

**σ — latent-to-infectious rate (E→I, SEIR and above)**

σ = 1 / (mean incubation period in days).

| Disease | Incubation (days) | σ (day⁻¹) | Source |
|---|---|---|---|
| COVID-19 (original) | 5.1 | 0.196 | Lauer et al., 2020 |
| COVID-19 (Omicron) | 3.4 | 0.294 | Backer et al., 2022 |
| Influenza | 2.0 | 0.500 | Lessler et al., 2009 |
| Ebola | 11.4 | 0.088 | WHO, 2014 |
| Measles | 12.5 | 0.080 | Anderson & May, 1991 |

---

**γ — recovery rate (I→R)**

γ = 1 / (mean infectious period in days).

| Disease | Infectious period (days) | γ (day⁻¹) |
|---|---|---|
| COVID-19 | 8–10 | 0.10–0.13 |
| Influenza | 4–5 | 0.20–0.25 |
| Measles | 8 | 0.125 |
| Ebola | 9–12 | 0.08–0.11 |

---

**Q_C — case detection probability**

Fraction of true infections that appear as reported cases.
This is almost always **below 0.50** for respiratory diseases —
often much lower in the early phase of an outbreak.

| Setting | Q_C estimate |
|---|---|
| Germany COVID-19 (2020) | 0.10–0.15 (Streeck et al., 2020) |
| Germany COVID-19 (2021) | 0.15–0.25 (mass testing era) |
| Influenza surveillance | 0.05–0.20 |
| Ebola (high-resource setting) | 0.60–0.90 |

---

**β_max — upper bound on transmission**

A sensible upper bound is β_max = R0_max × γ,
where R0_max is the highest plausible basic reproduction number.

| Disease / variant | R0 estimate | γ | β_max |
|---|---|---|---|
| COVID-19 original | 2.5–3.0 | 0.10 | 0.30 |
| COVID-19 Delta | 5–7 | 0.10 | 0.70 |
| COVID-19 Omicron | 10–15 | 0.13 | 1.95 |
| Influenza (seasonal) | 1.2–1.4 | 0.20 | 0.28 |
| Measles | 12–18 | 0.125 | 2.25 |

*Setting β_max too high allows the filter to simulate an unrealistically
large epidemic, depleting S and causing predicted cases to collapse to 0.*

---

**ω_R — waning rate**

Use 0 (permanent immunity) for short outbreaks or a single wave.
For multi-wave data set ω_R = 1 / (immunity duration in days).

| Scenario | Duration | ω_R |
|---|---|---|
| Single wave (classic SEIR) | ∞ | 0 |
| COVID-19 natural immunity | ~180 days | 1/180 ≈ 0.0056 |
| COVID-19 Omicron reinfection | ~90 days | 1/90 ≈ 0.011 |
| Seasonal influenza | ~365 days | 1/365 ≈ 0.0027 |

---

*Key references: Anderson & May (1991) — Infectious Diseases of Humans;
Lauer et al. (2020) Ann Intern Med; WHO Ebola response team (2014) NEJM.*
""")

                run_exp = st.button("Run BPF", type="primary", key="run_exp_btn")

                st.markdown("---")
                st.markdown("**Forecast & alert settings**")
                exp_forecast_days = st.slider(
                    "Forecast horizon (days)", 7, 60, 14, step=7,
                    key="exp_forecast_days",
                    help="Number of days to project forward after the last observation.",
                )
                _has_particles = st.session_state.get("exp_final_particles") is not None
                run_forecast_btn = st.button(
                    "Run Forecast",
                    key="run_forecast_btn",
                    disabled=not _has_particles,
                    help="Run BPF first, then click to generate a forward forecast.",
                )

                alert_threshold = st.slider(
                    "Alert threshold — P(R_eff > 1)", 0.50, 0.95, 0.80, step=0.05,
                    key="exp_alert_threshold",
                    help="Raise a red alert when this fraction of particles estimate R_eff > 1.",
                )

                st.markdown("---")
                st.markdown("**Intervention scenario**")
                exp_beta_reduction = st.slider(
                    "Transmission reduction (%)", 0, 80, 30, step=10,
                    key="exp_beta_reduction",
                    help="Counterfactual: reduce β by this % at the forecast start (e.g. NPI, behaviour change).",
                )
                run_scenario_btn = st.button(
                    "Run Scenario Forecast",
                    key="run_scenario_btn",
                    disabled=not _has_particles,
                    help="Run BPF first, then click to generate the counterfactual forecast.",
                )

            except Exception as e:
                st.error(f"Setup error: {str(e)[:200]}")
                run_exp = False
        else:
            if _ex_meta is None:
                st.info("Select a built-in dataset above or upload a CSV to get started.")
            run_exp = False

    with exp_col_r:
        mdl = None  # set during BPF run; also stored in session_state["exp_model_obj"]
        if df_user is not None and run_exp:
            # Build obs_df
            obs_exp = df_user[[date_col]].rename(columns={date_col: "date"}).copy()
            for ch, src in ch_map.items():
                obs_exp[ch] = df_user[src].values if src else np.nan

            # Build exog_df (vaccination)
            exog_exp = None
            if nu_col and nu_col != "(none)":
                exog_exp = df_user[[date_col, nu_col]].rename(
                    columns={date_col: "date", nu_col: "nu"})

            # Instantiate model  (I0_frac is overridden by auto-init in run_bpf)
            kw = dict(
                N=exp_pop,
                gamma=exp_gamma,
                beta_max=float(exp_beta_max),
                sigma_beta=float(exp_sigma_beta),
            )
            if model_key != "SIR":
                kw["sigma"] = exp_sigma
            if model_key not in ("SEIRV", "SEIARV"):
                kw["Q_C"] = exp_qc

            # Merge any dataset-specific model overrides (e.g. delta_init, phi_*)
            if _ex_meta and "bpf_model_kwargs" in _ex_meta:
                kw.update(_ex_meta["bpf_model_kwargs"])

            model_map = {
                "SIR":    SIRModel,
                "SEIR":   SEIRModel,
                "SEIRD":  SEIRDModel,
                "SEIRV":  SEIRVModel,
                "SEIARV": SEIARVModel,
                "SEIRHD": SEIRHDModel,
            }
            mdl_cls = model_map[model_key]
            try:
                mdl = mdl_cls(**{k: v for k, v in kw.items()
                                 if k in mdl_cls.__init__.__code__.co_varnames})
            except Exception:
                mdl = mdl_cls()
            # waning immunity — set after construction (safe across module cache)
            mdl.omega_r = float(exp_omega_r)

            # Dataset-level BPF overrides (phi dispersion and channel tempering)
            _bpf_phi = _ex_meta.get("bpf_phi") if _ex_meta else None
            _bpf_cw  = _ex_meta.get("bpf_channel_weights") if _ex_meta else None

            with st.spinner(f"Running {model_key} BPF on your data …"):
                try:
                    res_exp, _final_ptcls = _run_bpf(
                        mdl, obs_exp, N=int(exp_N),
                        exog_df=exog_exp,
                        seed=int(exp_seed),
                        burn_in=min(30, len(obs_exp)//5),
                        phi=_bpf_phi,
                        channel_weights=_bpf_cw,
                        progress=False,
                        return_particles=True,
                    )
                    st.session_state["exp_result"]          = res_exp
                    st.session_state["exp_model_key"]       = model_key
                    st.session_state["exp_state_names"]     = mdl.state_names
                    st.session_state["exp_param_names"]     = mdl.param_names
                    st.session_state["exp_obs_channels"]    = mdl.obs_channels
                    st.session_state["exp_N_used"]          = int(exp_N)
                    st.session_state["exp_final_particles"] = _final_ptcls
                    st.session_state["exp_last_date"]       = obs_exp["date"].iloc[-1]
                    st.session_state["exp_model_obj"]       = mdl   # for forecast
                    st.session_state["exp_forecast"]        = None  # reset on new BPF run
                except Exception as e:
                    _etype = type(e).__name__
                    st.error(
                        f"BPF run failed ({_etype}). "
                        "Check that your data has no missing dates, no all-zero columns, "
                        "and that the population size is plausible for your region. "
                        f"Detail: {str(e)[:200]}"
                    )
                    res_exp = None
        else:
            res_exp   = st.session_state.get("exp_result",    None)
            model_key = st.session_state.get("exp_model_key", "SEIR")

        # ── Forecast & scenario runs ───────────────────────────────────────
        _fcast_ptcls = st.session_state.get("exp_final_particles")
        _fcast_date  = st.session_state.get("exp_last_date")
        _fcast_mdl   = st.session_state.get("exp_model_obj")

        if run_forecast_btn and _fcast_ptcls is not None and _fcast_mdl is not None:
            with st.spinner(f"Forecasting {exp_forecast_days} days ahead …"):
                try:
                    _fcast_df = _forecast_bpf(
                        _fcast_mdl, _fcast_ptcls,
                        horizon=exp_forecast_days,
                        start_date=_fcast_date,
                        seed=int(exp_seed) + 1,
                    )
                    st.session_state["exp_forecast"]          = _fcast_df
                    st.session_state["exp_forecast_scenario"] = None  # reset scenario
                except Exception as _fe:
                    st.error(f"Forecast failed: {str(_fe)[:200]}")

        if run_scenario_btn and _fcast_ptcls is not None and _fcast_mdl is not None:
            _mult = 1.0 - exp_beta_reduction / 100.0
            with st.spinner(
                f"Running scenario forecast (−{exp_beta_reduction}% transmission) …"
            ):
                try:
                    _sc_df = _forecast_bpf(
                        _fcast_mdl, _fcast_ptcls,
                        horizon=exp_forecast_days,
                        start_date=_fcast_date,
                        beta_multiplier=_mult,
                        seed=int(exp_seed) + 2,
                    )
                    st.session_state["exp_forecast_scenario"] = _sc_df
                except Exception as _se:
                    st.error(f"Scenario forecast failed: {str(_se)[:200]}")

        _fcast_df  = st.session_state.get("exp_forecast",          None)
        _sc_df     = st.session_state.get("exp_forecast_scenario", None)

        # Recover model metadata (valid after run or on page reload)
        _state_names  = st.session_state.get("exp_state_names",  [])
        _param_names  = st.session_state.get("exp_param_names",  ["beta"])
        _obs_channels = st.session_state.get("exp_obs_channels", ["cases"])
        _N_used       = st.session_state.get("exp_N_used",       500)

        if res_exp is not None:
            st.success(
                f"BPF complete — {len(res_exp)} days  |  "
                f"mean ESS = {res_exp['ess'].mean():.0f} / {_N_used}  |  "
                f"model: **{model_key}**"
            )

            # ── Fit metrics table ─────────────────────────────────────────
            met_rows = []
            for ch in _obs_channels:
                obs_col  = f"obs_{ch}"
                pred_col = f"pred_{ch}_mean"
                if obs_col not in res_exp.columns or pred_col not in res_exp.columns:
                    continue
                mask = res_exp[obs_col].notna()
                if not mask.any():
                    continue
                y_obs  = res_exp.loc[mask, obs_col].values
                y_pred = res_exp.loc[mask, pred_col].values
                rmse     = float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
                mae      = float(np.mean(np.abs(y_obs - y_pred)))
                mean_obs = float(y_obs.mean())
                met_rows.append({
                    "Channel":      ch,
                    "RMSE":         f"{rmse:,.1f}",
                    "MAE":          f"{mae:,.1f}",
                    "Mean obs":     f"{mean_obs:,.1f}",
                    "RMSE / mean":  f"{rmse / (mean_obs + 1e-9) * 100:.1f}%",
                })
            if met_rows:
                st.markdown("**Fit metrics**")
                st.dataframe(pd.DataFrame(met_rows), hide_index=True,
                             use_container_width=True)

            # ── Epidemic alert status ─────────────────────────────────────
            if "P_growing" in res_exp.columns:
                st.markdown("---")
                st.markdown("**Epidemic alert**")

                _pg_now  = float(res_exp["P_growing"].iloc[-1])
                _pg_peak = float(res_exp["P_growing"].max())
                _days_above = int((res_exp["P_growing"] >= alert_threshold).sum())

                # Forecast alert: last P_growing value in forecast
                _pg_fcast = None
                if _fcast_df is not None and "P_growing" in _fcast_df.columns:
                    _pg_fcast = float(_fcast_df["P_growing"].iloc[-1])

                # Traffic-light banner
                if _pg_now >= alert_threshold:
                    st.error(
                        f"🔴 **ALERT — epidemic is GROWING**  "
                        f"P(R_eff > 1) = **{_pg_now:.0%}** "
                        f"(threshold {alert_threshold:.0%})  "
                        f"| {_days_above} days above threshold"
                    )
                elif _pg_now >= 0.50:
                    st.warning(
                        f"🟡 **CAUTION — growth uncertain**  "
                        f"P(R_eff > 1) = **{_pg_now:.0%}**"
                    )
                else:
                    st.success(
                        f"🟢 **Under control** — P(R_eff > 1) = **{_pg_now:.0%}**"
                    )

                # Key metrics row
                _al_c1, _al_c2, _al_c3 = st.columns(3)
                _al_c1.metric("P(growing) now",  f"{_pg_now:.0%}")
                _al_c2.metric("Peak P(growing)", f"{_pg_peak:.0%}")
                if _pg_fcast is not None:
                    _delta_pct = _pg_fcast - _pg_now
                    _al_c3.metric(
                        f"P(growing) in {len(_fcast_df)}d",
                        f"{_pg_fcast:.0%}",
                        delta=f"{_delta_pct:+.0%}",
                        delta_color="inverse",
                    )

                # P_growing time series chart
                _fig_pg = go.Figure()

                # Threshold fill for alert zone
                _fig_pg.add_hrect(
                    y0=alert_threshold, y1=1.0,
                    fillcolor="rgba(229,57,53,0.07)",
                    line_width=0,
                    annotation_text=f"Alert zone (>{alert_threshold:.0%})",
                    annotation_position="top right",
                )

                # Fit P_growing
                _fig_pg.add_trace(go.Scatter(
                    x=res_exp["date"], y=res_exp["P_growing"],
                    mode="lines", name="P(R_eff > 1) — fit",
                    line=dict(color="#1E88E5", width=2.0),
                    fill="tozeroy", fillcolor="rgba(30,136,229,0.08)",
                ))

                # Forecast P_growing — baseline
                if _fcast_df is not None and "P_growing" in _fcast_df.columns:
                    _fig_pg.add_trace(go.Scatter(
                        x=_fcast_df["date"], y=_fcast_df["P_growing"],
                        mode="lines", name="P(R_eff > 1) — forecast",
                        line=dict(color="#1E88E5", width=1.8, dash="dash"),
                    ))

                # Scenario P_growing
                if _sc_df is not None and "P_growing" in _sc_df.columns:
                    _fig_pg.add_trace(go.Scatter(
                        x=_sc_df["date"], y=_sc_df["P_growing"],
                        mode="lines",
                        name=f"P(R_eff > 1) — scenario (−{exp_beta_reduction}%)",
                        line=dict(color="#43A047", width=1.8, dash="dot"),
                    ))

                _fig_pg.add_hline(
                    y=alert_threshold, line_dash="dash", line_color="#E53935",
                    annotation_text=f"Alert threshold ({alert_threshold:.0%})",
                    annotation_position="bottom right",
                )
                _fig_pg.add_hline(
                    y=0.5, line_dash="dot", line_color="#888",
                    annotation_text="50 %",
                    annotation_position="bottom right",
                )
                _fig_pg.update_layout(
                    title="P(R_eff > 1) — probability the epidemic is growing",
                    xaxis_title="Date", yaxis_title="Probability",
                    yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                    height=260, template="plotly_white",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=-0.30, font=dict(size=11)),
                    margin=dict(t=40, b=55, l=55, r=20),
                )
                st.plotly_chart(_fig_pg, use_container_width=True)

            # ── Generate Report ───────────────────────────────────────────
            st.markdown("---")
            _rep_col1, _rep_col2 = st.columns([1, 3])
            with _rep_col1:
                _gen_report = st.button(
                    "📋 Generate Report",
                    key="btn_gen_report",
                    help="Produce a full statistical analysis and interpretation of the BPF results.",
                    type="primary",
                )
            with _rep_col2:
                if "exp_report_md" in st.session_state:
                    _dl_c1, _dl_c2 = st.columns(2)
                    with _dl_c1:
                        st.download_button(
                            "⬇️ Download (.md)",
                            data=st.session_state["exp_report_md"],
                            file_name=f"epifilter_report_{model_key}.md",
                            mime="text/markdown",
                            key="btn_dl_report",
                        )
                    with _dl_c2:
                        _html_src = _report_to_html(
                            st.session_state["exp_report_md"],
                            title=f"EpiSurveil Report — {model_key}",
                        )
                        st.download_button(
                            "⬇️ Download (.html)",
                            data=_html_src,
                            file_name=f"epifilter_report_{model_key}.html",
                            mime="text/html",
                            key="btn_dl_report_html",
                        )

            if _gen_report:
                _d_label = st.session_state.get("exp_disease_preset", "Custom")
                _c_label = st.session_state.get("exp_country_preset", "Custom")
                _fcast_df   = st.session_state.get("exp_forecast")
                _sc_df      = st.session_state.get("exp_forecast_scenario")
                _report_md = _build_epidemic_report(
                    res=res_exp,
                    model_key=model_key,
                    obs_channels=_obs_channels,
                    param_names=_param_names,
                    N_particles=_N_used,
                    N_pop=int(st.session_state.get("exp_pop", 1_000_000)),
                    gamma=float(st.session_state.get("exp_gamma", 0.10)),
                    beta_max=float(st.session_state.get("exp_beta_max", 0.80)),
                    disease_label=_d_label,
                    country_label=_c_label,
                    burn=30,
                    forecast_df=_fcast_df,
                    scenario_df=_sc_df,
                    alert_threshold=alert_threshold,
                    beta_reduction_pct=exp_beta_reduction,
                )
                st.session_state["exp_report_md"] = _report_md

            if "exp_report_md" in st.session_state:
                with st.expander("📋 Epidemic surveillance report", expanded=True):
                    st.markdown(st.session_state["exp_report_md"])

            # ── Observed vs predicted ─────────────────────────────────────
            obs_chs_avail = [
                ch for ch in _obs_channels
                if f"obs_{ch}" in res_exp.columns and f"pred_{ch}_mean" in res_exp.columns
            ]
            if obs_chs_avail:
                st.markdown("**Observed vs predicted**")

                # ── Diagnostic remark ─────────────────────────────────────
                with st.expander("How to interpret this chart — common failure modes", expanded=False):
                    st.markdown("""
**Reading the fit curve**

A good fit tracks the observed curve closely throughout the entire period.
Watch for these warning signs:

| Pattern | Likely cause | Fix |
|---|---|---|
| Prediction **drops to ~0** at the end while observations rise | **Susceptible depletion** — S is exhausted after multiple waves; β cannot produce new infections | Increase **ω_R** (e.g. 1/90 for Omicron) so R→S waning replenishes susceptibles |
| **β hits its ceiling** (β_mean ≈ β_max) for many consecutive days | β_max too low for the circulating variant | Set **β_max = R₀_max × γ** (e.g. Delta: 7 × 0.10 = 0.70) |
| Prediction **flat near zero from the start** | Too many initial susceptibles depleted, or wrong population size | Check **N** — use the actual country population (Germany: 83.2M) |
| Prediction overshoots then collapses | **Q_C too low** — model infers very high incidence to match reported cases | Raise **Q_C** toward 0.15–0.25 for Germany COVID 2020–21 |
| Wild oscillation / ESS near N (= particle collapse) | **σ_β too large** or **beta_max too wide** | Reduce σ_β (try 0.03–0.05); tighten β_max |
| Systematic underfit in summer troughs | **σ_β too small** — filter cannot pivot β fast enough | Raise σ_β slightly (0.05–0.07) |

**Parameter checklist before running**

- **N**: use the actual population of the country/region, not 1 M (Germany = 83 200 000; France = 68 000 000; UK = 67 000 000)
- **γ (recovery rate)**: ~0.10 for COVID-19 original/Delta (10-day infectious period), ~0.13 for Omicron (7–8 days), ~0.20–0.25 for influenza
- **σ (incubation rate)**: ~0.20 for COVID-19 original (5-day incubation), ~0.29 for Omicron (3.4 days)
- **Q_C (detection)**: Germany COVID 2020 ≈ 0.10–0.15 (Streeck et al. 2020), 2021 ≈ 0.15–0.25
- **β_max**: set to R₀_max × γ for the dominant variant (see plausibility guide above)
- **ω_R (waning)**: 0 if fitting a single isolated wave; 1/180 for natural immunity ~6 months; 1/90 for Omicron reinfection

> **Rule of thumb**: if the predicted curve tracks well for the first 1–2 waves but collapses later,
> the problem is almost always susceptible depletion. Increase ω_R first, then β_max.
""")

                # ── Auto-detect end-of-series collapse ────────────────────
                for _ch in obs_chs_avail:
                    _col = f"pred_{_ch}_mean"
                    _obs_col = f"obs_{_ch}"
                    if _col in res_exp.columns and _obs_col in res_exp.columns:
                        _n = len(res_exp)
                        _tail_n = max(_n // 5, 10)   # last 20% of the series
                        _tail_pred = res_exp[_col].iloc[-_tail_n:].mean()
                        _tail_obs  = res_exp[_obs_col].iloc[-_tail_n:].dropna().mean()
                        if (
                            _tail_obs > 10
                            and _tail_pred < _tail_obs * 0.10
                        ):
                            st.warning(
                                f"⚠️ **{_ch.capitalize()} — end-of-series collapse detected.** "
                                f"Predicted mean ({_tail_pred:,.0f}) is <10 % of observed mean "
                                f"({_tail_obs:,.0f}) over the last {_tail_n} days.  \n"
                                "This is almost always **susceptible depletion**: S → 0 after "
                                "multiple waves without waning immunity replenishing susceptibles.  \n"
                                "**Suggested fix**: increase **ω_R** (waning rate) to ~1/90 in the "
                                "Advanced panel, and verify **β_max ≥ R₀_max × γ** for the "
                                "dominant variant in your data."
                            )
                            break   # one warning is enough

                n_fit_cols = min(len(obs_chs_avail), 2)
                fit_cols   = st.columns(n_fit_cols)
                _CH_COLORS = {
                    "cases":  "#1E88E5",
                    "deaths": "#E53935",
                    "hosp":   "#8E24AA",
                    "icu":    "#00897B",
                }
                _BURN_EXP = 30   # skip first 30 rows — BPF settling from wide particle priors
                _res_plot = res_exp.iloc[_BURN_EXP:].copy() if len(res_exp) > _BURN_EXP else res_exp

                for ci, ch in enumerate(obs_chs_avail):
                    col_fit = fit_cols[ci % n_fit_cols]
                    mask    = _res_plot[f"obs_{ch}"].notna()
                    color   = _CH_COLORS.get(ch, "#555")

                    # Parse color to rgba for the credible band fill
                    _hex = color.lstrip("#")
                    _r, _g, _b = int(_hex[0:2], 16), int(_hex[2:4], 16), int(_hex[4:6], 16)
                    _fill_rgba = f"rgba({_r},{_g},{_b},0.15)"

                    fig_fit = go.Figure()

                    # ── 80% credible interval band (fit) ──────────────────
                    _q90_col = f"pred_{ch}_q90"
                    _q10_col = f"pred_{ch}_q10"
                    if _q90_col in _res_plot.columns and _q10_col in _res_plot.columns:
                        fig_fit.add_trace(go.Scatter(
                            x=_res_plot["date"], y=_res_plot[_q90_col],
                            mode="lines", line=dict(width=0),
                            showlegend=False, name="q90",
                        ))
                        fig_fit.add_trace(go.Scatter(
                            x=_res_plot["date"], y=_res_plot[_q10_col],
                            mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor=_fill_rgba,
                            name="80% CI",
                        ))

                    # ── Observed scatter ───────────────────────────────────
                    fig_fit.add_trace(go.Scatter(
                        x=_res_plot.loc[mask, "date"],
                        y=_res_plot.loc[mask, f"obs_{ch}"],
                        mode="markers",
                        name="Observed",
                        marker=dict(size=3, color="#333", opacity=0.55),
                    ))

                    # ── Predicted mean (fit) ───────────────────────────────
                    fig_fit.add_trace(go.Scatter(
                        x=_res_plot["date"],
                        y=_res_plot[f"pred_{ch}_mean"],
                        mode="lines",
                        name="Predicted",
                        line=dict(color=color, width=2.0),
                    ))

                    # ── Baseline forecast segment ──────────────────────────
                    if _fcast_df is not None and f"pred_{ch}_mean" in _fcast_df.columns:
                        _fc_fill = f"rgba({_r},{_g},{_b},0.08)"
                        fig_fit.add_trace(go.Scatter(
                            x=_fcast_df["date"], y=_fcast_df[f"pred_{ch}_q90"],
                            mode="lines", line=dict(width=0),
                            showlegend=False, name="fc_q90",
                        ))
                        fig_fit.add_trace(go.Scatter(
                            x=_fcast_df["date"], y=_fcast_df[f"pred_{ch}_q10"],
                            mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor=_fc_fill,
                            name="Forecast 80% CI",
                        ))
                        fig_fit.add_trace(go.Scatter(
                            x=_fcast_df["date"],
                            y=_fcast_df[f"pred_{ch}_mean"],
                            mode="lines",
                            name="Forecast (baseline)",
                            line=dict(color=color, width=1.8, dash="dash"),
                        ))

                    # ── Scenario forecast (green) ──────────────────────────
                    if _sc_df is not None and f"pred_{ch}_mean" in _sc_df.columns:
                        fig_fit.add_trace(go.Scatter(
                            x=_sc_df["date"], y=_sc_df[f"pred_{ch}_q90"],
                            mode="lines", line=dict(width=0),
                            showlegend=False, name="sc_q90",
                        ))
                        fig_fit.add_trace(go.Scatter(
                            x=_sc_df["date"], y=_sc_df[f"pred_{ch}_q10"],
                            mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor="rgba(67,160,71,0.10)",
                            name=f"Scenario 80% CI (−{exp_beta_reduction}%)",
                        ))
                        fig_fit.add_trace(go.Scatter(
                            x=_sc_df["date"],
                            y=_sc_df[f"pred_{ch}_mean"],
                            mode="lines",
                            name=f"Scenario (−{exp_beta_reduction}% transmission)",
                            line=dict(color="#43A047", width=1.8, dash="dot"),
                        ))

                    _title = ch.capitalize()
                    if _fcast_df is not None:
                        _title += f"  (+{len(_fcast_df)}d forecast)"
                    fig_fit.update_layout(
                        title=_title,
                        height=280,
                        template="plotly_white",
                        hovermode="x unified",
                        xaxis_title="Date",
                        yaxis_title=ch,
                        legend=dict(orientation="h", y=-0.28, font=dict(size=11)),
                        margin=dict(t=35, b=50, l=45, r=10),
                    )
                    col_fit.plotly_chart(fig_fit, use_container_width=True)

            # ── R_eff(t) ──────────────────────────────────────────────────
            fig_reff = go.Figure()
            # Fit credible band
            fig_reff.add_trace(go.Scatter(
                x=res_exp["date"], y=res_exp["R_eff_q90"],
                fill=None, mode="lines", line=dict(width=0),
                showlegend=False, name="R_eff 90%"))
            fig_reff.add_trace(go.Scatter(
                x=res_exp["date"], y=res_exp["R_eff_q10"],
                fill="tonexty", mode="lines", line=dict(width=0),
                fillcolor="rgba(30,136,229,0.15)", name="80% CI"))
            # Fit mean
            fig_reff.add_trace(go.Scatter(
                x=res_exp["date"], y=res_exp["R_eff_mean"],
                mode="lines", name="R_eff(t)",
                line=dict(color="#1E88E5", width=2.0)))
            # Forecast R_eff
            if _fcast_df is not None and "R_eff_mean" in _fcast_df.columns:
                fig_reff.add_trace(go.Scatter(
                    x=_fcast_df["date"], y=_fcast_df["R_eff_q90"],
                    fill=None, mode="lines", line=dict(width=0),
                    showlegend=False, name="fc_q90"))
                fig_reff.add_trace(go.Scatter(
                    x=_fcast_df["date"], y=_fcast_df["R_eff_q10"],
                    fill="tonexty", mode="lines", line=dict(width=0),
                    fillcolor="rgba(30,136,229,0.07)", name="Forecast 80% CI"))
                fig_reff.add_trace(go.Scatter(
                    x=_fcast_df["date"], y=_fcast_df["R_eff_mean"],
                    mode="lines", name="R_eff forecast",
                    line=dict(color="#1E88E5", width=1.8, dash="dash")))
            # Scenario R_eff (green)
            if _sc_df is not None and "R_eff_mean" in _sc_df.columns:
                fig_reff.add_trace(go.Scatter(
                    x=_sc_df["date"], y=_sc_df["R_eff_q90"],
                    fill=None, mode="lines", line=dict(width=0),
                    showlegend=False, name="sc_q90"))
                fig_reff.add_trace(go.Scatter(
                    x=_sc_df["date"], y=_sc_df["R_eff_q10"],
                    fill="tonexty", mode="lines", line=dict(width=0),
                    fillcolor="rgba(67,160,71,0.10)",
                    name=f"Scenario 80% CI (−{exp_beta_reduction}%)"))
                fig_reff.add_trace(go.Scatter(
                    x=_sc_df["date"], y=_sc_df["R_eff_mean"],
                    mode="lines", name=f"R_eff scenario (−{exp_beta_reduction}%)",
                    line=dict(color="#43A047", width=1.8, dash="dot")))
            fig_reff.add_hline(y=1.0, line_dash="dot", line_color="#888",
                               annotation_text="R_eff = 1")
            _reff_title = f"Effective reproduction number R_eff(t) — {model_key}"
            if _fcast_df is not None:
                _reff_title += f"  (+{len(_fcast_df)}d forecast)"
            fig_reff.update_layout(
                title=_reff_title,
                xaxis_title="Date", yaxis_title="R_eff(t)",
                height=320, template="plotly_white",
                hovermode="x unified",
                margin=dict(t=50, b=20, l=50, r=20),
            )
            st.plotly_chart(fig_reff, use_container_width=True)

            # ── Time-varying parameters (all, in a grid) ──────────────────
            param_cols_avail = [p for p in _param_names
                                if f"{p}_mean" in res_exp.columns]
            if param_cols_avail:
                st.markdown("**Time-varying parameters**")
                _PARAM_COLORS = {
                    "beta":    "#E53935",
                    "delta":   "#8E24AA",
                    "tau_i":   "#FB8C00",
                    "delta_h": "#E91E63",
                    "Q_C":     "#00897B",
                    "rho_c":   "#1565C0",
                }
                n_pcols = min(len(param_cols_avail), 2)
                p_cols  = st.columns(n_pcols)
                for pi, pn in enumerate(param_cols_avail):
                    col_p = p_cols[pi % n_pcols]
                    color = _PARAM_COLORS.get(pn, "#555")
                    fig_p = go.Figure()
                    if f"{pn}_q90" in res_exp.columns:
                        fig_p.add_trace(go.Scatter(
                            x=res_exp["date"], y=res_exp[f"{pn}_q90"],
                            fill=None, mode="lines", line=dict(width=0),
                            showlegend=False))
                        fig_p.add_trace(go.Scatter(
                            x=res_exp["date"], y=res_exp[f"{pn}_q10"],
                            fill="tonexty", mode="lines", line=dict(width=0),
                            fillcolor="rgba(128,128,128,0.12)",
                            showlegend=False))
                    fig_p.add_trace(go.Scatter(
                        x=res_exp["date"], y=res_exp[f"{pn}_mean"],
                        mode="lines", name=pn,
                        line=dict(color=color, width=1.8)))
                    fig_p.update_layout(
                        title=pn, height=230, template="plotly_white",
                        hovermode="x unified",
                        xaxis_title="Date", yaxis_title=pn,
                        margin=dict(t=30, b=20, l=40, r=10),
                        showlegend=False)
                    col_p.plotly_chart(fig_p, use_container_width=True)

            # ── ESS over time ─────────────────────────────────────────────
            fig_ess_exp = go.Figure()
            fig_ess_exp.add_trace(go.Scatter(
                x=res_exp["date"], y=res_exp["ess"],
                mode="lines", name="ESS",
                line=dict(color="#5C6BC0", width=1.4),
                fill="tozeroy", fillcolor="rgba(92,107,192,0.10)",
            ))
            fig_ess_exp.add_hline(
                y=0.45 * _N_used,
                line_dash="dot", line_color="#E53935",
                annotation_text=f"Resample threshold ({0.45 * _N_used:.0f})",
                annotation_position="top right",
            )
            fig_ess_exp.update_layout(
                title="Effective Sample Size (ESS) over time",
                height=220, template="plotly_white",
                hovermode="x unified",
                xaxis_title="Date", yaxis_title="ESS",
                margin=dict(t=40, b=20, l=50, r=20),
            )
            st.plotly_chart(fig_ess_exp, use_container_width=True)

            # ── Compartment trajectories (collapsible) ────────────────────
            with st.expander("Compartment trajectories"):
                COMP_COLORS = {
                    "S": "#1565C0", "V": "#00897B", "E": "#FDD835",
                    "A": "#FB8C00", "I": "#E53935", "H": "#8E24AA",
                    "C": "#D81B60", "R": "#43A047", "D": "#546E7A",
                }
                cols_comp = st.columns(min(len(_state_names), 3))
                for ci, sn in enumerate(_state_names):
                    col = cols_comp[ci % 3]
                    if f"{sn}_mean" not in res_exp.columns:
                        continue
                    fg = go.Figure()
                    color = COMP_COLORS.get(sn, "#888")
                    fg.add_trace(go.Scatter(
                        x=res_exp["date"], y=res_exp[f"{sn}_q90"],
                        fill=None, mode="lines", line=dict(width=0),
                        showlegend=False))
                    fg.add_trace(go.Scatter(
                        x=res_exp["date"], y=res_exp[f"{sn}_q10"],
                        fill="tonexty", mode="lines", line=dict(width=0),
                        fillcolor="rgba(128,128,128,0.15)", showlegend=False))
                    fg.add_trace(go.Scatter(
                        x=res_exp["date"], y=res_exp[f"{sn}_mean"],
                        mode="lines", name=sn,
                        line=dict(color=color, width=1.6)))
                    fg.update_layout(height=200, template="plotly_white",
                                     title=sn, showlegend=False,
                                     margin=dict(t=30, b=10, l=30, r=10),
                                     xaxis=dict(showticklabels=False))
                    col.plotly_chart(fg, use_container_width=True)

            # ── Download ──────────────────────────────────────────────────
            _dl_c1, _dl_c2, _dl_c3 = st.columns(3)
            with _dl_c1:
                csv_bytes = res_exp.to_csv(index=False).encode()
                st.download_button(
                    label="Download filter output (CSV)",
                    data=csv_bytes,
                    file_name=f"episurveil_{model_key}_output.csv",
                    mime="text/csv",
                )
            with _dl_c2:
                if _fcast_df is not None:
                    fcast_bytes = _fcast_df.to_csv(index=False).encode()
                    st.download_button(
                        label=f"Download forecast ({len(_fcast_df)}d) (CSV)",
                        data=fcast_bytes,
                        file_name=f"episurveil_{model_key}_forecast.csv",
                        mime="text/csv",
                    )
            with _dl_c3:
                if _sc_df is not None:
                    sc_bytes = _sc_df.to_csv(index=False).encode()
                    st.download_button(
                        label=f"Download scenario (−{exp_beta_reduction}%) (CSV)",
                        data=sc_bytes,
                        file_name=f"episurveil_{model_key}_scenario.csv",
                        mime="text/csv",
                    )
            st.caption(
                f"**Model:** {model_key}  |  "
                f"**States:** {_state_names}  |  "
                f"**Log-RW params:** {_param_names}  |  "
                f"**Channels:** {_obs_channels}"
            )

            # ── Feedback strip ────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 💬 Share your feedback")
            st.markdown(
                "Your input helps improve EpiSurveil. "
                "Did this run produce useful results?"
            )

            _fb_col1, _fb_col2 = st.columns([1, 2])
            with _fb_col1:
                _rating = st.feedback("thumbs", key="exp_feedback_rating")

            with _fb_col2:
                _fb_comment = st.text_area(
                    "Optional comment (model issues, feature requests, results quality …)",
                    height=80,
                    key="exp_feedback_text",
                    placeholder="e.g. 'β ceiling hit on Delta wave — need higher β_max option' …",
                    label_visibility="collapsed",
                )

            _fb_left, _fb_right = st.columns(2)
            with _fb_left:
                # Build pre-filled GitHub issue URL from rating + comment
                _rating_str = {0: "👎 Negative", 1: "👍 Positive"}.get(_rating, "Not rated")
                _model_ctx  = (f"Model: {model_key} | Disease: "
                               f"{st.session_state.get('exp_disease_preset', 'Custom')} | "
                               f"Country: {st.session_state.get('exp_country_preset', 'Custom')}")
                _issue_body = (
                    f"**Rating**: {_rating_str}%0A"
                    f"**Context**: {_model_ctx.replace(' ', '+')}%0A%0A"
                    f"**Comment**: {(_fb_comment or '(none)').replace(' ', '+').replace(chr(10), '%0A')}"
                )
                _issue_url = (
                    "https://github.com/YOUR-USERNAME/episurveil/issues/new"
                    f"?labels=feedback&title=[Feedback]+{model_key}"
                    f"&body={_issue_body}"
                )
                st.link_button(
                    "📬 Send feedback via GitHub Issue",
                    url=_issue_url,
                    help="Opens a pre-filled GitHub Issue with your rating and comment.",
                )
            with _fb_right:
                st.link_button(
                    "📖 View existing issues & discussions",
                    url="https://github.com/YOUR-USERNAME/episurveil/issues",
                )

# ===========================================================================
# TAB 11 — Live data connector
# ===========================================================================
with tab_live:
    st.subheader("Live epidemic data — real-time fetch and BPF fitting")
    st.caption(
        "Fetch the latest surveillance data from public APIs (WHO FluNet, ECDC, disease.sh) "
        "and run the BPF model directly on it, with no CSV preparation needed."
    )

    import importlib, episurveil.connectors.live_data as _lcd_mod
    importlib.reload(_lcd_mod)
    from episurveil.connectors.live_data import LIVE_SOURCES

    _live_col_l, _live_col_r = st.columns([1, 3])

    with _live_col_l:
        st.markdown("**Data source**")
        _live_source = st.selectbox(
            "Source", list(LIVE_SOURCES.keys()), key="live_source",
            help="Select which public surveillance API to fetch from.",
        )
        _src_meta = LIVE_SOURCES[_live_source]
        st.caption(_src_meta["description"])
        if _live_source == "COVID-19 (disease.sh)":
            st.warning(
                "**Data availability:** Most countries stopped mandatory COVID reporting in "
                "early 2023, so disease.sh data ends around that date. "
                "For current respiratory surveillance use **ILI Surveillance (ECDC)** "
                "or **Influenza lab-confirmed (ECDC)**, both updated weekly.",
                icon="⚠️",
            )

        _live_countries = _src_meta["countries_fn"]()
        _live_country = st.selectbox(
            "Country / region", _live_countries, key="live_country",
        )

        if _live_source == "COVID-19 (disease.sh)":
            _live_days = st.slider("Days of history", 60, 730, 180, step=30,
                                   key="live_days")
            _live_fetch_kw = {"days": _live_days}
        else:
            _live_seasons = st.slider("Seasons of history", 1, 5, 2, step=1,
                                      key="live_seasons")
            _live_fetch_kw = {"seasons": _live_seasons}

        _live_fetch_btn = st.button("Fetch data", type="primary", key="live_fetch_btn")

    with _live_col_r:
        if _live_fetch_btn:
            with st.spinner(f"Fetching {_live_source} data for {_live_country} …"):
                try:
                    _live_df = _src_meta["fetch_fn"](_live_country, **_live_fetch_kw)
                    st.session_state["live_df"]             = _live_df
                    st.session_state["live_fetched_source"]  = _live_source
                    st.session_state["live_fetched_country"] = _live_country
                    st.success(
                        f"Fetched {len(_live_df)} rows "
                        f"({_live_df['date'].min().date()} to "
                        f"{_live_df['date'].max().date()})"
                    )
                except Exception as _le:
                    st.error(f"Fetch failed: {str(_le)[:300]}")

        _live_data: "pd.DataFrame | None" = st.session_state.get("live_df")

        if _live_data is not None:
            _ld_source  = st.session_state.get("live_fetched_source",  _live_source)
            _ld_country = st.session_state.get("live_fetched_country", _live_country)
            st.markdown(f"**{_ld_source} — {_ld_country}**  "
                        f"({_live_data['date'].min().date()} to "
                        f"{_live_data['date'].max().date()}, "
                        f"{len(_live_data)} rows)")

            # ── Preview chart ─────────────────────────────────────────────
            import plotly.graph_objects as _go_live
            _live_src_meta = LIVE_SOURCES.get(_ld_source, {})
            _live_chs      = [c for c in _live_src_meta.get("obs_channels", ["cases"])
                              if c in _live_data.columns]

            _fig_live = _go_live.Figure()
            _COLORS_LIVE = ["#1976D2", "#D32F2F", "#388E3C"]
            for _ci, _ch in enumerate(_live_chs):
                _fig_live.add_trace(_go_live.Scatter(
                    x=_live_data["date"], y=_live_data[_ch],
                    name=_ch.capitalize(),
                    mode="lines",
                    line=dict(color=_COLORS_LIVE[_ci % len(_COLORS_LIVE)], width=1.5),
                ))
            _fig_live.update_layout(
                title=f"{_ld_source} — {_ld_country}",
                xaxis_title="Date", yaxis_title="Count",
                height=300, template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.30, font=dict(size=11)),
                margin=dict(t=45, b=55, l=55, r=20),
            )
            st.plotly_chart(_fig_live, use_container_width=True)

            # ── Data preview ──────────────────────────────────────────────
            with st.expander("Raw data preview (last 20 rows)", expanded=False):
                st.dataframe(
                    _live_data.tail(20).style.format(
                        {c: "{:,.1f}" for c in _live_chs}, na_rep="—"
                    ),
                    use_container_width=True,
                )

            st.download_button(
                "⬇️ Download CSV",
                data=_live_data.to_csv(index=False),
                file_name=f"live_{_ld_source.split('(')[0].strip().lower().replace(' ','_')}"
                          f"_{_ld_country.lower().replace(' ','_')}.csv",
                mime="text/csv",
                key="live_dl_csv",
            )

            # ── BPF run from live data ────────────────────────────────────
            st.markdown("---")
            st.markdown("**Run BPF on this live data**")
            _live_src_meta2 = LIVE_SOURCES.get(_ld_source, {})
            _suggested_model = _live_src_meta2.get("suggested_model", "SEIR")
            _suggested_N     = _live_src_meta2.get("suggested_N_pop", {}).get(
                _ld_country, 10_000_000
            )

            _lbpf_c1, _lbpf_c2, _lbpf_c3 = st.columns(3)
            with _lbpf_c1:
                _LIVE_MODELS = ["SIR", "SEIR", "SEIRD", "SEIRV", "SEIRHD"]
                _lbpf_model = st.selectbox(
                    "Model",
                    _LIVE_MODELS,
                    index=_LIVE_MODELS.index(_suggested_model)
                    if _suggested_model in _LIVE_MODELS else 1,
                    key="live_bpf_model",
                )
            with _lbpf_c2:
                _lbpf_N = st.number_input(
                    "Population N", value=int(_suggested_N),
                    step=100_000, min_value=100_000,
                    key="live_bpf_N",
                )
            with _lbpf_c3:
                _lbpf_particles = st.selectbox(
                    "Particles", [500, 1000, 2000, 4000], index=1,
                    key="live_bpf_particles",
                )

            _lbpf_c4, _lbpf_c5, _lbpf_c6 = st.columns(3)
            with _lbpf_c4:
                _lbpf_gamma = st.number_input(
                    "gamma (1/inf. period)", value=0.10, step=0.01,
                    min_value=0.01, max_value=1.0, key="live_bpf_gamma",
                )
            with _lbpf_c5:
                _lbpf_sigma = st.number_input(
                    "sigma (1/incub. period)", value=0.20, step=0.01,
                    min_value=0.01, max_value=1.0, key="live_bpf_sigma",
                )
            with _lbpf_c6:
                _lbpf_omega_r = st.number_input(
                    "omega_r (waning immunity)", value=0.0, step=0.002,
                    min_value=0.0, max_value=0.05, format="%.3f",
                    key="live_bpf_omega_r",
                    help="Rate R→S per day. 0 = permanent immunity. "
                         "1/180 ≈ 0.006 (6-month waning). "
                         "Use >0 for multi-wave flu/COVID fits.",
                )

            # SEIRD-specific: death rate and channel tuning
            if _lbpf_model == "SEIRD":
                with st.expander("SEIRD death-rate settings", expanded=True):
                    st.caption(
                        "**delta_init** = daily death rate per infectious person.  "
                        "IFR ≈ delta / (gamma + delta).  "
                        "Rule of thumb: delta ≈ IFR × gamma.  "
                        "Omicron (IFR ~0.1%): delta ≈ 0.0001.  "
                        "Pre-Omicron (IFR ~1%): delta ≈ 0.001.  "
                        "Lower **deaths weight** if death reporting is noisy or delayed."
                    )
                    _lbpf_dc1, _lbpf_dc2, _lbpf_dc3 = st.columns(3)
                    with _lbpf_dc1:
                        _lbpf_delta_init = st.number_input(
                            "delta_init", value=0.0001, step=0.0001,
                            min_value=0.00001, max_value=0.05,
                            format="%.5f", key="live_bpf_delta_init",
                            help="Daily death rate per infectious person. IFR ≈ delta/gamma.",
                        )
                        _implied_ifr = _lbpf_delta_init / (_lbpf_gamma + _lbpf_delta_init) * 100
                        st.caption(f"Implied IFR ≈ {_implied_ifr:.2f}%")
                    with _lbpf_dc2:
                        _lbpf_deaths_weight = st.slider(
                            "Deaths channel weight", 0.1, 1.0, 0.4, step=0.1,
                            key="live_bpf_deaths_weight",
                            help="Tempering weight for the deaths likelihood (1.0 = full trust).",
                        )
                    with _lbpf_dc3:
                        _lbpf_phi_deaths = st.select_slider(
                            "phi_deaths (NegBin)", [3, 5, 8, 10, 20, 50], value=5,
                            key="live_bpf_phi_deaths",
                            help="NegBin dispersion for deaths. Lower = wider, more tolerant of scale mismatch.",
                        )
            elif _lbpf_model == "SEIRV":
                with st.expander("SEIRV vaccination settings", expanded=True):
                    st.caption(
                        "**SEIRV** adds a Vaccinated (V) compartment. "
                        "Set **nu_const** > 0 to simulate ongoing vaccination pressure even without "
                        "external data. **epsilon** = vaccine efficacy (fraction of infectability blocked). "
                        "**omega_v** = V→S waning rate (1/180 ≈ 6-month waning)."
                    )
                    _lbpf_vc1, _lbpf_vc2, _lbpf_vc3 = st.columns(3)
                    with _lbpf_vc1:
                        _lbpf_epsilon = st.slider(
                            "Vaccine efficacy (epsilon)", 0.0, 1.0, 0.85, step=0.05,
                            key="live_bpf_epsilon",
                            help="Fraction reduction in infectability for vaccinated individuals.",
                        )
                    with _lbpf_vc2:
                        _lbpf_omega_v = st.number_input(
                            "omega_v (V→S waning)", value=1/180, step=0.001,
                            min_value=0.0, max_value=0.05, format="%.4f",
                            key="live_bpf_omega_v",
                            help="Rate at which vaccinated individuals return to susceptible. "
                                 "1/180 ≈ 0.0056 (6-month waning).",
                        )
                    with _lbpf_vc3:
                        _lbpf_nu_const = st.number_input(
                            "nu_const (daily vacc. rate)", value=0.0, step=0.001,
                            min_value=0.0, max_value=0.05, format="%.3f",
                            key="live_bpf_nu_const",
                            help="Constant fraction of S vaccinated per day. "
                                 "0 = no vaccination (SEIRV reduces to SEIR with waning V).",
                        )
                _lbpf_delta_init     = 0.005
                _lbpf_deaths_weight  = 1.0
                _lbpf_phi_deaths     = 10
                _lbpf_gamma_h        = 1 / 12
                _lbpf_los            = 8.0
                _lbpf_tau_i_init     = 0.005
                _lbpf_delta_h_init   = 0.02
                _lbpf_hosp_weight    = 0.6
                _lbpf_phi_hosp       = 15
            elif _lbpf_model == "SEIRHD":
                with st.expander("SEIRHD hospital & death settings", expanded=True):
                    st.caption(
                        "**SEIRHD** adds a Hospital (H) compartment. Fits cases + hosp occupancy + deaths. "
                        "If your data source lacks a **hosp** column the hospital likelihood is skipped "
                        "automatically (cases + deaths only). "
                        "LOS = average length of hospital stay (days, used to estimate occupancy)."
                    )
                    _lbpf_hc1, _lbpf_hc2, _lbpf_hc3 = st.columns(3)
                    with _lbpf_hc1:
                        _lbpf_gamma_h = st.number_input(
                            "gamma_h (1/hosp stay)", value=1/12, step=0.01,
                            min_value=0.01, max_value=1.0, format="%.3f",
                            key="live_bpf_gamma_h",
                            help="Recovery rate from hospital ward. 1/12 ≈ 12-day stay.",
                        )
                        _lbpf_los = st.number_input(
                            "LOS (days)", value=8.0, step=1.0,
                            min_value=1.0, max_value=30.0, format="%.0f",
                            key="live_bpf_los",
                            help="Mean hospital length of stay (scales hosp occupancy prediction).",
                        )
                    with _lbpf_hc2:
                        _lbpf_tau_i_init = st.number_input(
                            "tau_i_init (hosp. rate)", value=0.005, step=0.001,
                            min_value=0.0001, max_value=0.10, format="%.4f",
                            key="live_bpf_tau_i_init",
                            help="Initial daily hospitalisation rate per infectious person.",
                        )
                        _lbpf_delta_h_init = st.number_input(
                            "delta_h_init (in-hosp CFR)", value=0.02, step=0.005,
                            min_value=0.001, max_value=0.15, format="%.3f",
                            key="live_bpf_delta_h_init",
                            help="Initial daily in-hospital death rate.",
                        )
                    with _lbpf_hc3:
                        _lbpf_hosp_weight = st.slider(
                            "Hosp channel weight", 0.1, 1.0, 0.6, step=0.1,
                            key="live_bpf_hosp_weight",
                        )
                        _lbpf_deaths_weight = st.slider(
                            "Deaths channel weight", 0.1, 1.0, 0.4, step=0.1,
                            key="live_bpf_deaths_weight_hd",
                        )
                        _lbpf_phi_hosp   = st.select_slider(
                            "phi_hosp",  [5, 10, 15, 20, 50], value=15,
                            key="live_bpf_phi_hosp",
                        )
                        _lbpf_phi_deaths = st.select_slider(
                            "phi_deaths", [3, 5, 8, 10, 20, 50], value=5,
                            key="live_bpf_phi_deaths_hd",
                        )
                _lbpf_delta_init   = 0.005
                _lbpf_delta_h_init = _lbpf_delta_h_init  # already set above
            else:  # SIR / SEIR
                _lbpf_delta_init     = 0.005
                _lbpf_deaths_weight  = 1.0
                _lbpf_phi_deaths     = 10
                _lbpf_epsilon        = 0.85
                _lbpf_omega_v        = 1 / 180
                _lbpf_nu_const       = 0.0
                _lbpf_gamma_h        = 1 / 12
                _lbpf_los            = 8.0
                _lbpf_tau_i_init     = 0.005
                _lbpf_delta_h_init   = 0.02
                _lbpf_hosp_weight    = 0.6
                _lbpf_phi_hosp       = 15

            if _lbpf_model != "SEIRV":
                _lbpf_epsilon    = 0.85
                _lbpf_omega_v    = 1 / 180
                _lbpf_nu_const   = 0.0
            if _lbpf_model not in ("SEIRHD",):
                _lbpf_gamma_h      = 1 / 12
                _lbpf_los          = 8.0
                _lbpf_tau_i_init   = 0.005
                _lbpf_delta_h_init = 0.02
                _lbpf_hosp_weight  = 0.6
                _lbpf_phi_hosp     = 15

            _lbpf_run = st.button("Run BPF", type="primary", key="live_bpf_run")

            if _lbpf_run:
                from episurveil.inference.bpf_generic import run_bpf as _lbpf_fn
                if _lbpf_model == "SIR":
                    from episurveil.models.sir import SIRModel
                    _lbpf_mdl = SIRModel(
                        N=_lbpf_N, gamma=_lbpf_gamma,
                        omega_r=float(_lbpf_omega_r),
                    )
                    _lbpf_phi_dict = {}
                    _lbpf_cw_dict  = {}
                    _lbpf_exog_df  = None
                elif _lbpf_model == "SEIR":
                    from episurveil.models.seir import SEIRModel
                    _lbpf_mdl = SEIRModel(
                        N=_lbpf_N, sigma=_lbpf_sigma, gamma=_lbpf_gamma,
                        omega_r=float(_lbpf_omega_r),
                    )
                    _lbpf_phi_dict = {}
                    _lbpf_cw_dict  = {}
                    _lbpf_exog_df  = None
                elif _lbpf_model == "SEIRV":
                    from episurveil.models.seirv import SEIRVModel
                    _lbpf_mdl = SEIRVModel(
                        N=_lbpf_N, sigma=_lbpf_sigma, gamma=_lbpf_gamma,
                        epsilon=float(_lbpf_epsilon),
                        omega_v=float(_lbpf_omega_v),
                    )
                    _lbpf_phi_dict = {}
                    _lbpf_cw_dict  = {}
                    _lbpf_exog_df  = (
                        pd.DataFrame({
                            "date": _live_data["date"],
                            "nu":   float(_lbpf_nu_const),
                        })
                        if float(_lbpf_nu_const) > 0 else None
                    )
                elif _lbpf_model == "SEIRHD":
                    from episurveil.models.seirhd import SEIRHDModel
                    _lbpf_mdl = SEIRHDModel(
                        N=_lbpf_N, sigma=_lbpf_sigma, gamma_i=_lbpf_gamma,
                        gamma_h=float(_lbpf_gamma_h),
                        LOS=float(_lbpf_los),
                        tau_i_init=float(_lbpf_tau_i_init),
                        delta_h_init=float(_lbpf_delta_h_init),
                        phi_hosp=int(_lbpf_phi_hosp),
                        phi_deaths=int(_lbpf_phi_deaths),
                    )
                    _lbpf_phi_dict = {
                        "cases": 50,
                        "hosp":   int(_lbpf_phi_hosp),
                        "deaths": int(_lbpf_phi_deaths),
                    }
                    _lbpf_cw_dict  = {
                        "cases": 1.0,
                        "hosp":  float(_lbpf_hosp_weight),
                        "deaths": float(_lbpf_deaths_weight),
                    }
                    _lbpf_exog_df  = None
                else:  # SEIRD
                    from episurveil.models.seird import SEIRDModel
                    _lbpf_mdl = SEIRDModel(
                        N=_lbpf_N, sigma=_lbpf_sigma, gamma=_lbpf_gamma,
                        delta_init=float(_lbpf_delta_init),
                        phi_deaths=int(_lbpf_phi_deaths),
                    )
                    _lbpf_phi_dict = {"cases": 50, "deaths": int(_lbpf_phi_deaths)}
                    _lbpf_cw_dict  = {"cases": 1.0, "deaths": float(_lbpf_deaths_weight)}
                    _lbpf_exog_df  = None

                with st.spinner(f"Running BPF ({_lbpf_particles} particles) …"):
                    try:
                        _lbpf_res, _lbpf_ptcls = _lbpf_fn(
                            _lbpf_mdl, _live_data,
                            N=int(_lbpf_particles),
                            phi=_lbpf_phi_dict if _lbpf_phi_dict else None,
                            channel_weights=_lbpf_cw_dict if _lbpf_cw_dict else None,
                            exog_df=_lbpf_exog_df,
                            return_particles=True,
                        )
                        st.session_state["live_bpf_res"]     = _lbpf_res
                        st.session_state["live_bpf_ptcls"]   = _lbpf_ptcls
                        st.session_state["live_bpf_mdl"]     = _lbpf_mdl
                        st.session_state["live_bpf_last_dt"] = _live_data["date"].iloc[-1]
                        ess_pct = _lbpf_res["ess"].mean() / _lbpf_particles * 100
                        _last   = _lbpf_res.iloc[-1]
                        _lpg    = float(_last["P_growing"])
                        _append_alert_log([{
                            "timestamp":    pd.Timestamp.now().isoformat(timespec="seconds"),
                            "source":       _ld_source,
                            "country":      _ld_country,
                            "last_obs_date":str(_live_data["date"].iloc[-1].date()),
                            "R_eff_mean":   round(float(_last["R_eff_mean"]), 3),
                            "R_eff_q10":    round(float(_last["R_eff_q10"]),  3),
                            "R_eff_q90":    round(float(_last["R_eff_q90"]),  3),
                            "P_growing":    round(_lpg, 3),
                            "status":       ("Growing"   if _lpg >= alert_threshold else
                                             "Uncertain" if _lpg >= 0.50 else "Declining"),
                            "model":        _lbpf_model,
                            "N_particles":  int(_lbpf_particles),
                        }])
                        st.success(
                            f"BPF complete. Mean ESS = {ess_pct:.0f}%  |  "
                            f"Final R_eff = {_lbpf_res['R_eff_mean'].iloc[-1]:.2f}  |  "
                            f"P(growing) = {_lbpf_res['P_growing'].iloc[-1]:.0%}"
                        )
                    except Exception as _lbpf_err:
                        st.error(f"BPF failed: {str(_lbpf_err)[:300]}")

            _lbpf_res: "pd.DataFrame | None" = st.session_state.get("live_bpf_res")
            if _lbpf_res is not None:
                import plotly.graph_objects as _go2

                # Observed vs predicted
                _live_chs2 = [c for c in _live_src_meta2.get("obs_channels", ["cases"])
                              if f"obs_{c}" in _lbpf_res.columns]
                for _ch2 in _live_chs2:
                    _fig_lbpf = _go2.Figure()
                    _fig_lbpf.add_trace(_go2.Scatter(
                        x=_lbpf_res["date"], y=_lbpf_res[f"obs_{_ch2}"],
                        name="Observed", mode="markers",
                        marker=dict(size=4, color="#333"),
                    ))
                    _fig_lbpf.add_trace(_go2.Scatter(
                        x=_lbpf_res["date"], y=_lbpf_res[f"pred_{_ch2}_mean"],
                        name="Predicted (mean)", mode="lines",
                        line=dict(color="#1976D2", width=2),
                    ))
                    if f"pred_{_ch2}_q10" in _lbpf_res.columns:
                        _fig_lbpf.add_trace(_go2.Scatter(
                            x=pd.concat([_lbpf_res["date"], _lbpf_res["date"].iloc[::-1]]),
                            y=pd.concat([_lbpf_res[f"pred_{_ch2}_q90"],
                                         _lbpf_res[f"pred_{_ch2}_q10"].iloc[::-1]]),
                            fill="toself", fillcolor="rgba(25,118,210,0.15)",
                            line=dict(color="rgba(0,0,0,0)"), name="80% CI",
                        ))
                    _fig_lbpf.update_layout(
                        title=f"Live BPF — {_ch2.capitalize()} fit",
                        xaxis_title="Date", yaxis_title=_ch2.capitalize(),
                        height=320, template="plotly_white",
                        hovermode="x unified",
                        legend=dict(orientation="h", y=-0.30, font=dict(size=11)),
                        margin=dict(t=45, b=55, l=55, r=20),
                    )
                    st.plotly_chart(_fig_lbpf, use_container_width=True)

                # E and I compartment trajectories with 80% CI
                _EI_SPECS = [
                    ("E", "Exposed (E)",    "#F57C00", "rgba(245,124,0,0.15)"),
                    ("I", "Infectious (I)", "#C62828", "rgba(198,40,40,0.15)"),
                ]
                _ei_available = [
                    (col, label, col_color, ci_color)
                    for col, label, col_color, ci_color in _EI_SPECS
                    if f"{col}_mean" in _lbpf_res.columns
                ]
                if _ei_available:
                    with st.expander("Latent compartments — Exposed & Infectious (80% CI)",
                                     expanded=True):
                        for _ec, _el, _ecolor, _efill in _ei_available:
                            _fig_ei = _go2.Figure()
                            # 80% CI band
                            if f"{_ec}_q10" in _lbpf_res.columns:
                                _fig_ei.add_trace(_go2.Scatter(
                                    x=pd.concat([_lbpf_res["date"],
                                                 _lbpf_res["date"].iloc[::-1]]),
                                    y=pd.concat([_lbpf_res[f"{_ec}_q90"],
                                                 _lbpf_res[f"{_ec}_q10"].iloc[::-1]]),
                                    fill="toself", fillcolor=_efill,
                                    line=dict(color="rgba(0,0,0,0)"),
                                    name="80% CI", showlegend=True,
                                ))
                            # Posterior mean
                            _fig_ei.add_trace(_go2.Scatter(
                                x=_lbpf_res["date"],
                                y=_lbpf_res[f"{_ec}_mean"],
                                name=f"{_el} (mean)", mode="lines",
                                line=dict(color=_ecolor, width=2),
                            ))
                            _fig_ei.update_layout(
                                title=f"Live BPF — {_el} compartment",
                                xaxis_title="Date",
                                yaxis_title="Individuals",
                                height=280, template="plotly_white",
                                hovermode="x unified",
                                legend=dict(orientation="h", y=-0.32,
                                            font=dict(size=11)),
                                margin=dict(t=45, b=60, l=65, r=20),
                            )
                            st.plotly_chart(_fig_ei, use_container_width=True)

                # R_eff + P_growing
                _fig_lr = _go2.Figure()
                _fig_lr.add_trace(_go2.Scatter(
                    x=_lbpf_res["date"], y=_lbpf_res["R_eff_mean"],
                    name="R_eff", mode="lines",
                    line=dict(color="#7B1FA2", width=2),
                ))
                _fig_lr.add_hline(y=1.0, line_dash="dash",
                                  line_color="#e53935", opacity=0.6)
                _fig_lr.update_layout(
                    title="Effective reproduction number (live BPF)",
                    xaxis_title="Date", yaxis_title="R_eff",
                    height=280, template="plotly_white",
                    hovermode="x unified",
                    margin=dict(t=45, b=55, l=55, r=20),
                )
                st.plotly_chart(_fig_lr, use_container_width=True)

                _pg_now = float(_lbpf_res["P_growing"].iloc[-1])
                _color  = ("red" if _pg_now >= 0.80 else
                           "orange" if _pg_now >= 0.50 else "green")
                st.markdown(
                    f"**Current P(R_eff > 1) = :{_color}[{_pg_now:.0%}]**  "
                    f"({'Growing' if _pg_now >= 0.80 else 'Uncertain' if _pg_now >= 0.50 else 'Declining'})"
                )

                # Quick 14-day forecast
                if st.button("Quick 14-day forecast", key="live_quick_fcast"):
                    from episurveil.inference.bpf_generic import forecast_bpf as _live_fcast_fn
                    _l_ptcls = st.session_state.get("live_bpf_ptcls")
                    _l_mdl   = st.session_state.get("live_bpf_mdl")
                    _l_dt    = st.session_state.get("live_bpf_last_dt")
                    if _l_ptcls is not None and _l_mdl is not None:
                        _lfc = _live_fcast_fn(_l_mdl, _l_ptcls, horizon=14,
                                              start_date=_l_dt)
                        st.session_state["live_forecast"] = _lfc

                _lfc_df: "pd.DataFrame | None" = st.session_state.get("live_forecast")
                if _lfc_df is not None:
                    _live_chs3 = [c for c in _live_src_meta2.get("obs_channels", ["cases"])
                                  if f"pred_{c}_mean" in _lfc_df.columns]
                    for _ch3 in _live_chs3:
                        _fig_lfc = _go2.Figure()
                        # Historical tail
                        _tail = _lbpf_res.iloc[-30:]
                        _fig_lfc.add_trace(_go2.Scatter(
                            x=_tail["date"], y=_tail[f"obs_{_ch3}"],
                            name="Observed (last 30d)", mode="markers",
                            marker=dict(size=4, color="#333"),
                        ))
                        _fig_lfc.add_trace(_go2.Scatter(
                            x=_tail["date"], y=_tail[f"pred_{_ch3}_mean"],
                            name="Fitted (BPF)", mode="lines",
                            line=dict(color="#1976D2", width=1.5, dash="dot"),
                        ))
                        # Forecast
                        _fig_lfc.add_trace(_go2.Scatter(
                            x=_lfc_df["date"], y=_lfc_df[f"pred_{_ch3}_mean"],
                            name="14-day forecast", mode="lines",
                            line=dict(color="#e65100", width=2),
                        ))
                        if f"pred_{_ch3}_q10" in _lfc_df.columns:
                            _fig_lfc.add_trace(_go2.Scatter(
                                x=pd.concat([_lfc_df["date"], _lfc_df["date"].iloc[::-1]]),
                                y=pd.concat([_lfc_df[f"pred_{_ch3}_q90"],
                                             _lfc_df[f"pred_{_ch3}_q10"].iloc[::-1]]),
                                fill="toself",
                                fillcolor="rgba(230,81,0,0.15)",
                                line=dict(color="rgba(0,0,0,0)"), name="80% forecast CI",
                            ))
                        _fig_lfc.update_layout(
                            title=f"Live 14-day {_ch3} forecast",
                            xaxis_title="Date", yaxis_title=_ch3.capitalize(),
                            height=320, template="plotly_white",
                            hovermode="x unified",
                            legend=dict(orientation="h", y=-0.30, font=dict(size=11)),
                            margin=dict(t=45, b=55, l=55, r=20),
                        )
                        st.plotly_chart(_fig_lfc, use_container_width=True)

# ===========================================================================
# TAB 12 — Country comparison
# ===========================================================================
with tab_geo:
    st.subheader("Country comparison — R_eff & epidemic status across countries")
    st.caption(
        "Fetch the same surveillance source for multiple countries, run the BPF on each, "
        "and compare R_eff trajectories and current alert status side by side."
    )

    import importlib, episurveil.connectors.live_data as _lcd_geo
    importlib.reload(_lcd_geo)
    from episurveil.connectors.live_data import LIVE_SOURCES as _GEO_SOURCES
    import plotly.graph_objects as _go_geo

    # ── Settings ─────────────────────────────────────────────────────────────
    _geo_c1, _geo_c2 = st.columns([1, 2])
    with _geo_c1:
        _geo_source = st.selectbox(
            "Surveillance source",
            list(_GEO_SOURCES.keys()),
            key="geo_source",
            index=2,                       # default: ILI Surveillance (ECDC)
        )
        _geo_meta    = _GEO_SOURCES[_geo_source]
        _geo_all_ctr = _geo_meta["countries_fn"]()

    with _geo_c2:
        _geo_countries = st.multiselect(
            "Countries to compare (max 10)",
            _geo_all_ctr,
            default=[c for c in ["France", "Germany", "Italy", "Spain", "Poland"]
                     if c in _geo_all_ctr],
            max_selections=10,
            key="geo_countries",
        )

    # Fetch settings
    _geo_fc1, _geo_fc2 = st.columns(2)
    with _geo_fc1:
        if _geo_source == "COVID-19 (disease.sh)":
            _geo_seasons_or_days = st.slider("Days of history", 60, 730, 180,
                                             step=30, key="geo_days")
            _geo_fetch_kw = {"days": _geo_seasons_or_days}
        else:
            _geo_seasons_or_days = st.slider("Seasons of history", 1, 4, 2,
                                             key="geo_seasons")
            _geo_fetch_kw = {"seasons": _geo_seasons_or_days}
    with _geo_fc2:
        _geo_alert_thr = st.slider(
            "Alert threshold P(R_eff > 1)", 0.5, 0.95, 0.80, step=0.05,
            key="geo_alert_thr",
        )

    st.markdown("**BPF model parameters** (shared across all countries)")
    _gp1, _gp2, _gp3, _gp4 = st.columns(4)
    with _gp1:
        _geo_model = st.selectbox("Model", ["SEIR", "SIR", "SEIRD", "SEIRV", "SEIRHD"],
                                  key="geo_model")
    with _gp2:
        _geo_gamma = st.number_input("gamma", value=0.10, step=0.01,
                                     min_value=0.01, max_value=1.0,
                                     key="geo_gamma")
    with _gp3:
        _geo_sigma = st.number_input("sigma", value=0.20, step=0.01,
                                     min_value=0.01, max_value=1.0,
                                     key="geo_sigma")
    with _gp4:
        _geo_omega_r = st.number_input("omega_r (waning)", value=0.006,
                                       step=0.001, min_value=0.0, max_value=0.05,
                                       format="%.3f", key="geo_omega_r")

    _gp5, _gp6 = st.columns(2)
    with _gp5:
        _geo_particles = st.selectbox("Particles", [500, 1000, 2000],
                                      index=1, key="geo_particles")
    with _gp6:
        _geo_phi = st.select_slider("phi (NegBin)", [10, 20, 50, 100],
                                    value=50, key="geo_phi")

    _geo_run_btn = st.button(
        "Run comparison", type="primary", key="geo_run_btn",
        disabled=len(_geo_countries) == 0,
    )

    # ── Run BPF per country ───────────────────────────────────────────────────
    if _geo_run_btn and _geo_countries:
        from episurveil.inference.bpf_generic import run_bpf as _geo_bpf

        _geo_results   = {}
        _geo_errors    = {}
        _progress_bar  = st.progress(0, text="Starting…")

        for _gi, _gc in enumerate(_geo_countries):
            _progress_bar.progress(
                _gi / len(_geo_countries),
                text=f"Fetching & fitting {_gc} ({_gi+1}/{len(_geo_countries)})…"
            )
            try:
                # Fetch
                _geo_df = _geo_meta["fetch_fn"](_gc, **_geo_fetch_kw)

                # Build model
                _geo_N = _geo_meta.get("suggested_N_pop", {}).get(_gc, 10_000_000)
                if _geo_model == "SIR":
                    from episurveil.models.sir import SIRModel
                    _gmdl = SIRModel(N=_geo_N, gamma=_geo_gamma,
                                     omega_r=float(_geo_omega_r))
                elif _geo_model == "SEIR":
                    from episurveil.models.seir import SEIRModel
                    _gmdl = SEIRModel(N=_geo_N, sigma=_geo_sigma,
                                      gamma=_geo_gamma,
                                      omega_r=float(_geo_omega_r))
                elif _geo_model == "SEIRV":
                    from episurveil.models.seirv import SEIRVModel
                    _gmdl = SEIRVModel(N=_geo_N, sigma=_geo_sigma,
                                       gamma=_geo_gamma)
                elif _geo_model == "SEIRHD":
                    from episurveil.models.seirhd import SEIRHDModel
                    _gmdl = SEIRHDModel(N=_geo_N, sigma=_geo_sigma,
                                        gamma_i=_geo_gamma)
                else:  # SEIRD
                    from episurveil.models.seird import SEIRDModel
                    _gmdl = SEIRDModel(N=_geo_N, sigma=_geo_sigma,
                                       gamma=_geo_gamma)

                _geo_phi_dict = (
                    {"cases": int(_geo_phi), "hosp": int(_geo_phi),
                     "deaths": int(_geo_phi)}
                    if _geo_model == "SEIRHD"
                    else {"cases": int(_geo_phi)}
                )
                _geo_res = _geo_bpf(
                    _gmdl, _geo_df,
                    N=int(_geo_particles),
                    phi=_geo_phi_dict,
                    progress=False,
                )
                _geo_results[_gc] = _geo_res

            except Exception as _ge:
                _geo_errors[_gc] = str(_ge)[:120]

        _progress_bar.progress(1.0, text="Done.")
        st.session_state["geo_results"]    = _geo_results
        st.session_state["geo_errors"]     = _geo_errors
        st.session_state["geo_source_lbl"] = _geo_source

        # ── persist to alert log ──────────────────────────────────────────
        _now = pd.Timestamp.now().isoformat(timespec="seconds")
        _log_rows = []
        for _gc, _gres in _geo_results.items():
            _last = _gres.iloc[-1]
            _lpg  = float(_last["P_growing"])
            _log_rows.append({
                "timestamp":    _now,
                "source":       _geo_source,
                "country":      _gc,
                "last_obs_date":str(_gres["date"].iloc[-1].date()),
                "R_eff_mean":   round(float(_last["R_eff_mean"]), 3),
                "R_eff_q10":    round(float(_last["R_eff_q10"]),  3),
                "R_eff_q90":    round(float(_last["R_eff_q90"]),  3),
                "P_growing":    round(_lpg, 3),
                "status":       ("Growing"   if _lpg >= _geo_alert_thr else
                                 "Uncertain" if _lpg >= 0.50 else "Declining"),
                "model":        _geo_model,
                "N_particles":  int(_geo_particles),
            })
        if _log_rows:
            _append_alert_log(_log_rows)

        if _geo_errors:
            for _ec, _em in _geo_errors.items():
                st.warning(f"{_ec}: {_em}")

    # ── Display results ───────────────────────────────────────────────────────
    _geo_results: dict = st.session_state.get("geo_results", {})

    if _geo_results:
        _geo_src_lbl = st.session_state.get("geo_source_lbl", _geo_source)
        _PALETTE = [
            "#1976D2", "#D32F2F", "#388E3C", "#F57C00", "#7B1FA2",
            "#0097A7", "#C2185B", "#FBC02D", "#5D4037", "#455A64",
        ]

        # ── 1. Summary table ─────────────────────────────────────────────────
        st.markdown("### Current epidemic status")
        _tbl_rows = []
        for _ci, (_gc, _gres) in enumerate(_geo_results.items()):
            _pg   = float(_gres["P_growing"].iloc[-1])
            _reff = float(_gres["R_eff_mean"].iloc[-1])
            _r10  = float(_gres["R_eff_q10"].iloc[-1])
            _r90  = float(_gres["R_eff_q90"].iloc[-1])
            _last = _gres["date"].iloc[-1].date()
            _ess  = float(_gres["ess"].mean()) / int(_geo_particles) * 100

            _status = ("🔴 Growing"   if _pg >= _geo_alert_thr else
                       "🟡 Uncertain" if _pg >= 0.50 else
                       "🟢 Declining")
            _tbl_rows.append({
                "Country":   _gc,
                "Status":    _status,
                "R_eff":     f"{_reff:.2f}",
                "80% CI":    f"[{_r10:.2f}, {_r90:.2f}]",
                "P(growing)":f"{_pg:.0%}",
                "ESS %":     f"{_ess:.0f}%",
                "Last data": str(_last),
            })
        st.dataframe(
            pd.DataFrame(_tbl_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("---")

        # ── 2. R_eff comparison chart ─────────────────────────────────────────
        st.markdown("### R_eff trajectories")
        _fig_reff = _go_geo.Figure()
        _fig_reff.add_hline(y=1.0, line_dash="dash",
                            line_color="#e53935", opacity=0.5,
                            annotation_text="R_eff = 1",
                            annotation_position="bottom right")
        for _ci, (_gc, _gres) in enumerate(_geo_results.items()):
            _col = _PALETTE[_ci % len(_PALETTE)]
            # 80% CI band
            _ci_fill = (f"rgba({int(_col[1:3],16)},"
                        f"{int(_col[3:5],16)},"
                        f"{int(_col[5:7],16)},0.12)")
            _fig_reff.add_trace(_go_geo.Scatter(
                x=pd.concat([_gres["date"], _gres["date"].iloc[::-1]]),
                y=pd.concat([_gres["R_eff_q90"], _gres["R_eff_q10"].iloc[::-1]]),
                fill="toself",
                fillcolor=_ci_fill,
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{_gc} 80% CI", showlegend=False,
            ))
            # Posterior mean
            _fig_reff.add_trace(_go_geo.Scatter(
                x=_gres["date"], y=_gres["R_eff_mean"],
                name=_gc, mode="lines",
                line=dict(color=_col, width=2),
            ))
        _fig_reff.update_layout(
            title=f"R_eff — {_geo_src_lbl}",
            xaxis_title="Date", yaxis_title="R_eff",
            height=380, template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
            margin=dict(t=50, b=70, l=55, r=20),
        )
        st.plotly_chart(_fig_reff, use_container_width=True)

        # ── 3. P_growing comparison chart ────────────────────────────────────
        st.markdown("### P(R_eff > 1) — epidemic growth probability")
        _fig_pg = _go_geo.Figure()
        _fig_pg.add_hline(y=_geo_alert_thr, line_dash="dash",
                          line_color="#e53935", opacity=0.6,
                          annotation_text=f"Alert threshold {_geo_alert_thr:.0%}",
                          annotation_position="bottom right")
        for _ci, (_gc, _gres) in enumerate(_geo_results.items()):
            _col = _PALETTE[_ci % len(_PALETTE)]
            _fig_pg.add_trace(_go_geo.Scatter(
                x=_gres["date"], y=_gres["P_growing"],
                name=_gc, mode="lines",
                line=dict(color=_col, width=2),
            ))
        _fig_pg.update_layout(
            title=f"P(R_eff > 1) — {_geo_src_lbl}",
            xaxis_title="Date", yaxis_title="Probability",
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            height=340, template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
            margin=dict(t=50, b=70, l=55, r=20),
        )
        st.plotly_chart(_fig_pg, use_container_width=True)

        # ── 4. Cases fit per country (collapsible) ───────────────────────────
        with st.expander("Observed vs predicted — per country", expanded=False):
            for _ci, (_gc, _gres) in enumerate(_geo_results.items()):
                _col = _PALETTE[_ci % len(_PALETTE)]
                _fill_col = (
                    "rgba(" +
                    str(int(_col[1:3], 16)) + "," +
                    str(int(_col[3:5], 16)) + "," +
                    str(int(_col[5:7], 16)) + ",0.15)"
                )
                _obs_ch = [c.replace("obs_", "") for c in _gres.columns
                           if c.startswith("obs_")]
                for _ch in _obs_ch:
                    _fig_fit = _go_geo.Figure()
                    _fig_fit.add_trace(_go_geo.Scatter(
                        x=_gres["date"], y=_gres[f"obs_{_ch}"],
                        name="Observed", mode="markers",
                        marker=dict(size=4, color="#333"),
                    ))
                    _fig_fit.add_trace(_go_geo.Scatter(
                        x=_gres["date"], y=_gres[f"pred_{_ch}_mean"],
                        name="Predicted", mode="lines",
                        line=dict(color=_col, width=2),
                    ))
                    if f"pred_{_ch}_q10" in _gres.columns:
                        _fig_fit.add_trace(_go_geo.Scatter(
                            x=pd.concat([_gres["date"],
                                         _gres["date"].iloc[::-1]]),
                            y=pd.concat([_gres[f"pred_{_ch}_q90"],
                                         _gres[f"pred_{_ch}_q10"].iloc[::-1]]),
                            fill="toself", fillcolor=_fill_col,
                            line=dict(color="rgba(0,0,0,0)"),
                            name="80% CI",
                        ))
                    _fig_fit.update_layout(
                        title=f"{_gc} — {_ch.capitalize()} fit",
                        xaxis_title="Date",
                        yaxis_title=_ch.capitalize(),
                        height=260, template="plotly_white",
                        hovermode="x unified",
                        legend=dict(orientation="h", y=-0.32,
                                    font=dict(size=10)),
                        margin=dict(t=40, b=60, l=55, r=20),
                    )
                    st.plotly_chart(_fig_fit, use_container_width=True)

        # ── 5. CSV export ─────────────────────────────────────────────────────
        st.markdown("---")
        _export_rows = []
        for _gc, _gres in _geo_results.items():
            _last = _gres.iloc[-1]
            _export_rows.append({
                "country":   _gc,
                "source":    _geo_src_lbl,
                "date":      _last["date"].date(),
                "R_eff":     round(float(_last["R_eff_mean"]), 3),
                "R_eff_q10": round(float(_last["R_eff_q10"]), 3),
                "R_eff_q90": round(float(_last["R_eff_q90"]), 3),
                "P_growing": round(float(_last["P_growing"]), 3),
            })
        _export_df = pd.DataFrame(_export_rows)
        st.download_button(
            "⬇️ Download summary CSV",
            data=_export_df.to_csv(index=False),
            file_name=f"epi_comparison_{_geo_src_lbl.split('(')[0].strip().lower().replace(' ','_')}.csv",
            mime="text/csv",
            key="geo_dl_csv",
        )

# ===========================================================================
# TAB 13 — Alert history
# ===========================================================================
with tab_log:
    st.subheader("Alert history — longitudinal surveillance log")
    st.caption(
        "Every BPF run (Live Data tab or Country Comparison tab) appends a row here. "
        "Tracks how R_eff and epidemic status evolve across sessions."
    )

    import plotly.graph_objects as _go_log

    _log_df = _read_alert_log()

    if _log_df.empty:
        st.info(
            "No entries yet. Run the BPF in the **Live data** or **Country comparison** "
            "tab to start logging surveillance results.",
            icon="ℹ️",
        )
    else:
        # ── Controls ──────────────────────────────────────────────────────────
        _log_c1, _log_c2, _log_c3 = st.columns(3)
        with _log_c1:
            _log_sources = ["All"] + sorted(_log_df["source"].unique())
            _log_src_sel = st.selectbox("Filter source", _log_sources,
                                        key="log_src_sel")
        with _log_c2:
            _log_countries_all = sorted(_log_df["country"].unique())
            _log_ctr_sel = st.multiselect("Filter countries",
                                          _log_countries_all,
                                          default=_log_countries_all[:5],
                                          key="log_ctr_sel")
        with _log_c3:
            _log_n = st.slider("Show last N runs", 10, 200, 50, step=10,
                               key="log_n")

        # Apply filters
        _ldf = _log_df.copy()
        if _log_src_sel != "All":
            _ldf = _ldf[_ldf["source"] == _log_src_sel]
        if _log_ctr_sel:
            _ldf = _ldf[_ldf["country"].isin(_log_ctr_sel)]
        _ldf = _ldf.sort_values("timestamp").tail(_log_n)

        # ── Summary metrics ───────────────────────────────────────────────────
        if not _ldf.empty:
            _m1, _m2, _m3, _m4 = st.columns(4)
            _growing = (_ldf.drop_duplicates("country", keep="last")
                            ["status"] == "Growing").sum()
            _declining = (_ldf.drop_duplicates("country", keep="last")
                              ["status"] == "Declining").sum()
            _m1.metric("Total log entries", len(_log_df))
            _m2.metric("Countries tracked",
                       _ldf["country"].nunique())
            _m3.metric("🔴 Currently growing", int(_growing))
            _m4.metric("🟢 Currently declining", int(_declining))

        st.markdown("---")

        # ── R_eff longitudinal chart ──────────────────────────────────────────
        st.markdown("### R_eff over time — by country")
        _LOGPAL = [
            "#1976D2","#D32F2F","#388E3C","#F57C00","#7B1FA2",
            "#0097A7","#C2185B","#FBC02D","#5D4037","#455A64",
        ]
        _fig_log = _go_log.Figure()
        _fig_log.add_hline(y=1.0, line_dash="dash",
                           line_color="#e53935", opacity=0.5)
        for _li, _lc in enumerate(sorted(_ldf["country"].unique())):
            _sub = _ldf[_ldf["country"] == _lc].sort_values("timestamp")
            _lcol = _LOGPAL[_li % len(_LOGPAL)]
            # CI band
            _lfill = (f"rgba({int(_lcol[1:3],16)},"
                      f"{int(_lcol[3:5],16)},"
                      f"{int(_lcol[5:7],16)},0.12)")
            _fig_log.add_trace(_go_log.Scatter(
                x=list(_sub["timestamp"]) + list(_sub["timestamp"].iloc[::-1]),
                y=list(_sub["R_eff_q90"]) + list(_sub["R_eff_q10"].iloc[::-1]),
                fill="toself", fillcolor=_lfill,
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{_lc} CI", showlegend=False,
            ))
            _fig_log.add_trace(_go_log.Scatter(
                x=_sub["timestamp"], y=_sub["R_eff_mean"],
                name=_lc, mode="lines+markers",
                line=dict(color=_lcol, width=2),
                marker=dict(size=5),
            ))
        _fig_log.update_layout(
            xaxis_title="Run timestamp", yaxis_title="R_eff",
            height=380, template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
            margin=dict(t=30, b=70, l=55, r=20),
        )
        st.plotly_chart(_fig_log, use_container_width=True)

        # ── P_growing longitudinal chart ──────────────────────────────────────
        st.markdown("### P(R_eff > 1) over time")
        _fig_lpg = _go_log.Figure()
        _fig_lpg.add_hline(y=0.80, line_dash="dash",
                           line_color="#e53935", opacity=0.5,
                           annotation_text="80% alert",
                           annotation_position="bottom right")
        for _li, _lc in enumerate(sorted(_ldf["country"].unique())):
            _sub = _ldf[_ldf["country"] == _lc].sort_values("timestamp")
            _lcol = _LOGPAL[_li % len(_LOGPAL)]
            _fig_lpg.add_trace(_go_log.Scatter(
                x=_sub["timestamp"], y=_sub["P_growing"],
                name=_lc, mode="lines+markers",
                line=dict(color=_lcol, width=2),
                marker=dict(size=5),
            ))
        _fig_lpg.update_layout(
            xaxis_title="Run timestamp", yaxis_title="P(growing)",
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            height=320, template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
            margin=dict(t=30, b=70, l=55, r=20),
        )
        st.plotly_chart(_fig_lpg, use_container_width=True)

        # ── Status table ──────────────────────────────────────────────────────
        st.markdown("### Log entries")
        _disp = _ldf.sort_values("timestamp", ascending=False).copy()
        _disp["R_eff"] = _disp.apply(
            lambda r: f"{r['R_eff_mean']:.2f} [{r['R_eff_q10']:.2f}, {r['R_eff_q90']:.2f}]",
            axis=1,
        )
        _disp["P(growing)"] = _disp["P_growing"].map("{:.0%}".format)
        _disp["timestamp"]  = _disp["timestamp"].astype(str).str[:19]
        st.dataframe(
            _disp[["timestamp", "source", "country", "last_obs_date",
                   "R_eff", "P(growing)", "status", "model"]],
            use_container_width=True,
            hide_index=True,
        )

        # ── Download + clear ──────────────────────────────────────────────────
        _dl_col, _clr_col = st.columns([2, 1])
        with _dl_col:
            st.download_button(
                "⬇️ Download full log (CSV)",
                data=_log_df.to_csv(index=False),
                file_name="episurveil_alert_log.csv",
                mime="text/csv",
                key="log_dl",
            )
        with _clr_col:
            if st.button("🗑️ Clear log", key="log_clear",
                         help="Permanently deletes the alert log file."):
                if _ALERT_LOG_PATH.exists():
                    _ALERT_LOG_PATH.unlink()
                st.rerun()

# ===========================================================================
# TAB 14 — Methodology
# ===========================================================================
with tab_help:
    st.subheader("Methodology & user guide")

    st.markdown("""
EpiSurveil is an interactive epidemic surveillance platform that uses a
**Bootstrap Particle Filter (BPF)** to estimate time-varying transmission
parameters from surveillance data in real time.
""")

    with st.expander("1. Bootstrap Particle Filter — core algorithm", expanded=True):
        st.markdown(r"""
### Sequential Bayesian inference

The BPF approximates the posterior filtering distribution $p(x_t, \theta_t \mid y_{1:t})$
with $N$ weighted particles $\{(x_t^{(i)}, \theta_t^{(i)}),\, w_t^{(i)}\}_{i=1}^N$.

**One time step $t-1 \to t$:**

**1. Propagate** — advance each particle one day through the compartmental ODE:
$$x_{t}^{(i)} = f\!\left(x_{t-1}^{(i)},\, \theta_{t-1}^{(i)}\right) + \text{noise}$$

**2. Parameters** — evolve as independent log-random walks:
$$\log \theta_t^{(i)} = \log \theta_{t-1}^{(i)} + \xi_t^{(i)}, \qquad \xi_t^{(i)} \sim \mathcal{N}(0,\, \sigma_{\mathrm{RW}}^2)$$

**3. Weight update** — accumulate tempered log-weights from every active channel $c$:
$$\log w_t^{(i)} \leftarrow \log w_{t-1}^{(i)} + \sum_c \alpha_c \cdot \log p\!\left(y_{c,t} \mid \mu_{c,t}^{(i)},\, \phi_c\right)$$
where $\alpha_c \in (0,1]$ is the **tempering weight** (reduce for noisy / delayed channels).
Channels with $y_{c,t} = \mathrm{NaN}$ are silently skipped.

The likelihood is a **Negative Binomial** (mean $\mu$, dispersion $\phi$, variance $\mu + \mu^2/\phi$):
$$\log p(y \mid \mu, \phi) = \log\Gamma(y+\phi) - \log\Gamma(\phi) - \log\Gamma(y+1) + \phi\log\tfrac{\phi}{\phi+\mu} + y\log\tfrac{\mu}{\phi+\mu}$$

**4. Normalise** — $\tilde{w}_t^{(i)} = w_t^{(i)} \big/ \sum_j w_t^{(j)}$

**5. Resample** — when $\mathrm{ESS}_t = \bigl(\sum_i (\tilde{w}_t^{(i)})^2\bigr)^{-1} < 0.45\,N$,
apply **systematic resampling** $\mathcal{O}(N)$; reset weights to $1/N$.

**Key outputs:**
- $R_{\mathrm{eff}}(t)$ — effective reproduction number, posterior mean + 80% CI
- $P(\text{growing}) = \sum_i \tilde{w}_t^{(i)}\,\mathbf{1}[R_{\mathrm{eff}}^{(i)} > 1]$ — probability the epidemic is expanding
""")

    with st.expander("2. Compartmental models available"):
        st.markdown(r"""
| Model | States | Parameters | Best for |
|-------|--------|------------|----------|
| **SIR** | S, I, R | $\beta_t$ | Fast screening; no incubation |
| **SEIR** | S, E, I, R | $\beta_t$ | Standard flu/COVID; incubation lag |
| **SEIRD** | S, E, I, R, D | $\beta_t$, $\delta_t$ | Multi-channel (cases + deaths); time-varying IFR |
| **SEIRV** | S, E, I, R, V | $\beta_t$ | Vaccination impact; waning immunity |
| **SEIARV** | S, E, A, I, R, V | $\beta_t$, $Q_{C,t}$ | Under-reporting estimation |
| **SEIRHD** | S, E, I, H, R, D | $\beta_t$, $\tau_{i,t}$, $\delta_{h,t}$ | Hospital surge planning |
| **SVEAIHCRD** | 9 states | 5 params | Full national COVID-19 surveillance |

**R_eff formulas:**

| Model | $R_{\mathrm{eff}}$ |
|-------|--------------------|
| SIR/SEIR | $\beta_t S_t / (N \gamma)$ |
| SEIRD | $\beta_t S_t / [N(\gamma + \delta_t)]$ |
| SEIRV | $\beta_t [S_t + (1-\varepsilon) V_t] / (N \gamma)$ |
""")

    with st.expander("3. Parameter guide"):
        st.markdown(r"""
| Parameter | Symbol | Typical value | Meaning |
|-----------|--------|---------------|---------|
| Infectious period | $1/\gamma$ | 7–14 days → $\gamma \approx 0.07\text{–}0.14$ | Average time from infection to recovery |
| Incubation period | $1/\sigma$ | 3–7 days → $\sigma \approx 0.14\text{–}0.33$ | Latent (exposed) period before infectiousness |
| Waning immunity (R→S) | $\omega_r$ | 0 – 0.006 | Rate recovered individuals return to susceptible. $1/180 \approx 6$ months |
| Death rate | $\delta$ | Omicron: $10^{-4}$; pre-Omicron: $10^{-3}$ | IFR $\approx \delta / (\gamma + \delta)$ |
| Vaccine efficacy | $\varepsilon$ | 0.70 – 0.95 | Fraction of infectability blocked in V |
| Waning (V→S) | $\omega_v$ | $1/180 \approx 0.0056$ | Rate vaccinated return to susceptible |
| Daily vacc. rate | $\nu$ | 0 – 0.003 | Fraction of S vaccinated per day |
| Log-RW noise | $\sigma_{\mathrm{RW}}$ | 0.03 – 0.06 | Flexibility of parameter drift per day |
| NegBin dispersion | $\phi$ | 5 – 100 | Lower = wider, more tolerant of outliers |
| Tempering weight | $\alpha$ | 0.4 – 1.0 | Down-weight noisy/delayed channels (deaths) |
| Particles | $N$ | 500 – 4000 | More = smoother but slower |
""")

    with st.expander("4. Observation model (Negative-Binomial likelihood)"):
        st.markdown(r"""
Each observed count $y_t^c$ is modelled as $y_t^c \sim \mathrm{NegBin}(\mu_t^c,\, \phi_c)$.

**Predicted means by model and channel:**

| Channel | Model | Formula | $\phi$ (default) | $\alpha$ (default) |
|---------|-------|---------|-----------------|---------------------|
| cases | all | $Q_C \cdot \sigma \cdot E_t$ | 50–80 | 1.00 |
| deaths | SEIRD, SEIRHD | $\delta_t \cdot I_t$ | 5–10 | 0.40 |
| deaths | SVEAIHCRD | $\delta_H H_t + \delta_C C_t$ | 10 | 0.40 |
| hosp occupancy | SEIRHD | $\tau_{i,t} \cdot I_t \cdot \mathrm{LOS}$ | 15 | 0.60 |
| hosp proxy | SVEAIHCRD | $\tau_I\,I_t \times \mathrm{LOS}$ | 15 | 0.10 |
| ICU occupancy | SVEAIHCRD | $C_t$ | 12 | 0.60 |

**NegBin variance:** $\mathrm{Var} = \mu + \mu^2/\phi$. Lower $\phi$ = heavier tails, more tolerant of outliers.

**Weekly data** (ECDC sentinel): counts are divided by 7 and forward-filled daily
so the BPF (daily time step) sees an approximately constant rate across each week.
Missing weeks (off-season) remain NaN and are skipped.

**Case reporting delay** (SVEAIHCRD only): a Gamma convolution kernel (mean 5 d, SD 2 d)
is applied to simulated incidence before comparing to reported cases,
accounting for the typical lag between infection onset and confirmed report.
""")

    with st.expander("5. Data sources"):
        st.markdown("""
| Source | Data type | Coverage | Update frequency |
|--------|-----------|----------|-----------------|
| **disease.sh** | Daily COVID-19 confirmed cases + deaths | 180+ countries | Frozen ~March 2023 (reporting ended) |
| **ECDC sentinel (detections)** | Weekly lab-confirmed influenza | EU/EEA + Norway | Updated weekly, current |
| **ECDC ILI/ARI rates** | Weekly GP consultation rate per 100k | EU/EEA + Norway | Updated weekly, current |
| **SVEAIHCRD filter** | Pre-computed BPF for Germany 2020–2023 | Germany only | Static CSV bundled with app |

ILI = Influenza-Like Illness (fever + cough/sore throat).
ARI = Acute Respiratory Infection (broader; used by Germany, Bulgaria, Cyprus when ILI unavailable).

Daily case counts for ILI/ARI are estimated as: `rate_per_100k / 100000 × population / 7`.
""")

    with st.expander("6. Alert thresholds and status labels"):
        st.markdown(r"""
| Status | Condition | Colour |
|--------|-----------|--------|
| **Growing** | $P(R_{\mathrm{eff}} > 1) \geq \theta$ | Red |
| **Uncertain** | $0.50 \leq P(R_{\mathrm{eff}} > 1) < \theta$ | Amber |
| **Declining** | $P(R_{\mathrm{eff}} > 1) < 0.50$ | Green |

Default threshold $\theta = 0.80$. Adjust in the Country comparison tab.

The **Alert history** tab records every BPF run across sessions so you can
track how $R_{\mathrm{eff}}$ and epidemic status evolve over calendar time.
""")

    with st.expander("7. ESS and particle collapse"):
        st.markdown(r"""
**Effective Sample Size (ESS):**
$$\mathrm{ESS}_t = \frac{1}{\sum_{i=1}^N (w_t^{(i)})^2}$$

- $\mathrm{ESS} \approx N$: particles are equally weighted (healthy filter).
- $\mathrm{ESS} \ll N$: weight degeneracy — one particle dominates.
- Resampling is triggered when $\mathrm{ESS} < 0.45 N$.

**Low ESS usually means:**
- Model is misspecified (wrong $\gamma$, $\sigma$, or $Q_C$).
- Observation counts are scale-mismatched (check population $N$).
- NegBin dispersion $\phi$ is too small (over-penalises moderate deviations).

**Fixes:** Increase particles, widen $\sigma_{\mathrm{RW}}$, reduce $\phi$, or adjust $N$.
""")

    with st.expander("8. Short-term forecast and scenario comparison"):
        st.markdown(r"""
After a BPF run, the final particle cloud is propagated forward for up to 30 days
**without likelihood updates** (equal weights). This gives an ensemble forecast
that reflects current parameter uncertainty.

**Scenario comparison** applies a $\beta$ multiplier at $t = 0$ of the forecast
horizon (e.g. $-30\%$ transmission → `multiplier = 0.70`) to simulate an NPI or
contact reduction and compare projected incidence against the baseline.
""")

    st.info(
        "For questions, issues, or feature requests, open an issue on the project repository.",
        icon="ℹ️",
    )
