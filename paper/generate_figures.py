"""
Generate all paper figures for EpiSurveil main.tex.

Figures produced (saved to paper/figures/):
  fig1_filter_output.pdf   — 4-panel filter output with 80% CI
  fig2_parameters.pdf      — 5 dynamic parameter trajectories
  fig3_ess.pdf             — ESS over time with resampling events
  fig4_pareto.pdf          — Pareto frontier (deaths averted vs NPI cost)
  fig5_pfmpc.pdf           — PF-MPC control + counterfactual trajectories
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

DATA_DIR = ROOT / "data" / "processed"
FILTER_CSV = DATA_DIR / "sveaihcrd_filter_output.csv"

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.linewidth":   0.4,
    "grid.alpha":       0.4,
})

BLUE   = "#2166AC"
RED    = "#D6604D"
GREEN  = "#4DAC26"
TEAL   = "#35978F"
ORANGE = "#F4A582"
GREY   = "#888888"

# Known pandemic events for annotations
EVENTS = {
    "Alpha\n(Dec'20)":   "2020-12-19",
    "Delta\n(May'21)":   "2021-05-01",
    "Omicron\n(Nov'21)": "2021-11-26",
}
SARI_CUTOFF = pd.Timestamp("2023-04-07")


def _fmt_k(ax, axis="y"):
    """Format axis ticks as k (thousands)."""
    def _f(x, _): return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"
    if axis == "y":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_f))
    else:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_f))


def _date_ax(ax, df, major="year", minor="month"):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.set_xlim(df["date"].iloc[0], df["date"].iloc[-1])


def _sari_band(ax, ymax):
    ax.axvspan(SARI_CUTOFF, ax.get_xlim()[1] if hasattr(ax, '_sari_set')
               else pd.Timestamp("2024-10-31"),
               color=GREY, alpha=0.08, zorder=0)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Filter output — 4-panel
# ─────────────────────────────────────────────────────────────────────────────
def fig1_filter_output(df):
    print("  Figure 1: filter output …")
    channels = [
        ("pred_cases",  "obs_cases",  "Daily reported cases",     "Cases / day",       BLUE),
        ("pred_icu",    "obs_icu",    "ICU occupancy",             "ICU beds",          RED),
        ("pred_deaths", "obs_deaths", "Daily deaths",              "Deaths / day",      "#7B2D8B"),
        ("pred_hosp",   "obs_hosp",   "Hospital occupancy (proxy)","Hosp. beds",        TEAL),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 9), sharex=True)
    fig.subplots_adjust(hspace=0.35, left=0.10, right=0.97, top=0.96, bottom=0.06)

    for ax, (pred, obs, title, ylabel, col) in zip(axes, channels):
        mn  = df[f"{pred}_mean"]
        q10 = df[f"{pred}_q10"]
        q90 = df[f"{pred}_q90"]
        yob = df[obs]
        dates = df["date"]

        ax.fill_between(dates, q10, q90, color=col, alpha=0.20, label="80% CI")
        ax.plot(dates, mn,  color=col, lw=1.2, label="Posterior mean")
        ax.scatter(dates[yob.notna()], yob[yob.notna()],
                   s=1.5, color="black", alpha=0.55, label="Observed", zorder=5)

        # SARI extension shading
        ax.axvspan(SARI_CUTOFF, dates.iloc[-1], color=GREY, alpha=0.08, zorder=0)
        if ax is axes[0]:
            ax.text(SARI_CUTOFF + pd.Timedelta(days=10),
                    ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1,
                    "SARI\nsentinel", fontsize=6, color=GREY, va="top")

        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        _date_ax(ax, df)
        if pred in ("pred_cases", "pred_hosp"):
            _fmt_k(ax)

    axes[-1].set_xlabel("Date")

    # shared legend on top panel
    handles = [
        Line2D([0],[0], color=BLUE, lw=1.2, label="Posterior mean"),
        plt.fill_between([], [], [], color=BLUE, alpha=0.20, label="80% CI"),
        Line2D([0],[0], color="black", lw=0, marker="o", ms=3, label="Observed"),
    ]
    axes[0].legend(handles=handles, loc="upper right", framealpha=0.7)

    fig.savefig(FIG_DIR / "fig1_filter_output.pdf", bbox_inches="tight")
    plt.close(fig)
    print("    → fig1_filter_output.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Five dynamic parameters
# ─────────────────────────────────────────────────────────────────────────────
def fig2_parameters(df):
    print("  Figure 2: dynamic parameters …")

    params = [
        ("beta_mean",    r"$\beta_t$ — transmission rate",                 BLUE),
        ("tau_i_mean",   r"$\tau_{I,t}$ — hosp. rate from $I$",           RED),
        ("delta_h_mean", r"$\delta_{H,t}$ — in-hospital CFR",             GREEN),
        ("rho_c_mean",   r"$\rho_{C,t}$ — ICU fraction of hosp.",         ORANGE),
        ("q_c_mean",     r"$Q_{C,t}$ — case detection probability",       TEAL),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(7.5, 10), sharex=True)
    fig.subplots_adjust(hspace=0.38, left=0.12, right=0.97, top=0.96, bottom=0.06)

    for ax, (col, title, color) in zip(axes, params):
        ax.plot(df["date"], df[col], color=color, lw=1.0)
        ax.axvspan(SARI_CUTOFF, df["date"].iloc[-1], color=GREY, alpha=0.08, zorder=0)
        ax.set_title(title, loc="left", fontweight="bold")
        _date_ax(ax, df)

        # Variant annotations on first panel only
        if col == "beta_mean":
            for label, date in EVENTS.items():
                ts = pd.Timestamp(date)
                if df["date"].iloc[0] <= ts <= df["date"].iloc[-1]:
                    ax.axvline(ts, color=GREY, lw=0.7, ls="--", alpha=0.7)
                    ax.text(ts + pd.Timedelta(days=5),
                            ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1,
                            label, fontsize=6, color=GREY, va="top")

    axes[-1].set_xlabel("Date")
    axes[2].set_ylabel("Rate (day⁻¹)", labelpad=2)

    fig.savefig(FIG_DIR / "fig2_parameters.pdf", bbox_inches="tight")
    plt.close(fig)
    print("    → fig2_parameters.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: ESS over time
# ─────────────────────────────────────────────────────────────────────────────
def fig3_ess(df):
    print("  Figure 3: ESS …")
    N      = 2000
    thresh = int(0.45 * N)

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.14)

    resample_mask = df["ess"] < thresh

    ax.plot(df["date"], df["ess"], color=BLUE, lw=0.6, alpha=0.85, label="ESS")
    ax.scatter(df["date"][resample_mask], df["ess"][resample_mask],
               s=4, color=RED, alpha=0.7, zorder=5, label="Resampled")
    ax.axhline(thresh, color=RED, lw=0.8, ls="--",
               label=f"Threshold ({thresh})")
    ax.axhline(N, color=GREY, lw=0.6, ls=":", alpha=0.6)
    ax.text(df["date"].iloc[5], N * 1.01, f"N={N}", fontsize=6, color=GREY)

    ax.axvspan(SARI_CUTOFF, df["date"].iloc[-1], color=GREY, alpha=0.08, zorder=0)
    ax.set_ylabel("Effective sample size")
    ax.set_xlabel("Date")
    ax.set_title("ESS over 1,673-day panel — resampling events marked in red",
                 loc="left", fontweight="bold")
    ax.set_ylim(0, N * 1.08)
    _date_ax(ax, df)
    ax.legend(loc="lower right", framealpha=0.7)

    n_resamp = resample_mask.sum()
    ax.text(0.02, 0.06,
            f"{n_resamp} resampling events  |  mean ESS = {df['ess'].mean():.0f}",
            transform=ax.transAxes, fontsize=7, color=GREY)

    fig.savefig(FIG_DIR / "fig3_ess.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"    → fig3_ess.pdf  ({n_resamp} resampling events)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Compartment trajectories (E, A, I, H, C, S, V, R, D)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_compartments(df):
    print("  Figure 4: compartment trajectories …")

    # Four grouped panels
    groups = [
        {
            "title": "Infectious pool  —  E, A, I",
            "comps": [
                ("E", "Exposed $E_t$",             "#E69F00"),
                ("A", "Asymptomatic $A_t$",         "#56B4E9"),
                ("I", "Symptomatic $I_t$",           RED),
            ],
        },
        {
            "title": "Hospital burden  —  H, C",
            "comps": [
                ("H", "Hospitalised $H_t$",          BLUE),
                ("C", "ICU $C_t$",                   "#D55E00"),
            ],
        },
        {
            "title": "Population immunity  —  S, V",
            "comps": [
                ("S", "Susceptible $S_t$",           GREY),
                ("V", "Vaccinated $V_t$",            GREEN),
            ],
        },
        {
            "title": "Outcomes  —  R, D",
            "comps": [
                ("R", "Recovered $R_t$",             TEAL),
                ("D", "Cumulative deaths $D_t$",     "#7B2D8B"),
            ],
        },
    ]

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 10), sharex=True)
    fig.subplots_adjust(hspace=0.40, left=0.12, right=0.97, top=0.96, bottom=0.06)

    for ax, grp in zip(axes, groups):
        for (comp, label, col) in grp["comps"]:
            mn  = df[f"{comp}_mean"]
            q10 = df[f"{comp}_q10"]
            q90 = df[f"{comp}_q90"]
            ax.fill_between(df["date"], q10, q90,
                            color=col, alpha=0.18)
            ax.plot(df["date"], mn, color=col, lw=1.1, label=label)

        # SARI extension shading
        ax.axvspan(SARI_CUTOFF, df["date"].iloc[-1],
                   color=GREY, alpha=0.08, zorder=0)

        ax.set_title(grp["title"], loc="left", fontweight="bold")
        _date_ax(ax, df)
        _fmt_k(ax)
        ax.legend(loc="upper right", framealpha=0.7, ncol=len(grp["comps"]))

    axes[-1].set_xlabel("Date")

    # variant lines on infectious panel
    for label, date in EVENTS.items():
        ts = pd.Timestamp(date)
        if df["date"].iloc[0] <= ts <= df["date"].iloc[-1]:
            axes[0].axvline(ts, color=GREY, lw=0.7, ls="--", alpha=0.7)
            ylim = axes[0].get_ylim()
            axes[0].text(ts + pd.Timedelta(days=5), ylim[1] * 0.88,
                         label, fontsize=6, color=GREY, va="top")

    fig.savefig(FIG_DIR / "fig4_compartments.pdf", bbox_inches="tight")
    plt.close(fig)
    print("    → fig4_compartments.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Pareto frontier
# ─────────────────────────────────────────────────────────────────────────────
def _extract_state(df, t0):
    """Extract x0 and parameters from filter CSV at date t0."""
    row = df[df["date"] == pd.Timestamp(t0)]
    if row.empty:
        # fallback: nearest date
        idx = (df["date"] - pd.Timestamp(t0)).abs().idxmin()
        row = df.iloc[[idx]]
    row = row.iloc[0]
    comp_names = ["S", "V", "E", "A", "I", "H", "C", "R", "D"]
    x0        = np.array([float(row.get(f"{c}_mean", 0.0)) for c in comp_names])
    beta_base = float(row.get("beta_mean",    0.44))
    tau_i     = float(row.get("tau_i_mean",   0.0021))
    delta_h   = float(row.get("delta_h_mean", 0.019))
    q_c_base  = float(row.get("rho_c_mean",   0.35))
    return x0, beta_base, tau_i, delta_h, q_c_base


def fig5_pareto(df):
    print("  Figure 5: Pareto frontier …")
    from episurveil.control.optimal_control import run_pareto_sweep, H_MAX_DE, C_MAX_DE, ALPHA_NPI

    t0 = "2020-10-01"
    x0, beta_base, tau_i, delta_h, q_c_base = _extract_state(df, t0)
    H0 = x0[5]; C0 = x0[6]
    h_max = max(H_MAX_DE, H0 * 1.05)
    c_max = max(C_MAX_DE, C0 * 1.05)

    print(f"    running pareto sweep from {t0} …")
    pareto = run_pareto_sweep(
        x0=x0, beta_base=beta_base, tau_i=tau_i, delta_h=delta_h,
        q_c_base=q_c_base, T=90,
        h_max=h_max, c_max=c_max,
        alpha_npi=ALPHA_NPI, n_points=8,
        start_date=t0,
    )
    if not pareto:
        print("    pareto sweep returned empty — skipping fig4")
        return

    da   = np.array([p["deaths_averted"]     for p in pareto])
    wu   = np.array([p["w_u"]                for p in pareto])
    tpct = np.array([p.get("testing_contribution_pct", 50.0) for p in pareto])
    mu   = np.array([p.get("mean_u", np.nan) for p in pareto])
    mv   = np.array([p.get("mean_v", np.nan) for p in pareto])

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    fig.subplots_adjust(left=0.13, right=0.88, top=0.92, bottom=0.13)

    sc = ax.scatter(wu, da, c=tpct, cmap="RdYlGn_r", s=60,
                    vmin=0, vmax=100, zorder=5, edgecolors="white", lw=0.4)
    ax.plot(wu, da, color=GREY, lw=0.8, alpha=0.5, zorder=3)

    for i, p in enumerate(pareto):
        ax.annotate(
            f"NPI {p.get('npi_contribution_pct', 50):.0f}%\ntest {p.get('testing_contribution_pct', 50):.0f}%",
            (wu[i], da[i]),
            textcoords="offset points", xytext=(6, 3),
            fontsize=5.5, color="#444444",
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Testing contribution (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xscale("log")
    ax.set_xlabel("NPI cost weight $w_u$")
    ax.set_ylabel("Deaths averted (vs. no-control baseline)")
    ax.set_title(f"Pareto frontier — start {t0}, $T=90$ days, $w_v=2$",
                 loc="left", fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)

    fig.savefig(FIG_DIR / "fig5_pareto.pdf", bbox_inches="tight")
    plt.close(fig)
    print("    → fig5_pareto.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: PF-MPC
# ─────────────────────────────────────────────────────────────────────────────
def fig6_pfmpc(df):
    print("  Figure 6: PF-MPC …")
    from episurveil.control.pf_mpc import run_pf_mpc

    t0    = "2021-09-01"
    T_sim = 60
    H     = 14

    steps_done = [0]
    def _prog(s, T): steps_done[0] = s; print(f"\r    step {s}/{T}   ", end="", flush=True)

    r = run_pf_mpc(df, t0=t0, T_sim=T_sim, H=H, progress_cb=_prog)
    print()

    dates_ctrl = pd.to_datetime(r.dates)
    dates_T    = dates_ctrl[:-1]   # length T_sim

    fig, axes = plt.subplots(5, 1, figsize=(7.5, 11), sharex=True)
    fig.subplots_adjust(hspace=0.38, left=0.12, right=0.97, top=0.96, bottom=0.06)

    # --- panel 1: NPI control u*(t) ---
    ax = axes[0]
    ax.fill_between(dates_T, 0, r.u_mpc, color=BLUE, alpha=0.30)
    ax.plot(dates_T, r.u_mpc, color=BLUE, lw=1.2, label="$u^*(t)$ — NPI intensity")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("NPI intensity $u_t$")
    ax.set_title("NPI control $u^*(t)$", loc="left", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.7)

    # --- panel 2: testing control v*(t) ---
    ax = axes[1]
    ax.fill_between(dates_T, 0, r.v_mpc, color=TEAL, alpha=0.30)
    ax.plot(dates_T, r.v_mpc, color=TEAL, lw=1.2, label="$v^*(t)$ — testing intensity")
    ax.plot(dates_T, r.q_c_traj, color=GREEN, lw=1.0, ls="--",
            label="$Q_{C,t}^*$ — effective detection")
    ax.plot(dates_T, r.q_c_base_bpf, color=RED, lw=0.8, ls=":",
            label="$\\hat{Q}_{C,t}$ — BPF baseline")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Testing / detection")
    ax.set_title("Testing control $v^*(t)$ and effective detection $Q_{C,t}^*$",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.7)

    # --- panel 3: β trajectory ---
    ax = axes[2]
    ax.plot(dates_T, r.beta_bpf,  color=RED,  lw=1.0, ls="--",
            label=r"$\hat{\beta}_t$ — BPF (uncontrolled)")
    ax.plot(dates_T, r.beta_eff,  color=BLUE, lw=1.2,
            label=r"$\beta_t^{\mathrm{eff}}$ — under MPC")
    ax.fill_between(dates_T, r.beta_eff, r.beta_bpf, color=BLUE, alpha=0.12)
    ax.set_ylabel("Transmission rate $\\beta$")
    ax.set_title("Effective transmission rate under MPC vs. BPF background",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.7)

    # --- panels 4-5: H and D trajectories ---
    comps = [
        ("H", "Hospital occupancy",  "Hosp. beds",   RED),
        ("D", "Cumulative deaths",   "Deaths",        "#7B2D8B"),
    ]
    for ax, (comp, title, ylabel, col) in zip(axes[3:], comps):
        bl = r.traj_baseline[comp]
        mc = r.traj_mpc[comp]
        ax.plot(dates_ctrl, bl, color=GREY, lw=1.2, ls="--", label="No intervention")
        ax.plot(dates_ctrl, mc, color=col,  lw=1.4,           label="PF-MPC policy")
        ax.fill_between(dates_ctrl, mc, bl, color=col, alpha=0.15)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.7)
        _fmt_k(ax)

    axes[-1].set_xlabel("Date")

    # summary annotation
    fig.text(0.12, 0.005,
             f"PF-MPC: $T_{{\\rm sim}}={r.T_sim}$ days, $H={r.H}$ days  |  "
             f"Deaths averted: {r.deaths_averted:,}  |  "
             f"Hosp-days averted: {r.hosp_days_averted:,}  |  "
             f"Convergence: {r.converged.sum()}/{r.T_sim} steps",
             fontsize=6.5, color=GREY)

    fig.savefig(FIG_DIR / "fig6_pfmpc.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"    → fig6_pfmpc.pdf  (deaths averted: {r.deaths_averted:,})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading filter output …")
    df = pd.read_csv(FILTER_CSV, parse_dates=["date"])
    print(f"  {len(df)} rows  |  {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    fig1_filter_output(df)
    fig2_parameters(df)
    fig3_ess(df)
    fig4_compartments(df)
    fig5_pareto(df)
    fig6_pfmpc(df)

    print(f"\nAll figures written to: {FIG_DIR}")
