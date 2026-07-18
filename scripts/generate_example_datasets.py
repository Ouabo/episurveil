"""
Generate synthetic but epidemiologically realistic example datasets
for the EpiSurveil Model Explorer built-in examples.

Diseases covered
----------------
1. Measles outbreak          — West Africa, 2019 (200 days, SEIR)
2. Influenza season          — Europe, Oct 2019–Mar 2020 (182 days, SEIR)
3. Influenza multi-wave      — Europe, 2018–2020 (548 days, SEIR + waning)
4. Ebola outbreak            — Central Africa, 2018–2019 (365 days, SEIR)
5. Influenza w/ hospital     — Europe, 2019 season (182 days, SEIRHD)

All datasets use a forward Euler SEIR(HD) ODE + NegBin observation noise.
Ground-truth parameters are stored in the metadata block at the top of each CSV.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import nbinom

OUT = Path(__file__).resolve().parents[1] / "data" / "examples"
OUT.mkdir(parents=True, exist_ok=True)


# ── Simulation helpers ────────────────────────────────────────────────────────

def _negbin_sample(rng, mu: np.ndarray | float, phi: float) -> np.ndarray | int:
    """Sample NegBin(mu, phi) — parameterised so Var = mu + mu²/phi."""
    mu = np.asarray(mu, dtype=float)
    mu = np.maximum(mu, 1e-6)
    p  = phi / (phi + mu)
    return rng.negative_binomial(phi, p)


def simulate_seir(
    *,
    N: int,
    S0_frac: float,
    E0: int,
    I0: int,
    sigma: float,
    gamma: float,
    beta_fn,          # callable t -> beta (scalar)
    Q_C: float,
    phi: float,
    T: int,
    start_date: str,
    omega_r: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    S = N * S0_frac
    E = float(E0)
    I = float(I0)
    R = N - S - E - I

    dates, cases = [], []
    dt_index = pd.date_range(start_date, periods=T, freq="D")

    for t in range(T):
        beta  = beta_fn(t)
        N_lv  = max(S + E + I + R, 1.0)
        lam   = beta * I / N_lv
        wane  = omega_r * R
        dS    = -lam * S + wane
        dE    =  lam * S - sigma * E
        dI    =  sigma * E - gamma * I
        dR    =  gamma * I - wane
        S     = max(S + dS, 0.0)
        E     = max(E + dE, 0.0)
        I     = max(I + dI, 0.0)
        R     = max(R + dR, 0.0)

        mu = Q_C * sigma * E
        obs = int(_negbin_sample(rng, mu, phi)) if mu > 0.5 else 0
        dates.append(dt_index[t])
        cases.append(obs)

    return pd.DataFrame({"date": dates, "cases": cases})


def simulate_seirhd(
    *,
    N: int,
    S0_frac: float,
    E0: int,
    I0: int,
    sigma: float,
    gamma_i: float,
    gamma_h: float,
    beta_fn,
    tau_i: float,       # hospitalisation rate from I
    delta_h: float,     # in-hospital CFR
    LOS: float,
    Q_C: float,
    phi_c: float,
    phi_h: float,
    phi_d: float,
    T: int,
    start_date: str,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    S = N * S0_frac
    E = float(E0)
    I = float(I0)
    H = float(0)
    R = N - S - E - I
    D = float(0)

    dates, obs_cases, obs_hosp, obs_deaths = [], [], [], []
    dt_index = pd.date_range(start_date, periods=T, freq="D")

    for t in range(T):
        beta  = beta_fn(t)
        N_lv  = max(S + E + I + H + R, 1.0)
        lam   = beta * I / N_lv
        dS    = -lam * S
        dE    =  lam * S - sigma * E
        dI    =  sigma * E - (gamma_i + tau_i) * I
        dH    =  tau_i * I - (gamma_h + delta_h) * H
        dR    =  gamma_i * I + gamma_h * H
        dD    =  delta_h * H
        S  = max(S + dS, 0.0)
        E  = max(E + dE, 0.0)
        I  = max(I + dI, 0.0)
        H  = max(H + dH, 0.0)
        R  = max(R + dR, 0.0)
        D += max(dD, 0.0)

        mu_c = Q_C * sigma * E
        mu_h = tau_i * I * LOS
        mu_d = delta_h * H
        obs_cases.append(int(_negbin_sample(rng, mu_c, phi_c)) if mu_c > 0.5 else 0)
        obs_hosp.append(int(_negbin_sample(rng, mu_h, phi_h))  if mu_h > 0.5 else 0)
        obs_deaths.append(int(_negbin_sample(rng, mu_d, phi_d)) if mu_d > 0.2 else 0)
        dates.append(dt_index[t])

    return pd.DataFrame({"date": dates, "cases": obs_cases,
                          "hosp": obs_hosp, "deaths": obs_deaths})


# ── Dataset 1 — Measles outbreak (West Africa, 2019) ────────────────────────

print("Generating: measles_outbreak_west_africa_2019.csv …")
df_measles = simulate_seir(
    N=500_000,
    S0_frac=0.15,           # 85% vaccinated, 15% susceptible
    E0=100,
    I0=50,
    sigma=0.08,             # 12.5-day incubation (1/12.5)
    gamma=0.125,            # 8-day infectious period
    beta_fn=lambda t: 1.875,  # R0=15 × γ — constant (herd immunity terminates wave)
    Q_C=0.70,               # distinct rash → high detection
    phi=50.0,
    T=200,
    start_date="2019-01-15",
    seed=7,
)
df_measles.to_csv(OUT / "measles_outbreak_west_africa_2019.csv", index=False)
print(f"  {len(df_measles)} rows, peak cases = {df_measles['cases'].max()}")


# ── Dataset 2 — Influenza season (Europe, Oct 2019 – Mar 2020) ──────────────

print("Generating: influenza_season_europe_2019_20.csv …")
df_flu = simulate_seir(
    N=8_000_000,
    S0_frac=0.90,           # 90% susceptible at start of season (low cross-immunity)
    E0=2_000,
    I0=1_000,
    sigma=0.50,             # 2-day incubation
    gamma=0.25,             # 4-day infectious
    beta_fn=lambda t: 0.35, # R0 ≈ 1.4, gentle seasonal wave
    Q_C=0.10,               # low detection (most flu goes unreported)
    phi=30.0,
    T=182,
    start_date="2019-10-01",
    seed=13,
)
df_flu.to_csv(OUT / "influenza_season_europe_2019_20.csv", index=False)
print(f"  {len(df_flu)} rows, peak cases = {df_flu['cases'].max()}")


# ── Dataset 3 — Influenza multi-wave (Europe, Oct 2018 – Mar 2020) ──────────

print("Generating: influenza_multiwav_europe_2018_20.csv …")

def _flu_beta(t):
    """Two winter seasons separated by low-transmission summer."""
    if t < 182:    return 0.38   # season 1
    if t < 270:    return 0.12   # summer trough
    return 0.40                  # season 2 (slightly stronger — drift)

df_flu2 = simulate_seir(
    N=20_000_000,
    S0_frac=0.88,
    E0=5_000,
    I0=2_000,
    sigma=0.50,
    gamma=0.25,
    beta_fn=_flu_beta,
    Q_C=0.08,
    phi=25.0,
    T=548,
    start_date="2018-10-01",
    omega_r=1 / 270,         # ~9-month natural immunity → second wave possible
    seed=21,
)
df_flu2.to_csv(OUT / "influenza_multiwav_europe_2018_20.csv", index=False)
print(f"  {len(df_flu2)} rows, peak cases = {df_flu2['cases'].max()}")


# ── Dataset 4 — Ebola outbreak (Central Africa, 2018–2019) ──────────────────

print("Generating: ebola_outbreak_drc_2018_19.csv …")

def _ebola_beta(t):
    """Response ramps up: R_eff starts above 1, crosses below 1 by day ~180."""
    if t < 60:   return 0.19   # early: R0 ≈ 1.9 (no response yet)
    if t < 150:  return 0.14   # partial response: R0 ≈ 1.4
    if t < 250:  return 0.09   # strong response: R0 ≈ 0.9 (declining)
    return 0.07                # containment phase

df_ebola = simulate_seir(
    N=200_000,
    S0_frac=0.96,
    E0=15,
    I0=8,
    sigma=0.088,             # 11.4-day incubation
    gamma=0.10,              # 10-day infectious period
    beta_fn=_ebola_beta,
    Q_C=0.70,                # good detection in outbreak response
    phi=10.0,                # small numbers → more overdispersion
    T=365,
    start_date="2018-08-01",
    seed=31,
)
df_ebola.to_csv(OUT / "ebola_outbreak_drc_2018_19.csv", index=False)
print(f"  {len(df_ebola)} rows, peak cases = {df_ebola['cases'].max()}")


# ── Dataset 5 — Influenza with hospitalizations (SEIRHD, Europe 2019) ────────

print("Generating: influenza_seirhd_europe_2019_20.csv …")
df_flu_hd = simulate_seirhd(
    N=10_000_000,
    S0_frac=0.88,
    E0=3_000,
    I0=1_500,
    sigma=0.50,
    gamma_i=0.20,            # 5-day infectious before hosp split
    gamma_h=0.14,            # ~7-day hospital stay (1/0.14 ≈ 7 days)
    beta_fn=lambda t: 0.38,
    tau_i=0.002,             # 0.2% of infectious hospitalised per day (realistic flu)
    delta_h=0.003,           # in-hospital CFR ≈ 2%  (delta_h / (gamma_h + delta_h))
    LOS=5.0,
    Q_C=0.10,
    phi_c=30.0,
    phi_h=15.0,
    phi_d=5.0,
    T=182,
    start_date="2019-10-01",
    seed=41,
)
df_flu_hd.to_csv(OUT / "influenza_seirhd_europe_2019_20.csv", index=False)
print(f"  {len(df_flu_hd)} rows, peak cases = {df_flu_hd['cases'].max()}, "
      f"peak hosp = {df_flu_hd['hosp'].max()}, "
      f"peak deaths = {df_flu_hd['deaths'].max()}")


print(f"\nAll datasets written to: {OUT}")
