"""
Download and preprocess real epidemiological datasets for EpiSurveil Model Explorer.

Sources
-------
1. Ebola West Africa 2014-2016
   - cmrivers/ebola GitHub (CC0 / public domain)
   - Cumulative cases/deaths by country → differenced to weekly new cases
   - Covers: Guinea (most complete), Sierra Leone, Liberia

2. Influenza Europe weekly ILI surveillance (EU-ECDC)
   - EU-ECDC/Respiratory_viruses_weekly_data GitHub (ECDC open data)
   - Weekly ILI consultation rate per 100k → approximate case count
   - Covers: Germany, France, Netherlands (past seasons)

Output
------
All CSVs written to data/examples/real/ in the simple format expected by the
Model Explorer:  date (YYYY-MM-DD), cases [, deaths]
Weekly datasets keep one row per week; the BPF runs episode-by-episode.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "examples" / "real"
OUT.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fetch(url: str) -> pd.DataFrame:
    print(f"  Fetching {url} …")
    try:
        return pd.read_csv(url)
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Ebola West Africa 2014-2016  (Guinea, Sierra Leone, Liberia)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n-- Ebola West Africa 2014-2016 ------------------------------------------")
URL_EBOLA = (
    "https://raw.githubusercontent.com/cmrivers/ebola/master/country_timeseries.csv"
)
raw = _fetch(URL_EBOLA)

if raw is not None:
    raw["date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    raw = raw.sort_values("date").set_index("date")

    for country_col, label, pop, fname in [
        ("Guinea",      "Guinea",       2_000_000, "ebola_guinea_2014_16.csv"),
        ("SierraLeone", "Sierra Leone", 3_000_000, "ebola_sierraleone_2014_16.csv"),
    ]:
        c_col = f"Cases_{country_col}"
        d_col = f"Deaths_{country_col}"
        if c_col not in raw.columns:
            print(f"  Column {c_col} not found, skipping {label}")
            continue

        sub = raw[[c_col, d_col]].copy()
        sub.columns = ["cum_cases", "cum_deaths"]

        # Keep only rows that have at least one non-NaN value
        sub = sub[sub["cum_cases"].notna() | sub["cum_deaths"].notna()]

        # Resample to daily, then linearly interpolate cumulative counts between
        # actual reporting dates.  This converts the WHO situation-report
        # "cumulative total on date X" series into a smooth daily incidence
        # curve without the 0-spike artefact that forward-fill + diff creates.
        sub = sub.resample("D").first()
        sub["cum_cases"]  = sub["cum_cases"].interpolate(method="linear")
        sub["cum_deaths"] = sub["cum_deaths"].interpolate(method="linear")

        # Diff interpolated cumulative → smooth daily new cases / deaths.
        # Keep as float and set sub-threshold values to NaN: days where
        # interpolation rounds to zero are reporting gaps, not true zero-
        # case days.  BPF skips NaN rows in the likelihood (no weight update).
        sub["cases"]  = sub["cum_cases"].diff().clip(lower=0)
        sub["deaths"] = sub["cum_deaths"].diff().clip(lower=0)
        sub.loc[sub["cases"]  < 0.5, "cases"]  = np.nan
        sub.loc[sub["deaths"] < 0.5, "deaths"] = np.nan

        sub = sub.iloc[1:][["cases", "deaths"]].reset_index()

        sub.to_csv(OUT / fname, index=False)
        print(f"  {fname}: {len(sub)} rows, "
              f"peak cases={sub['cases'].max()}, "
              f"peak deaths={sub['deaths'].max()}, "
              f"total cases={sub['cases'].sum()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Influenza Europe — EU-ECDC weekly ILI consultation rates
# ═══════════════════════════════════════════════════════════════════════════════

print("\n-- Influenza Europe (ECDC weekly ILI) -----------------------------------")
URL_FLU = (
    "https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data"
    "/main/data/ILIARIRates.csv"
)
raw_flu = _fetch(URL_FLU)

if raw_flu is not None:
    # Keep primary care syndromic, ILI consultation rate, total age group
    flu = raw_flu[
        (raw_flu["survtype"] == "primary care syndromic") &
        (raw_flu["indicator"] == "ILIconsultationrate") &
        (raw_flu["age"] == "total")
    ].copy()

    def _yearweek_to_date(yw: str) -> pd.Timestamp:
        """'2024-W40' → Monday of that ISO week."""
        year, wk = yw.split("-W")
        return pd.Timestamp.fromisocalendar(int(year), int(wk), 1)

    flu["date"] = flu["yearweek"].apply(_yearweek_to_date)
    flu = flu.sort_values(["date"])

    # Available countries in the dataset
    available = sorted(flu["countryname"].unique())
    print(f"  Available countries: {available}")

    # Seasons to extract per country
    # Find the earliest date available to pick valid seasons
    earliest = flu["date"].min()
    latest   = flu["date"].max()
    print(f"  Date range in dataset: {earliest.date()} to {latest.date()}")

    TARGETS = [
        # (country, N_pop, season_start, season_end, output_fname)
        ("France",      68_000_000, "2023-10-01", "2024-03-31",
         "influenza_france_2023_24_real.csv"),
        ("Netherlands", 17_900_000, "2022-10-01", "2023-03-31",
         "influenza_netherlands_2022_23_real.csv"),
        ("Spain",       47_000_000, "2023-10-01", "2024-03-31",
         "influenza_spain_2023_24_real.csv"),
        ("Poland",      38_000_000, "2023-10-01", "2024-03-31",
         "influenza_poland_2023_24_real.csv"),
    ]

    for country, N_pop, t0, t1, fname in TARGETS:
        if country not in available:
            print(f"  {country} not in dataset, skipping")
            continue

        sub = flu[
            (flu["countryname"] == country) &
            (flu["date"] >= t0) &
            (flu["date"] <= t1)
        ][["date", "value"]].copy()

        if sub.empty:
            print(f"  No data for {country} {t0}–{t1}, skipping")
            continue

        # ILI rate (per 100k per week) -> daily case count then forward-fill to
        # produce a daily series so the BPF runs with standard per-day parameters.
        sub["daily_cases"] = (sub["value"] / 100_000 * N_pop / 7).round().astype(int)

        # Forward-fill: each Monday value applies to Mon-Sun of that week
        sub = sub.set_index("date")[["daily_cases", "value"]]
        daily = sub.resample("D").first().ffill().reset_index()
        daily.columns = ["date", "cases", "ILI_rate_per_100k"]
        daily["cases"] = daily["cases"].astype(int)

        daily.to_csv(OUT / fname, index=False)
        print(f"  {fname}: {len(daily)} daily rows (forward-filled from {len(sub)} weeks), "
              f"peak cases/day~{daily['cases'].max()}, "
              f"peak ILI rate={daily['ILI_rate_per_100k'].max():.1f}/100k")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Ebola DRC 2018-2020  (Andersen Lab / WHO)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n-- Ebola DRC 2018-2020 (Andersen Lab / WHO) -----------------------------")
URL_EBOLA_DRC = (
    "https://raw.githubusercontent.com/andersen-lab/ebola-drc-epidemiology"
    "/master/dataonly_ebolaDRC.csv"
)
raw_drc = _fetch(URL_EBOLA_DRC)

if raw_drc is not None:
    # Wide format: location rows, date-range columns (weekly periods)
    # Identify date columns (they look like "5 Aug 2018 - 12 Aug 2018")
    meta_cols = [c for c in raw_drc.columns if " - " not in c]
    date_cols  = [c for c in raw_drc.columns if " - " not in c is False or " - " in c]
    # Re-detect: date columns contain " - "
    date_cols = [c for c in raw_drc.columns if " - " in c]

    if not date_cols:
        print("  Could not detect date columns — skipping")
    else:
        # Sum confirmed cases across all districts
        confirmed = raw_drc[raw_drc.iloc[:, 2] == "Confirmed"] if raw_drc.shape[1] > 2 else raw_drc
        # Find which column has case definition
        case_def_col = None
        for col in raw_drc.columns[:5]:
            if raw_drc[col].isin(["Confirmed", "Probable"]).any():
                case_def_col = col
                break

        if case_def_col is None:
            print("  Could not find case-definition column — skipping")
        else:
            confirmed = raw_drc[raw_drc[case_def_col] == "Confirmed"]
            total_by_week = confirmed[date_cols].sum(axis=0)

            # Parse start date from "5 Aug 2018 - 12 Aug 2018"
            dates = []
            for dc in date_cols:
                start_str = dc.split(" - ")[0].strip()
                try:
                    dates.append(pd.to_datetime(start_str, format="%d %b %Y"))
                except Exception:
                    dates.append(pd.NaT)

            drc_df = pd.DataFrame({"date": dates, "cases": total_by_week.values})
            drc_df = drc_df.dropna(subset=["date"]).sort_values("date")
            drc_df["cases"] = drc_df["cases"].clip(lower=0).round().astype(int)

            fname = "ebola_drc_2018_20_real.csv"
            drc_df.to_csv(OUT / fname, index=False)
            print(f"  {fname}: {len(drc_df)} rows (weekly), "
                  f"peak cases={drc_df['cases'].max()}, "
                  f"total cases={drc_df['cases'].sum()}")


print(f"\nAll real datasets written to: {OUT}")
print("\nNOTE: These datasets contain weekly rows — the BPF will treat each row")
print("as one time step. Use the 'Weekly data' label when running the Model Explorer.")
