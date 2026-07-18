"""
Live epidemic data connectors for EpiSurveil Model Explorer.

Sources
-------
1. COVID-19 — disease.sh open API (no auth, daily counts)
2. Influenza — WHO FluNet public CSV download
3. ECDC Respiratory — GitHub CSV (weekly ILI + flu confirmation)

All functions return a DataFrame with columns:
  date (datetime64[ns]), cases (float), deaths (float, optional)
in chronological order, ready to feed directly into run_bpf().

NaN values mark reporting gaps; the BPF skips them automatically.
"""
from __future__ import annotations

import io
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

_TIMEOUT = 20  # seconds per HTTP request


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "EpiSurveil/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return r.read()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  COVID-19  —  disease.sh
# ─────────────────────────────────────────────────────────────────────────────

_DISEASE_SH_BASE = "https://disease.sh/v3/covid-19"

_COVID_COUNTRIES: dict[str, str] = {
    "Germany":        "germany",
    "France":         "france",
    "United Kingdom": "uk",
    "United States":  "usa",
    "Brazil":         "brazil",
    "India":          "india",
    "Italy":          "italy",
    "Spain":          "spain",
    "Japan":          "japan",
    "South Korea":    "south%20korea",
    "South Africa":   "south%20africa",
    "Nigeria":        "nigeria",
    "Pakistan":       "pakistan",
    "Indonesia":      "indonesia",
    "Mexico":         "mexico",
    "Canada":         "canada",
    "Australia":      "australia",
    "Argentina":      "argentina",
    "Poland":         "poland",
    "Netherlands":    "netherlands",
}


def covid19_countries() -> list[str]:
    """Return the list of supported COVID-19 countries."""
    return sorted(_COVID_COUNTRIES)


def fetch_covid19(
    country: str,
    days: int = 180,
) -> pd.DataFrame:
    """
    Fetch daily COVID-19 cases and deaths from disease.sh.

    Parameters
    ----------
    country : display name from covid19_countries()
    days    : how many recent days to return (max ~365)

    Returns
    -------
    DataFrame with columns: date, cases, deaths
    """
    slug = _COVID_COUNTRIES.get(country)
    if slug is None:
        raise ValueError(f"Unknown country '{country}'. Use covid19_countries() for valid names.")

    url = f"{_DISEASE_SH_BASE}/historical/{slug}?lastdays={days}"
    raw = _get(url)

    import json
    data = json.loads(raw)

    timeline = data.get("timeline", data)
    cases_raw  = timeline.get("cases",  {})
    deaths_raw = timeline.get("deaths", {})

    # date keys are "M/D/YY"
    def _parse(d: dict) -> pd.Series:
        idx = pd.to_datetime(list(d.keys()), format="%m/%d/%y")
        return pd.Series(list(d.values()), index=idx, dtype=float)

    s_cases  = _parse(cases_raw)
    s_deaths = _parse(deaths_raw)

    df = pd.DataFrame({"cum_cases": s_cases, "cum_deaths": s_deaths})
    df = df.sort_index()

    # Diff to get daily new counts
    df["cases"]  = df["cum_cases"].diff().clip(lower=0)
    df["deaths"] = df["cum_deaths"].diff().clip(lower=0)
    df.loc[df["cases"]  < 0.5, "cases"]  = np.nan
    df.loc[df["deaths"] < 0.5, "deaths"] = np.nan
    df = df.iloc[1:][["cases", "deaths"]].reset_index()
    df.columns = ["date", "cases", "deaths"]
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(how="all", subset=["cases", "deaths"])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Influenza lab-confirmed  —  ECDC sentinel surveillance (EU/EEA)
#     Source: EU-ECDC/Respiratory_viruses_weekly_data GitHub (updated weekly)
#     Replaces WHO FluNet which returned 404 after their API migration.
# ─────────────────────────────────────────────────────────────────────────────

_ECDC_SENTINEL_URL = (
    "https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data"
    "/main/data/sentinelTestsDetectionsPositivity.csv"
)

# Countries available in the sentinel dataset (EU/EEA + Norway)
_ECDC_SENTINEL_COUNTRIES: list[str] = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
    "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Malta", "Netherlands", "Norway", "Poland", "Portugal", "Romania",
    "Slovakia", "Slovenia", "Spain", "Sweden",
]

# Display name → name used in the CSV (Czechia vs "Czech Republic")
_ECDC_SENTINEL_NAME_MAP: dict[str, str] = {
    "Czech Republic": "Czechia",
}


def influenza_countries() -> list[str]:
    """Return the list of supported countries for ECDC sentinel flu data."""
    return sorted(_ECDC_SENTINEL_COUNTRIES)


def _isoweek_to_date(yearweek: str) -> pd.Timestamp:
    """'2024-W40' → Monday of that ISO week."""
    year, wk = yearweek.split("-W")
    return pd.Timestamp.fromisocalendar(int(year), int(wk), 1)


def fetch_influenza_flunet(
    country: str,
    seasons: int = 2,
) -> pd.DataFrame:
    """
    Fetch weekly lab-confirmed influenza detections from ECDC sentinel surveillance.

    Parameters
    ----------
    country : display name from influenza_countries()
    seasons : number of recent flu seasons (~52 weeks each) to include

    Returns
    -------
    DataFrame with columns: date (Monday of ISO week), cases (weekly detections)
    """
    if country not in _ECDC_SENTINEL_COUNTRIES:
        raise ValueError(
            f"'{country}' not available. Use influenza_countries() for valid names."
        )
    csv_name = _ECDC_SENTINEL_NAME_MAP.get(country, country)

    raw = _get(_ECDC_SENTINEL_URL)
    df  = pd.read_csv(io.BytesIO(raw))

    sub = df[
        (df["countryname"]      == csv_name) &
        (df["pathogen"]         == "Influenza") &
        (df["pathogensubtype"]  == "total") &
        (df["indicator"]        == "detections") &
        (df["age"]              == "total")
    ].copy()

    if sub.empty:
        raise ValueError(f"No sentinel flu data found for '{country}'.")

    sub["date"] = sub["yearweek"].apply(_isoweek_to_date)
    sub = sub.sort_values("date")

    sub["weekly"] = pd.to_numeric(sub["value"], errors="coerce").clip(lower=0)
    sub.loc[sub["weekly"] < 0.5, "weekly"] = np.nan

    cutoff = sub["date"].max() - pd.Timedelta(weeks=seasons * 52)
    sub = sub[sub["date"] >= cutoff]

    # Convert weekly counts to daily: spread each week's total over 7 days.
    # ffill(limit=6) fills Mon→Sun of each valid week (6 gap days) but does
    # NOT bleed into a subsequent NaN week (7th+ day exceeds limit).
    # NaN off-season weeks remain NaN so the BPF skips their likelihood update.
    sub["daily"] = sub["weekly"] / 7.0
    daily = (
        sub.set_index("date")[["daily"]]
        .resample("D").first()
        .ffill(limit=6)
        .reset_index()
    )
    daily.columns = ["date", "cases"]
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ECDC Respiratory surveillance  —  GitHub CSV (ILI + ARI)
# ─────────────────────────────────────────────────────────────────────────────

_ECDC_ILI_URL = (
    "https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data"
    "/main/data/ILIARIRates.csv"
)

_ECDC_COUNTRIES: dict[str, str] = {
    "Austria":         "Austria",
    "Belgium":         "Belgium",
    "Bulgaria":        "Bulgaria",
    "Croatia":         "Croatia",
    "Cyprus":          "Cyprus",
    "Czech Republic":  "Czechia",
    "Denmark":         "Denmark",
    "Estonia":         "Estonia",
    "Finland":         "Finland",
    "France":          "France",
    "Germany":         "Germany",
    "Greece":          "Greece",
    "Hungary":         "Hungary",
    "Iceland":         "Iceland",
    "Ireland":         "Ireland",
    "Italy":           "Italy",
    "Latvia":          "Latvia",
    "Lithuania":       "Lithuania",
    "Luxembourg":      "Luxembourg",
    "Malta":           "Malta",
    "Netherlands":     "Netherlands",
    "Norway":          "Norway",
    "Poland":          "Poland",
    "Portugal":        "Portugal",
    "Romania":         "Romania",
    "Slovakia":        "Slovakia",
    "Slovenia":        "Slovenia",
    "Spain":           "Spain",
    "Sweden":          "Sweden",
}

# Countries that only report ARI (Acute Respiratory Infection), not ILI
_ARI_ONLY: set[str] = {"Germany", "Bulgaria", "Cyprus"}

_ECDC_POP: dict[str, int] = {
    "Austria": 9_100_000,   "Belgium": 11_600_000,  "Bulgaria": 6_900_000,
    "Croatia": 3_900_000,   "Cyprus":  920_000,      "Czechia": 10_900_000,
    "Denmark": 5_900_000,   "Estonia": 1_400_000,    "Finland": 5_500_000,
    "France":  68_000_000,  "Germany": 83_200_000,   "Greece": 10_600_000,
    "Hungary": 9_800_000,   "Iceland": 380_000,      "Ireland": 5_100_000,
    "Italy":   60_300_000,  "Latvia":  1_900_000,    "Lithuania": 2_900_000,
    "Luxembourg": 660_000,  "Malta":   540_000,      "Netherlands": 17_900_000,
    "Norway":  5_500_000,   "Poland":  38_000_000,   "Portugal": 10_300_000,
    "Romania": 19_000_000,  "Slovakia": 5_500_000,   "Slovenia": 2_100_000,
    "Spain":   47_400_000,  "Sweden":  10_500_000,
}


def ecdc_ili_countries() -> list[str]:
    """Return the list of supported ECDC ILI countries."""
    return sorted(_ECDC_COUNTRIES)


def fetch_ecdc_ili(
    country: str,
    seasons: int = 2,
) -> pd.DataFrame:
    """
    Fetch weekly ILI consultation rate from ECDC and convert to daily case counts.

    Parameters
    ----------
    country : display name from ecdc_ili_countries()
    seasons : number of recent flu seasons (each ~52 weeks) to include

    Returns
    -------
    DataFrame with columns: date, cases (daily forward-filled), ILI_rate_per_100k
    """
    ecdc_name = _ECDC_COUNTRIES.get(country)
    if ecdc_name is None:
        raise ValueError(f"Unknown country '{country}'. Use ecdc_ili_countries() for valid names.")

    raw  = _get(_ECDC_ILI_URL)
    df   = pd.read_csv(io.BytesIO(raw))

    # Prefer ILI; fall back to ARI for countries that only report ARI
    # (Germany, Bulgaria, Cyprus report ARIconsultationrate only)
    _base = df[
        (df["countryname"] == ecdc_name) &
        (df["survtype"] == "primary care syndromic") &
        (df["age"] == "total")
    ]
    sub = _base[_base["indicator"] == "ILIconsultationrate"].copy()
    _used_indicator = "ILI"
    if sub.empty:
        sub = _base[_base["indicator"] == "ARIconsultationrate"].copy()
        _used_indicator = "ARI"
    if sub.empty:
        raise ValueError(
            f"No ECDC ILI/ARI data for '{country}' ('{ecdc_name}'). "
            "Country may not be in the sentinel network."
        )

    def _yw_to_date(yw: str) -> pd.Timestamp:
        year, wk = yw.split("-W")
        return pd.Timestamp.fromisocalendar(int(year), int(wk), 1)

    sub["date"] = sub["yearweek"].apply(_yw_to_date)
    sub = sub.sort_values("date")

    cutoff = sub["date"].max() - pd.Timedelta(weeks=seasons * 52)
    sub = sub[sub["date"] >= cutoff]

    N = _ECDC_POP.get(ecdc_name, _ECDC_POP.get(country, 10_000_000))
    sub["daily_cases"] = (sub["value"] / 100_000 * N / 7).round().astype(int)
    sub = sub.set_index("date")[["daily_cases", "value"]]
    daily = sub.resample("D").first().ffill().reset_index()
    rate_col = f"{_used_indicator}_rate_per_100k"
    daily.columns = ["date", "cases", rate_col]
    daily["cases"] = daily["cases"].astype(float)
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# Registry for the UI
# ─────────────────────────────────────────────────────────────────────────────

LIVE_SOURCES: dict[str, dict] = {
    "COVID-19 (disease.sh)": {
        "description": "Daily confirmed cases and deaths from disease.sh open API. "
                       "Data ends ~March 2023 for most countries (governments stopped mandatory reporting).",
        "countries_fn": covid19_countries,
        "fetch_fn":     fetch_covid19,
        "obs_channels": ["cases", "deaths"],
        "suggested_model": "SEIRD",
        "suggested_N_pop": {
            "Germany": 83_200_000, "France": 68_000_000,
            "United Kingdom": 67_000_000, "United States": 330_000_000,
            "Brazil": 215_000_000, "India": 1_400_000_000,
            "Italy": 60_300_000, "Spain": 47_400_000,
            "Japan": 125_000_000, "South Korea": 52_000_000,
            "South Africa": 60_000_000, "Nigeria": 220_000_000,
            "Pakistan": 230_000_000, "Indonesia": 275_000_000,
            "Mexico": 128_000_000, "Canada": 38_000_000,
            "Australia": 26_000_000, "Argentina": 45_000_000,
            "Poland": 38_000_000, "Netherlands": 17_900_000,
        },
    },
    "Influenza lab-confirmed (ECDC)": {
        "description": "Weekly lab-confirmed influenza detections from ECDC sentinel GP network "
                       "(EU/EEA + Norway). Updated weekly, current to present.",
        "countries_fn": influenza_countries,
        "fetch_fn":     fetch_influenza_flunet,
        "obs_channels": ["cases"],
        "suggested_model": "SEIR",
        "suggested_N_pop": {k: _ECDC_POP.get(_ECDC_SENTINEL_NAME_MAP.get(k, k), 10_000_000)
                            for k in _ECDC_SENTINEL_COUNTRIES},
    },
    "ILI Surveillance (ECDC)": {
        "description": "Weekly ILI consultation rate (EU primary care, ECDC) converted to daily counts.",
        "countries_fn": ecdc_ili_countries,
        "fetch_fn":     fetch_ecdc_ili,
        "obs_channels": ["cases"],
        "suggested_model": "SEIR",
        "suggested_N_pop": {k: _ECDC_POP.get(v, 10_000_000)
                            for k, v in _ECDC_COUNTRIES.items()},
    },
}
