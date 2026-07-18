"""Extend germany_integrated_panel.csv from March 2023 to last available SARI date.

Germany stopped mandatory daily COVID-19 case reporting on 7 April 2023.
After that date, only weekly SARI hospitalization sentinel data is available
(RKI COVID-SARI-Hospitalisierungsinzidenz, updated through Oct 2024 in local data).

Extension strategy:
  - reported_cases, deaths, icu_occupancy : NaN  (no data; filter skips these)
  - hospitalization_proxy                 : SARI-calibrated daily estimate
  - vaccinated_pct / fully_vaccinated_pct : frozen at last known values (77.82 / 76.24)

Calibration: compute scale factor from the last 8 weeks of overlap between
the existing hosp_proxy and the SARI incidence, then apply uniformly.

Usage:
  python scripts/extend_panel.py
  python scripts/run_sveaihcrd_validation.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT           = Path(__file__).resolve().parents[1]
PANEL_CSV      = ROOT / "data/processed/germany_integrated_panel.csv"
SARI_CSV       = ROOT / "data/processed/rki_hospitalization_canonical.csv"
N_POPULATION   = 83_200_000
REPORTING_END  = pd.Timestamp("2023-04-07")  # Germany stopped daily reporting


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load existing panel
    # ------------------------------------------------------------------
    panel = pd.read_csv(PANEL_CSV, parse_dates=["date"])
    panel_end = panel["date"].max()
    print(f"Existing panel: {panel['date'].min().date()} to {panel_end.date()} ({len(panel)} rows)")

    # ------------------------------------------------------------------
    # 2. Load SARI weekly sentinel data (all-age 00+ group)
    # ------------------------------------------------------------------
    sari_all = pd.read_csv(SARI_CSV, parse_dates=["date"])
    # signal_type may be 'hospitalization_incidence' or similar
    sari_all = sari_all[sari_all["signal_type"].str.contains("hospitalization", case=False)]
    # Keep only rows that have a single 'all-ages' value; the canonical already
    # filtered to 00+ in harmonize_rki_hospitalizations.py
    sari_weekly = (
        sari_all[["date", "value"]]
        .rename(columns={"value": "sari_incidence"})
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    sari_end = sari_weekly["date"].max()
    print(f"SARI data:      {sari_weekly['date'].min().date()} to {sari_end.date()} ({len(sari_weekly)} weeks)")

    # ------------------------------------------------------------------
    # 3. Calibrate SARI to hosp_proxy using the last 8 overlapping weeks
    # ------------------------------------------------------------------
    # Panel hosp_proxy is in persons (absolute count, ≈ occupancy)
    # SARI incidence is per 100,000 to convert to persons: sari * N_pop / 1e5
    overlap_start = panel_end - pd.Timedelta(weeks=8)
    panel_overlap = panel[panel["date"] >= overlap_start][["date", "hospitalization_proxy"]].copy()
    sari_overlap  = sari_weekly[sari_weekly["date"] >= overlap_start].copy()

    # Merge: assign each panel date the most recent weekly SARI value
    panel_overlap_s = pd.merge_asof(
        panel_overlap.sort_values("date"),
        sari_overlap.sort_values("date"),
        on="date",
        direction="backward",
    ).dropna()

    if len(panel_overlap_s) >= 4:
        sari_abs = panel_overlap_s["sari_incidence"] * N_POPULATION / 1e5
        hosp_p   = panel_overlap_s["hospitalization_proxy"]
        ratio = (hosp_p / sari_abs.replace(0, np.nan)).dropna()
        scale_factor = float(ratio.median())
        print(f"SARI calibration: scale_factor = {scale_factor:.3f}  "
              f"(median over {len(ratio)} overlapping points)")
    else:
        # Fallback: use last known hosp_proxy / last known SARI
        last_hosp = float(panel["hospitalization_proxy"].dropna().iloc[-1])
        last_sari = float(sari_weekly["sari_incidence"].iloc[-1])
        scale_factor = last_hosp / (last_sari * N_POPULATION / 1e5)
        print(f"SARI calibration fallback: scale_factor = {scale_factor:.3f}")

    # ------------------------------------------------------------------
    # 4. Build daily extension skeleton (day after panel_end to sari_end)
    # ------------------------------------------------------------------
    ext_dates = pd.date_range(panel_end + pd.Timedelta(days=1), sari_end, freq="D")
    if len(ext_dates) == 0:
        print("Nothing to extend — panel already reaches SARI end date.")
        return

    ext = pd.DataFrame({"date": ext_dates})

    # Assign daily SARI by carrying each weekly value forward 7 days
    ext = pd.merge_asof(ext, sari_weekly, on="date", direction="backward")
    ext["hosp_proxy_sari"] = ext["sari_incidence"] * N_POPULATION / 1e5 * scale_factor

    # ------------------------------------------------------------------
    # 5. Freeze vaccination at last known values
    # ------------------------------------------------------------------
    last_vacc       = float(panel["vaccinated_pct"].dropna().iloc[-1])
    last_fully_vacc = float(panel["fully_vaccinated_pct"].dropna().iloc[-1])
    ext["vaccinated_pct"]       = last_vacc
    ext["fully_vaccinated_pct"] = last_fully_vacc

    # ------------------------------------------------------------------
    # 6. All other channels are NaN (not observed)
    # ------------------------------------------------------------------
    ext["reported_cases"]            = np.nan
    ext["deaths"]                    = np.nan
    ext["icu_occupancy"]             = np.nan
    ext["hospitalization_proxy"]     = ext["hosp_proxy_sari"]
    ext["admissions_proxy"]          = ext["hosp_proxy_sari"]
    ext["rki_weekly_hospitalization_incidence"] = ext["sari_incidence"]

    # Carry over metadata columns
    ext["disease"]           = "COVID-19"
    ext["country"]           = "Germany"
    ext["region_id"]         = "DE"
    ext["source_combination"] = (
        "SARI sentinel only (RKI COVID-SARI-Hospitalisierungsinzidenz); "
        "cases/deaths/ICU not available after April 2023"
    )

    # Keep same columns as existing panel
    ext = ext[[c for c in panel.columns if c in ext.columns] +
               [c for c in ext.columns if c not in panel.columns]]
    ext = ext.reindex(columns=panel.columns)

    # ------------------------------------------------------------------
    # 7. Append and save
    # ------------------------------------------------------------------
    combined = pd.concat([panel, ext], ignore_index=True)
    combined.to_csv(PANEL_CSV, index=False)
    print(f"\nExtended panel: {combined['date'].min().date()} to {combined['date'].max().date()} "
          f"({len(combined)} rows, +{len(ext)} extension days)")
    print(f"Wrote {PANEL_CSV}")

    # Quick sanity check at the join boundary
    boundary = combined[combined["date"].between(panel_end - pd.Timedelta(days=3),
                                                  panel_end + pd.Timedelta(days=3))]
    print("\nBoundary check (hosp_proxy at join):")
    print(boundary[["date", "reported_cases", "hospitalization_proxy",
                     "vaccinated_pct", "icu_occupancy"]].to_string(index=False))


if __name__ == "__main__":
    main()
