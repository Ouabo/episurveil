"""Integrate the existing German DIVI/RKI/OWID validation panel."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/"data"/"external"/"germany_real_covid_validation_input.csv"
df=pd.read_csv(src); df["date"]=pd.to_datetime(df["date"],errors="coerce")
keep={"date":"date","reported_cases":"reported_cases","new_deaths":"deaths","icu_patients":"icu_occupancy","hosp_proxy":"hospitalization_proxy","people_vaccinated_per_hundred":"vaccinated_pct","people_fully_vaccinated_per_hundred":"fully_vaccinated_pct"}
out=df[list(keep)].rename(columns=keep)
out["admissions_proxy"] = out["hospitalization_proxy"]
weekly_path=ROOT/"data/processed/rki_hospitalization_canonical.csv"
if weekly_path.exists():
    wk=pd.read_csv(weekly_path,parse_dates=["date"])[["date","value"]].rename(columns={"value":"rki_weekly_hospitalization_incidence"}).sort_values("date")
    out=pd.merge_asof(out.sort_values("date"),wk,on="date",direction="backward")
out["disease"]="COVID-19"; out["country"]="Germany"; out["region_id"]="DE"; out["source_combination"]="RKI cases/deaths + DIVI ICU + RKI admissions proxy + OWID vaccination"
out=out.sort_values("date"); target=ROOT/"data"/"processed"/"germany_integrated_panel.csv"; target.parent.mkdir(parents=True,exist_ok=True); out.to_csv(target,index=False); print(target,len(out),out.date.min(),out.date.max())
