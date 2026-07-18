from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data"/"raw"/"rki_covid_hospitalizations.tsv",sep="\t")
df=df[df["agegroup"].eq("00+")].copy()
df["date"]=pd.to_datetime(df["date"]+"-1",format="%G-W%V-%u",errors="coerce")
out=pd.DataFrame({"date":df["date"],"disease":"COVID-19","country":"Germany","region_id":"DE","region_name":"Germany","signal_type":"hospitalization_incidence","value":pd.to_numeric(df["sari_covid19_incidence"],errors="coerce"),"population":None,"source":"RKI COVID-SARI-Hospitalisierungsinzidenz","is_smoothed":False,"is_imputed":False,"quality_flag":"weekly_sentinel"}).dropna(subset=["date","value"])
target=ROOT/"data"/"processed"/"rki_hospitalization_canonical.csv"; target.parent.mkdir(parents=True,exist_ok=True); out.to_csv(target,index=False); print(target,len(out))
