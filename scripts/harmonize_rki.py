"""Harmonize the compact RKI national deaths/cases file into canonical long form."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data"/"raw"/"rki_covid_cases.csv")
df["date"]=pd.to_datetime(df["Berichtsdatum"],errors="coerce")
rows=[]
for signal,col in (("reported_cases","Faelle_gesamt"),("deaths","Todesfaelle_neu")):
    rows.append(pd.DataFrame({"date":df["date"],"disease":"COVID-19","country":"Germany","region_id":"DE","region_name":"Germany","signal_type":signal,"value":pd.to_numeric(df[col],errors="coerce"),"population":None,"source":"RKI COVID-19-Todesfaelle_in_Deutschland","is_smoothed":False,"is_imputed":False,"quality_flag":"raw"}))
out=pd.concat(rows,ignore_index=True).dropna(subset=["date","value"])
target=ROOT/"data"/"processed"/"rki_covid_canonical.csv"; target.parent.mkdir(parents=True,exist_ok=True); out.to_csv(target,index=False)
print(target,len(out))
