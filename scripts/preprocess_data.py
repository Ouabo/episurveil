"""Convert canonical surveillance files to model-ready long form."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
for f in (ROOT/"data"/"raw").glob("*.csv"):
    df=pd.read_csv(f)
    if "date" in df: df["date"]=pd.to_datetime(df["date"],errors="coerce",utc=True)
    if "value" in df: df["value"]=pd.to_numeric(df["value"],errors="coerce")
    df=df.dropna(subset=[c for c in ("date","value") if c in df])
    out=ROOT/"data"/"processed"/f.name; out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False); print(out)
