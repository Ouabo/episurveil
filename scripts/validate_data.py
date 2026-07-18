import sys; sys.path.insert(0,"src")
from pathlib import Path
import pandas as pd
from episurveil.data_schema import validate_columns
ROOT=Path(__file__).resolve().parents[1]
for f in (ROOT/"data"/"raw").glob("*.csv"):
    try:
        df=pd.read_csv(f); validate_columns(df); print(f.name,"valid",len(df))
    except Exception as exc: print(f.name,"not canonical:",exc)
