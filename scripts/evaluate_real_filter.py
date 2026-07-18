from pathlib import Path
import json, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; panel=pd.read_csv(ROOT/"data/processed/germany_integrated_panel.csv"); pred=pd.read_csv(ROOT/"data/processed/germany_real_filter_output.csv")
y=panel["reported_cases"].to_numpy(float); m=pred["pred_cases_mean"].to_numpy(float); lo=pred["pred_cases_q10"].to_numpy(float); hi=pred["pred_cases_q90"].to_numpy(float)
mask=np.isfinite(y)&np.isfinite(m); metrics={"n":int(mask.sum()),"rmse":float(np.sqrt(np.mean((y[mask]-m[mask])**2))),"mae":float(np.mean(np.abs(y[mask]-m[mask]))),"coverage_80":float(np.mean((y[mask]>=lo[mask])&(y[mask]<=hi[mask]))),"mean_interval_width":float(np.mean(hi[mask]-lo[mask])),"mean_ess":float(np.nanmean(pred["ess"]))}
(ROOT/"data/processed/germany_real_filter_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2))
