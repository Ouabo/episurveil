from pathlib import Path
import json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; panel=pd.read_csv(ROOT/"data/processed/germany_integrated_panel.csv"); pred=pd.read_csv(ROOT/"data/processed/germany_multichannel_filter_output.csv")
mapping={"cases_mean":"reported_cases","deaths_mean":"deaths","icu_mean":"icu_occupancy","hospitalization_proxy_mean":"hospitalization_proxy"}; metrics={}
for pcol,ocol in mapping.items():
    y=panel[ocol].to_numpy(float); p=pred[pcol].to_numpy(float); m=np.isfinite(y)&np.isfinite(p); metrics[ocol]={"rmse":float(np.sqrt(np.mean((y[m]-p[m])**2))),"mae":float(np.mean(np.abs(y[m]-p[m]))),"mean_observed":float(np.mean(y[m])),"mean_predicted":float(np.mean(p[m]))}
metrics["mean_ess"]=float(pred.ess.mean()); (ROOT/"data/processed/germany_multichannel_filter_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2))
