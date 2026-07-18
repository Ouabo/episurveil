"""Train/test observation-scale calibration using chronological splitting."""
from pathlib import Path
import json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; panel=pd.read_csv(ROOT/"data/processed/germany_integrated_panel.csv"); pred=pd.read_csv(ROOT/"data/processed/germany_multichannel_filter_output.csv"); n=len(panel); cut=int(.7*n); mapping={"reported_cases":"cases_mean","deaths":"deaths_mean","icu_occupancy":"icu_mean","hospitalization_proxy":"hospitalization_proxy_mean"}
scales={k:float(panel[k].iloc[:cut].mean()/max(pred[v].iloc[:cut].mean(),1e-9)) for k,v in mapping.items()}; metrics={}
for obs,pcol in mapping.items():
    y=panel[obs].iloc[cut:].to_numpy(float); p=pred[pcol].iloc[cut:].to_numpy(float)*scales[obs]; metrics[obs]={"scale":scales[obs],"test_rmse":float(np.sqrt(np.mean((y-p)**2))),"test_mae":float(np.mean(np.abs(y-p))),"test_mean_observed":float(y.mean()),"test_mean_predicted":float(p.mean())}
out={"train_fraction":.7,"metrics":metrics,"method":"chronological train/test moment calibration"}; (ROOT/"data/processed/germany_heldout_scale_calibration.json").write_text(json.dumps(out,indent=2),encoding="utf-8"); print(json.dumps(out,indent=2))
