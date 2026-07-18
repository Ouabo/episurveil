"""Diagnostic moment calibration of observation scales from joint-filter output."""
from pathlib import Path
import json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; panel=pd.read_csv(ROOT/"data/processed/germany_integrated_panel.csv"); pred=pd.read_csv(ROOT/"data/processed/germany_multichannel_filter_output.csv")
def ratio(obs,fit):
    a=float(np.nanmean(panel[obs])); b=float(np.nanmean(pred[fit])); return a/max(b,1e-9)
scales={"case_reporting_correction":ratio("reported_cases","cases_mean"),"death_scale":ratio("deaths","deaths_mean"),"icu_scale":ratio("icu_occupancy","icu_mean"),"hospitalization_scale":ratio("hospitalization_proxy","hospitalization_proxy_mean"),"method":"observed/predicted mean ratios; diagnostic moment calibration, not posterior inference"}
(ROOT/"data/processed/germany_observation_scale_calibration.json").write_text(json.dumps(scales,indent=2),encoding="utf-8"); print(json.dumps(scales,indent=2))
