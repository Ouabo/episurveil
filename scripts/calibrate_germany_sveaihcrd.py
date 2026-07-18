"""Derive transparent SVEAIHCRD initialization values from German surveillance means."""
from pathlib import Path
import json, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; df=pd.read_csv(ROOT/"data/processed/germany_integrated_panel.csv")
q=.709; tau=.0021; gamma_h=1/12; tau_h=.03; delta_h=.019; gamma_c=1/10; delta_c=.01; los=8.
mean_cases=float(df.reported_cases.mean()); mean_hosp=float(df.hospitalization_proxy.mean()); mean_icu=float(df.icu_occupancy.mean()); mean_deaths=float(df.deaths.mean())
H=max(mean_hosp,1.); C=max(mean_icu,1.); I=max((gamma_h+tau_h+delta_h)*H/tau,1.); E=max(mean_cases/(q*.62*(1/4.5)),1.)
N=83_000_000.; S=max(N-E-I-H-C,1.); result={"initial_state":[S,0.,E,0.,I,H,C,0.,0.],"reporting_probability":q,"tau_i":tau,"delta_h":delta_h,"rho_icu":mean_icu/max(H,1.),"hospital_length_of_stay_days":los,"panel_means":{"cases":mean_cases,"hospitalization_proxy":mean_hosp,"icu":mean_icu,"deaths":mean_deaths},"notes":"Initialization adapter; not a posterior estimate."}
(ROOT/"data/processed/germany_sveaihcrd_initialization.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
