import sys; sys.path.insert(0,"src")
from pathlib import Path
import json
import os
import numpy as np,pandas as pd
from episurveil.models.sveaihcrd import Parameters,rhs
from episurveil.inference.particle_filter import sir_filter_multichannel
ROOT=Path(__file__).resolve().parents[1]; df=pd.read_csv(ROOT/"data/processed/germany_integrated_panel.csv"); p=Parameters(tau_i=.0021,delta_h=.019)
process_noise_scale=float(os.getenv("PROCESS_NOISE_SCALE", "0.003"))
scale_path=ROOT/"data/processed/germany_likelihood_calibration.json"
scales=json.loads(scale_path.read_text())["scales"] if (scale_path.exists() and os.getenv("USE_LIKELIHOOD_SCALES") == "1") else {"reported_cases":1.,"deaths":1.,"icu_occupancy":1.,"hospitalization_proxy":1.}
x0=np.array([82108587.,0.,357346.,0.,523893.,8314.,1860.,0.,0.])
coverage=df["vaccinated_pct"].fillna(0).to_numpy(float); daily_v=np.maximum(np.diff(np.r_[coverage[0],coverage])/100.,0.)
def transition(x,rng,row):
    nu=float(np.clip(row.get("daily_vaccination_rate",0.),0.,.02)); pp=Parameters(tau_i=p.tau_i,delta_h=p.delta_h,nu=nu)
    return np.maximum(x+rhs(0,x,pp)+rng.normal(0,process_noise_scale*np.sqrt(np.maximum(x,1)),len(x)),0.)
means={"cases":lambda x:scales["reported_cases"]*.709*.62*(1/4.5)*x[:,2],"deaths":lambda x:scales["deaths"]*(p.delta_h*x[:,5]+p.delta_c*x[:,6]),"icu":lambda x:scales["icu_occupancy"]*x[:,6],"hospitalization_proxy":lambda x:scales["hospitalization_proxy"]*x[:,5]}
rows=df[["reported_cases","deaths","icu_occupancy","hospitalization_proxy"]].rename(columns={"reported_cases":"cases","icu_occupancy":"icu"}).to_dict("records")
for i,row in enumerate(rows): row["daily_vaccination_rate"]=daily_v[i]
res=sir_filter_multichannel(rows,transition,means,x0,n_particles=250,seed=20260716,weights={"cases":1.,"deaths":.4,"icu":.6,"hospitalization_proxy":.1},dispersions={"cases":80,"deaths":10,"icu":12,"hospitalization_proxy":10})
case_factor=scales["reported_cases"]*.709*.62*(1/4.5)
out=pd.DataFrame({"date":df.date,
 "cases_mean":[r["mean"][2]*case_factor for r in res],"cases_q10":[r["q10"][2]*case_factor for r in res],"cases_q90":[r["q90"][2]*case_factor for r in res],
 "deaths_mean":[scales["deaths"]*(r["mean"][5]*p.delta_h+r["mean"][6]*p.delta_c) for r in res],"deaths_q10":[scales["deaths"]*(r["q10"][5]*p.delta_h+r["q10"][6]*p.delta_c) for r in res],"deaths_q90":[scales["deaths"]*(r["q90"][5]*p.delta_h+r["q90"][6]*p.delta_c) for r in res],
 "icu_mean":[scales["icu_occupancy"]*r["mean"][6] for r in res],"icu_q10":[scales["icu_occupancy"]*r["q10"][6] for r in res],"icu_q90":[scales["icu_occupancy"]*r["q90"][6] for r in res],
 "hospitalization_proxy_mean":[scales["hospitalization_proxy"]*r["mean"][5] for r in res],"hospitalization_proxy_q10":[scales["hospitalization_proxy"]*r["q10"][5] for r in res],"hospitalization_proxy_q90":[scales["hospitalization_proxy"]*r["q90"][5] for r in res],"ess":[r["ess"] for r in res]})
out.to_csv(ROOT/"data/processed/germany_multichannel_filter_output.csv",index=False); print(out.tail(1).to_dict("records")[0])
