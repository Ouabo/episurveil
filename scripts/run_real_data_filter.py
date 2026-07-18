"""Run a transparent case-channel SIR filter on the integrated German panel."""
import sys; sys.path.insert(0,"src")
from pathlib import Path
import numpy as np, pandas as pd
from episurveil.models.sveaihcrd import Parameters,rhs
from episurveil.inference.particle_filter import sir_filter
ROOT=Path(__file__).resolve().parents[1]; df=pd.read_csv(ROOT/"data/processed/germany_integrated_panel.csv")
p=Parameters(tau_i=.0021,delta_h=.019); x0=np.array([83_000_000.,0,100.,100.,1000.,100.,20.,0.,0.])
def transition(x,rng): return np.maximum(x+rhs(0,x,p)+rng.normal(0,.01*np.sqrt(np.maximum(x,1)),len(x)),0.)
res=sir_filter(df["reported_cases"].fillna(0).to_numpy(),transition,lambda x: np.maximum(.709*.62*(1/4.5)*x[:,2],1e-3),x0,n_particles=300,seed=20260716)
out=pd.DataFrame({"date":df.date,"pred_cases_mean":[r["mean"][2]*.709*.62*(1/4.5) for r in res],"pred_cases_q10":[r["q10"][2]*.709*.62*(1/4.5) for r in res],"pred_cases_q90":[r["q90"][2]*.709*.62*(1/4.5) for r in res],"ess":[r["ess"] for r in res]})
out.to_csv(ROOT/"data/processed/germany_real_filter_output.csv",index=False); print(out.tail(1).to_dict("records")[0])
