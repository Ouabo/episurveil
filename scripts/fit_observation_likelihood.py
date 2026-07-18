from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom

ROOT = Path(__file__).resolve().parents[1]
panel = pd.read_csv(ROOT / "data/processed/germany_integrated_panel.csv")
pred = pd.read_csv(ROOT / "data/processed/germany_multichannel_filter_output.csv")
cut = int(0.70 * len(panel))
mapping = {
    "reported_cases": "cases_mean",
    "deaths": "deaths_mean",
    "icu_occupancy": "icu_mean",
    "hospitalization_proxy": "hospitalization_proxy_mean",
}
phi = {"reported_cases": 80.0, "deaths": 10.0, "icu_occupancy": 12.0, "hospitalization_proxy": 10.0}
keys = list(mapping)

def objective(log_scales):
    total = 0.0
    for i, key in enumerate(keys):
        y = np.maximum(np.rint(panel[key].iloc[:cut].to_numpy(float)), 0).astype(int)
        base = np.maximum(pred[mapping[key]].iloc[:cut].to_numpy(float), 1e-6)
        mu = np.maximum(np.exp(log_scales[i]) * base, 1e-6)
        r = phi[key]
        total -= np.sum(nbinom.logpmf(y, r, r / (r + mu)))
    return float(total)

fit = minimize(objective, np.zeros(len(keys)), method="L-BFGS-B", bounds=[(-3, 2)] * len(keys))
scales = {key: float(np.exp(fit.x[i])) for i, key in enumerate(keys)}
out = {
    "success": bool(fit.success),
    "objective": float(fit.fun),
    "train_fraction": 0.70,
    "scales": scales,
    "dispersion": phi,
    "method": "negative-binomial maximum likelihood on training split",
}
(ROOT / "data/processed/germany_likelihood_calibration.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
