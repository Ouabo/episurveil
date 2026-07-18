import sys; sys.path.insert(0, "src")
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom
from episurveil.models.sveaihcrd import Parameters, rhs
from episurveil.observations.delays import delayed_incidence, occupancy_from_admissions

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/germany_integrated_panel.csv")
df = df.copy()
df["reported_cases"] = delayed_incidence(df["reported_cases"].to_numpy(float), mean_delay=5., sd_delay=2.)
df["hospitalization_proxy"] = occupancy_from_admissions(df["hospitalization_proxy"].to_numpy(float), mean_los=8.)
train = df.iloc[:int(.7 * len(df))]
xbase = np.array([82108587., 0., 1000., 0., 1000., 0., 0., 0., 0.])

def objective(z):
    beta0, beta1, q, tau_h, hosp_obs_scale, h0, c0 = z
    x = xbase.copy(); x[5], x[6] = h0, c0
    ll = 0.
    # Weakly informative initialization priors from the first observed DIVI/RKI levels.
    ll += -0.5 * ((np.log(max(h0, 1e-6)) - np.log(8314.)) / 1.0) ** 2
    ll += -0.5 * ((np.log(max(c0, 1e-6)) - np.log(1860.)) / 1.0) ** 2
    # Generate latent epidemic compartments during the unobserved pre-surveillance window.
    for j in range(30):
        beta = beta0 + (beta1-beta0) * j / 29.
        x = np.maximum(x + rhs(0., x, Parameters(beta=beta, tau_i=.0021, tau_h=tau_h, delta_h=.019)), 1e-9)
    for j, (_, row) in enumerate(train.iterrows()):
        beta = beta0 + (beta1-beta0) * j / max(len(train)-1, 1)
        p = Parameters(beta=beta, tau_i=.0021, tau_h=tau_h, delta_h=.019)
        x = np.maximum(x + rhs(0., x, p), 1e-9)
        mu = [q*.62*(1/4.5)*x[2], p.delta_h*x[5] + p.delta_c*x[6], x[6], hosp_obs_scale*q*.62*(1/4.5)*x[2]*7.]
        # RKI COVID-SARI values are weekly incidence per 100,000; convert to German counts.
        rki_weekly_counts = row.rki_weekly_hospitalization_incidence * (82108587. / 100000.)
        y = [row.reported_cases, row.deaths, row.icu_occupancy, rki_weekly_counts]
        for yy, mm, rr in zip(y, mu, [80., 10., 12., 10.]):
            mm = max(float(mm), 1e-6)
            ll += nbinom.logpmf(max(int(round(yy)), 0), rr, rr/(rr+mm))
    return float(-ll)

res = minimize(objective, [0.35, .35, .709, .03, 1., 8314., 1860.], method="Powell", bounds=[(.05, 1.), (.05, 1.), (.1, 1.), (.005, .2), (.001, 10.), (0., 50000.), (0., 20000.)], options={"maxiter": 40})
out = {"success": bool(res.success), "objective": float(res.fun), "parameters": {"beta_start": float(res.x[0]), "beta_end": float(res.x[1]), "q": float(res.x[2]), "tau_h": float(res.x[3]), "hosp_obs_scale": float(res.x[4]), "H0": float(res.x[5]), "C0": float(res.x[6])}, "train_fraction": .7, "likelihood": "joint negative-binomial cases/deaths/DIVI ICU/RKI weekly incidence with sentinel observation scale"}
(ROOT / "data/processed/germany_joint_data_likelihood_calibration.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
