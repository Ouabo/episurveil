# EpiSurveil Control Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-live-blue?logo=streamlit)](https://episurveil.streamlit.app)

> **Modular Python framework for epidemic surveillance, real-time Bayesian parameter estimation, and particle-filter-guided optimal intervention planning.**

Validated on Germany COVID-19 (10 March 2020 – 7 October 2024, 1 673 days) with four simultaneous observation channels (cases, ICU, hospitalizations, deaths).

---

## Why EpiSurveil?

| What you need | What EpiSurveil provides |
|---|---|
| Track how β, R_eff change day-by-day | Bootstrap Particle Filter with log-random-walk parameters — no linearisation, no Gaussian assumption |
| Fit multiple data streams at once | Multi-channel NegBin likelihood with per-channel tempering weights |
| Handle missing / delayed data | Reporting-delay convolution; NaN-safe likelihood (missing days skipped) |
| Measure uncertainty, not just point estimates | Full posterior distribution at every time step (80% credible intervals on all states and parameters) |
| Go from surveillance to policy | Two-lever optimal control (NPI + testing) and PF-MPC receding-horizon controller built in |
| Explore without coding | Interactive Streamlit dashboard — upload CSV, pick disease preset, run BPF, download report |

---

## Live demo

```
https://episurveil.streamlit.app
```

Upload your own CSV (date + cases columns minimum) and get R_eff(t), β(t), compartment trajectories, fit metrics, and an auto-generated epidemic analysis report — no installation needed.

---

## Quick start

```bash
git clone https://github.com/Ouabo/episurveil.git
cd episurveil
pip install -e .
pytest tests/ -q        # 32 tests, ~6 s
streamlit run app/Home.py
```

### Fit a SEIR model to case data (3 lines)

```python
import pandas as pd
from episurveil.models.seir import SEIRModel
from episurveil.inference.bpf_generic import run_bpf

obs_df = pd.read_csv("your_cases.csv", parse_dates=["date"])
model  = SEIRModel(N=83_200_000, sigma=1/5, gamma=1/10, Q_C=0.15)
result = run_bpf(model, obs_df, N=2000, seed=42)

print(result[["date", "beta_mean", "R_eff_mean", "R_eff_q10", "R_eff_q90"]])
```

`result` is a plain `pd.DataFrame` — one row per day, posterior mean + 80% CI for every state and parameter.

---

## Models

| Model | States | Time-varying params | Observation channels | Primary use |
|---|---|---|---|---|
| `SIR` | 3 | β_t | cases | Rapid proof-of-concept |
| `SEIR` | 4 | β_t | cases | Incubation lag, R_eff(t) |
| `SEIRD` | 5 | β_t, δ_t | cases, deaths | Time-varying IFR; variant severity |
| `SEIRV` | 5 | β_t | cases (+ vaccination exog.) | Vaccination impact; herd threshold |
| `SEIARV` | 6 | β_t, Q_C_t | cases | Under-reporting; asymptomatic spread |
| `SEIRHD` | 6 | β_t, τ_I_t, δ_H_t | cases, hosp, deaths | Hospital surge planning |
| `SVEAIHCRD` | 9 | β_t, τ_I_t, δ_H_t, ρ_C_t, Q_C_t | cases, ICU, hosp, deaths | Full pipeline + control |

All models support **waning immunity** (`omega_r` parameter, R → S flow) for multi-wave fitting.

Add your own model by subclassing `EpiModel` — the BPF works with any implementation automatically.

---

## Dashboard (14 tabs)

```bash
streamlit run app/Home.py
```

| Tab | Content |
|---|---|
| Filter output | Observed vs predicted for all 4 channels with 80% CI |
| Compartments | S, V, E, A, I, H, C, R, D trajectories |
| Parameters | β_t, τ_I_t, δ_H_t, ρ_C_t time series with credible intervals |
| Metrics | RMSE, MAE, 80% coverage per channel |
| Fixed parameters | Transparency table of all fixed values used |
| Model description | SVEAIHCRD ODEs, observation channels, data coverage, optimal control |
| ESS | Effective sample size over time; resampling diagnostics |
| Optimal control | Two-lever NPI + testing optimisation; Pareto frontier |
| Model comparison | SEIR vs SEIRHD vs SVEAIHCRD side-by-side (R_eff, β, E_t) |
| **Model explorer** | **Upload your own CSV → select model + disease preset → run BPF → auto report** |
| **Live data** | **Fetch ECDC/disease.sh data → run BPF → 14-day forecast** |
| **Country comparison** | **R_eff & alert status across up to 10 countries simultaneously** |
| **Alert history** | **Longitudinal log of all BPF runs across sessions** |
| Methodology | BPF algorithm, NegBin likelihood, model reference, parameter guide |

### Model explorer disease presets

Select a country (auto-fills population N) and a disease/variant (auto-fills σ, γ, Q_C, β_max, ω_R):

| Preset | γ | σ | Q_C | β_max | ω_R |
|---|---|---|---|---|---|
| COVID-19 Original 2020 | 0.10 | 0.20 | 0.12 | 0.40 | 0 |
| COVID-19 Delta 2021 | 0.10 | 0.25 | 0.20 | 0.70 | 1/180 |
| COVID-19 Omicron 2022 | 0.13 | 0.29 | 0.25 | 1.95 | 1/90 |
| Influenza (seasonal) | 0.25 | 0.50 | 0.10 | 0.35 | 1/365 |
| Measles | 0.125 | 0.08 | 0.70 | 2.40 | 0 |
| Ebola | 0.10 | 0.09 | 0.70 | 0.22 | 0 |

---

## Auto-generated epidemic report

After running the BPF in the Model explorer, click **📋 Generate Report** to get a structured markdown report with:

- **Fit quality** — RMSE / MAE / MAPE per channel with Excellent / Good / Fair / Poor grade
- **Epidemic trajectory** — wave count, R_eff peak dates, phase table (Growth / Decline / Transition)
- **Transmission β(t)** — range, ceiling hits, implied R₀ range
- **Filter health (ESS)** — mean/min ESS, resampling events, collapse detection
- **Compartment summary** — peak values and dates for each compartment
- **Interpretation** — auto-generated narrative from statistics
- **Suggested adjustments** — actionable fixes when issues are detected

Reports are downloadable as `.md` files.

---

## More examples

### Time-varying IFR with deaths channel (SEIRD)

```python
from episurveil.models.seird import SEIRDModel

model  = SEIRDModel(N=83_200_000, Q_C=0.15)
result = run_bpf(model, obs_df, N=2000,
                 channel_weights={"cases": 1.0, "deaths": 0.4})

result["IFR_t"] = result["delta_mean"] / (1/10 + result["delta_mean"])
```

### Vaccination impact (SEIRV)

```python
from episurveil.models.seirv import SEIRVModel

# exog_df: date + nu (daily vaccination rate, 0–1 scale)
model  = SEIRVModel(N=83_200_000, epsilon=0.85, omega_v=1/180)
result = run_bpf(model, obs_df, N=2000, exog_df=exog_df)
```

### Hospital surge planning — three channels (SEIRHD)

```python
from episurveil.models.seirhd import SEIRHDModel

model  = SEIRHDModel(N=83_200_000, LOS=8.0)
result = run_bpf(model, obs_df, N=2000,
                 channel_weights={"cases": 1.0, "hosp": 0.1, "deaths": 0.4})

result["hosp_rate"] = result["tau_i_mean"] / (0.10 + result["tau_i_mean"])
result["CFR"]       = result["delta_h_mean"] / (1/12 + result["delta_h_mean"])
```

### Optimal intervention (NPI + testing)

```python
from episurveil.control.optimal_control import run_optimal_control, ControlWeights
import numpy as np, pandas as pd

df  = pd.read_csv("data/processed/sveaihcrd_filter_output.csv", parse_dates=["date"])
row = df[df["date"] == "2020-10-01"].iloc[0]
x0  = np.array([row[f"{c}_mean"] for c in ["S","V","E","A","I","H","C","R","D"]])

result = run_optimal_control(
    x0=x0,
    beta_base=float(row["beta_mean"]),
    tau_i=float(row["tau_i_mean"]),
    delta_h=float(row["delta_h_mean"]),
    q_c_base=float(row["rho_c_mean"]),
    T=90,
    weights=ControlWeights(w_u=50, w_v=2),
)
print(f"Deaths averted : {result.deaths_averted:,}")
```

---

## BPF output columns

Every `run_bpf()` call returns a `pd.DataFrame` with one row per day:

| Column | Description |
|---|---|
| `{state}_mean / _q10 / _q90` | Posterior mean and 80% CI for each compartment |
| `{param}_mean / _q10 / _q90` | Posterior mean and 80% CI for each time-varying parameter |
| `R_eff_mean / _q10 / _q90` | Effective reproduction number with 80% CI |
| `ess` | Effective sample size (diagnostic — healthy: > 0.45 × N) |
| `obs_{channel}` | Original observations (NaN = missing, skipped in likelihood) |
| `pred_{channel}_mean` | Posterior predictive mean |

---

## Project structure

```
episurveil/
├── app/
│   ├── Home.py                   # Streamlit dashboard (14 tabs)
│   └── requirements.txt          # Runtime deps (used by Streamlit Cloud)
├── src/episurveil/
│   ├── models/
│   │   ├── base.py               # EpiModel abstract base class
│   │   ├── sir.py / seir.py / seird.py / seirv.py / seiarv.py / seirhd.py
│   │   ├── sveaihcrd.py          # Full 9-compartment model
│   │   └── registry.py           # list_models(), get_model()
│   ├── connectors/
│   │   └── live_data.py          # disease.sh + ECDC live data connectors
│   ├── inference/
│   │   ├── bpf_generic.py        # Generic BPF — works with any EpiModel
│   │   └── particle_filter.py    # SVEAIHCRD-specific BPF (validated run)
│   ├── control/
│   │   ├── optimal_control.py    # L-BFGS-B + Pareto sweep
│   │   └── pf_mpc.py             # Receding-horizon PF-MPC
│   └── observations/
│       ├── likelihoods.py        # NegBin / Poisson log-likelihoods
│       └── delays.py             # Reporting-delay gamma convolution
├── data/processed/
│   └── sveaihcrd_filter_output.csv   # Pre-computed 1 673-day Germany run
├── tests/                        # 32-test suite
├── .streamlit/config.toml        # Server config (upload limit, XSRF, no telemetry)
└── requirements.txt              # Pinned dependencies
```

---

## Installation

```bash
# From source (recommended — editable install)
git clone https://github.com/Ouabo/episurveil.git
cd episurveil
pip install -e .

# Verify
pytest tests/ -q
```

Requirements: Python ≥ 3.10, numpy 2.2, scipy 1.15, pandas 2.3, streamlit 1.57, plotly 6.7.

---

## Contact

**Florent Ouabo Kamkumo, PhD**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/florent-ouabo-kamkumo-phd-ba39472b8/)

For research collaborations, questions, or feedback:
- **LinkedIn** — preferred for research inquiries and collaborations
- [Open a GitHub Issue](https://github.com/Ouabo/episurveil/issues/new?labels=feedback&title=[Feedback]) — for bugs and feature requests

Feedback on the following is especially welcome:
- Filter collapse edge cases (new disease / unusual data)
- Missing disease presets (epidemiological parameters for diseases not in the list)
- Model comparison artefacts
- Report quality and interpretation accuracy

---

## Citation

If you use EpiSurveil in your work, please cite:

```bibtex
@software{kamkumo2026episurveil,
  author  = {Kamkumo, Florent Ouabo},
  title   = {{EpiSurveil}: A Modular Python Framework for Epidemic Surveillance,
             Sequential Bayesian Estimation, and Particle-Filter-Guided
             Optimal Intervention Planning},
  year    = {2026},
  url     = {https://github.com/Ouabo/episurveil},
}
```

Paper preprint available on request — contact via LinkedIn.

---

## License

MIT — see [LICENSE](LICENSE).
