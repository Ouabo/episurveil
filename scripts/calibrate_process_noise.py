from pathlib import Path
import os, subprocess, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
panel = pd.read_csv(ROOT / "data/processed/germany_integrated_panel.csv")
scales = [0.003, 0.01, 0.02, 0.05]
rows = []
for scale in scales:
    env = os.environ.copy()
    env["PROCESS_NOISE_SCALE"] = str(scale)
    subprocess.run([sys.executable, "scripts/run_multichannel_filter.py"], cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    pred = pd.read_csv(ROOT / "data/processed/germany_multichannel_filter_output.csv")
    channels = [("reported_cases", "cases"), ("deaths", "deaths"), ("icu_occupancy", "icu"), ("hospitalization_proxy", "hospitalization_proxy")]
    cov = []
    for obs, name in channels:
        lo, hi = pred[f"{name}_q10"], pred[f"{name}_q90"]
        cov.append(np.mean((panel[obs] >= lo) & (panel[obs] <= hi)))
    rows.append({"process_noise_scale": scale, "mean_coverage_80_band": float(np.mean(cov)), **{f"coverage_{n}": float(c) for (_, n), c in zip(channels, cov)}})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "data/processed/process_noise_calibration.csv", index=False)
print(out.to_string(index=False))
