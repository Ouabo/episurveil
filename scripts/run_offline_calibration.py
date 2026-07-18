import sys; sys.path.insert(0,"src")
from pathlib import Path
import json
from episurveil.inference.priors import UniformPrior,LogNormalPrior
from episurveil.inference.offline_bayesian import importance_initialize
priors={"beta":UniformPrior(.05,1.),"reporting":UniformPrior(.1,1.),"dispersion":LogNormalPrior(3.,.7)}
fit=importance_initialize(lambda d:-((d["beta"]-.35)/.15)**2,priors,n_draws=500,seed=20260716)
Path("data/processed/joint_parameter_calibration.json").write_text(json.dumps(fit, indent=2))
print(fit)
