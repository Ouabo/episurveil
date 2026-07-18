import sys; sys.path.insert(0,"src")
import numpy as np
from episurveil.control.policies import capacity_risk_policy
forecast=np.array([80,90,110,130]); print(capacity_risk_policy(forecast,100))
