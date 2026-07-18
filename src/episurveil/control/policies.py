import numpy as np
def capacity_risk_policy(icu_forecast,capacity,intensity=.2):
    exceed=np.maximum(np.asarray(icu_forecast)-capacity,0); return np.clip(intensity*exceed/np.maximum(capacity,1),0,1)
