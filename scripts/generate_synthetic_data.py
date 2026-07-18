import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd
from episurveil.models.sveaihcrd import Parameters,simulate
out=simulate(np.array([999000.,500.,100.,100.,250.,30.,10.,10.,0.]),np.arange(120),Parameters())
pd.DataFrame(out,columns=["S","V","E","A","I","H","C","R","D"]).to_csv("data/synthetic/sveaihcrd.csv",index=False)
