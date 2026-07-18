"""Deterministic posterior-mean forecast smoke experiment."""
import sys; sys.path.insert(0,"src")
import numpy as np
from episurveil.models.sveaihcrd import Parameters,simulate
x0=np.array([999000.,500.,100.,100.,250.,30.,10.,10.,0.]); out=simulate(x0,np.arange(30),Parameters())
print({"horizon":14,"final_infectious":float(out[-1,4]),"final_icu":float(out[-1,6])})
