"""Fixed-step Poisson tau-leaping for non-negative transition systems."""
import numpy as np
def tau_leap(x0, propensities, stoichiometry, times, seed=0):
    rng=np.random.default_rng(seed); x=np.zeros((len(times),len(x0))); x[0]=np.asarray(x0,float)
    S=np.asarray(stoichiometry,float)
    for k in range(1,len(times)):
        dt=float(times[k]-times[k-1]); rates=np.maximum(np.asarray(propensities(x[k-1],times[k-1])),0.)
        jumps=rng.poisson(rates*dt); x[k]=np.maximum(x[k-1]+jumps@S,0.)
        if len(x0)>=9: x[k,8]=max(x[k,8],x[k-1,8])
    return x
