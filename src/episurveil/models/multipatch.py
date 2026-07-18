"""Multipatch force of infection utilities."""
import numpy as np
def force_of_infection(states, beta, mobility, eta_a=.5):
    states=np.asarray(states,float); living=np.maximum(states[:,:8].sum(axis=1),1e-12)
    infectious=(states[:,4]+eta_a*states[:,3])/living; M=np.asarray(mobility,float)
    if M.shape != (len(states),len(states)): raise ValueError("mobility dimension must equal patches")
    return np.asarray(beta)*(M@infectious)
