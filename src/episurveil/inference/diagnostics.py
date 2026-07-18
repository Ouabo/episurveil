import numpy as np
def effective_sample_size(weights): return float(1/np.sum(np.asarray(weights)**2))
def posterior_summary(draws):
    a=np.asarray(draws,float); return {"mean":float(a.mean()),"sd":float(a.std(ddof=1)),"q05":float(np.quantile(a,.05)),"q50":float(np.quantile(a,.5)),"q95":float(np.quantile(a,.95))}
