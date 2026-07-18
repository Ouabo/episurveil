"""Euler diffusion approximation with projected non-negative states."""
import numpy as np
def euler_maruyama(x0,drift,covariance,times,seed=0):
    rng=np.random.default_rng(seed); x=np.zeros((len(times),len(x0))); x[0]=x0
    for k in range(1,len(times)):
        dt=times[k]-times[k-1]; q=np.asarray(covariance(x[k-1],times[k-1])); L=np.linalg.cholesky(q+1e-10*np.eye(len(x0)))
        x[k]=np.maximum(x[k-1]+dt*np.asarray(drift(x[k-1],times[k-1]))+np.sqrt(dt)*L@rng.normal(size=len(x0)),0.)
    return x
