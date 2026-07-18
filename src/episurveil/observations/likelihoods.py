import numpy as np
from scipy.stats import poisson, nbinom
from scipy.special import gammaln

def loglik(y, mu, family="negative_binomial", dispersion=20.):
    """Per-particle log-likelihood.

    y   : scalar observation (may be a smoothed float; rounded for discrete families)
    mu  : array of predicted means, one per particle
    """
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    if family == "poisson":
        # Round to nearest non-negative integer for discrete PMF
        k = max(int(round(float(y))), 0)
        return poisson.logpmf(k, mu)
    if family == "negative_binomial":
        # Continuous NegBin log-PMF via gammaln (handles non-integer k gracefully)
        k = np.maximum(np.asarray(y, dtype=float), 0.0)
        r = float(dispersion)
        # log P(k|r,mu) = gammaln(k+r) - gammaln(r) - gammaln(k+1)
        #                 + r*log(r/(r+mu)) + k*log(mu/(r+mu))
        lp = (gammaln(k + r) - gammaln(r) - gammaln(k + 1)
              + r * np.log(r / (r + mu))
              + k * np.log(mu / (r + mu)))
        return lp
    raise ValueError(f"unsupported observation family: {family}")
def incidence_observation(x, reporting=.7): return reporting*np.maximum(x[...,2],0)
