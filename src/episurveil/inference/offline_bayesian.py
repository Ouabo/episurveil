import numpy as np
def prior_predictive(priors,n_draws=1000,seed=0):
    rng=np.random.default_rng(seed); return [{k:v.sample(rng) for k,v in priors.items()} for _ in range(n_draws)]
def importance_initialize(loglikelihood,priors,n_draws=2000,seed=0):
    draws=prior_predictive(priors,n_draws,seed); logw=np.array([loglikelihood(d) for d in draws]); logw-=np.max(logw); w=np.exp(logw); w/=w.sum()
    return {k:float(sum(w[i]*d[k] for i,d in enumerate(draws))) for k in priors}
