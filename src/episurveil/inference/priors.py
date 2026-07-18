from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class UniformPrior:
    lower: float; upper: float
    def logpdf(self,x): return 0.0 if self.lower<=x<=self.upper else -np.inf
    def sample(self,rng): return float(rng.uniform(self.lower,self.upper))
@dataclass(frozen=True)
class LogNormalPrior:
    meanlog: float; sdlog: float
    def logpdf(self,x):
        if x<=0: return -np.inf
        z=(np.log(x)-self.meanlog)/self.sdlog; return float(-np.log(x*self.sdlog*np.sqrt(2*np.pi))-.5*z*z)
    def sample(self,rng): return float(rng.lognormal(self.meanlog,self.sdlog))
