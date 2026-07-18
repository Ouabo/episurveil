import numpy as np
from episurveil.observations.likelihoods import loglik
def test_likelihood_shapes(): assert loglik(np.array([1,2]),np.array([1.,2.])).shape==(2,)
