from episurveil.inference.priors import UniformPrior
from episurveil.inference.offline_bayesian import importance_initialize
def test_offline_initializer():
    fit=importance_initialize(lambda d:-(d["x"]-.7)**2,{"x":UniformPrior(0,1)},200,3)
    assert 0<=fit["x"]<=1
