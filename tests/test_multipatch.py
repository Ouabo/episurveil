import numpy as np
from episurveil.models.multipatch_sim import simulate_multipatch
def test_multipatch_shape():
    x=np.zeros((2,9)); x[:,0]=1000; x[0,4]=10
    out=simulate_multipatch(x,np.arange(5),np.array([.3,.3]),np.eye(2))
    assert out.shape==(5,2,9) and np.all(out>=0)
