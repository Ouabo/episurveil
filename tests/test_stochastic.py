import numpy as np
from episurveil.stochastic.tau_leaping import tau_leap
from episurveil.observations.delays import delay_convolution
def test_tau_leaping_nonnegative():
    S=np.array([[-1,1],[0,0]],float); out=tau_leap(np.array([100.,0.]),lambda x,t:[.2*x[0],0],S,np.arange(10),seed=2)
    assert np.all(out>=0)
def test_delay_convolution_preserves_length():
    y=delay_convolution([1,2,3],[.2,.8]); assert len(y)==3 and y[0]>=0
