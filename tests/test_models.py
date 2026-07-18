import numpy as np
from episurveil.models.sveaihcrd import Parameters, rhs, simulate, mass_balance
def test_nonnegative_and_deaths_monotone():
    x0=np.array([9990,0,0,0,10,0,0,0,0.]); out=simulate(x0,np.arange(0,20),Parameters())
    assert np.all(out>=0); assert np.all(np.diff(out[:,8])>=-1e-10)
def test_living_mass_changes_only_by_deaths():
    x0=np.array([9990,0,0,0,10,0,0,0,0.]); out=simulate(x0,np.arange(0,20),Parameters())
    assert mass_balance(out)[-1] <= mass_balance(out)[0]+1e-8
