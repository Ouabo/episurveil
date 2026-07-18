import numpy as np
from episurveil.observations.multi_signal import joint_loglik
def test_joint_loglik_missing_channel():
    z=joint_loglik({"cases":10.,"icu":np.nan},{"cases":10.,"icu":2.}); assert np.isfinite(z)
