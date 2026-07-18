# Current limitations

The present implementation is a validated foundation, not yet a complete decision-support system.
The core SVEAIHCRD ODE, count likelihoods, SIR filter, and baseline risk policy are implemented and
tested. CTMC support is a minimal smoke implementation; tau-leaping, full diffusion covariance
construction, offline PyMC calibration, reporting-delay convolution, multipatch filtering, and
stochastic MPC/CVaR require additional tests and domain review. No clinical or public-health decision
should be based on the prototype application.
