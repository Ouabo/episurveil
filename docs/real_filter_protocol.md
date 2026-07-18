# Real-data filter protocol

`run_real_data_filter.py` runs a reproducible 300-particle case-channel SIR filter on the integrated
German panel. `evaluate_real_filter.py` computes RMSE, MAE, 80% interval coverage, interval width, and
mean ESS. These are engineering diagnostics for the current prototype. They must not be presented as
final epidemiological performance until offline calibration, reporting-delay treatment, and a proper
multi-signal likelihood have been completed.

The current multi-channel run aligns OWID vaccination coverage but does not yet inject daily
vaccination increments into the transition kernel. This is deliberately recorded as an implementation
limitation; the next calibration revision must use vaccination increments as time-dependent $S\to V$
inputs.
