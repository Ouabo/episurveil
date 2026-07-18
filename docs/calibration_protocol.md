# Calibration protocol

The first calibration pass estimates observation-scale corrections from multi-channel mean residuals.
For channel $k$, the diagnostic correction is $c_k=\overline{Y}_k/\overline{\widehat{Y}}_k$.

These corrections identify mis-scaled observation equations, but they are not posterior parameter
estimates. A publication-quality analysis must replace them with held-out Bayesian calibration or
likelihood-based optimization and must rerun rolling-origin evaluation.
