# Held-out calibration

The held-out calibration script estimates channel scales using only the first 70% of the chronological
panel and evaluates the corrected predictions on the final 30%. This is a diagnostic safeguard against
using the full evaluation period to tune observation scales. It does not replace joint Bayesian
calibration of biological parameters.
