"""Multi-channel joint log-likelihood for tempered particle-filter weight updates.

Default dispersions follow the Germany calibration conventions:
    cases  phi=80   (high overdispersion; wide weekend/reporting variation)
    icu    phi=12   (tighter; DIVI register is complete)
    deaths phi=10
    hosp   phi=15   (admissions proxy; intermediate noise)
All other channels fall back to phi=20.
"""
import numpy as np
from .likelihoods import loglik

# Per-channel dispersion defaults (override via dispersions= argument).
_DEFAULT_DISPERSIONS: dict[str, float] = {
    "cases":  80.0,
    "icu":    12.0,
    "deaths": 10.0,
    "hosp":   15.0,
}


def joint_loglik(
    observed: dict,
    means: dict,
    families: dict | None = None,
    dispersions: dict | None = None,
    weights: dict | None = None,
) -> float:
    """Tempered joint log-likelihood for one time step.

    Parameters
    ----------
    observed : dict[str, float]
        Observed values keyed by channel name.  ``None`` or ``nan`` values
        are excluded from the computation.
    means : dict[str, float | ndarray]
        Predicted means keyed by channel name.
    families : dict[str, str] or None
        Per-channel observation family.  Defaults to ``"negative_binomial"``.
    dispersions : dict[str, float] or None
        Per-channel dispersion phi.  Falls back to ``_DEFAULT_DISPERSIONS``
        then to 20.0.
    weights : dict[str, float] or None
        Per-channel tempering weight alpha_k.  Defaults to 1.0 (no tempering).

    Returns
    -------
    float
        Weighted sum of per-channel log-likelihoods.
    """
    families    = families    or {}
    weights     = weights     or {}
    # Merge caller-supplied dispersions over the defaults
    _disp = {**_DEFAULT_DISPERSIONS, **(dispersions or {})}

    total = 0.0
    for k, y in observed.items():
        if k not in means:
            continue
        if y is None:
            continue
        try:
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(yf):
            continue
        alpha = float(weights.get(k, 1.0))
        phi   = float(_disp.get(k, 20.0))
        fam   = families.get(k, "negative_binomial")
        total += alpha * float(loglik(yf, means[k], family=fam, dispersion=phi))
    return total
