"""Reporting-delay convolution utilities.

Typical usage in a filter script
---------------------------------
    from episurveil.observations.delays import apply_case_delay

    # Pre-process the case series before feeding rows to the filter:
    df["reported_cases_delayed"] = apply_case_delay(
        df["reported_cases"].to_numpy(), mean_delay=5.0, sd_delay=2.0
    )

The delay is applied to the *full* series before the online filter loop so
that each particle weight update uses the delay-adjusted observation.
"""
import numpy as np
def delay_convolution(incidence, delays):
    incidence=np.asarray(incidence,float); delays=np.asarray(delays,float); delays=np.maximum(delays,0); delays/=max(delays.sum(),1e-12)
    return np.convolve(incidence,delays,mode="full")[:len(incidence)]
def observed_with_missing(values, missing_mask):
    values=np.asarray(values,float).copy(); values[np.asarray(missing_mask,bool)]=np.nan; return values

def occupancy_from_admissions(admissions, mean_los=8.0):
    """Convert daily admissions to an occupancy proxy using a geometric LOS kernel."""
    admissions = np.asarray(admissions, float)
    horizon = max(1, int(np.ceil(6 * mean_los)))
    stay = (1.0 / mean_los) * (1.0 - 1.0 / mean_los) ** np.arange(horizon)
    stay /= stay.sum()
    return np.convolve(np.maximum(admissions, 0.0), stay, mode="full")[:len(admissions)] * mean_los

def delayed_incidence(incidence, mean_delay=5.0, sd_delay=2.0):
    """Apply a discretized gamma reporting-delay kernel to an incidence series."""
    incidence = np.asarray(incidence, float)
    horizon = max(1, int(np.ceil(mean_delay + 5 * sd_delay)))
    d = np.arange(horizon, dtype=float)
    shape = max((mean_delay / max(sd_delay, 1e-6)) ** 2, 1e-3)
    scale = max(sd_delay**2 / max(mean_delay, 1e-6), 1e-6)
    kernel = d ** (shape - 1) * np.exp(-d / scale)
    kernel[0] = 0.0
    kernel /= max(kernel.sum(), 1e-12)
    return delay_convolution(incidence, kernel)

def weekly_sum(daily_values):
    """Aggregate aligned daily incidence into non-overlapping epidemiological weeks."""
    values = np.asarray(daily_values, float)
    n = (len(values) // 7) * 7
    return values[:n].reshape(-1, 7).sum(axis=1)


def apply_case_delay(
    case_series: np.ndarray,
    mean_delay: float = 5.0,
    sd_delay: float = 2.0,
) -> np.ndarray:
    """Convenience wrapper: apply a gamma reporting-delay kernel to a case series.

    Returns the delay-adjusted series (same length as input).  Pass this as
    the ``cases`` column in the observation rows fed to
    ``sir_filter_multichannel`` to align the observation model with
    surveillance reporting practice.

    Parameters
    ----------
    case_series : array-like
        Daily observed case counts (raw or smoothed).
    mean_delay : float
        Mean reporting delay in days (German RKI: typically 3--7 days).
    sd_delay : float
        Standard deviation of the reporting-delay distribution.
    """
    return delayed_incidence(np.asarray(case_series, float), mean_delay, sd_delay)
