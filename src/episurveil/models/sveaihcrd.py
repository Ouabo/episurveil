"""Deterministic SVEAIHCRD epidemic dynamics."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Parameters:
    beta: float = 0.35; eta_a: float = 0.5; sigma: float = 1/4.5
    kappa: float = 0.62; gamma_a: float = 1/7; gamma_i: float = 1/8
    tau_i: float = 0.02; gamma_h: float = 1/12; tau_h: float = 0.03
    delta_h: float = 0.005; gamma_c: float = 1/10; delta_c: float = 0.01
    nu: float = 0.0; omega_v: float = 1/180; omega_r: float = 1/365
    vaccine_efficacy: float = 0.75

NAMES = ("S", "V", "E", "A", "I", "H", "C", "R", "D")

def rhs(t: float, x: np.ndarray, p: Parameters) -> np.ndarray:
    S,V,E,A,I,H,C,R,D = np.asarray(x, dtype=float)
    living = max(S+V+E+A+I+H+C+R, 1e-12)
    lam = p.beta * (I + p.eta_a*A) / living
    inf_v = (1-p.vaccine_efficacy)*lam*V
    # nu(t) is a population-level per-capita rate applied uniformly to all
    # eligible compartments: S, E, A, R (people who don't know they are
    # infectious and would present for vaccination).
    # I, H, C are excluded (visibly ill / hospitalised).
    # Mass balance: nu*(S+E+A+R) enter V; same amount leaves S+E+A+R.
    return np.array([
        -lam*S - p.nu*S + p.omega_v*V + p.omega_r*R,
        p.nu*(S+E+A+R) - inf_v - p.omega_v*V,
        lam*S + inf_v - p.sigma*E - p.nu*E,
        (1-p.kappa)*p.sigma*E - p.gamma_a*A - p.nu*A,
        p.kappa*p.sigma*E - (p.gamma_i+p.tau_i)*I,
        p.tau_i*I - (p.gamma_h+p.tau_h+p.delta_h)*H,
        p.tau_h*H - (p.gamma_c+p.delta_c)*C,
        p.gamma_a*A + p.gamma_i*I + p.gamma_h*H + p.gamma_c*C - p.omega_r*R - p.nu*R,
        p.delta_h*H + p.delta_c*C], dtype=float)

def simulate(x0: np.ndarray, times: np.ndarray, p: Parameters, dt: float = 0.25) -> np.ndarray:
    """Forward Euler simulation with non-negativity projection."""
    out = np.zeros((len(times), 9)); out[0] = np.asarray(x0, dtype=float)
    for k in range(1, len(times)):
        n = max(1, int(np.ceil((times[k]-times[k-1])/dt))); h=(times[k]-times[k-1])/n
        x = out[k-1].copy()
        for j in range(n):
            x = np.maximum(x + h*rhs(times[k-1]+j*h, x, p), 0.0)
            x[8] = max(x[8], out[k-1,8])
        out[k] = x
    return out

def mass_balance(x: np.ndarray) -> np.ndarray:
    return np.sum(x[..., :8], axis=-1)
