"""
Abstract base class for all epidemic models used with the generic BPF.

Every model exposes:
  - state_names   : compartment labels  (e.g. ['S','E','I','R'])
  - param_names   : time-varying parameter labels (e.g. ['beta'])
  - obs_channels  : observation channel labels (e.g. ['cases','deaths'])
  - init_particles: draw N initial augmented-state particles
  - step          : propagate one day (log-RW params + ODE)
  - observation_mean: expected observations per channel per particle
  - R_eff         : effective reproduction number per particle

Particle layout (1-D per particle, length = n_states + n_params):
  particles[:, :n_states]   → compartment counts  (S, E, I, ...)
  particles[:, n_states:]   → log values of parameters (log β, ...)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class EpiModel(ABC):

    # ── descriptors ────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def state_names(self) -> list[str]:
        """Ordered compartment names."""

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Ordered names of time-varying log-RW parameters."""

    @property
    @abstractmethod
    def obs_channels(self) -> list[str]:
        """Observation channel keys (must match DataFrame column names)."""

    # derived sizes
    @property
    def n_states(self) -> int:
        return len(self.state_names)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def augmented_dim(self) -> int:
        return self.n_states + self.n_params

    # ── BPF interface ───────────────────────────────────────────────────────
    @abstractmethod
    def init_particles(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """Return (N, augmented_dim) initial particles."""

    @abstractmethod
    def step(
        self,
        particles: np.ndarray,
        t: int,
        exog: dict | None = None,
    ) -> np.ndarray:
        """
        Propagate N particles one day.

        Parameters
        ----------
        particles : (N, augmented_dim)
        t         : time index (0-based day)
        exog      : dict of exogenous scalars at step t (e.g. {'nu': 0.002})

        Returns
        -------
        particles : (N, augmented_dim) updated in-place OR new array
        """

    @abstractmethod
    def observation_mean(self, particles: np.ndarray) -> dict[str, np.ndarray]:
        """
        Expected value of each observation channel for each particle.

        Returns
        -------
        dict channel_name → (N,) array of expected values
        """

    @abstractmethod
    def R_eff(self, particles: np.ndarray) -> np.ndarray:
        """
        Effective reproduction number R_eff(t) for each particle.

        Returns
        -------
        (N,) array
        """

    # ── helpers shared by all subclasses ────────────────────────────────────
    def _log_rw_step(
        self,
        log_vals: np.ndarray,
        sigmas: np.ndarray,
        log_min: np.ndarray,
        log_max: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Update log-RW parameters with clipping. (N, n_params) in/out."""
        noise = rng.standard_normal(log_vals.shape) * sigmas[None, :]
        return np.clip(log_vals + noise, log_min[None, :], log_max[None, :])

    def _euler(self, x: np.ndarray, dx: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Non-negative Euler step."""
        return np.maximum(x + dt * dx, 0.0)
