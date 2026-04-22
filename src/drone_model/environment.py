from __future__ import annotations

import numpy as np

from .config import EnvironmentConfig


class DrydenWindModel:
    """
    Simplified Dryden-style gust generator using first-order shaping filters.

    This keeps continuous, correlated turbulence without pulling in the full
    aerospace transfer-function machinery for milestone 1.
    """

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.state = np.zeros(3, dtype=float)
        self.rng = np.random.default_rng(config.seed)

    def reset(self) -> None:
        self.state.fill(0.0)

    def update(self, dt: float) -> np.ndarray:
        V = max(self.config.nominal_airspeed, 0.1)
        lengths = np.maximum(self.config.dryden_scale_lengths, 1e-3)
        sigmas = np.maximum(self.config.turbulence_intensities, 0.0)
        alpha = V / lengths
        noise = self.rng.standard_normal(3)
        self.state += (-alpha * self.state) * dt + np.sqrt(2.0 * alpha) * sigmas * np.sqrt(dt) * noise
        return self.state.copy()
