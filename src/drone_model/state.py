from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DroneState:
    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray
    rates: np.ndarray

    def copy(self) -> "DroneState":
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            attitude=self.attitude.copy(),
            rates=self.rates.copy(),
        )

    def add_scaled(self, other: "DroneState", scale: float) -> "DroneState":
        return DroneState(
            position=self.position + scale * other.position,
            velocity=self.velocity + scale * other.velocity,
            attitude=self.attitude + scale * other.attitude,
            rates=self.rates + scale * other.rates,
        )

    @classmethod
    def zeros(cls) -> "DroneState":
        return cls(
            position=np.zeros(3, dtype=float),
            velocity=np.zeros(3, dtype=float),
            attitude=np.zeros(3, dtype=float),
            rates=np.zeros(3, dtype=float),
        )
