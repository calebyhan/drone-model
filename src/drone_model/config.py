from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DroneConfig:
    mass: float = 0.249
    gravity: float = 9.81
    arm_length: float = 0.08
    inertia: np.ndarray = field(
        default_factory=lambda: np.diag([1.7e-3, 1.7e-3, 2.8e-3]).astype(float)
    )
    thrust_coefficient: float = 3.4e-7
    yaw_drag_coefficient: float = 1.2e-8
    min_omega: float = 0.0
    max_omega: float = 3200.0
    max_tilt_rad: float = np.deg2rad(35.0)
    max_total_thrust: float = 8.5
    linear_drag: np.ndarray = field(
        default_factory=lambda: np.array([0.18, 0.18, 0.28], dtype=float)
    )
    angular_drag: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.04], dtype=float)
    )
    rotor_spin_directions: np.ndarray = field(
        default_factory=lambda: np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
    )


@dataclass(frozen=True)
class PIDGains:
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    integral_limit: np.ndarray


@dataclass(frozen=True)
class ControlConfig:
    position_gains: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=np.array([0.65, 0.65, 0.85], dtype=float),
            ki=np.array([0.0, 0.0, 0.0], dtype=float),
            kd=np.array([0.0, 0.0, 0.0], dtype=float),
            integral_limit=np.array([1.5, 1.5, 1.0], dtype=float),
        )
    )
    attitude_gains: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=np.array([3.2, 3.2, 1.2], dtype=float),
            ki=np.array([0.0, 0.0, 0.0], dtype=float),
            kd=np.array([0.0, 0.0, 0.0], dtype=float),
            integral_limit=np.array([0.35, 0.35, 0.25], dtype=float),
        )
    )
    rate_gains: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=np.array([0.012, 0.012, 0.01], dtype=float),
            ki=np.array([0.0, 0.0, 0.0], dtype=float),
            kd=np.array([0.0, 0.0, 0.0], dtype=float),
            integral_limit=np.array([0.4, 0.4, 0.3], dtype=float),
        )
    )
    max_velocity: np.ndarray = field(
        default_factory=lambda: np.array([4.0, 4.0, 2.5], dtype=float)
    )
    max_acceleration: np.ndarray = field(
        default_factory=lambda: np.array([2.2, 2.2, 2.2], dtype=float)
    )
    position_damping: np.ndarray = field(
        default_factory=lambda: np.array([2.3, 2.3, 3.5], dtype=float)
    )
    max_rate: np.ndarray = field(
        default_factory=lambda: np.array([3.5, 3.5, 2.0], dtype=float)
    )
    max_torque: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.01], dtype=float)
    )
    yaw_target: float = 0.0


@dataclass(frozen=True)
class EnvironmentConfig:
    nominal_airspeed: float = 8.0
    dryden_scale_lengths: np.ndarray = field(
        default_factory=lambda: np.array([45.0, 45.0, 20.0], dtype=float)
    )
    turbulence_intensities: np.ndarray = field(
        default_factory=lambda: np.array([0.9, 0.9, 0.5], dtype=float)
    )
    seed: int = 442


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.01
    duration: float = 15.0
    initial_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    initial_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    initial_attitude: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    initial_rates: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    target_position: np.ndarray = field(
        default_factory=lambda: np.array([4.0, 4.0, 2.5], dtype=float)
    )


@dataclass(frozen=True)
class ModelConfig:
    drone: DroneConfig = field(default_factory=DroneConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def build_default_config() -> ModelConfig:
    """Return a baseline configuration inspired by a sub-250 g quadcopter."""

    return ModelConfig()
