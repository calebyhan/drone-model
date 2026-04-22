from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DroneConfig
from .state import DroneState


def rotation_matrix_from_euler(attitude: np.ndarray) -> np.ndarray:
    phi, theta, psi = attitude
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    return np.array(
        [
            [cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi],
            [spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi],
            [-stheta, ctheta * sphi, ctheta * cphi],
        ],
        dtype=float,
    )


def euler_rates_matrix(attitude: np.ndarray) -> np.ndarray:
    phi, theta, _ = attitude
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta = np.cos(theta)
    ttheta = np.tan(theta)

    if abs(ctheta) < 1e-3:
        ctheta = np.sign(ctheta) * 1e-3 if ctheta != 0.0 else 1e-3

    return np.array(
        [
            [1.0, sphi * ttheta, cphi * ttheta],
            [0.0, cphi, -sphi],
            [0.0, sphi / ctheta, cphi / ctheta],
        ],
        dtype=float,
    )


@dataclass
class ControlCommand:
    total_thrust: float
    body_torque: np.ndarray


@dataclass
class ActuationOutput:
    motor_omegas: np.ndarray
    total_thrust: float
    body_torque: np.ndarray
    saturated: bool


class DroneDynamics:
    def __init__(self, config: DroneConfig):
        self.config = config
        arm = config.arm_length / np.sqrt(2.0)
        kf = config.thrust_coefficient
        km = config.yaw_drag_coefficient
        spin = config.rotor_spin_directions

        self.allocation_matrix = np.array(
            [
                [kf, kf, kf, kf],
                [arm * kf, -arm * kf, -arm * kf, arm * kf],
                [-arm * kf, -arm * kf, arm * kf, arm * kf],
                [km * spin[0], km * spin[1], km * spin[2], km * spin[3]],
            ],
            dtype=float,
        )
        self.inverse_allocation = np.linalg.inv(self.allocation_matrix)

    def mix(self, command: ControlCommand) -> ActuationOutput:
        desired = np.concatenate(([command.total_thrust], command.body_torque))
        omega_sq = self.inverse_allocation @ desired
        omega_sq = np.clip(
            omega_sq,
            self.config.min_omega ** 2,
            self.config.max_omega ** 2,
        )
        omegas = np.sqrt(omega_sq)
        realized = self.allocation_matrix @ omega_sq

        realized_thrust = float(np.clip(realized[0], 0.0, self.config.max_total_thrust))
        realized_torque = realized[1:].astype(float)
        saturated = bool(np.any(np.abs(realized - desired) > 1e-6))

        return ActuationOutput(
            motor_omegas=omegas,
            total_thrust=realized_thrust,
            body_torque=realized_torque,
            saturated=saturated,
        )

    def derivatives(
        self,
        state: DroneState,
        actuation: ActuationOutput,
        wind_velocity_world: np.ndarray,
    ) -> DroneState:
        mass = self.config.mass
        gravity = self.config.gravity
        inertia = self.config.inertia
        rotation = rotation_matrix_from_euler(state.attitude)

        thrust_body = np.array([0.0, 0.0, actuation.total_thrust], dtype=float)
        thrust_world = rotation @ thrust_body
        relative_air_velocity = state.velocity - wind_velocity_world
        drag_world = -self.config.linear_drag * relative_air_velocity
        gravity_force = np.array([0.0, 0.0, -mass * gravity], dtype=float)
        net_force = thrust_world + gravity_force + drag_world
        linear_accel = net_force / mass

        angular_drag = -self.config.angular_drag * state.rates
        coriolis = np.cross(state.rates, inertia @ state.rates)
        angular_accel = np.linalg.solve(
            inertia,
            actuation.body_torque + angular_drag - coriolis,
        )

        attitude_dot = euler_rates_matrix(state.attitude) @ state.rates

        return DroneState(
            position=state.velocity.copy(),
            velocity=linear_accel,
            attitude=attitude_dot,
            rates=angular_accel,
        )
