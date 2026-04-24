from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ControlConfig, DroneConfig, PIDGains
from .dynamics import ControlCommand
from .state import DroneState


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class PID:
    gains: PIDGains
    deriv_filter_alpha: float = 0.7
    integral: np.ndarray | None = None
    prev_error: np.ndarray | None = None
    filtered_derivative: np.ndarray | None = None
    initialized: bool = False

    def __post_init__(self) -> None:
        if self.integral is None:
            self.integral = np.zeros_like(self.gains.kp, dtype=float)
        if self.prev_error is None:
            self.prev_error = np.zeros_like(self.gains.kp, dtype=float)
        if self.filtered_derivative is None:
            self.filtered_derivative = np.zeros_like(self.gains.kp, dtype=float)

    def reset(self) -> None:
        self.integral.fill(0.0)
        self.prev_error.fill(0.0)
        self.filtered_derivative.fill(0.0)
        self.initialized = False

    def update(self, error: np.ndarray, dt: float) -> np.ndarray:
        self.integral = np.clip(
            self.integral + error * dt,
            -self.gains.integral_limit,
            self.gains.integral_limit,
        )
        if not self.initialized or dt <= 0.0:
            self.filtered_derivative = np.zeros_like(error)
            self.initialized = True
        else:
            raw_derivative = (error - self.prev_error) / dt
            self.filtered_derivative = (
                self.deriv_filter_alpha * raw_derivative
                + (1.0 - self.deriv_filter_alpha) * self.filtered_derivative
            )
        self.prev_error = error.copy()
        return (
            self.gains.kp * error
            + self.gains.ki * self.integral
            + self.gains.kd * self.filtered_derivative
        )


@dataclass
class ControllerSnapshot:
    desired_attitude: np.ndarray
    desired_rates: np.ndarray
    desired_acceleration: np.ndarray


class CascadedController:
    def __init__(self, drone_config: DroneConfig, control_config: ControlConfig):
        self.drone_config = drone_config
        self.control_config = control_config
        self.position_pid = PID(control_config.position_gains)
        self.attitude_pid = PID(control_config.attitude_gains)
        self.rate_pid = PID(control_config.rate_gains)
        self.last_snapshot = ControllerSnapshot(
            desired_attitude=np.zeros(3, dtype=float),
            desired_rates=np.zeros(3, dtype=float),
            desired_acceleration=np.zeros(3, dtype=float),
        )

    def compute_command(
        self,
        state: DroneState,
        target_position: np.ndarray,
        dt: float,
    ) -> ControlCommand:
        position_error = target_position - state.position
        desired_acceleration = (
            self.position_pid.update(position_error, dt)
            - self.control_config.position_damping * state.velocity
        )
        desired_acceleration = np.clip(
            desired_acceleration,
            -self.control_config.max_acceleration,
            self.control_config.max_acceleration,
        )

        psi_des = self.control_config.yaw_target
        g = self.drone_config.gravity
        phi_des = (
            desired_acceleration[0] * np.sin(psi_des)
            - desired_acceleration[1] * np.cos(psi_des)
        ) / g
        theta_des = (
            desired_acceleration[0] * np.cos(psi_des)
            + desired_acceleration[1] * np.sin(psi_des)
        ) / g

        desired_attitude = np.array(
            [
                np.clip(phi_des, -self.drone_config.max_tilt_rad, self.drone_config.max_tilt_rad),
                np.clip(theta_des, -self.drone_config.max_tilt_rad, self.drone_config.max_tilt_rad),
                psi_des,
            ],
            dtype=float,
        )

        attitude_error = wrap_angle(desired_attitude - state.attitude)
        desired_rates = self.attitude_pid.update(attitude_error, dt)
        desired_rates = np.clip(
            desired_rates,
            -self.control_config.max_rate,
            self.control_config.max_rate,
        )

        rate_error = desired_rates - state.rates
        commanded_torque = self.rate_pid.update(rate_error, dt)
        commanded_torque = np.clip(
            commanded_torque,
            -self.control_config.max_torque,
            self.control_config.max_torque,
        )

        commanded_thrust = self.drone_config.mass * (
            self.drone_config.gravity + desired_acceleration[2]
        )
        tilt_compensation = np.cos(state.attitude[0]) * np.cos(state.attitude[1])
        if abs(tilt_compensation) > 1e-3:
            commanded_thrust /= max(tilt_compensation, 1e-3)
        commanded_thrust = float(np.clip(commanded_thrust, 0.0, self.drone_config.max_total_thrust))

        self.last_snapshot = ControllerSnapshot(
            desired_attitude=desired_attitude,
            desired_rates=desired_rates,
            desired_acceleration=desired_acceleration,
        )
        return ControlCommand(total_thrust=commanded_thrust, body_torque=commanded_torque)
