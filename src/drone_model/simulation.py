from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .control import CascadedController
from .dynamics import ActuationOutput, DroneDynamics
from .environment import DrydenWindModel
from .state import DroneState


@dataclass
class SimulationResult:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray
    rates: np.ndarray
    motor_omegas: np.ndarray
    wind: np.ndarray
    total_thrust: np.ndarray
    body_torque: np.ndarray
    target_position: np.ndarray
    saturation: np.ndarray


class SimulationRunner:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.dynamics = DroneDynamics(config.drone)
        self.controller = CascadedController(config.drone, config.control)
        self.environment = DrydenWindModel(config.environment)

    def initial_state(self) -> DroneState:
        sim = self.config.simulation
        return DroneState(
            position=sim.initial_position.copy(),
            velocity=sim.initial_velocity.copy(),
            attitude=sim.initial_attitude.copy(),
            rates=sim.initial_rates.copy(),
        )

    def _rk4_step(
        self,
        state: DroneState,
        actuation: ActuationOutput,
        wind: np.ndarray,
        dt: float,
    ) -> DroneState:
        k1 = self.dynamics.derivatives(state, actuation, wind)
        k2 = self.dynamics.derivatives(state.add_scaled(k1, 0.5 * dt), actuation, wind)
        k3 = self.dynamics.derivatives(state.add_scaled(k2, 0.5 * dt), actuation, wind)
        k4 = self.dynamics.derivatives(state.add_scaled(k3, dt), actuation, wind)

        next_state = state.copy()
        next_state.position += (dt / 6.0) * (
            k1.position + 2.0 * k2.position + 2.0 * k3.position + k4.position
        )
        next_state.velocity += (dt / 6.0) * (
            k1.velocity + 2.0 * k2.velocity + 2.0 * k3.velocity + k4.velocity
        )
        next_state.attitude += (dt / 6.0) * (
            k1.attitude + 2.0 * k2.attitude + 2.0 * k3.attitude + k4.attitude
        )
        next_state.rates += (dt / 6.0) * (
            k1.rates + 2.0 * k2.rates + 2.0 * k3.rates + k4.rates
        )
        next_state.attitude[2] = ((next_state.attitude[2] + np.pi) % (2.0 * np.pi)) - np.pi
        return next_state

    def run(self) -> SimulationResult:
        dt = self.config.simulation.dt
        duration = self.config.simulation.duration
        target_position = self.config.simulation.target_position.copy()
        num_steps = int(duration / dt) + 1

        time = np.linspace(0.0, duration, num_steps)
        position = np.zeros((num_steps, 3), dtype=float)
        velocity = np.zeros((num_steps, 3), dtype=float)
        attitude = np.zeros((num_steps, 3), dtype=float)
        rates = np.zeros((num_steps, 3), dtype=float)
        motor_omegas = np.zeros((num_steps, 4), dtype=float)
        wind = np.zeros((num_steps, 3), dtype=float)
        total_thrust = np.zeros(num_steps, dtype=float)
        body_torque = np.zeros((num_steps, 3), dtype=float)
        saturation = np.zeros(num_steps, dtype=bool)

        state = self.initial_state()
        self.environment.reset()

        for i, current_time in enumerate(time):
            current_wind = self.environment.update(dt) if i > 0 else np.zeros(3, dtype=float)
            command = self.controller.compute_command(state, target_position, dt)
            actuation = self.dynamics.mix(command)

            position[i] = state.position
            velocity[i] = state.velocity
            attitude[i] = state.attitude
            rates[i] = state.rates
            motor_omegas[i] = actuation.motor_omegas
            wind[i] = current_wind
            total_thrust[i] = actuation.total_thrust
            body_torque[i] = actuation.body_torque
            saturation[i] = actuation.saturated

            if i < num_steps - 1:
                state = self._rk4_step(state, actuation, current_wind, dt)

        return SimulationResult(
            time=time,
            position=position,
            velocity=velocity,
            attitude=attitude,
            rates=rates,
            motor_omegas=motor_omegas,
            wind=wind,
            total_thrust=total_thrust,
            body_torque=body_torque,
            target_position=target_position,
            saturation=saturation,
        )
