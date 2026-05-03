# Drone Model

DATA 442 final project - a 6-DOF quadcopter simulation with cascaded PID control and Dryden atmospheric turbulence.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Scripts

### `run_sim.py` - batch simulation

Runs a fixed-duration simulation and prints summary telemetry (final position error, RMS motor speeds, wind statistics).

```bash
python scripts/run_sim.py
```

### `view_sim_ui.py` - interactive UI

Opens a live matplotlib window with real-time controls.

```bash
python scripts/view_sim_ui.py
```

The UI includes:
- **3D environment** - live drone position, flight trail, rotor geometry, and wind quiver
- **PID state panel** - roll/pitch/yaw attitude bars and per-motor w readouts with saturation warnings
- **Telemetry plots** - scrolling position error, motor speeds, and Dryden wind components
- **Target XYZ sliders** - move the setpoint in real-time while the drone is flying
- **RESET button** - restart the drone from its initial conditions

https://github.com/user-attachments/assets/8553fa0e-c74c-4799-92b1-c6ded5b91901

## Architecture

```
src/drone_model/
├── config.py        # Frozen dataclass configs (drone, control, environment, simulation)
├── state.py         # DroneState: position, velocity, attitude (ZYX Euler), body rates
├── dynamics.py      # 6-DOF rigid-body EOM, motor allocation matrix, actuation mixing
├── control.py       # Cascaded PID controller with wind feedforward
├── environment.py   # Dryden gust model (first-order shaping filter)
└── simulation.py    # RK4 integration loop, SimulationResult
```

## Control System

The controller uses a three-stage cascade running at 100 Hz (dt = 0.01 s):

1. **Position -> acceleration** - PID on XYZ error; output is desired linear acceleration
2. **Attitude -> rates** - converts desired acceleration to roll/pitch setpoints via small-angle inversion; PID on attitude error
3. **Rates -> torque** - PID on body-rate error; output is commanded body torque

**Wind feedforward** is applied at stage 1: the expected aerodynamic drag from the current wind velocity (`F = linear_drag * wind`) is subtracted from the desired acceleration so the drone anticipates disturbances rather than waiting for position error to accumulate.

Commanded thrust and torques are fed through a 4x4 motor allocation matrix (inverted analytically) to produce individual motor speeds, clamped to [0, 3200] rad/s.

## Wind Model

The `DrydenWindModel` generates spatially correlated turbulence using first-order Ornstein-Uhlenbeck shaping filters:

```
dw = -a*w*dt + sqrt(2a)*s*sqrt(dt)*N(0,1)
```

where `a = V / L` (airspeed over scale length). Default parameters model low-altitude turbulence at ~8 m/s with scale lengths of 45 m (horizontal) and 20 m (vertical).

## Configuration

All parameters live in `src/drone_model/config.py`. The defaults model a sub-250 g quadcopter (Crazyflie-class):

| Parameter | Value | Notes |
|---|---|---|
| Mass | 0.249 kg | |
| Arm length | 0.08 m | motor-to-center |
| Max rotor speed | 3200 rad/s | |
| Max tilt | 35 deg | pitch/roll limit |
| Sim timestep | 0.01 s | 100 Hz |
| Sim duration | 15 s | batch mode |
| RNG seed | 442 | reproducible wind |
