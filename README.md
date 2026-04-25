# Drone Model

DATA 442 final project

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_sim.py
python scripts/view_sim_ui.py
```

`run_sim.py` prints summary telemetry.

`view_sim_ui.py` opens an interactive live simulation with real-time controls:


The UI includes:
- **3D environment** - live drone position, flight trail, rotor geometry, and wind quiver
- **PID state panel** - roll/pitch/yaw attitude bars and per-motor ω readouts with saturation warnings
- **Telemetry plots** - scrolling position error, motor speeds, and Dryden wind components
- **Target XYZ sliders** - move the setpoint in real-time while the drone is flying
- **RESET button** - restart the drone from its initial conditions
