# Drone Model

DATA 442 final project

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_sim.py
python scripts/view_flight.py
```

`run_sim.py` prints summary telemetry.

`view_flight.py` opens a very simple 3D animation of the drone moving through space toward the target.

You can also overlay wind in the viewer:

```bash
python scripts/view_flight.py --wind-style both
```

Options are `none`, `arrows`, `heatmap`, or `both`.
