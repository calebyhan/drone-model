from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drone_model import SimulationRunner, build_default_config


def main() -> None:
    config = build_default_config()
    runner = SimulationRunner(config)
    result = runner.run()

    final_position = result.position[-1]
    final_velocity = result.velocity[-1]
    final_attitude_deg = np.rad2deg(result.attitude[-1])
    target = result.target_position
    position_error = np.linalg.norm(final_position - target)
    peak_wind = np.max(np.linalg.norm(result.wind, axis=1))
    peak_motor_speed = float(np.max(result.motor_omegas))
    saturation_fraction = float(np.mean(result.saturation))

    print("Drone simulation complete.")
    print(f"Final position: {final_position.round(3)} m")
    print(f"Target position: {target.round(3)} m")
    print(f"Final velocity: {final_velocity.round(3)} m/s")
    print(f"Final attitude: {final_attitude_deg.round(2)} deg")
    print(f"Position error magnitude: {position_error:.3f} m")
    print(f"Peak wind magnitude: {peak_wind:.3f} m/s")
    print(f"Peak motor speed: {peak_motor_speed:.1f} rad/s")
    print(f"Motor saturation fraction: {saturation_fraction:.3f}")


if __name__ == "__main__":
    main()
