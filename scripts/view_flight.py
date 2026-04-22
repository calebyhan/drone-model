from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drone_model import SimulationRunner, build_default_config
from drone_model.dynamics import rotation_matrix_from_euler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple 3D drone flight viewer.")
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Use every Nth simulation sample in the animation.",
    )
    parser.add_argument(
        "--save-frame",
        type=Path,
        help="Save the first rendered frame to an image instead of opening a window.",
    )
    parser.add_argument(
        "--wind-style",
        choices=["none", "arrows", "heatmap", "both"],
        default="both",
        help="How to visualize wind along the flight path.",
    )
    return parser.parse_args()


def set_equal_3d_axes(ax: plt.Axes, points: np.ndarray, target: np.ndarray) -> None:
    stacked = np.vstack([points, target[None, :]])
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(np.max(maxs - mins) * 0.6, 1.5)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)


def drone_arm_segments(position: np.ndarray, attitude: np.ndarray, arm_length: float) -> tuple[np.ndarray, np.ndarray]:
    rotation = rotation_matrix_from_euler(attitude)
    body_x = rotation @ np.array([1.0, 0.0, 0.0], dtype=float)
    body_y = rotation @ np.array([0.0, 1.0, 0.0], dtype=float)

    x_segment = np.vstack(
        [position - arm_length * body_x, position + arm_length * body_x]
    )
    y_segment = np.vstack(
        [position - arm_length * body_y, position + arm_length * body_y]
    )
    return x_segment, y_segment


def add_wind_heatmap(
    ax: plt.Axes,
    positions: np.ndarray,
    wind_vectors: np.ndarray,
) -> tuple[plt.Artist, np.ndarray]:
    wind_speed = np.linalg.norm(wind_vectors, axis=1)
    heatmap = ax.scatter(
        [],
        [],
        [],
        c=[],
        cmap="YlOrRd",
        alpha=0.65,
        s=20,
        vmin=0.0,
        vmax=max(float(np.max(wind_speed)), 1e-6),
        label="Wind Speed",
    )
    colorbar = plt.colorbar(heatmap, ax=ax, pad=0.08, shrink=0.72)
    colorbar.set_label("Wind speed (m/s)")
    return heatmap, wind_speed


def add_static_wind_arrows(
    ax: plt.Axes,
    positions: np.ndarray,
    wind_vectors: np.ndarray,
) -> tuple[plt.Artist | None, float]:
    wind_speed = np.linalg.norm(wind_vectors, axis=1)
    max_wind = float(np.max(wind_speed)) if len(wind_speed) else 0.0
    arrow_scale = 0.9 if max_wind < 1e-6 else 0.9 / max_wind
    return None, arrow_scale


def main() -> None:
    args = parse_args()

    config = build_default_config()
    result = SimulationRunner(config).run()

    stride = max(args.stride, 1)
    positions = result.position[::stride]
    attitudes = result.attitude[::stride]
    times = result.time[::stride]
    wind_vectors = result.wind[::stride]
    target = result.target_position

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("Drone Flight Path", fontsize=15)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    set_equal_3d_axes(ax, positions, target)
    ax.view_init(elev=24, azim=42)

    ax.scatter(*positions[0], color="#1f77b4", s=40, label="Start")
    ax.scatter(*target, color="#d62728", s=60, marker="*", label="Target")

    wind_sample_step = max(len(positions) // 30, 1)
    wind_positions = positions[::wind_sample_step]
    wind_samples = wind_vectors[::wind_sample_step]
    wind_arrow_scale = 0.0

    heatmap = None
    heatmap_speeds = None
    if args.wind_style in {"heatmap", "both"}:
        heatmap, heatmap_speeds = add_wind_heatmap(ax, positions, wind_vectors)
    static_wind_quiver = None
    if args.wind_style in {"arrows", "both"}:
        static_wind_quiver, wind_arrow_scale = add_static_wind_arrows(ax, wind_positions, wind_samples)

    trail_line, = ax.plot([], [], [], color="#1f77b4", linewidth=2.0, label="Flight Path")
    drone_x_line, = ax.plot([], [], [], color="#111111", linewidth=3.0)
    drone_y_line, = ax.plot([], [], [], color="#ff7f0e", linewidth=3.0)
    drone_point, = ax.plot([], [], [], marker="o", color="#2ca02c", markersize=7)
    wind_line, = ax.plot([], [], [], color="#9467bd", linewidth=2.5, label="Current Wind")
    time_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)

    ax.legend(loc="upper right")

    def update(frame: int):
        nonlocal static_wind_quiver
        path = positions[: frame + 1]
        position = positions[frame]
        attitude = attitudes[frame]
        wind_vector = wind_vectors[frame]
        x_segment, y_segment = drone_arm_segments(position, attitude, config.drone.arm_length)

        trail_line.set_data(path[:, 0], path[:, 1])
        trail_line.set_3d_properties(path[:, 2])

        drone_x_line.set_data(x_segment[:, 0], x_segment[:, 1])
        drone_x_line.set_3d_properties(x_segment[:, 2])

        drone_y_line.set_data(y_segment[:, 0], y_segment[:, 1])
        drone_y_line.set_3d_properties(y_segment[:, 2])

        drone_point.set_data([position[0]], [position[1]])
        drone_point.set_3d_properties([position[2]])

        if heatmap is not None and heatmap_speeds is not None:
            heat_path = path
            heatmap._offsets3d = (
                heat_path[:, 0],
                heat_path[:, 1],
                np.zeros(len(heat_path)),
            )
            heatmap.set_array(heatmap_speeds[: frame + 1])

        if args.wind_style in {"arrows", "both"}:
            if static_wind_quiver is not None:
                static_wind_quiver.remove()

            visible_arrow_count = min(frame // wind_sample_step + 1, len(wind_positions))
            if visible_arrow_count > 0:
                current_positions = wind_positions[:visible_arrow_count]
                current_wind_samples = wind_samples[:visible_arrow_count] * wind_arrow_scale
                static_wind_quiver = ax.quiver(
                    current_positions[:, 0],
                    current_positions[:, 1],
                    current_positions[:, 2],
                    current_wind_samples[:, 0],
                    current_wind_samples[:, 1],
                    current_wind_samples[:, 2],
                    length=1.0,
                    normalize=False,
                    arrow_length_ratio=0.25,
                    color="#9467bd",
                    alpha=0.35,
                    linewidth=1.0,
                )

        if args.wind_style in {"arrows", "both"}:
            wind_tip = position + wind_arrow_scale * wind_vector
            wind_line.set_data([position[0], wind_tip[0]], [position[1], wind_tip[1]])
            wind_line.set_3d_properties([position[2], wind_tip[2]])
        else:
            wind_line.set_data([], [])
            wind_line.set_3d_properties([])
        time_text.set_text(f"t = {times[frame]:.2f} s")

        artists = [trail_line, drone_x_line, drone_y_line, drone_point, wind_line, time_text]
        if heatmap is not None:
            artists.append(heatmap)
        if static_wind_quiver is not None:
            artists.append(static_wind_quiver)
        return tuple(artists)

    if args.save_frame:
        update(0)
        fig.savefig(args.save_frame, dpi=160, bbox_inches="tight")
        print(f"Saved frame to {args.save_frame}")
    else:
        animation = FuncAnimation(
            fig,
            update,
            frames=len(positions),
            interval=max(int(1000 * config.simulation.dt * stride), 1),
            blit=False,
            repeat=False,
        )
        plt.show()
        del animation


if __name__ == "__main__":
    main()
