#!/usr/bin/env python3
"""DATA 442 — Drone Simulation UI  (interactive live simulation)

Controls
- Target XYZ sliders: move the setpoint in real-time
- RESET button: restart drone from initial conditions
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drone_model import SimulationRunner, build_default_config
from drone_model.dynamics import rotation_matrix_from_euler

# Color palette
BG = "#0f0f0f"
PANEL = "#161616"
GRID = "#2a2a2a"
TXT = "#d0d0d0"
ACCENT = "#f0a030"
BLUE = "#4a9eff"
RED = "#ff4d4d"
GREEN = "#44cc88"
PURPLE = "#bb88ff"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "text.color": TXT,
    "axes.edgecolor": GRID,
    "xtick.color": TXT,
    "ytick.color": TXT,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "font.family": "monospace",
    "font.size": 8,
})

_ROTOR_VIS = 3.5    # visual arm scale relative to physical arm_length
MAX_HIST   = 2000   # rolling history depth (~20 s at dt = 0.01)
WIN_STEPS  = 1000   # telemetry window (10 s at dt = 0.01)

# Geometry helpers 
def _arm_diagonals(pos: np.ndarray, att: np.ndarray, arm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = rotation_matrix_from_euler(att)
    a = arm * _ROTOR_VIS
    tips = np.array([[a, a, 0.0], [-a, a, 0.0], [-a, -a, 0.0], [a, -a, 0.0]])
    rots = np.array([pos + R @ t for t in tips])
    return rots[[0, 2]], rots[[1, 3]], rots   # diag1, diag2, all-4

def _3d_axis_widget(ax, origin: np.ndarray, scale: float) -> None:
    for vec, col, lbl in zip(
        [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])],
        [RED, GREEN, BLUE], ["X", "Y", "Z"],
    ):
        ax.quiver(*origin, *(vec * scale), color=col, linewidth=1.3, arrow_length_ratio=0.35, normalize=False)
        ax.text(*(origin + vec * scale * 1.3), lbl, color=col, fontsize=6.5)


def _style_ax2d(ax: plt.Axes, title: str = "") -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, alpha=0.8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    if title:
        ax.set_title(title, fontsize=7.5, pad=4)


# Entry point 
def main() -> None:
    parser = argparse.ArgumentParser(description="DATA 442 – Drone Simulation UI")
    parser.add_argument("--stride", type=int, default=5, help="Simulation steps per animation frame (default 5).")
    parser.add_argument("--save-frame", type=Path, default=None, help="Save first frame to file instead of opening window.")
    args = parser.parse_args()

    config = build_default_config()
    runner = SimulationRunner(config)
    dynamics = runner.dynamics
    ctrl = runner.controller
    env = runner.environment
    dt = config.simulation.dt
    arm = config.drone.arm_length
    max_om = config.drone.max_omega
    max_tilt = config.drone.max_tilt_rad
    init_pos = config.simulation.initial_position.copy()
    init_tgt = config.simulation.target_position.copy()

    # Live simulation state (mutable containers for closure mutation)
    sim_state = [runner.initial_state()]
    t_sim = [0.0]
    current_target = [init_tgt.copy()]

    # Rolling history deques – auto-drop oldest when full
    hist_t = deque(maxlen=MAX_HIST)
    hist_pos = deque(maxlen=MAX_HIST)
    hist_att = deque(maxlen=MAX_HIST)
    hist_om = deque(maxlen=MAX_HIST)
    hist_w = deque(maxlen=MAX_HIST)
    hist_sat = deque(maxlen=MAX_HIST)

    def step_sim(n: int) -> None:
        for _ in range(n):
            w = env.update(dt)
            cmd = ctrl.compute_command(sim_state[0], current_target[0], dt)
            act = dynamics.mix(cmd)
            hist_t.append(t_sim[0])
            hist_pos.append(sim_state[0].position.copy())
            hist_att.append(sim_state[0].attitude.copy())
            hist_om.append(act.motor_omegas.copy())
            hist_w.append(w.copy())
            hist_sat.append(act.saturated)
            sim_state[0] = runner._rk4_step(sim_state[0], act, w, dt)
            t_sim[0] += dt

    def do_reset(_event=None) -> None:
        sim_state[0] = runner.initial_state()
        ctrl.position_pid.reset()
        ctrl.attitude_pid.reset()
        ctrl.rate_pid.reset()
        env.reset()
        t_sim[0] = 0.0
        hist_t.clear();   hist_pos.clear(); hist_att.clear()
        hist_om.clear();  hist_w.clear();   hist_sat.clear()
        # Prime with one step so artists never read from empty deques
        step_sim(1)

    # Prime history before building artists
    step_sim(1)

    # Figure / GridSpec 
    fig = plt.figure(figsize=(13.5, 8.6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("DATA 442 — DRONE SIMULATION UI", color=TXT, fontsize=10, fontweight="bold",
        x=0.02, ha="left", y=0.987,
    )

    # Leave bottom=0.14 so slider axes sit comfortably below
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1, 15, 10],
        hspace=0.13,
        left=0.03, right=0.97, top=0.958, bottom=0.14,
    )

    ax_hdr = fig.add_subplot(gs[0])
    ax_hdr.set_facecolor(BG); ax_hdr.axis("off")

    gs_mid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1], width_ratios=[63, 37], wspace=0.05,
    )
    ax3d = fig.add_subplot(gs_mid[0], projection="3d")
    ax_pid = fig.add_subplot(gs_mid[1])

    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[2], wspace=0.40,
    )
    ax_err = fig.add_subplot(gs_bot[0])
    ax_ctl = fig.add_subplot(gs_bot[1])
    ax_wnd = fig.add_subplot(gs_bot[2])

    # Slider / button widget axes 
    ax_ctrl_lbl = fig.add_axes([0.03, 0.098, 0.50, 0.018])
    ax_ctrl_lbl.set_facecolor(BG); ax_ctrl_lbl.axis("off")
    ax_ctrl_lbl.text(0, 0.5, "— SET TARGET —", transform=ax_ctrl_lbl.transAxes, color="#666666", fontsize=7, va="center")

    ax_sl_x = fig.add_axes([0.07, 0.052, 0.23, 0.026])
    ax_sl_y = fig.add_axes([0.36, 0.052, 0.23, 0.026])
    ax_sl_z = fig.add_axes([0.65, 0.052, 0.17, 0.026])
    ax_btn  = fig.add_axes([0.86, 0.038, 0.10, 0.045])

    sl_x = Slider(ax_sl_x, "X", -2.0, 8.0, valinit=init_tgt[0], color=BLUE)
    sl_y = Slider(ax_sl_y, "Y", -2.0, 8.0, valinit=init_tgt[1], color=GREEN)
    sl_z = Slider(ax_sl_z, "Z", 0.0, 6.0, valinit=init_tgt[2], color=ACCENT)
    btn_reset = Button(ax_btn, "RESET", color=GRID, hovercolor="#3a3a3a")

    for sl in (sl_x, sl_y, sl_z):
        sl.label.set_color(TXT)
        sl.valtext.set_color(TXT)
        sl.ax.set_facecolor(PANEL)

    btn_reset.label.set_color(RED)
    btn_reset.label.set_fontweight("bold")
    btn_reset.label.set_fontsize(9)

    def on_slider(_val: float) -> None:
        current_target[0] = np.array([sl_x.val, sl_y.val, sl_z.val])

    sl_x.on_changed(on_slider)
    sl_y.on_changed(on_slider)
    sl_z.on_changed(on_slider)
    btn_reset.on_clicked(do_reset)

    # 3-D environment
    ax3d.set_facecolor(PANEL)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(GRID)
    ax3d.tick_params(labelsize=6)
    ax3d.set_xlabel("X (m)", fontsize=6.5, labelpad=1)
    ax3d.set_ylabel("Y (m)", fontsize=6.5, labelpad=1)
    ax3d.set_zlabel("Z (m)", fontsize=6.5, labelpad=1)
    ax3d.set_title("3D ENVIRONMENT", fontsize=8, pad=5)
    ax3d.view_init(elev=22, azim=38)

    # Fixed bounds that cover the full slider range
    ax3d.set_xlim(-2.5, 8.5)
    ax3d.set_ylim(-2.5, 8.5)
    ax3d.set_zlim(0.0,  6.5)

    # XYZ widget (static)
    _3d_axis_widget(ax3d, np.array([-2.0, -2.0, 0.2]), 0.9)

    # Start dot (static)
    ax3d.scatter(*init_pos, color=BLUE, s=25, zorder=5)

    # Legend proxy lines
    ax3d.plot([], [], [], ls="--", color=BLUE,     lw=1.5,  label="flight path")
    ax3d.scatter([], [], [], c=ACCENT,    s=50, label="front rotors")
    ax3d.scatter([], [], [], c="#555555", s=50, label="rear rotors")
    ax3d.legend(
        loc="upper left", fontsize=6,
        facecolor=PANEL, edgecolor=GRID, labelcolor=TXT,
        framealpha=0.85, handlelength=1.5, markerscale=0.8,
    )

    # Dynamic target marker (follows sliders)
    target_sc  = ax3d.scatter([], [], [], c=RED, s=80, marker="*", zorder=5)
    tcross_h,  = ax3d.plot([], [], [], color=RED, lw=0.9)
    tcross_v,  = ax3d.plot([], [], [], color=RED, lw=0.9)
    target_lbl = ax3d.text(0, 0, 0, "", color=RED, fontsize=6.5)

    # Dynamic drone artists
    trail,   = ax3d.plot([], [], [], ls="--", color=BLUE,     lw=1.5, alpha=0.75)
    arm1_ln, = ax3d.plot([], [], [], color=TXT,      lw=2.8)
    arm2_ln, = ax3d.plot([], [], [], color=TXT,      lw=2.8)
    front_r  = ax3d.scatter([], [], [], c=ACCENT,    s=110, zorder=6, depthshade=False)
    rear_r   = ax3d.scatter([], [], [], c="#555555",  s=110, zorder=6, depthshade=False)
    body_c   = ax3d.scatter([], [], [], c="white",    s=60,  zorder=7, depthshade=False)
    alt_vl,  = ax3d.plot([], [], [], ls=":", color=TXT, lw=0.8, alpha=0.4)
    alt_txt  = ax3d.text(0, 0, 0, "", fontsize=6.5, ha="center")
    wind_q   = [None]   # removable quiver

    # PID state panel 
    ax_pid.set_facecolor(PANEL)
    ax_pid.set_xlim(-1.05, 1.05)
    ax_pid.set_ylim(-0.8, 12.0)
    ax_pid.axis("off")
    ax_pid.text(-1.02, 11.65, "PID STATE", fontsize=8.5, fontweight="bold", va="top")

    att_cfg = [
        ("roll φ",  RED,    9.0, max_tilt),
        ("pitch θ", GREEN,  6.5, max_tilt),
        ("yaw ψ",   ACCENT, 4.0, np.pi),
    ]
    att_txts = []
    att_bars = []
    for lbl, col, y, _ in att_cfg:
        ax_pid.text(-1.0, y + 0.72, lbl, color="#888888", fontsize=7)
        vt = ax_pid.text(-1.0, y - 0.05, "+0.0°", fontsize=10.5, fontweight="bold", va="top")
        att_txts.append(vt)
        ax_pid.plot([-0.95, 0.95], [y, y], color=GRID, lw=7, solid_capstyle="butt")
        bl, = ax_pid.plot([0.0, 0.0], [y, y], color=col, lw=7, solid_capstyle="butt")
        att_bars.append((bl, col))

    ax_pid.text(-1.0, 3.0, "motor ω (rad/s)", color="#888888", fontsize=7)
    om_txts = []
    for x, y, lbl in [
        (-1.0, 1.75, "ω1"), (0.02, 1.75, "ω2"),
        (-1.0, 0.25, "ω3"), (0.02, 0.25, "ω4"),
    ]:
        ax_pid.text(x + 0.04, y + 0.52, lbl, color="#777777", fontsize=6.5)
        t = ax_pid.text(
            x + 0.04, y, "0", fontsize=10.5, fontweight="bold",
            bbox=dict(facecolor=GRID, edgecolor=GRID, boxstyle="round,pad=0.22"),
        )
        om_txts.append(t)
    sat_txt = ax_pid.text(-1.0, -0.58, "", color=RED, fontsize=7)

    # Telemetry axes 
    for ax, ttl in [
        (ax_err, "position error (m)"),
        (ax_ctl, "control effort / saturation"),
        (ax_wnd, "dryden wind (m/s)"),
    ]:
        _style_ax2d(ax, ttl)

    ax_ctl.axhline(max_om, color=RED, lw=0.8, ls="--", alpha=0.65)
    ax_ctl.text(1.0, max_om, " MAX",
                transform=ax_ctl.get_yaxis_transform(),
                color=RED, fontsize=6.5, va="center")
    ax_wnd.axhline(0.0, color=GRID, lw=0.6)

    err_ln,  = ax_err.plot([], [], color=PURPLE, lw=1.5)
    ctl_l1,  = ax_ctl.plot([], [], color=ACCENT, lw=1.2, label="ω1")
    ctl_l4,  = ax_ctl.plot([], [], color=RED,    lw=1.2, label="ω4")
    wnd_u,   = ax_wnd.plot([], [], color=GREEN,  lw=1.2, label="U")
    wnd_v,   = ax_wnd.plot([], [], color=PURPLE, lw=1.2, label="V")
    for ax in (ax_ctl, ax_wnd):
        ax.legend(fontsize=6, facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, loc="upper left", framealpha=0.85)

    # Header bar 
    hdr_txt = ax_hdr.text(
        0.01, 0.45, "",
        transform=ax_hdr.transAxes, fontsize=9,
        va="center", fontfamily="monospace",
    )
    ax_hdr.text(
        0.785, 0.45, " FLYING ",
        transform=ax_hdr.transAxes, color=BG,
        fontsize=8.5, va="center", fontweight="bold",
        bbox=dict(facecolor=GREEN, edgecolor="none", boxstyle="round,pad=0.25"),
    )
    ax_hdr.text(
        0.875, 0.45, " PID: CASCADE ",
        transform=ax_hdr.transAxes, fontsize=8.5, va="center",
        bbox=dict(facecolor=PANEL, edgecolor=GREEN, boxstyle="round,pad=0.25", linewidth=1.2),
    )

    # Animation update 
    stride = max(args.stride, 1)

    def update(_frame: int) -> tuple:
        step_sim(stride)

        p     = hist_pos[-1]
        at    = hist_att[-1]
        w     = hist_w[-1]
        om    = hist_om[-1]
        sat   = hist_sat[-1]
        t_now = hist_t[-1]
        ct    = current_target[0]

        # header
        wm = float(np.linalg.norm(w))
        hdr_txt.set_text(
            f"t = {t_now:6.1f} s    alt = {p[2]:.1f} m    wind = {wm:.1f} m/s"
        )

        # trail (last 400 recorded steps)
        trail_arr = np.array(list(hist_pos)[-400:])
        trail.set_data(trail_arr[:, 0], trail_arr[:, 1])
        trail.set_3d_properties(trail_arr[:, 2])

        # drone geometry
        d1, d2, rots = _arm_diagonals(p, at, arm)
        arm1_ln.set_data(d1[:, 0], d1[:, 1]); arm1_ln.set_3d_properties(d1[:, 2])
        arm2_ln.set_data(d2[:, 0], d2[:, 1]); arm2_ln.set_3d_properties(d2[:, 2])
        front_r._offsets3d = (rots[:2, 0], rots[:2, 1], rots[:2, 2])
        rear_r._offsets3d  = (rots[2:, 0], rots[2:, 1], rots[2:, 2])
        body_c._offsets3d  = ([p[0]], [p[1]], [p[2]])

        # altitude indicator
        alt_vl.set_data([p[0], p[0]], [p[1], p[1]])
        alt_vl.set_3d_properties([0.0, p[2]])
        alt_txt.set_position_3d((p[0], p[1], 0.0))
        alt_txt.set_text(f"{p[2]:.1f}m")

        # dynamic target marker
        target_sc._offsets3d = ([ct[0]], [ct[1]], [ct[2]])
        tcross_h.set_data([ct[0] - 0.25, ct[0] + 0.25], [ct[1], ct[1]])
        tcross_h.set_3d_properties([ct[2], ct[2]])
        tcross_v.set_data([ct[0], ct[0]], [ct[1] - 0.25, ct[1] + 0.25])
        tcross_v.set_3d_properties([ct[2], ct[2]])
        target_lbl.set_position_3d((ct[0] + 0.15, ct[1] + 0.15, ct[2]))
        target_lbl.set_text("TARGET")

        # wind quiver (remove → redraw)
        if wind_q[0] is not None:
            wind_q[0].remove()
            wind_q[0] = None
        if wm > 0.15:
            sc_w = min(0.6 / wm, 1.2)
            wind_q[0] = ax3d.quiver(
                p[0], p[1], p[2],
                w[0] * sc_w, w[1] * sc_w, w[2] * sc_w,
                color=RED, lw=1.5, arrow_length_ratio=0.3, normalize=False,
            )

        # PID attitude bars
        at_deg = np.rad2deg(at)
        for val, vt, (bl, _), (_, _, _, lim) in zip(
            at_deg, att_txts, att_bars, att_cfg
        ):
            sign = "+" if val >= 0 else ""
            vt.set_text(f"{sign}{val:.1f}°")
            lim_deg = np.rad2deg(lim)
            n = float(np.clip(val / lim_deg, -1.0, 1.0)) * 0.95
            bl.set_xdata([0.0, n])

        # motor omega boxes
        near_sat = False
        for om_j, t_art in zip(om, om_txts):
            ratio = om_j / max_om
            if ratio > 0.92:
                clr = RED; near_sat = True
            elif ratio > 0.72:
                clr = ACCENT
            else:
                clr = TXT
            t_art.set_text(str(int(om_j)))
            t_art.set_color(clr)
            bp = t_art.get_bbox_patch()
            if bp is not None:
                bp.set_edgecolor(RED if ratio > 0.92 else GRID)
        sat_txt.set_text("▲ ω near sat." if (near_sat or sat) else "")

        # scrolling telemetry (last WIN_STEPS recorded steps)
        n_w   = min(WIN_STEPS, len(hist_t))
        tw    = np.array(list(hist_t)[-n_w:])
        pos_w = np.array(list(hist_pos)[-n_w:])
        om_w  = np.array(list(hist_om)[-n_w:])
        wnd_w = np.array(list(hist_w)[-n_w:])

        errs = np.linalg.norm(pos_w - ct, axis=1)
        err_ln.set_data(tw, errs)
        ax_err.set_xlim(tw[0], max(tw[-1], tw[0] + 0.01))
        ax_err.set_ylim(0.0, max(float(errs.max()), 0.3) * 1.18)

        ctl_l1.set_data(tw, om_w[:, 0])
        ctl_l4.set_data(tw, om_w[:, 3])
        ax_ctl.set_xlim(tw[0], max(tw[-1], tw[0] + 0.01))
        ax_ctl.set_ylim(0.0, max_om * 1.05)

        wnd_u.set_data(tw, wnd_w[:, 0])
        wnd_v.set_data(tw, wnd_w[:, 1])
        ax_wnd.set_xlim(tw[0], max(tw[-1], tw[0] + 0.01))
        wmax = max(float(np.abs(wnd_w[:, :2]).max()), 0.5)
        ax_wnd.set_ylim(-wmax * 1.25, wmax * 1.25)

        return ()

    interval_ms = max(int(1000 * dt * stride), 16)

    if args.save_frame:
        update(0)
        fig.savefig(args.save_frame, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"Saved to {args.save_frame}")
        return

    # frames=None → itertools.count() → runs indefinitely
    anim = FuncAnimation(
        fig, update,
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
    del anim


if __name__ == "__main__":
    main()