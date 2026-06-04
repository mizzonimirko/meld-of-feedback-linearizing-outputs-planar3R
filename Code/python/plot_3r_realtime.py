from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Rectangle, Patch


ACTIVE_INTERVALS_DEFAULT = np.array([
    [0, 4],
    [4, 7],
    [7, 9],
    [9, 13],
    [13, 18],
    [18, 22],
], dtype=float)

ACTIVE_VARS_DEFAULT = [
    ["q3", "pb1x", "pb1y"],
    ["q2", "pb2x", "pb2y"],
    ["q3", "pb1x", "pb1y"],
    ["q3", "pb2x", "pb2y"],
    ["q1", "pb2x", "pb2y"],
    ["q1", "q2", "q3"],
]


# ---------------------------------------------------------------------
# Paper-style colors
# ---------------------------------------------------------------------
COLOR_ACTUAL = "black"
COLOR_DESIRED = "#B22222"
COLOR_ACTIVE = "#B22222"
COLOR_SWITCH = "0.25"

LINK_GRAY = "#696969"
JOINT_DARK = "#232323"
BASKET_BLUE = "#1F4E79"
ACTIVE_BALL = "#B22222"
ACTIVE_RING = "#1F4E79"
ACTIVE_CROSSHAIR = "#232323"
INACTIVE_BALL = "#AAB2B9"
TABLE_BORDER = "#828282"

TRACE_1 = "#5A7896"
TRACE_2 = "#5F7D69"
DESIRED_RED = "#962D2D"

MELD_COLORS = [
    "#FDE0DD",
    "#FCC5C0",
    "#FCAE91",
    "#FB6A4A",
    "#DE2D26",
    "#A50F15",
]

MELD_LABELS = [
    r"$\sigma_1$",
    r"$\sigma_2$",
    r"$\sigma_3$",
    r"$\sigma_4$",
    r"$\sigma_5$",
    r"$\sigma_6$",
]


def mpl_color(c):
    if isinstance(c, str):
        return c
    arr = np.array(c, dtype=float)
    return arr / 255.0


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def matlab_cellstr_to_list(x):
    arr = np.array(x).squeeze()
    out = []

    if arr.ndim == 0:
        arr = np.array([arr.item()])

    for item in arr:
        if isinstance(item, str):
            s = item
        elif isinstance(item, np.ndarray):
            flat = item.squeeze()
            if flat.dtype.kind in {"U", "S"}:
                s = "".join(str(v) for v in flat.flat)
            else:
                s = str(flat)
        else:
            s = str(item)

        out.append([v.strip() for v in s.split(",") if v.strip()])

    return out


def require_key(container, key, file_path):
    if key not in container:
        available = ", ".join(sorted(container.keys()))
        raise KeyError(
            f"Missing variable '{key}' in {file_path}.\n"
            f"Available variables are:\n{available}"
        )
    return container[key]


def load_data(path: Path):
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".mat":
        # Important: convert Path to str for scipy compatibility.
        m = loadmat(str(path), squeeze_me=True)

        active_vars = matlab_cellstr_to_list(m["activeVars"]) if "activeVars" in m else ACTIVE_VARS_DEFAULT

        return {
            "tvec": np.array(require_key(m, "tvec", path)).squeeze(),
            "q": np.array(require_key(m, "q", path)),
            "qd": np.array(require_key(m, "qd", path)),
            "pb1": np.array(require_key(m, "pb1", path)),
            "pb2": np.array(require_key(m, "pb2", path)),
            "b1d": np.array(require_key(m, "b1d", path)),
            "b2d": np.array(require_key(m, "b2d", path)),
            "l1": float(np.array(require_key(m, "l1", path)).squeeze()),
            "activeIntervals": np.array(m.get("activeIntervals", ACTIVE_INTERVALS_DEFAULT), dtype=float),
            "activeVars": active_vars,
            "switching_instants": np.array(require_key(m, "switching_instants", path)).squeeze(),
            "p_des": np.array(require_key(m, "p_des", path)),
        }

    if path.suffix.lower() == ".npz":
        # Important: convert Path to str for numpy compatibility.
        data = np.load(str(path), allow_pickle=True)
        active_vars_raw = data.get("activeVars", None)

        if active_vars_raw is None:
            active_vars = ACTIVE_VARS_DEFAULT
        else:
            active_vars = [[v.strip() for v in str(s).split(",")] for s in active_vars_raw]

        return {
            "tvec": data["tvec"].squeeze(),
            "q": data["q"],
            "qd": data["qd"],
            "pb1": data["pb1"],
            "pb2": data["pb2"],
            "b1d": data["b1d"],
            "b2d": data["b2d"],
            "l1": float(np.array(data["l1"]).squeeze()),
            "activeIntervals": data.get("activeIntervals", ACTIVE_INTERVALS_DEFAULT),
            "activeVars": active_vars,
            "switching_instants": data["switching_instants"].squeeze(),
            "p_des": data["p_des"],
        }

    raise ValueError(f"Unsupported file: {path}")


def ensure_shape(data):
    n = len(data["tvec"])

    for key, rows in [
        ("q", 3),
        ("qd", 3),
        ("pb1", 2),
        ("pb2", 2),
        ("b1d", 2),
        ("b2d", 2),
    ]:
        x = np.array(data[key], dtype=float)

        if x.ndim == 1:
            raise ValueError(f"{key} should be 2D, got {x.shape}")

        if x.shape[0] != rows and x.shape[-1] == rows:
            x = x.T

        if x.shape[0] != rows:
            raise ValueError(f"{key} should have shape ({rows}, N), got {x.shape}")

        if x.shape[1] != n:
            raise ValueError(f"{key} has {x.shape[1]} samples but tvec has {n}")

        data[key] = x

    p_des = np.array(data["p_des"], dtype=float)

    if p_des.ndim == 1:
        p_des = p_des.reshape(2, -1)

    if p_des.shape[0] != 2 and p_des.shape[-1] == 2:
        p_des = p_des.T

    data["p_des"] = p_des

    active_intervals = np.array(data["activeIntervals"], dtype=float)
    if active_intervals.shape[0] != 6 and active_intervals.shape[-1] == 6:
        active_intervals = active_intervals.T
    data["activeIntervals"] = active_intervals

    return data


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
def forward_kinematics(q, L):
    q1, q2, q3 = q

    x0, y0 = 0.0, 0.0

    x1 = L[0] * np.cos(q1)
    y1 = L[0] * np.sin(q1)

    x2 = x1 + L[1] * np.cos(q1 + q2)
    y2 = y1 + L[1] * np.sin(q1 + q2)

    x3 = x2 + L[2] * np.cos(q1 + q2 + q3)
    y3 = y2 + L[2] * np.sin(q1 + q2 + q3)

    return np.array([
        [x0, y0],
        [x1, y1],
        [x2, y2],
        [x3, y3],
    ])


def active_interval_at(t, active_intervals):
    for i, (a, b) in enumerate(active_intervals):
        if a <= t <= b:
            return i
    return None


def visible_target_index(t, switching_instants, dt):
    idx = None

    for s, ts in enumerate(switching_instants):
        if t >= ts - 0.5 * dt:
            idx = s

    return idx


def active_target_position(data, k, target_idx):
    """Use moving desired basket trajectories in intervals 2 and 3."""
    t = float(data["tvec"][k])
    active_intervals = data["activeIntervals"]

    bx = float(data["p_des"][0, target_idx])
    by = float(data["p_des"][1, target_idx])

    # Python index 1 = MATLAB interval 2.
    if active_intervals[1, 0] <= t <= active_intervals[1, 1]:
        p_nom = data["p_des"][:, target_idx]
        b1 = data["b1d"][:, k]
        b2 = data["b2d"][:, k]

        d1 = np.sum((b1 - p_nom) ** 2)
        d2 = np.sum((b2 - p_nom) ** 2)

        if d1 <= d2:
            bx, by = float(b1[0]), float(b1[1])
        else:
            bx, by = float(b2[0]), float(b2[1])

    # Python index 2 = MATLAB interval 3.
    elif active_intervals[2, 0] <= t <= active_intervals[2, 1]:
        bx = float(data["b1d"][0, k])
        by = float(data["b1d"][1, k])

    return bx, by


# ---------------------------------------------------------------------
# Plot setup helpers
# ---------------------------------------------------------------------
def setup_time_axis(ax, tvec, y, yd, active_intervals, switch_times, ylabel, var_name, active_vars):
    y_all = np.concatenate([np.ravel(y), np.ravel(yd)])
    finite = y_all[np.isfinite(y_all)]

    if finite.size == 0:
        y_min, y_max = -1.0, 1.0
    else:
        y_min, y_max = np.min(finite), np.max(finite)

    pad = 0.08 * max(y_max - y_min, 1e-6)
    y_min -= pad
    y_max += pad

    for i, (a, b) in enumerate(active_intervals):
        color = MELD_COLORS[i % len(MELD_COLORS)]
        alpha = 0.38 if var_name in active_vars[i] else 0.16
        ax.axvspan(a, b, color=color, alpha=alpha, linewidth=0)

    for ts in switch_times:
        ax.axvline(ts, color=COLOR_SWITCH, linestyle="--", linewidth=0.8, alpha=0.55)

    ax.set_xlim(tvec[0], tvec[-1])
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, which="major", linewidth=0.35, alpha=0.35)
    ax.tick_params(axis="both", labelsize=9)

    return y_min, y_max


def draw_basket(ax, center, radius=0.20):
    outer = Circle(center, radius, fill=False, edgecolor=mpl_color(BASKET_BLUE), linewidth=2.2)
    inner = Circle(center, radius * 0.72, fill=False, edgecolor=mpl_color(BASKET_BLUE), linewidth=1.2)
    ax.add_patch(outer)
    ax.add_patch(inner)
    return outer, inner


def update_basket_patches(patches, center, radius=0.20):
    outer, inner = patches
    outer.center = center
    outer.radius = radius
    inner.center = center
    inner.radius = radius * 0.72


# ---------------------------------------------------------------------
# Main animation
# ---------------------------------------------------------------------
def animate(data, frame_step=1, save_mp4=None, fps=60, realtime=False):
    tvec = data["tvec"]
    q = data["q"]
    qd = data["qd"]
    pb1 = data["pb1"]
    pb2 = data["pb2"]
    b1d = data["b1d"]
    b2d = data["b2d"]

    active_intervals = data["activeIntervals"]
    active_vars = data["activeVars"]
    switching_instants = data["switching_instants"]
    p_des = data["p_des"]

    L = np.array([data["l1"], data["l1"], data["l1"]], dtype=float)
    switch_times = np.unique(active_intervals.ravel())

    variables = [
        {"data": q, "des": qd, "idx": 0, "label": r"$q_1$ [rad]", "name": "q1"},
        {"data": q, "des": qd, "idx": 1, "label": r"$q_2$ [rad]", "name": "q2"},
        {"data": q, "des": qd, "idx": 2, "label": r"$q_3$ [rad]", "name": "q3"},
        {"data": pb1, "des": b1d, "idx": 0, "label": r"$p^B_{1,x}$ [m]", "name": "pb1x"},
        {"data": pb1, "des": b1d, "idx": 1, "label": r"$p^B_{1,y}$ [m]", "name": "pb1y"},
        {"data": pb2, "des": b2d, "idx": 0, "label": r"$p^B_{2,x}$ [m]", "name": "pb2x"},
        {"data": pb2, "des": b2d, "idx": 1, "label": r"$p^B_{2,y}$ [m]", "name": "pb2y"},
    ]

    fig = plt.figure(figsize=(20, 8.5), constrained_layout=False)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.14)

    left = outer[0].subgridspec(7, 1, hspace=0.08)
    ax_vars = [fig.add_subplot(left[i, 0]) for i in range(7)]

    ax_anim = fig.add_subplot(outer[1])
    fig.patch.set_facecolor("white")

    time_title = fig.suptitle(r"$t = 0.00\,\mathrm{s}$", fontsize=22, y=0.985)

    data_lines = []
    des_lines = []
    active_lines = []
    curr_markers = []

    for i, var in enumerate(variables):
        ax = ax_vars[i]
        x = var["data"][var["idx"], :]
        xd = var["des"][var["idx"], :]

        setup_time_axis(
            ax,
            tvec,
            x,
            xd,
            active_intervals,
            switch_times,
            var["label"],
            var["name"],
            active_vars,
        )

        actual_line, = ax.plot([], [], color=COLOR_ACTUAL, linewidth=1.5)
        desired_line, = ax.plot([], [], color=COLOR_DESIRED, linestyle="--", linewidth=1.35)
        active_line, = ax.plot([], [], color=COLOR_ACTIVE, linewidth=2.4)
        marker, = ax.plot([], [], marker="o", color=COLOR_ACTIVE, markersize=4.5)

        data_lines.append(actual_line)
        des_lines.append(desired_line)
        active_lines.append(active_line)
        curr_markers.append(marker)

        if i < len(variables) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time [s]", fontsize=11)

    ax_anim.set_aspect("equal", adjustable="box")
    ax_anim.set_xlim(-1.0, 8.4)
    ax_anim.set_ylim(-3.0, 5.6)
    ax_anim.set_xlabel(r"$x$ [m]", fontsize=14)
    ax_anim.set_ylabel(r"$y$ [m]", fontsize=14)
    ax_anim.tick_params(axis="both", labelsize=11)
    ax_anim.grid(True, linewidth=0.35, alpha=0.35)
    ax_anim.set_facecolor("#FAFAFA")

    table = Rectangle(
        (-1.0, -3.0),
        9.4,
        8.6,
        fill=False,
        edgecolor=mpl_color(TABLE_BORDER),
        linewidth=1.2,
    )
    ax_anim.add_patch(table)

    ax_anim.scatter(
        p_des[0, :],
        p_des[1, :],
        s=80,
        color=mpl_color(INACTIVE_BALL),
        edgecolors="none",
        zorder=2,
    )

    link_line, = ax_anim.plot([], [], color=LINK_GRAY, linewidth=11, solid_capstyle="round", zorder=4)
    link_highlight, = ax_anim.plot([], [], color="#AAAAAA", linewidth=2.3, solid_capstyle="round", zorder=5)
    joint_scatter = ax_anim.scatter([], [], s=[], color=JOINT_DARK, zorder=6)

    basket1 = draw_basket(ax_anim, (0, 0), radius=0.20)
    basket2 = draw_basket(ax_anim, (0, 0), radius=0.20)

    active_ball = ax_anim.scatter([], [], s=220, color=mpl_color(ACTIVE_BALL), zorder=7)

    active_ring = Circle(
        (0, 0),
        0.30,
        fill=False,
        edgecolor=mpl_color(ACTIVE_RING),
        linewidth=2.2,
        zorder=6,
    )
    ax_anim.add_patch(active_ring)

    cross_h, = ax_anim.plot([], [], color=mpl_color(ACTIVE_CROSSHAIR), linewidth=1.35, zorder=8)
    cross_v, = ax_anim.plot([], [], color=mpl_color(ACTIVE_CROSSHAIR), linewidth=1.35, zorder=8)

    desired_lines = []
    for _ in range(3):
        line, = ax_anim.plot([], [], color=DESIRED_RED, linestyle="--", linewidth=2.2, zorder=3)
        desired_lines.append(line)

    trace1_line, = ax_anim.plot([], [], color=mpl_color(TRACE_1), linewidth=1.1, alpha=0.75, zorder=1)
    trace2_line, = ax_anim.plot([], [], color=mpl_color(TRACE_2), linewidth=1.1, alpha=0.75, zorder=1)

    meld_handles = [
        Patch(
            facecolor=MELD_COLORS[i],
            edgecolor="none",
            alpha=0.55,
            label=MELD_LABELS[i],
        )
        for i in range(len(MELD_LABELS))
    ]

    curve_handles = [
        plt.Line2D([0], [0], color=COLOR_ACTUAL, linewidth=2.0, label="Actual"),
        plt.Line2D([0], [0], color=COLOR_DESIRED, linestyle="--", linewidth=1.8, label="Desired"),
        plt.Line2D([0], [0], color=COLOR_ACTIVE, linewidth=2.4, label="Active variable"),
        plt.Line2D([0], [0], color=COLOR_SWITCH, linestyle="--", linewidth=1.3, label=r"$t_k$"),
        plt.Line2D([0], [0], color=LINK_GRAY, linewidth=7, label="Robot"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=mpl_color(ACTIVE_BALL),
            markeredgecolor="black",
            markersize=10,
            label="Active target",
        ),
        plt.Line2D([0], [0], color=mpl_color(BASKET_BLUE), linewidth=2.2, label="Baskets"),
    ]

    legend = fig.legend(
        handles=meld_handles + curve_handles,
        loc="lower center",
        ncol=7,
        frameon=True,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.015),
        handlelength=2.2,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("0.25")

    fig.subplots_adjust(bottom=0.20)

    frame_indices = np.arange(0, len(tvec), frame_step)

    def update(frame_number):
        k = int(frame_indices[frame_number])
        t = float(tvec[k])

        curr = active_interval_at(t, active_intervals)
        time_title.set_text(rf"$t = {t:.2f}\,\mathrm{{s}}$")

        for v, var in enumerate(variables):
            actual = var["data"][var["idx"], :]
            desired = var["des"][var["idx"], :]

            data_lines[v].set_data(tvec[: k + 1], actual[: k + 1])

            y_des = np.full(k + 1, np.nan)
            y_active = np.full(k + 1, np.nan)

            if curr is not None and var["name"] in active_vars[curr]:
                mask = (tvec[: k + 1] >= active_intervals[curr, 0]) & (
                    tvec[: k + 1] <= active_intervals[curr, 1]
                )

                y_des[mask] = desired[: k + 1][mask]
                y_active[mask] = actual[: k + 1][mask]

            des_lines[v].set_data(tvec[: k + 1], y_des)
            active_lines[v].set_data(tvec[: k + 1], y_active)
            curr_markers[v].set_data([t], [actual[k]])

        pts = forward_kinematics(q[:, k], L)
        xs = pts[:, 0]
        ys = pts[:, 1]

        link_line.set_data(xs, ys)
        link_highlight.set_data(xs, ys)
        joint_scatter.set_offsets(pts)
        joint_scatter.set_sizes([170, 135, 115, 95])

        link2_mid = 0.5 * (pts[1] + pts[2])
        end_eff = pts[3]

        update_basket_patches(basket1, link2_mid, radius=0.20)
        update_basket_patches(basket2, end_eff, radius=0.20)

        dt = tvec[1] - tvec[0] if len(tvec) > 1 else 0.01
        target_idx = visible_target_index(t, switching_instants, dt)

        if target_idx is not None and target_idx < p_des.shape[1]:
            bx, by = active_target_position(data, k, target_idx)

            active_ball.set_offsets([[bx, by]])
            active_ring.center = (bx, by)
            active_ring.set_visible(True)

            cross_h.set_data([bx - 0.36, bx + 0.36], [by, by])
            cross_v.set_data([bx, bx], [by - 0.36, by + 0.36])
        else:
            active_ball.set_offsets(np.empty((0, 2)))
            active_ring.set_visible(False)
            cross_h.set_data([], [])
            cross_v.set_data([], [])

        if k > 2:
            trace1_line.set_data(pb1[0, : k + 1], pb1[1, : k + 1])
            trace2_line.set_data(pb2[0, : k + 1], pb2[1, : k + 1])

        for line in desired_lines:
            line.set_data([], [])

        if curr is not None:
            q_des = q[:, k].copy()

            for j in range(3):
                if f"q{j + 1}" in active_vars[curr]:
                    q_des[j] = qd[j, k]

            pts_des = forward_kinematics(q_des, L)

            for j in range(3):
                if f"q{j + 1}" in active_vars[curr]:
                    desired_lines[j].set_data(
                        [pts[j, 0], pts_des[j + 1, 0]],
                        [pts[j, 1], pts_des[j + 1, 1]],
                    )

        artists = (
            data_lines
            + des_lines
            + active_lines
            + curr_markers
            + [
                link_line,
                link_highlight,
                joint_scatter,
                active_ball,
                active_ring,
                cross_h,
                cross_v,
                trace1_line,
                trace2_line,
            ]
            + desired_lines
            + list(basket1)
            + list(basket2)
        )

        return artists

    interval_ms = 1000.0 / fps if realtime else 1

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval_ms,
        blit=False,
    )

    if save_mp4:
        print(f"Saving high-quality MP4 to {save_mp4}")

        writer = FFMpegWriter(
            fps=fps,
            bitrate=18000,
            codec="libx264",
            extra_args=[
                "-pix_fmt", "yuv420p",
                "-crf", "15",
                "-preset", "slow",
            ],
        )

        anim.save(save_mp4, writer=writer, dpi=240)
        print(f"Saved {save_mp4}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=Path)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--save-mp4", type=str, default=None)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Use real-time playback speed in the matplotlib window.",
    )

    args = parser.parse_args()

    data = ensure_shape(load_data(args.data_file))

    animate(
        data,
        frame_step=args.frame_step,
        save_mp4=args.save_mp4,
        fps=args.fps,
        realtime=args.realtime,
    )


if __name__ == "__main__":
    main()