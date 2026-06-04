from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy import rec
import rerun as rr
from scipy.io import loadmat

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

# Elsevier / journal-style palette
LINK_GRAY = [105, 105, 105]
JOINT_DARK = [35, 35, 35]

TABLE_LINE = [215, 215, 215]
TABLE_BORDER = [130, 130, 130]

INACTIVE_BALL = [170, 178, 185]

# Active target styling
ACTIVE_BALL = [178, 34, 34]          # target ball: deep red
ACTIVE_RING = [31, 78, 121]          # ring around target: dark academic blue
ACTIVE_CROSSHAIR = [35, 35, 35]      # crosshair: near black

BASKET_BLUE = [31, 78, 121]
TRACE_1 = [90, 120, 150]
TRACE_2 = [95, 125, 105]

DESIRED_RED = [150, 45, 45]

MOVING_OBSTACLE_TRACE = [178, 34, 34]      # same family as active ball
MOVING_OBSTACLE_START = [35, 35, 35]       # dark start marker
MOVING_OBSTACLE_END = [31, 78, 121]        # blue current/end marker


def rr_points3d(positions, radii=None, colors=None):
    """Compatibility wrapper for old/new Rerun Points3D APIs."""
    kwargs = {}
    if radii is not None:
        kwargs["radii"] = radii
    if colors is not None:
        kwargs["colors"] = colors

    try:
        return rr.Points3D(positions=positions, **kwargs)
    except TypeError:
        try:
            return rr.Points3D(points=positions, **kwargs)
        except TypeError:
            # Very old API or incompatible colors: drop colors first.
            kwargs.pop("colors", None)
            try:
                return rr.Points3D(positions=positions, **kwargs)
            except TypeError:
                return rr.Points3D(points=positions, **kwargs)

def rr_line_strips3d(strips, radii=None, colors=None):
    """Compatibility wrapper for old/new Rerun LineStrips3D APIs."""
    kwargs = {}
    if radii is not None:
        kwargs["radii"] = radii
    if colors is not None:
        kwargs["colors"] = colors

    try:
        return rr.LineStrips3D(strips=strips, **kwargs)
    except TypeError:
        try:
            return rr.LineStrips3D(strips, **kwargs)
        except TypeError:
            kwargs.pop("colors", None)
            try:
                return rr.LineStrips3D(strips=strips, **kwargs)
            except TypeError:
                return rr.LineStrips3D(strips)

def make_recording(name: str, save_rrd: str | None = None, spawn: bool = True):
    """Create a Rerun recording. Can optionally save an .rrd recording."""
    if hasattr(rr, "RecordingStream"):
        rec = rr.RecordingStream(name)

        if save_rrd is not None:
            # Different Rerun SDK versions expose saving differently.
            if hasattr(rec, "save"):
                rec.save(save_rrd)
            elif hasattr(rr, "save"):
                rr.save(save_rrd)
            else:
                print("Warning: this rerun-sdk version does not expose save(); opening viewer only.")

        if spawn:
            rec.spawn()
        return rec

    rr.init(name, spawn=spawn)
    if save_rrd is not None and hasattr(rr, "save"):
        rr.save(save_rrd)
    return rr

def set_frame_time(rec, frame_idx: int, time_seconds: float):
    if hasattr(rec, "set_time"):
        try:
            rec.set_time("frame_idx", sequence=frame_idx)
            rec.set_time("sim_time", duration=float(time_seconds))
            return
        except TypeError:
            pass

    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame_idx", frame_idx)
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("sim_time", float(time_seconds))

def log(rec, path, obj, static=False):
    try:
        rec.log(path, obj, static=static)
    except TypeError:
        rec.log(path, obj)

def matlab_cellstr_to_list(x):
    arr = np.array(x).squeeze()
    out = []
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

def load_data(path: Path):
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
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

    if path.suffix.lower() == ".mat":
        m = loadmat(path, squeeze_me=True)
        active_vars = matlab_cellstr_to_list(m["activeVars"]) if "activeVars" in m else ACTIVE_VARS_DEFAULT
        return {
            "tvec": np.array(m["tvec"]).squeeze(),
            "q": np.array(m["q"]),
            "qd": np.array(m["qd"]),
            "pb1": np.array(m["pb1"]),
            "pb2": np.array(m["pb2"]),
            "b1d": np.array(m["b1d"]),
            "b2d": np.array(m["b2d"]),
            "l1": float(np.array(m["l1"]).squeeze()),
            "activeIntervals": np.array(m.get("activeIntervals", ACTIVE_INTERVALS_DEFAULT), dtype=float),
            "activeVars": active_vars,
            "switching_instants": np.array(m["switching_instants"]).squeeze(),
            "p_des": np.array(m["p_des"]),
        }

    raise ValueError(f"Unsupported input file: {path}")

def ensure_shape(data):
    n = len(data["tvec"])

    for key, rows in [("q", 3), ("qd", 3), ("pb1", 2), ("pb2", 2), ("b1d", 2), ("b2d", 2)]:
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

    return data

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
        [x0, y0, 0.0],
        [x1, y1, 0.0],
        [x2, y2, 0.0],
        [x3, y3, 0.0],
    ])

def circle_points(center_xy, radius, n=64, z=0.0):
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return np.column_stack([
        center_xy[0] + radius * np.cos(theta),
        center_xy[1] + radius * np.sin(theta),
        z * np.ones_like(theta),
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

def log_table_plane(rec):
    """Log a clean planar table instead of a busy background."""
    z = -0.04
    xmin, xmax = -1.0, 8.4
    ymin, ymax = -3.0, 5.6

    # Main table border.
    border = [[
        [xmin, ymin, z],
        [xmax, ymin, z],
        [xmax, ymax, z],
        [xmin, ymax, z],
        [xmin, ymin, z],
    ]]
    log(
        rec,
        "world/table/border",
        rr_line_strips3d(border, radii=[0.018], colors=[TABLE_BORDER]),
        static=True,
    )

    # Sparse, calm table grid.
    grid = []
    for x in np.arange(np.ceil(xmin), np.floor(xmax) + 1, 1.0):
        grid.append([[x, ymin, z], [x, ymax, z]])
    for y in np.arange(np.ceil(ymin), np.floor(ymax) + 1, 1.0):
        grid.append([[xmin, y, z], [xmax, y, z]])

    log(
        rec,
        "world/table/grid",
        rr_line_strips3d(grid, radii=[0.0015], colors=[TABLE_LINE]),
        static=True,
    )

    # Add a very subtle planar rectangle outline. This is intentionally not a filled surface
    # because it works across older Rerun versions.
    inner = [[
        [xmin + 0.15, ymin + 0.15, z + 0.002],
        [xmax - 0.15, ymin + 0.15, z + 0.002],
        [xmax - 0.15, ymax - 0.15, z + 0.002],
        [xmin + 0.15, ymax - 0.15, z + 0.002],
        [xmin + 0.15, ymin + 0.15, z + 0.002],
    ]]
    log(
        rec,
        "world/table/inner_frame",
        rr_line_strips3d(inner, radii=[0.01], colors=[TABLE_BORDER]),
        static=True,
    )

def log_static_context(rec, data):
    p_des = data["p_des"]

    # Calm table plane instead of a distracting background.
    log_table_plane(rec)

    # Show all target balls as muted balls.
    if p_des.size:
        targets = np.column_stack([
            p_des[0, :],
            p_des[1, :],
            0.12 * np.ones(p_des.shape[1]),
        ])

        log(
            rec,
            "world/targets/inactive_balls",
            rr_points3d(
                targets,
                radii=[0.105] * len(targets),
                colors=[INACTIVE_BALL] * len(targets),
            ),
            static=True,
        )

        # Subtle rings for all target locations.
        for i in range(p_des.shape[1]):
            ring = circle_points([p_des[0, i], p_des[1, i]], 0.17, z=0.12)
            log(
                rec,
                f"world/targets/target_{i+1}_ring",
                rr_line_strips3d([ring], radii=[0.006], colors=[INACTIVE_BALL]),
                static=True,
            )

def log_time_series(rec, data, k):
    if not hasattr(rr, "Scalar"):
        return

    variables = {
        "q1": data["q"][0, k],
        "q2": data["q"][1, k],
        "q3": data["q"][2, k],
        "pb1x": data["pb1"][0, k],
        "pb1y": data["pb1"][1, k],
        "pb2x": data["pb2"][0, k],
        "pb2y": data["pb2"][1, k],
    }

    desired = {
        "q1": data["qd"][0, k],
        "q2": data["qd"][1, k],
        "q3": data["qd"][2, k],
        "pb1x": data["b1d"][0, k],
        "pb1y": data["b1d"][1, k],
        "pb2x": data["b2d"][0, k],
        "pb2y": data["b2d"][1, k],
    }

    for name, value in variables.items():
        log(rec, f"plots/actual/{name}", rr.Scalar(float(value)))
    for name, value in desired.items():
        log(rec, f"plots/desired/{name}", rr.Scalar(float(value)))

def rectangle_link_points(p0, p1, width, z=0.025):
    """Return a closed rectangle around the segment p0 -> p1."""
    p0 = np.array(p0[:2], dtype=float)
    p1 = np.array(p1[:2], dtype=float)
    d = p1 - p0
    n = np.linalg.norm(d)

    if n < 1e-9:
        return np.column_stack([np.repeat(p0[0], 5), np.repeat(p0[1], 5), z * np.ones(5)])

    u = d / n
    normal = np.array([-u[1], u[0]])
    hw = 0.5 * width

    corners2 = np.array([
        p0 + hw * normal,
        p1 + hw * normal,
        p1 - hw * normal,
        p0 - hw * normal,
        p0 + hw * normal,
    ])

    return np.column_stack([corners2[:, 0], corners2[:, 1], z * np.ones(len(corners2))])

def log_link_body(rec, path, p0, p1, width=0.20):
    """Log one link as a mechanical planar bar."""
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)

    p0[2] = 0.05
    p1[2] = 0.05

    # Main link body
    log(
        rec,
        f"{path}/main_bar",
        rr_line_strips3d(
            [[p0.tolist(), p1.tolist()]],
            radii=[width * 0.36],
            colors=[LINK_GRAY],
        ),
    )

    # Dark side outline
    outline = rectangle_link_points(p0, p1, width, z=0.075)
    log(
        rec,
        f"{path}/outline",
        rr_line_strips3d(
            [outline],
            radii=[0.010],
            colors=[JOINT_DARK],
        ),
    )

    # Thin central highlight line
    mid_line = [
        [p0[0], p0[1], 0.105],
        [p1[0], p1[1], 0.105],
    ]
    log(
        rec,
        f"{path}/center_highlight",
        rr_line_strips3d(
            [mid_line],
            radii=[0.006],
            colors=[[170, 170, 170]],
        ),
    )
def log_joint_hub(rec, path, center, radius=0.12):
    """Log a more realistic planar revolute joint hub."""
    c = np.array(center, dtype=float)
    c[2] = 0.10

    # Outer dark ring
    outer = circle_points(c[:2], radius, n=72, z=0.11)
    log(
        rec,
        path + "/outer_ring",
        rr_line_strips3d([outer], radii=[0.026], colors=[JOINT_DARK]),
    )

    # Inner light metal ring
    inner = circle_points(c[:2], radius * 0.68, n=72, z=0.115)
    log(
        rec,
        path + "/inner_ring",
        rr_line_strips3d([inner], radii=[0.018], colors=[LINK_GRAY]),
    )

    # Central axle
    log(
        rec,
        path + "/axle",
        rr_points3d([c], radii=[radius * 0.38], colors=[JOINT_DARK]),
    )

    # Small bolt markers around the joint
    bolt_radius = radius * 0.78
    bolt_points = []
    for a in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False):
        bolt_points.append([
            c[0] + bolt_radius * np.cos(a),
            c[1] + bolt_radius * np.sin(a),
            0.13,
        ])

    log(
        rec,
        path + "/bolts",
        rr_points3d(
            bolt_points,
            radii=[radius * 0.075] * len(bolt_points),
            colors=[JOINT_DARK] * len(bolt_points),
        ),
    )
def log_basket_realistic(rec, path, center_xy, radius=0.20):
    """Log a catcher basket as blue outer and inner rings."""
    outer = circle_points(center_xy, radius, n=64, z=0.09)
    inner = circle_points(center_xy, radius * 0.72, n=64, z=0.09)

    log(rec, path + "/outer", rr_line_strips3d([outer], radii=[0.022], colors=[BASKET_BLUE]))
    log(rec, path + "/inner", rr_line_strips3d([inner], radii=[0.010], colors=[BASKET_BLUE]))
    log(
        rec,
        path + "/center",
        rr_points3d([[center_xy[0], center_xy[1], 0.09]], radii=[0.030], colors=[BASKET_BLUE]),
    )

def log_current_active_ball(rec, data, target_idx, k=None):
    """Log the currently active target ball.

    Moving intervals use the exported desired basket position.
    Static intervals use p_des.
    """
    if k is None:
        bx = float(data["p_des"][0, target_idx])
        by = float(data["p_des"][1, target_idx])
    else:
        t = float(data["tvec"][k])
        active_intervals = data["activeIntervals"]

        # Default: static target.
        bx = float(data["p_des"][0, target_idx])
        by = float(data["p_des"][1, target_idx])

        # Interval 2: MATLAB moves either b1d or b2d depending on closest basket.
        # Python index 1 = MATLAB interval 2.
        if active_intervals[1, 0] <= t <= active_intervals[1, 1]:
            p_nom = data["p_des"][:, target_idx]

            b1 = data["b1d"][:, k]
            b2 = data["b2d"][:, k]

            d1 = np.sum((b1 - p_nom) ** 2)
            d2 = np.sum((b2 - p_nom) ** 2)

            if d1 <= d2:
                bx = float(b1[0])
                by = float(b1[1])
            else:
                bx = float(b2[0])
                by = float(b2[1])

        # Interval 3: MATLAB explicitly moves b1d.
        # Python index 2 = MATLAB interval 3.
        elif active_intervals[2, 0] <= t <= active_intervals[2, 1]:
            bx = float(data["b1d"][0, k])
            by = float(data["b1d"][1, k])

    target = np.array([[bx, by, 0.25]])

    log(
        rec,
        "world/active_target/ball",
        rr_points3d(target, radii=[0.22], colors=[ACTIVE_BALL]),
    )

    active_ring = circle_points([bx, by], 0.30, z=0.25)
    log(
        rec,
        "world/active_target/ring",
        rr_line_strips3d(
            [active_ring],
            radii=[0.010],
            colors=[ACTIVE_RING],
        ),
    )

    cross = [
        [[bx - 0.36, by, 0.25], [bx + 0.36, by, 0.25]],
        [[bx, by - 0.36, 0.25], [bx, by + 0.36, 0.25]],
    ]

    log(
        rec,
        "world/active_target/crosshair",
        rr_line_strips3d(
            cross,
            radii=[0.007],
            colors=[ACTIVE_CROSSHAIR],
        ),
    )
    
def log_moving_obstacle_trace(rec, data, k):
    """Trace only the moving target during interval 3.

    This avoids drawing an artificial line from the moving target
    to the next static target after the switching instant.
    """
    tvec = data["tvec"]
    t = float(tvec[k])
    active_intervals = data["activeIntervals"]

    # Python index 2 = MATLAB interval 3.
    t0 = float(active_intervals[2, 0])
    t1 = float(active_intervals[2, 1])

    # Before the moving interval, show nothing.
    if t < t0:
        return

    # After the moving interval, keep only the completed interval-3 path.
    t_end = min(t, t1)

    mask = (tvec >= t0) & (tvec <= t_end)

    if np.count_nonzero(mask) < 2:
        return

    xs = data["b1d"][0, mask]
    ys = data["b1d"][1, mask]
    zs = 0.23 * np.ones_like(xs)

    trace = np.column_stack([xs, ys, zs])

    log(
        rec,
        "world/moving_target/interval_3_trace",
        rr_line_strips3d(
            [trace],
            radii=[0.012],
            colors=[MOVING_OBSTACLE_TRACE],
        ),
    )

    start = trace[0:1]
    log(
        rec,
        "world/moving_target/interval_3_start",
        rr_points3d(
            start,
            radii=[0.08],
            colors=[MOVING_OBSTACLE_START],
        ),
    )

    # Show current marker only while the target is actually moving.
    if t <= t1:
        current = trace[-1:]
        log(
            rec,
            "world/moving_target/interval_3_current",
            rr_points3d(
                current,
                radii=[0.10],
                colors=[MOVING_OBSTACLE_END],
            ),
        )
        """Trace only the moving target during interval 3.

    This avoids drawing an artificial line from the moving target
    to the next static target after the switching instant.
    """
    tvec = data["tvec"]
    t = float(tvec[k])
    active_intervals = data["activeIntervals"]

    # Python index 2 = MATLAB interval 3.
    t0 = float(active_intervals[2, 0])
    t1 = float(active_intervals[2, 1])

    # Do not draw anything before the moving interval.
    if t < t0:
        return

    # Important:
    # after the moving interval, keep the completed trace fixed only up to t1.
    t_end = min(t, t1)

    mask = (tvec >= t0) & (tvec <= t_end)

    if np.count_nonzero(mask) < 2:
        return

    xs = data["b1d"][0, mask]
    ys = data["b1d"][1, mask]
    zs = 0.23 * np.ones_like(xs)

    trace = np.column_stack([xs, ys, zs])

    log(
        rec,
        "world/moving_target/interval_3_trace",
        rr_line_strips3d(
            [trace],
            radii=[0.012],
            colors=[MOVING_OBSTACLE_TRACE],
        ),
    )

    # Start marker of the moving segment.
    start = trace[0:1]
    log(
        rec,
        "world/moving_target/interval_3_start",
        rr_points3d(
            start,
            radii=[0.08],
            colors=[MOVING_OBSTACLE_START],
        ),
    )

    # Current marker only while the obstacle is actually moving.
    if t <= t1:
        current = trace[-1:]
        log(
            rec,
            "world/moving_target/interval_3_current",
            rr_points3d(
                current,
                radii=[0.10],
                colors=[MOVING_OBSTACLE_END],
            ),
        )
        """Trace the path of the moving obstacle/target during interval 3.

    In MATLAB interval 3:
        b1d = p_des(:,3) + v_item * (t - switching_instants(3))

    In Python:
        activeIntervals[2] is interval 3.
    """
    t = float(data["tvec"][k])
    active_intervals = data["activeIntervals"]

    t0 = float(active_intervals[2, 0])
    t1 = float(active_intervals[2, 1])

    # Only show the moving trace during and after the moving interval.
    if t < t0:
        return

    # Collect all samples from the start of the moving interval up to now,
    # but do not go beyond the end of the moving interval.
    tvec = data["tvec"]
    mask = (tvec >= t0) & (tvec <= min(t, t1))

    if np.count_nonzero(mask) < 2:
        return

    xs = data["b1d"][0, mask]
    ys = data["b1d"][1, mask]
    zs = 0.23 * np.ones_like(xs)

    trace = np.column_stack([xs, ys, zs])

    log(
        rec,
        "world/moving_target/trace",
        rr_line_strips3d(
            [trace],
            radii=[0.012],
            colors=[MOVING_OBSTACLE_TRACE],
        ),
    )

    # Start marker of the moving obstacle path.
    start = trace[0:1]
    log(
        rec,
        "world/moving_target/start",
        rr_points3d(
            start,
            radii=[0.08],
            colors=[MOVING_OBSTACLE_START],
        ),
    )

    # Current point on the moving obstacle path.
    current = trace[-1:]
    log(
        rec,
        "world/moving_target/current",
        rr_points3d(
            current,
            radii=[0.10],
            colors=[MOVING_OBSTACLE_END],
        ),
    )


def log_robot_frame(rec, data, k, basket_radius=0.20):
    """Log a serious-looking planar 3R robot frame."""
    tvec = data["tvec"]
    t = float(tvec[k])
    dt = float(tvec[1] - tvec[0]) if len(tvec) > 1 else 0.01

    L = np.array([data["l1"], data["l1"], data["l1"]], dtype=float)
    qk = data["q"][:, k]
    pts = forward_kinematics(qk, L)
    # Base pedestal
    base_outer = circle_points(pts[0, :2], 0.34, n=96, z=0.02)
    base_inner = circle_points(pts[0, :2], 0.22, n=96, z=0.04)

    log(
        rec,
        "world/robot/base/outer",
        rr_line_strips3d([base_outer], radii=[0.030], colors=[JOINT_DARK]),
    )

    log(
        rec,
        "world/robot/base/inner",
        rr_line_strips3d([base_inner], radii=[0.020], colors=[LINK_GRAY]),
    )
    # Clean gray link bodies.
    link_widths = [0.22, 0.22, 0.22]
    for i in range(3):
        log_link_body(
            rec,
            f"world/robot/link_{i+1}",
            pts[i],
            pts[i + 1],
            width=link_widths[i],
        )

    # Dark joint hubs.
    joint_radii = [0.16, 0.14, 0.13, 0.11]
    for i, p in enumerate(pts):
        log_joint_hub(rec, f"world/robot/joint_{i}", p, radius=joint_radii[i])

    # Two baskets/grippers:
    # basket 1 at the midpoint of link 2, basket 2 at the end effector.
    link2_mid = 0.5 * (pts[1, :2] + pts[2, :2])
    end_eff = pts[3, :2]

    log_basket_realistic(rec, "world/robot/basket_1", link2_mid, radius=basket_radius)
    log_basket_realistic(rec, "world/robot/basket_2", end_eff, radius=basket_radius)

    # Mounting brackets from the robot to the basket centers.
    log(
        rec,
        "world/robot/basket_mounts",
        rr_line_strips3d(
            [
                [[link2_mid[0], link2_mid[1], 0.07], [pts[2, 0], pts[2, 1], 0.07]],
                [[end_eff[0], end_eff[1], 0.07], [pts[3, 0], pts[3, 1], 0.07]],
            ],
            radii=[0.014, 0.014],
            colors=[BASKET_BLUE, BASKET_BLUE],
        ),
    )

    # Active ball for the current time window.
# Active ball for the current time window.
    target_idx = visible_target_index(t, data["switching_instants"], dt)
    if target_idx is not None and target_idx < data["p_des"].shape[1]:
     log_current_active_ball(rec, data, target_idx, k)

# Trace the moving obstacle/target path during interval 3.
    #log_moving_obstacle_trace(rec, data, k)
    # Trajectory traces for the two catching points.
    if k > 2:
        trace1 = np.column_stack([
            data["pb1"][0, : k + 1],
            data["pb1"][1, : k + 1],
            0.03 * np.ones(k + 1),
        ])
        trace2 = np.column_stack([
            data["pb2"][0, : k + 1],
            data["pb2"][1, : k + 1],
            0.03 * np.ones(k + 1),
        ])
        log(
            rec,
            "world/traces/basket_1_path",
            rr_line_strips3d([trace1], radii=[0.006], colors=[TRACE_1]),
        )
        log(
            rec,
            "world/traces/basket_2_path",
            rr_line_strips3d([trace2], radii=[0.006], colors=[TRACE_2]),
        )

    # Desired active-link overlays.
    curr = active_interval_at(t, data["activeIntervals"])
    if curr is not None:
        active_vars = data["activeVars"][curr]
        q_des = qk.copy()

        for j in range(3):
            if f"q{j+1}" in active_vars:
                q_des[j] = data["qd"][j, k]

        pts_des = forward_kinematics(q_des, L)

        desired_segments = []
        for j in range(3):
            if f"q{j+1}" in active_vars:
                desired_segments.append([
                    [pts[j, 0], pts[j, 1], 0.12],
                    [pts_des[j + 1, 0], pts_des[j + 1, 1], 0.12],
                ])

        if desired_segments:
            log(
                rec,
                "world/robot/desired_active_links",
                rr_line_strips3d(desired_segments, radii=[0.020], colors=[DESIRED_RED]),
            )

        if hasattr(rr, "TextLog"):
            log(rec, "status/active_variables", rr.TextLog(", ".join(active_vars)))

def visualize(path: Path, frame_step=1, save_rrd: str | None = None, spawn: bool = True):
    data = ensure_shape(load_data(path))
    rec = make_recording("planar-3r-catching-serious", save_rrd=save_rrd, spawn=spawn)

    log_static_context(rec, data)

    print("Rerun planar 3R serious visualizer")
    print(f"  samples: {len(data['tvec'])}")
    print(f"  duration: {data['tvec'][0]:.2f} to {data['tvec'][-1]:.2f} s")
    print(f"  link length: {data['l1']}")
    if save_rrd:
        print(f"  recording: {save_rrd}")

    for k in range(0, len(data["tvec"]), frame_step):
        t = float(data["tvec"][k])
        set_frame_time(rec, k, t)
        log_robot_frame(rec, data, k)
        log_time_series(rec, data, k)

    if hasattr(rec, "flush"):
        rec.flush()

    print("Finished logging all frames to Rerun.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=Path)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--save-rrd", type=str, default=None)
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not open the Rerun viewer. Useful when only saving an .rrd recording.",
    )
    args = parser.parse_args()

    visualize(
        args.data_file,
        frame_step=args.frame_step,
        save_rrd=args.save_rrd,
        spawn=not args.no_spawn,
    )

if __name__ == "__main__":
    main()
