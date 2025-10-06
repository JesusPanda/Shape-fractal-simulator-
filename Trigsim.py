# Recursive right-triangle viewer — CUDA GPU + Taichi kernels
# Changes vs Vulkan version:
#   • Force CUDA backend: ti.init(arch=ti.cuda)
#   • Capacity math fixed for max depth so buffers never overflow.
#   • Same smooth, cursor-anchored zoom and GUI.
# Note: Requires an NVIDIA GPU with a working CUDA stack. No CPU fallback is configured.

import math
import time
import numpy as np
import taichi as ti

# -------- Taichi init: force CUDA GPU --------
ti.init(arch=ti.cuda, debug=False)

# ---------- constants ----------
RES = (1200, 1200)
MAX_ITERS = 22
MAX_DEPTH = MAX_ITERS
POW2 = 1 << MAX_DEPTH
# segments: 3 base (legs+altitude) + 2 connectors per processed node; nodes processed = 2^d - 1
SEG_CAP  = 3 + 2 * (POW2 - 1)         # = 8193 for d=12
# hypotenuses: 1 base + 3 per processed node
HYPO_CAP = 1 + 3 * (POW2 - 1)         # = 12286 for d=12
# nodes needed at most at last level
NODES_CAP = POW2                       # = 4096 for d=12

# ---------- fields (GPU) ----------
# segment arrays (non-hypotenuse)
seg_begin = ti.Vector.field(2, ti.f32, shape=SEG_CAP)
seg_end   = ti.Vector.field(2, ti.f32, shape=SEG_CAP)
# hypotenuse arrays
hyp_begin = ti.Vector.field(2, ti.f32, shape=HYPO_CAP)
hyp_end   = ti.Vector.field(2, ti.f32, shape=HYPO_CAP)
# counts
seg_count = ti.field(ti.i32, shape=())
hyp_count = ti.field(ti.i32, shape=())

# node arrays for BFS-style expansion
node_R  = ti.Vector.field(2, ti.f32, shape=NODES_CAP)
node_H0 = ti.Vector.field(2, ti.f32, shape=NODES_CAP)
node_H1 = ti.Vector.field(2, ti.f32, shape=NODES_CAP)
next_R  = ti.Vector.field(2, ti.f32, shape=NODES_CAP)
next_H0 = ti.Vector.field(2, ti.f32, shape=NODES_CAP)
next_H1 = ti.Vector.field(2, ti.f32, shape=NODES_CAP)
node_count = ti.field(ti.i32, shape=())

# store A,B,C for HUD markers
A_pt = ti.Vector.field(2, ti.f32, shape=())
B_pt = ti.Vector.field(2, ti.f32, shape=())
C_pt = ti.Vector.field(2, ti.f32, shape=())

# ---------- math helpers on GPU ----------
@ti.func
def foot_of_perp(P, A, B):
    AB = B - A
    denom = AB.dot(AB)
    t = 0.0
    if denom != 0:
        t = (P - A).dot(AB) / denom
    return A + t * AB

# ---------- base build on GPU ----------
@ti.kernel
def reset_and_build_base(angle_deg: ti.f32):
    seg_count[None] = 0
    hyp_count[None] = 0

    # clamp angle
    a = angle_deg
    if a < 5.0:  a = 5.0
    if a > 85.0: a = 85.0
    ang = a * ti.math.pi / 180.0

    # construct base triangle with right angle at B
    B = ti.Vector([0.15, 0.15])
    AB_len = 0.70
    BC_len = AB_len * ti.tan(ang)

    if BC_len > 0.70:
        s = 0.70 / BC_len
        AB_len *= s
        BC_len *= s

    A = ti.Vector([B[0] + AB_len, B[1]])
    C = ti.Vector([B[0], B[1] + BC_len])

    # persist points
    A_pt[None] = A
    B_pt[None] = B
    C_pt[None] = C

    # legs as segments
    i = ti.atomic_add(seg_count[None], 1)
    if i < SEG_CAP:
        seg_begin[i] = A
        seg_end[i]   = B
    i = ti.atomic_add(seg_count[None], 1)
    if i < SEG_CAP:
        seg_begin[i] = B
        seg_end[i]   = C

    # altitude from B to AC
    K = foot_of_perp(B, A, C)
    i = ti.atomic_add(seg_count[None], 1)
    if i < SEG_CAP:
        seg_begin[i] = B
        seg_end[i]   = K

    # base hypotenuse in hyp-array so it renders on top
    j = ti.atomic_add(hyp_count[None], 1)
    if j < HYPO_CAP:
        hyp_begin[j] = A
        hyp_end[j]   = C

    # init node list with one node: R=B, hyp=(A,C)
    node_R[0]  = B
    node_H0[0] = A
    node_H1[0] = C
    node_count[None] = 1

# ---------- one-level expansion on GPU ----------
@ti.kernel
def expand_once(old_count: ti.i32):
    for i in range(old_count):
        R  = node_R[i]
        H0 = node_H0[i]
        H1 = node_H1[i]

        F = foot_of_perp(R, H0, H1)
        v = R - F
        H0p = H0 + v
        H1p = H1 + v

        # connectors
        sidx = ti.atomic_add(seg_count[None], 1)
        if sidx < SEG_CAP:
            seg_begin[sidx] = H0
            seg_end[sidx]   = H0p
        sidx = ti.atomic_add(seg_count[None], 1)
        if sidx < SEG_CAP:
            seg_begin[sidx] = H1
            seg_end[sidx]   = H1p

        # hypotenuses: translated and the two child ones
        hidx = ti.atomic_add(hyp_count[None], 1)
        if hidx < HYPO_CAP:
            hyp_begin[hidx] = H0p
            hyp_end[hidx]   = H1p
        hidx = ti.atomic_add(hyp_count[None], 1)
        if hidx < HYPO_CAP:
            hyp_begin[hidx] = R
            hyp_end[hidx]   = H0
        hidx = ti.atomic_add(hyp_count[None], 1)
        if hidx < HYPO_CAP:
            hyp_begin[hidx] = R
            hyp_end[hidx]   = H1

        # write next-level nodes deterministically
        j = 2 * i
        next_R[j]  = H0p
        next_H0[j] = R
        next_H1[j] = H0
        next_R[j+1]  = H1p
        next_H0[j+1] = R
        next_H1[j+1] = H1

# copy next->current and set node_count to new count
@ti.kernel
def commit_next(old_count: ti.i32):
    new_count = old_count * 2
    for i in range(new_count):
        node_R[i]  = next_R[i]
        node_H0[i] = next_H0[i]
        node_H1[i] = next_H1[i]
    node_count[None] = new_count

# ---------- CPU-side UI and draw ----------
cam_center = [0.5, 0.5]
zoom = 1.0
zoom_target = 1.0
wheel_accum = 0.0
zoom_anchor = None

ZOOM_STEP = 1.02
ZOOM_MIN, ZOOM_MAX = 0.2, 50.0
SMOOTH_ZOOM_RATE = 12.0


def clamp(x, a, b):
    return a if x < a else b if x > b else x


def to_screen_arr(pts, center, z):
    x = (pts[:, 0] - center[0]) * z + 0.5
    y = (pts[:, 1] - center[1]) * z + 0.5
    return np.stack([x, y], axis=1)

# GUI
gui = ti.GUI("Recursive Right-Triangle (CUDA + Kernels)", res=RES, background_color=0x0)
angle_slider = gui.slider("angle_deg", 5, 85)
angle_slider.value = 30

iter_buffer = "2"
rmb_dragging = False
last_mouse = (0.0, 0.0)
TARGET_FPS = 15
BS_REPEAT_INTERVAL = 0.08
bs_repeat_cooldown = 0.0

# cache keys
last_angle_key = None
last_depth = None

while gui.running:
    frame_start = time.time()

    # events
    for e in gui.get_events():
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        elif e.key == 'r' and e.type == ti.GUI.PRESS:
            cam_center = [0.5, 0.5]
            zoom = 1.0
            zoom_target = 1.0
            wheel_accum = 0.0
            zoom_anchor = None
        elif e.key == ti.GUI.WHEEL:
            dz = e.delta[1] if hasattr(e, "delta") else 0.0
            wheel_accum += dz
            steps = int(wheel_accum)
            if steps != 0:
                zoom_target *= (ZOOM_STEP ** steps)
                zoom_target = clamp(zoom_target, ZOOM_MIN, ZOOM_MAX)
                wheel_accum -= steps
                zoom_anchor = gui.get_cursor_pos()
        else:
            if isinstance(e.key, str) and e.type == ti.GUI.PRESS:
                if e.key.isdigit():
                    if not (e.key == '0' and len(iter_buffer) == 0):
                        iter_buffer += e.key

    # BACKSPACE repeat
    if gui.is_pressed(ti.GUI.BACKSPACE):
        if bs_repeat_cooldown <= 0.0 and len(iter_buffer) > 0:
            iter_buffer = iter_buffer[:-1]
            bs_repeat_cooldown = BS_REPEAT_INTERVAL
    else:
        bs_repeat_cooldown = 0.0

    # RMB pan
    if gui.is_pressed(ti.GUI.RMB):
        cur = gui.get_cursor_pos()
        if rmb_dragging:
            dx = cur[0] - last_mouse[0]
            dy = cur[1] - last_mouse[1]
            cam_center[0] -= dx / zoom
            cam_center[1] -= dy / zoom
        last_mouse = cur
        rmb_dragging = True
    else:
        rmb_dragging = False

    # parameters
    angle_deg = float(angle_slider.value)
    try:
        depth = int(iter_buffer) if len(iter_buffer) > 0 else 0
    except ValueError:
        depth = 0
    depth = max(0, min(MAX_ITERS, depth))

    angle_key = round(angle_deg, 2)

    # smooth zoom step
    prev_zoom = zoom
    alpha = 1.0 - math.exp(-SMOOTH_ZOOM_RATE * (1.0 / 15.0))
    zoom += (zoom_target - zoom) * alpha
    if zoom_anchor is not None and abs(zoom - prev_zoom) > 1e-9:
        sx, sy = zoom_anchor
        wx = (sx - 0.5) / prev_zoom + cam_center[0]
        wy = (sy - 0.5) / prev_zoom + cam_center[1]
        cam_center[0] = wx - (sx - 0.5) / zoom
        cam_center[1] = wy - (sy - 0.5) / zoom
        if abs(zoom_target - zoom) < 1e-6:
            zoom_anchor = None

    # rebuild on change using GPU kernels
    if angle_key != last_angle_key or depth != last_depth:
        reset_and_build_base(angle_deg)
        count = 1
        for _ in range(depth):
            expand_once(count)
            commit_next(count)
            count *= 2
        last_angle_key = angle_key
        last_depth = depth

    # copy arrays to CPU for drawing
    sc = min(int(seg_count.to_numpy().item()), SEG_CAP)
    hc = min(int(hyp_count.to_numpy().item()), HYPO_CAP)

    if sc > 0:
        sb = seg_begin.to_numpy()[:sc]
        se = seg_end.to_numpy()[:sc]
        sb = to_screen_arr(sb, cam_center, zoom)
        se = to_screen_arr(se, cam_center, zoom)
        if hasattr(gui, "lines"):
            gui.lines(begin=sb, end=se, color=0xDDDDDD, radius=1)
        else:
            for i in range(sb.shape[0]):
                gui.line(begin=tuple(sb[i]), end=tuple(se[i]), color=0xDDDDDD, radius=1)

    if hc > 0:
        hb = hyp_begin.to_numpy()[:hc]
        he = hyp_end.to_numpy()[:hc]
        hb = to_screen_arr(hb, cam_center, zoom)
        he = to_screen_arr(he, cam_center, zoom)
        if hasattr(gui, "lines"):
            gui.lines(begin=hb, end=he, color=0xB0C4DE, radius=1)
        else:
            for i in range(hb.shape[0]):
                gui.line(begin=tuple(hb[i]), end=tuple(he[i]), color=0xB0C4DE, radius=1)

    # draw A,B,C markers
    A_np = A_pt.to_numpy()
    B_np = B_pt.to_numpy()
    C_np = C_pt.to_numpy()
    if A_np.size == 2:
        pts = np.stack([A_np, B_np, C_np], axis=0)
        sp = to_screen_arr(pts, cam_center, zoom)
        for (x, y), col in zip(sp, [0xFF5555, 0x55FF55, 0x5555FF]):
            gui.circle(pos=(float(x), float(y)), radius=4, color=col)

    # HUD
    gui.text(f"angle A = {angle_deg:.1f}°", pos=(0.02, 0.97), color=0xCCCCCC)
    gui.text(f"iterations = {depth}  [digits/BACKSPACE]  cap={MAX_ITERS}", pos=(0.02, 0.94), color=0xCCCCCC)
    gui.text(f"buffer: '{iter_buffer}'", pos=(0.02, 0.91), color=0x777777)
    gui.text("CUDA GPU kernels | RMB pan | Wheel zoom | R reset | 15 FPS", pos=(0.02, 0.88), color=0x888888)

    gui.show()

    # frame pacing
    elapsed = time.time() - frame_start
    bs_repeat_cooldown = max(0.0, bs_repeat_cooldown - elapsed)
    remain = (1.0 / 15.0) - elapsed
    if remain > 0:
        time.sleep(remain)

