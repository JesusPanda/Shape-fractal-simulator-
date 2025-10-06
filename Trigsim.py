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
SEG_CAP  = 3 + 2 * (POW2 - 1)
# hypotenuses: 1 base + 3 per processed node
HYPO_CAP = 1 + 3 * (POW2 - 1)
# nodes needed at most at last level
NODES_CAP = POW2

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

# node arrays for BFS-style expansion (ping-pong buffers)
node_R  = ti.Vector.field(2, ti.f32, shape=(2, NODES_CAP))
node_H0 = ti.Vector.field(2, ti.f32, shape=(2, NODES_CAP))
node_H1 = ti.Vector.field(2, ti.f32, shape=(2, NODES_CAP))
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
    seg_begin[i], seg_end[i] = A, B
    i = ti.atomic_add(seg_count[None], 1)
    seg_begin[i], seg_end[i] = B, C

    # altitude from B to AC
    K = foot_of_perp(B, A, C)
    i = ti.atomic_add(seg_count[None], 1)
    seg_begin[i], seg_end[i] = B, K

    # base hypotenuse in hyp-array so it renders on top
    j = ti.atomic_add(hyp_count[None], 1)
    hyp_begin[j], hyp_end[j] = A, C

    # init node list with one node: R=B, hyp=(A,C)
    node_R[0, 0]  = B
    node_H0[0, 0] = A
    node_H1[0, 0] = C
    node_count[None] = 1

# ---------- one-level expansion on GPU ----------
@ti.kernel
def expand_once(current_buf: ti.i32, next_buf: ti.i32, count: ti.i32):
    for i in range(count):
        R  = node_R[current_buf, i]
        H0 = node_H0[current_buf, i]
        H1 = node_H1[current_buf, i]

        F = foot_of_perp(R, H0, H1)
        v = R - F
        H0p = H0 + v
        H1p = H1 + v

        # connectors
        sidx = ti.atomic_add(seg_count[None], 2)
        seg_begin[sidx], seg_end[sidx] = H0, H0p
        seg_begin[sidx+1], seg_end[sidx+1] = H1, H1p

        # hypotenuses: translated and the two child ones
        hidx = ti.atomic_add(hyp_count[None], 3)
        hyp_begin[hidx],   hyp_end[hidx]   = H0p, H1p
        hyp_begin[hidx+1], hyp_end[hidx+1] = R, H0
        hyp_begin[hidx+2], hyp_end[hidx+2] = R, H1

        # write next-level nodes deterministically
        j = 2 * i
        node_R[next_buf, j], node_H0[next_buf, j], node_H1[next_buf, j] = H0p, R, H0
        node_R[next_buf, j+1], node_H0[next_buf, j+1], node_H1[next_buf, j+1] = H1p, R, H1

# ---------- CPU-side UI and draw ----------
cam_center = ti.Vector([0.5, 0.5])
zoom = 1.0
zoom_target = 1.0
wheel_accum = 0.0
zoom_anchor = None

ZOOM_STEP = 1.02
ZOOM_MIN, ZOOM_MAX = 0.2, 50.0
SMOOTH_ZOOM_RATE = 12.0

@ti.kernel
def transform_to_screen(
    pts_in: ti.template(),
    pts_out: ti.template(),
    center: ti.types.vector(2, ti.f32),
    z: ti.f32,
    n: ti.i32
):
    for i in range(n):
        p = (pts_in[i] - center) * z + 0.5
        pts_out[i] = p

# GUI
gui = ti.GUI("Recursive Right-Triangle (CUDA + Kernels)", res=RES, background_color=0x0)
angle_slider = gui.slider("angle_deg", 5, 85)
angle_slider.value = 30

iter_buffer = "2"
rmb_dragging = False
last_mouse = (0.0, 0.0)
bs_repeat_cooldown = 0.0
BS_REPEAT_INTERVAL = 0.08

# cache keys
last_angle_key = None
last_depth = None

# buffers for drawing
seg_begin_screen = ti.Vector.field(2, ti.f32, shape=SEG_CAP)
seg_end_screen = ti.Vector.field(2, ti.f32, shape=SEG_CAP)
hyp_begin_screen = ti.Vector.field(2, ti.f32, shape=HYPO_CAP)
hyp_end_screen = ti.Vector.field(2, ti.f32, shape=HYPO_CAP)
abc_pts_screen = ti.Vector.field(2, ti.f32, shape=3)

while gui.running:
    frame_start = time.time()

    # events
    for e in gui.get_events(ti.GUI.PRESS, ti.GUI.WHEEL):
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        elif e.key == 'r':
            cam_center, zoom, zoom_target, wheel_accum, zoom_anchor = ti.Vector([0.5, 0.5]), 1.0, 1.0, 0.0, None
        elif e.key == ti.GUI.WHEEL:
            wheel_accum += e.delta[1]
            steps = int(wheel_accum)
            if steps != 0:
                zoom_target *= (ZOOM_STEP ** steps)
                zoom_target = min(max(zoom_target, ZOOM_MIN), ZOOM_MAX)
                wheel_accum -= steps
                zoom_anchor = ti.Vector(gui.get_cursor_pos())
        elif isinstance(e.key, str) and e.key.isdigit():
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
        cur = ti.Vector(gui.get_cursor_pos())
        if rmb_dragging:
            cam_center -= (cur - last_mouse) / zoom
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
    depth = min(MAX_ITERS, max(0, depth))
    angle_key = round(angle_deg, 2)

    # smooth zoom step
    prev_zoom = zoom
    alpha = 1.0 - math.exp(-SMOOTH_ZOOM_RATE * (1.0 / 60.0)) # assume 60fps
    zoom += (zoom_target - zoom) * alpha
    if zoom_anchor is not None and abs(zoom - prev_zoom) > 1e-9:
        wx, wy = (zoom_anchor - 0.5) / prev_zoom + cam_center
        cam_center = ti.Vector([wx - (zoom_anchor.x - 0.5) / zoom, wy - (zoom_anchor.y - 0.5) / zoom])
        if abs(zoom_target - zoom) < 1e-6:
            zoom_anchor = None

    # rebuild on change using GPU kernels
    if angle_key != last_angle_key or depth != last_depth:
        reset_and_build_base(angle_deg)
        count = 1
        current_buf, next_buf = 0, 1
        for _ in range(depth):
            expand_once(current_buf, next_buf, count)
            count *= 2
            node_count[None] = count
            current_buf, next_buf = next_buf, current_buf # swap buffers
        last_angle_key, last_depth = angle_key, depth

    # transform geometry for drawing
    sc = seg_count[None]
    hc = hyp_count[None]
    if sc > 0:
        transform_to_screen(seg_begin, seg_begin_screen, cam_center, zoom, sc)
        transform_to_screen(seg_end, seg_end_screen, cam_center, zoom, sc)
    if hc > 0:
        transform_to_screen(hyp_begin, hyp_begin_screen, cam_center, zoom, hc)
        transform_to_screen(hyp_end, hyp_end_screen, cam_center, zoom, hc)

    # transform A,B,C markers
    abc_pts = ti.Vector.field(2, ti.f32, shape=3)
    abc_pts[0], abc_pts[1], abc_pts[2] = A_pt[None], B_pt[None], C_pt[None]
    transform_to_screen(abc_pts, abc_pts_screen, cam_center, zoom, 3)

    # draw calls
    if sc > 0:
        gui.lines(begin=seg_begin_screen, end=seg_end_screen, radius=1, color=0xDDDDDD)
    if hc > 0:
        gui.lines(begin=hyp_begin_screen, end=hyp_end_screen, radius=1, color=0xB0C4DE)

    # draw markers from screen-transformed buffer
    gui.circles(abc_pts_screen, radius=4, palette=[0xFF5555, 0x55FF55, 0x5555FF], palette_indices=[0,1,2])

    # HUD
    gui.text(f"angle A = {angle_deg:.1f}°", pos=(0.02, 0.97), color=0xCCCCCC)
    gui.text(f"iterations = {depth}  [digits/BACKSPACE]  cap={MAX_ITERS}", pos=(0.02, 0.94), color=0xCCCCCC)
    gui.text(f"buffer: '{iter_buffer}'", pos=(0.02, 0.91), color=0x777777)
    gui.text("CUDA GPU kernels | RMB pan | Wheel zoom | R reset", pos=(0.02, 0.88), color=0x888888)
    gui.text(f"FPS: {gui.fps:.1f}", pos=(0.02, 0.85), color=0x888888)

    gui.show()

    # update timers
    elapsed = time.time() - frame_start
    bs_repeat_cooldown = max(0.0, bs_repeat_cooldown - elapsed)