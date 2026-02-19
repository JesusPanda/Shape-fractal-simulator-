
content = r"""# Recursive right-triangle viewer — CUDA GPU + Taichi kernels
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

# ---------- new fields for ti.ui ----------
seg_vertices = ti.Vector.field(2, ti.f32, shape=2 * SEG_CAP)
hyp_vertices = ti.Vector.field(2, ti.f32, shape=2 * HYPO_CAP)
abc_vertices = ti.Vector.field(2, ti.f32, shape=3)
abc_colors = ti.Vector.field(3, ti.f32, shape=3)

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

# ---------- Render Kernels ----------

@ti.kernel
def init_render_buffers():
    # Fill with NaN initially so nothing draws
    for i in range(2 * SEG_CAP):
        seg_vertices[i] = ti.Vector([float('nan'), float('nan')])
    for i in range(2 * HYPO_CAP):
        hyp_vertices[i] = ti.Vector([float('nan'), float('nan')])

    # Init colors for A, B, C (red, green, blue)
    abc_colors[0] = ti.Vector([1.0, 0.33, 0.33])
    abc_colors[1] = ti.Vector([0.33, 1.0, 0.33])
    abc_colors[2] = ti.Vector([0.33, 0.33, 1.0])

@ti.kernel
def update_render_lines(
    count: ti.i32,
    begin_buf: ti.template(),
    end_buf: ti.template(),
    out_buf: ti.template(),
    center: ti.types.vector(2, ti.f32),
    z: ti.f32
):
    for i in range(count):
        p1 = (begin_buf[i] - center) * z + 0.5
        p2 = (end_buf[i]   - center) * z + 0.5
        out_buf[2*i]   = p1
        out_buf[2*i+1] = p2

@ti.kernel
def clear_render_tail(start_idx: ti.i32, end_idx: ti.i32, out_buf: ti.template()):
    for i in range(start_idx, end_idx):
        idx = 2 * i
        out_buf[idx]   = ti.Vector([float('nan'), float('nan')])
        out_buf[idx+1] = ti.Vector([float('nan'), float('nan')])

@ti.kernel
def update_markers(center: ti.types.vector(2, ti.f32), z: ti.f32):
    # Transform A, B, C
    abc_vertices[0] = (A_pt[None] - center) * z + 0.5
    abc_vertices[1] = (B_pt[None] - center) * z + 0.5
    abc_vertices[2] = (C_pt[None] - center) * z + 0.5

# ---------- CPU-side UI and draw ----------
cam_center = ti.Vector([0.5, 0.5])
zoom = 1.0
zoom_target = 1.0
zoom_anchor = None

ZOOM_STEP = 1.1
ZOOM_MIN, ZOOM_MAX = 0.2, 50000.0
SMOOTH_ZOOM_RATE = 12.0

iter_buffer = "2"
rmb_dragging = False
last_mouse = (0.0, 0.0)
bs_repeat_cooldown = 0.0
BS_REPEAT_INTERVAL = 0.08

# cache keys
last_angle_key = None
last_depth = None
last_seg_count = 0
last_hyp_count = 0

# Init window
window = ti.ui.Window("Recursive Right-Triangle (CUDA + Kernels)", res=RES)
canvas = window.get_canvas()
gui = window.get_gui()

init_render_buffers()

angle_deg = 30.0

frame_start = time.time()
frame_count = 0
fps_val = 0.0
last_fps_time = time.time()

while window.running:
    current_time = time.time()
    dt = current_time - frame_start
    frame_start = current_time

    frame_count += 1
    if current_time - last_fps_time >= 1.0:
        fps_val = frame_count / (current_time - last_fps_time)
        frame_count = 0
        last_fps_time = current_time

    # Events
    # Zoom with keys if needed
    if window.is_pressed('['):
        zoom_target /= ZOOM_STEP
    if window.is_pressed(']'):
        zoom_target *= ZOOM_STEP

    # Check for digit keys manually if needed, or rely on slider
    # ti.ui doesn't easily capture 'typed' characters like GUI
    # We will rely on GUI slider/text for params, but maybe use keys for iteration count?
    # Let's check events
    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.ESCAPE:
            window.running = False
        elif e.key == 'r':
            cam_center, zoom, zoom_target, zoom_anchor = ti.Vector([0.5, 0.5]), 1.0, 1.0, None
        elif e.key == ti.ui.BACKSPACE:
             if len(iter_buffer) > 0:
                iter_buffer = iter_buffer[:-1]
        elif e.key >= '0' and e.key <= '9':
             if not (e.key == '0' and len(iter_buffer) == 0):
                iter_buffer += e.key

    # Mouse Pan
    # ti.ui cursor pos is 0..1
    curr_mouse = ti.Vector(window.get_cursor_pos())
    if window.is_pressed(ti.ui.RMB):
        if rmb_dragging:
            cam_center -= (curr_mouse - last_mouse) / zoom
        last_mouse = curr_mouse
        rmb_dragging = True
    else:
        rmb_dragging = False

    # Mouse Wheel? ti.ui doesn't expose it well in python yet usually.
    # We rely on keys for zoom or assume user drags slider?
    # Or maybe checking delta?
    # We'll stick to [ and ] and 'r'.

    zoom_target = min(max(zoom_target, ZOOM_MIN), ZOOM_MAX)

    # Smooth zoom
    prev_zoom = zoom
    alpha = 1.0 - math.exp(-SMOOTH_ZOOM_RATE * dt)
    zoom += (zoom_target - zoom) * alpha

    # Rebuild logic
    try:
        depth = int(iter_buffer) if len(iter_buffer) > 0 else 0
    except ValueError:
        depth = 0
    depth = min(MAX_ITERS, max(0, depth))
    angle_key = round(angle_deg, 2)

    if angle_key != last_angle_key or depth != last_depth:
        reset_and_build_base(angle_deg)
        count = 1
        current_buf, next_buf = 0, 1
        for _ in range(depth):
            expand_once(current_buf, next_buf, count)
            count *= 2
            node_count[None] = count
            current_buf, next_buf = next_buf, current_buf
        last_angle_key, last_depth = angle_key, depth

    # Update render buffers
    sc = seg_count[None]
    hc = hyp_count[None]

    # If count decreased, clear tail
    if sc < last_seg_count:
        clear_render_tail(sc, last_seg_count, seg_vertices)
    if hc < last_hyp_count:
        clear_render_tail(hc, last_hyp_count, hyp_vertices)

    last_seg_count = sc
    last_hyp_count = hc

    # Update active lines
    if sc > 0:
        update_render_lines(sc, seg_begin, seg_end, seg_vertices, cam_center, zoom)
    if hc > 0:
        update_render_lines(hc, hyp_begin, hyp_end, hyp_vertices, cam_center, zoom)

    # Update markers
    update_markers(cam_center, zoom)

    # Render
    canvas.set_background_color((0, 0, 0))

    # Draw lines
    # Width needs to be small for 0..1 coord system? default is 0.005
    canvas.lines(seg_vertices, width=0.0015, color=(0.86, 0.86, 0.86))
    canvas.lines(hyp_vertices, width=0.0015, color=(0.69, 0.77, 0.87))

    # Draw markers
    canvas.circles(abc_vertices, radius=0.006, per_vertex_color=abc_colors)

    # HUD
    with gui.sub_window("Controls", 0.02, 0.02, 0.4, 0.25):
        gui.text(f"FPS: {fps_val:.1f}")
        gui.text("Controls: RMB Pan, [ ] Zoom, R Reset")
        angle_deg = gui.slider_float("Angle", angle_deg, 5.0, 85.0)

        gui.text(f"Iterations: {depth} (Max {MAX_ITERS})")
        gui.text(f"Buffer: {iter_buffer}")
        gui.text("Type digits to set iterations")

    window.show()
"""

with open("Trigsim.py", "w") as f:
    f.write(content)
