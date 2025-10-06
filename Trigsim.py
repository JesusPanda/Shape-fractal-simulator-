# Faster recursive right-triangle viewer: hypotenuse always visible
# Changes vs your snippet:
#   • Draw order swapped: draw generic segments first, then hypotenuses last, so they are not overdrawn by AB/BC.
#   • Everything else preserved: cache, batched draw, typed iterations, BACKSPACE repeat, 15 FPS cap.
# Controls:
#   RMB drag = pan | Mouse wheel = zoom | R = reset view
#   Angle slider sets angle at A (deg)
#   Type digits to set iteration depth directly (capped by MAX_ITERS)

import math
import time
import numpy as np
import taichi as ti

ti.init()

# ---------- vector utils ----------
def v_add(a, b):  return (a[0]+b[0], a[1]+b[1])
def v_sub(a, b):  return (a[0]-b[0], a[1]-b[1])
def v_mul(s, a):  return (s*a[0], s*a[1])
def v_dot(a, b):  return a[0]*b[0] + a[1]*b[1]
def v_len2(a):    return v_dot(a, a)

def foot_of_perp(P, A, B):
    AB = v_sub(B, A)
    denom = v_len2(AB)
    if denom == 0.0:
        return A
    t = v_dot(v_sub(P, A), AB) / denom
    return v_add(A, v_mul(t, AB))

# ---------- geometry ----------
def build_initial_triangle(angle_deg):
    B = (0.15, 0.15)
    angle = math.radians(max(5.0, min(85.0, angle_deg)))
    AB_len = 0.70
    BC_len = AB_len * math.tan(angle)
    if BC_len > 0.70:
        s = 0.70 / BC_len
        AB_len *= s
        BC_len *= s
    A = (B[0] + AB_len, B[1])
    C = (B[0],          B[1] + BC_len)
    return A, B, C

def draw_triangle_edges(A, B, C, segs_out):
    segs_out.append((A, B)); segs_out.append((B, C)); segs_out.append((A, C))
    K = foot_of_perp(B, A, C)
    segs_out.append((B, K))

def translate_hyp_and_children(R, H0, H1, depth, segs_out, hypos_out):
    F = foot_of_perp(R, H0, H1)
    v = v_sub(R, F)
    H0p = v_add(H0, v)
    H1p = v_add(H1, v)

    # connectors
    segs_out.append((H0, H0p))
    segs_out.append((H1, H1p))

    # hypotenuses of the two child triangles
    hypos_out.append((R, H0))
    hypos_out.append((R, H1))

    if depth > 1:
        translate_hyp_and_children(H0p, R, H0, depth - 1, segs_out, hypos_out)
        translate_hyp_and_children(H1p, R, H1, depth - 1, segs_out, hypos_out)

def pairs_to_arrays(pairs):
    if not pairs:
        z = np.zeros((0, 2), dtype=np.float32)
        return z, z
    a = np.array([p for p, _ in pairs], dtype=np.float32)
    b = np.array([q for _, q in pairs], dtype=np.float32)
    return a, b

# ---------- view (pan + zoom) ----------
cam_center = [0.5, 0.5]
zoom = 1.0

def to_screen_arr(pts, center, z):
    x = (pts[:, 0] - center[0]) * z + 0.5
    y = (pts[:, 1] - center[1]) * z + 0.5
    return np.stack([x, y], axis=1)

# ---------- GUI ----------
res = (900, 900)
gui = ti.GUI("Recursive Right-Triangle Translation (fast)", res=res, background_color=0x0)

angle_slider = gui.slider("angle_deg", 5, 85)
angle_slider.value = 30

MAX_ITERS = 12
iter_buffer = "2"  # empty => depth 0

# input state
rmb_dragging = False
last_mouse = (0.0, 0.0)

# backspace repeat
TARGET_FPS = 15
TARGET_DT = 1.0 / TARGET_FPS
bs_repeat_cooldown = 0.0
BS_REPEAT_INTERVAL = 0.08

# ---------- caching ----------
last_angle_key = None
last_depth = None
seg_starts = seg_ends = None
hypo_starts = hypo_ends = None
A_pt = B_pt = C_pt = None

while gui.running:
    frame_start = time.time()

    # ---- input ----
    for e in gui.get_events():
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        elif e.key == 'r' and e.type == ti.GUI.PRESS:
            cam_center = [0.5, 0.5]
            zoom = 1.0
        elif e.key == ti.GUI.WHEEL:
            dz = e.delta[1] if hasattr(e, "delta") else 0.0
            if dz != 0.0:
                zoom *= (1.1 ** dz)
                zoom = max(0.2, min(zoom, 50.0))
        else:
            if isinstance(e.key, str) and e.type == ti.GUI.PRESS:
                if e.key.isdigit():
                    if not (e.key == '0' and len(iter_buffer) == 0):
                        iter_buffer += e.key

    # BACKSPACE press/hold
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

    # ---- parameters ----
    angle_deg = float(angle_slider.value)
    try:
        depth = int(iter_buffer) if len(iter_buffer) > 0 else 0
    except ValueError:
        depth = 0
    depth = max(0, min(MAX_ITERS, depth))

    # quantize angle to avoid rebuilds on tiny jitter
    angle_key = round(angle_deg, 2)

    # ---- rebuild geometry only when angle/depth change ----
    if angle_key != last_angle_key or depth != last_depth:
        segs, hypos = [], []
        A, B, C = build_initial_triangle(angle_deg)
        draw_triangle_edges(A, B, C, segs)
        if depth >= 1:
            translate_hyp_and_children(B, A, C, depth, segs, hypos)

        seg_starts, seg_ends = pairs_to_arrays(segs)
        hypo_starts, hypo_ends = pairs_to_arrays(hypos)
        A_pt, B_pt, C_pt = np.array(A, dtype=np.float32), np.array(B, dtype=np.float32), np.array(C, dtype=np.float32)

        last_angle_key = angle_key
        last_depth = depth

    # ---- draw ----
    HYP_COLOR = 0xB0C4DE
    has_lines = hasattr(gui, "lines")

    # 1) draw generic segments first
    if seg_starts is not None and seg_starts.shape[0] > 0:
        sb = to_screen_arr(seg_starts, cam_center, zoom)
        se = to_screen_arr(seg_ends,   cam_center, zoom)
        if has_lines:
            gui.lines(begin=sb, end=se, color=0xDDDDDD, radius=1)
        else:
            for i in range(sb.shape[0]):
                gui.line(begin=tuple(sb[i]), end=tuple(se[i]), color=0xDDDDDD, radius=1)

    # 2) then draw hypotenuses on top so they remain visible at every step
    if hypo_starts is not None and hypo_starts.shape[0] > 0:
        hb = to_screen_arr(hypo_starts, cam_center, zoom)
        he = to_screen_arr(hypo_ends,   cam_center, zoom)
        if has_lines:
            gui.lines(begin=hb, end=he, color=HYP_COLOR, radius=1)
        else:
            for i in range(hb.shape[0]):
                gui.line(begin=tuple(hb[i]), end=tuple(he[i]), color=HYP_COLOR, radius=1)

    if A_pt is not None:
        pts = np.stack([A_pt, B_pt, C_pt], axis=0)
        sp = to_screen_arr(pts, cam_center, zoom)
        for (x, y), col in zip(sp, [0xFF5555, 0x55FF55, 0x5555FF]):
            gui.circle(pos=(float(x), float(y)), radius=4, color=col)

    # HUD
    gui.text(f"angle A = {angle_deg:.1f}°", pos=(0.02, 0.97), color=0xCCCCCC)
    gui.text(f"iterations = {depth}  [digits/BACKSPACE]  cap={MAX_ITERS}", pos=(0.02, 0.94), color=0xCCCCCC)
    gui.text(f"buffer: '{iter_buffer}'", pos=(0.02, 0.91), color=0x777777)
    gui.text("RMB pan | Wheel zoom | R reset | 15 FPS", pos=(0.02, 0.88), color=0x888888)

    gui.show()

    # ---- timers ----
    elapsed = time.time() - frame_start
    bs_repeat_cooldown = max(0.0, bs_repeat_cooldown - elapsed)
    remain = (1.0 / 15.0) - elapsed
    if remain > 0:
        time.sleep(remain)
