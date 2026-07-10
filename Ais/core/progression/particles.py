# Global particle pool, integrated once per frame and drawn on the ImGui
# foreground draw list. Kinds: "dot" (brush/box bursts) and "confetti".
from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import imgui


Color = Tuple[float, float, float]

MAX_PARTICLES = 2400

# Global "make it pop" knobs, applied by the brush/box dot emitters.
SIZE_MUL = 1.3
LIFETIME_MUL = 2.0
COUNT_MUL = 2

# Confetti is the celebration centrepiece: more numerous than dots.
CONFETTI_SIZE_MUL = 1.3
CONFETTI_LIFETIME_MUL = 2.0
CONFETTI_COUNT_MUL = 4


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    age: float
    lifetime: float
    color: Color
    size: float
    gravity: float = 0.0
    drag: float = 0.0
    spin: float = 0.0
    angle: float = 0.0
    kind: str = "dot"   # "dot" | "confetti"
    # world particles store position/velocity in world units (y in screen-down
    # convention) and are transformed to screen at draw time so they track the
    # scene under zoom/pan. Screen particles (confetti) live in pixels.
    world: bool = False
    fade_bias: float = 0.0   # per-confetti offset (fraction of screen) for the fade band


_particles: List[Particle] = []


def _clamp_pool() -> None:
    if len(_particles) > MAX_PARTICLES:
        del _particles[: len(_particles) - MAX_PARTICLES]


def _jitter_color(
    base: Color,
    h_amp: float = 0.05,
    s_amp: float = 0.10,
    v_amp: float = 0.10,
) -> Color:
    h_amp = min(0.15, max(0.0, h_amp))
    h, s, v = colorsys.rgb_to_hsv(*base)
    h = (h + random.uniform(-h_amp, h_amp)) % 1.0
    s = max(0.0, min(1.0, s + random.uniform(-s_amp, s_amp)))
    v = max(0.0, min(1.0, v + random.uniform(-v_amp, v_amp)))
    return colorsys.hsv_to_rgb(h, s, v)


def palette_color(palette: str, base: Color) -> Color:
    # Map a feature's base colour through a named cosmetic palette (see cosmetics).
    if palette == "gold":
        return (1.0, random.uniform(0.78, 0.86), random.uniform(0.24, 0.34))
    if palette == "mono":
        v = random.uniform(0.9, 1.0)
        return (v, v, min(1.0, v + 0.02))
    if palette == "warm":
        return colorsys.hsv_to_rgb(random.uniform(0.02, 0.11), random.uniform(0.75, 0.95), 1.0)
    if palette == "cool":
        return colorsys.hsv_to_rgb(random.uniform(0.50, 0.62), random.uniform(0.55, 0.80), 1.0)
    if palette == "prism":
        return colorsys.hsv_to_rgb(random.random(), random.uniform(0.70, 0.90), 1.0)
    return base   # "feature" / unknown


def emit_brush_ring(
    cx: float,
    cy: float,
    radius: float,
    color: Color,
    n: int = 1,
    h_amp: float = 0.05,
    world: bool = False,
    palette: str = "feature",
    size_mul: float = 1.0,
) -> None:
    if radius < 1.0:
        radius = 1.0
    if world:
        cy = -cy   # store in screen-down convention; draw() flips back
    for _ in range(n):
        ang = random.uniform(0.0, 2.0 * math.pi)
        r = radius + random.uniform(-1.5, 1.5)
        x = cx + math.cos(ang) * r
        y = cy + math.sin(ang) * r
        out_speed = random.uniform(8.0, 28.0)
        _particles.append(Particle(
            x=x,
            y=y,
            vx=math.cos(ang) * out_speed,
            vy=math.sin(ang) * out_speed - 6.0,
            age=0.0,
            lifetime=random.uniform(0.70, 1.50) * LIFETIME_MUL,
            color=_jitter_color(palette_color(palette, color), h_amp=h_amp),
            size=random.uniform(1.6, 2.8) * SIZE_MUL * size_mul,
            gravity=24.0,
            drag=2.4,
            kind="dot",
            world=world,
        ))
    _clamp_pool()


def emit_box_outline_burst(
    cx: float,
    cy: float,
    size: float,
    color: Color,
    n: int = 18,
    h_amp: float = 0.05,
    world: bool = False,
    palette: str = "feature",
    size_mul: float = 1.0,
) -> None:
    if size < 4.0:
        size = 4.0
    if world:
        cy = -cy   # store in screen-down convention; draw() flips back
    half = size * 0.5
    for _ in range(n * COUNT_MUL):
        side = random.randint(0, 3)
        t = random.uniform(-1.0, 1.0)
        if side == 0:      # top
            x, y = cx + t * half, cy - half
        elif side == 1:    # right
            x, y = cx + half, cy + t * half
        elif side == 2:    # bottom
            x, y = cx + t * half, cy + half
        else:              # left
            x, y = cx - half, cy + t * half
        # Fly outward from the box centre (so corners spread diagonally), with a
        # little jitter and an upward pop; gravity then arcs them back down.
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy) or 1.0
        speed = random.uniform(70.0, 180.0)
        vx = dx / d * speed + random.uniform(-22.0, 22.0)
        vy = dy / d * speed + random.uniform(-22.0, 22.0) - 45.0
        _particles.append(Particle(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            age=0.0,
            lifetime=random.uniform(1.10, 2.00) * LIFETIME_MUL,
            color=_jitter_color(palette_color(palette, color), h_amp=h_amp),
            size=random.uniform(2.0, 3.4) * SIZE_MUL * size_mul,
            gravity=110.0,
            drag=1.6,
            kind="dot",
            world=world,
        ))
    _clamp_pool()


def emit_confetti(screen_w: int, color: Color, n: int = 55, palette: str = "feature",
                  shape: str = "rect", size_mul: float = 1.0) -> None:
    kind = "confetti_dot" if shape == "dot" else "confetti"
    for _ in range(n * CONFETTI_COUNT_MUL):
        x = random.uniform(0.0, max(1.0, screen_w))
        _particles.append(Particle(
            x=x,
            y=-12.0 + random.uniform(-30, 30),
            vx=random.uniform(-60.0, 60.0),
            vy=random.uniform(36.0, 140.0),          # slight drop-speed variation
            age=0.0,
            lifetime=random.uniform(7.2, 12.6) * CONFETTI_LIFETIME_MUL,
            color=_jitter_color(palette_color(palette, color), h_amp=0.10),
            size=random.uniform(3.2, 6.4) * CONFETTI_SIZE_MUL * size_mul,
            gravity=random.uniform(120.0, 165.0),    # varies fall a little over time
            drag=0.25,
            spin=random.uniform(-6.0, 6.0),
            angle=random.uniform(0.0, 2 * math.pi),
            kind=kind,
            fade_bias=random.uniform(-0.09, 0.09),   # each fades at a slightly different height
        ))
    _clamp_pool()


def emit_burst(cx: float, cy: float, color: Color, n: int = 40) -> None:
    """Firework-like radial spark burst at a screen point."""
    for _ in range(n):
        ang = random.uniform(0.0, 2.0 * math.pi)
        spd = random.uniform(120.0, 380.0)
        _particles.append(Particle(
            x=cx,
            y=cy,
            vx=math.cos(ang) * spd,
            vy=math.sin(ang) * spd,
            age=0.0,
            lifetime=random.uniform(0.8, 1.6),
            color=_jitter_color(color, h_amp=0.12),
            size=random.uniform(2.0, 4.0) * SIZE_MUL,
            gravity=240.0,
            drag=1.4,
            kind="dot",
        ))
    _clamp_pool()


def tick(dt: float) -> None:
    if not _particles or dt <= 0.0:
        return
    if dt > 0.1:
        dt = 0.1
    survivors = []
    for p in _particles:
        p.age += dt
        if p.age >= p.lifetime:
            continue
        if p.drag > 0:
            damp = max(0.0, 1.0 - p.drag * dt)
            p.vx *= damp
            p.vy *= damp
        p.vy += p.gravity * dt
        p.x += p.vx * dt
        p.y += p.vy * dt
        if p.spin:
            p.angle += p.spin * dt
        survivors.append(p)
    _particles[:] = survivors


def draw(camera=None, screen_h: float = 0.0) -> None:
    if not _particles:
        return
    dl = imgui.get_foreground_draw_list()
    # confetti fades out as it falls past mid-screen instead of raining to the bottom
    fade_start = screen_h * 0.40
    fade_end = screen_h * 0.60
    # World->screen affine (from the 2D camera), precomputed once for the frame.
    m00 = m01 = m03 = m10 = m11 = m13 = hw = hh = 0.0
    if camera is not None:
        M = camera.view_projection_matrix
        m00 = float(M[0, 0]); m01 = float(M[0, 1]); m03 = float(M[0, 3])
        m10 = float(M[1, 0]); m11 = float(M[1, 1]); m13 = float(M[1, 3])
        hw = camera.projection_width / 2.0
        hh = camera.projection_height / 2.0
    for p in _particles:
        if p.world and camera is not None:
            wx, wy = p.x, -p.y   # undo the screen-down anchoring
            ox = m00 * wx + m01 * wy + m03
            oy = m10 * wx + m11 * wy + m13
            px = (1.0 + ox) * hw
            py = (1.0 - oy) * hh
            psize = p.size   # position tracks the scene; size stays screen-constant
        else:
            px, py, psize = p.x, p.y, p.size
        u = max(0.0, 1.0 - (p.age / p.lifetime))
        a = u * u
        if screen_h > 0.0 and p.kind.startswith("confetti"):
            fs = fade_start + p.fade_bias * screen_h   # per-particle fade height
            fe = fade_end + p.fade_bias * screen_h
            if fe > fs:
                a *= max(0.0, min(1.0, (fe - py) / (fe - fs)))
        col = imgui.get_color_u32_rgba(p.color[0], p.color[1], p.color[2], a)
        if p.kind == "confetti":
            s = psize
            c = math.cos(p.angle)
            sn = math.sin(p.angle)
            hx, hy = s, s * 0.45
            x0 = -hx * c - -hy * sn; y0 = -hx * sn + -hy * c
            x1 =  hx * c - -hy * sn; y1 =  hx * sn + -hy * c
            x2 =  hx * c -  hy * sn; y2 =  hx * sn +  hy * c
            x3 = -hx * c -  hy * sn; y3 = -hx * sn +  hy * c
            dl.add_quad_filled(
                px + x0, py + y0,
                px + x1, py + y1,
                px + x2, py + y2,
                px + x3, py + y3,
                col,
            )
        elif p.kind == "confetti_dot":
            dl.add_circle_filled(px, py, psize * 0.75, col, 12)
        else:
            dl.add_circle_filled(
                px, py, psize * 1.6,
                imgui.get_color_u32_rgba(p.color[0], p.color[1], p.color[2], a * 0.35),
                10,
            )
            dl.add_circle_filled(px, py, psize, col, 10)


def alive() -> int:
    return len(_particles)


def clear() -> None:
    _particles.clear()
