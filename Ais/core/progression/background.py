# Living background: soft feature-colour shapes SPAWNED by your annotations.
#
# There is no pre-built pool and no XP/recency tracking. Every annotation action
# (a brush tick, a box, a copy) drops ONE new shape, which is given:
#   - a colour: the active tool's colour, with a little hue jitter. Rarely a
#     sibling's colour (another feature in the tomogram) or the inverted colour.
#   - a random size and velocity
#   - a fade-in: it starts invisible and eases in (it doesn't pop), drifts, and
#     eventually fades back out and dies.
# So the field is just a short-lived echo of what you are drawing. It is empty
# until you act. Opt-in via the Background cosmetic.
#
# Styles (from the equipped cosmetic's params):
#   "blob"   - big soft gaussian washes (Aurora)
#   "bokeh"  - small crisp discs that gently avoid the cursor
#
# Rendered in a GL pre-pass BEHIND the tomogram (fills the papery margins), mostly
# screen-space with a slight camera parallax. The GL draw lives in
# Renderer.render_background; this module hands it, each frame, a list of
# (x, y, radius, colour, angle, alpha) + base colour + intensity + a shape id.
from __future__ import annotations

import colorsys
import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import Ais.core.config as cfg
from . import cosmetics


Color = Tuple[float, float, float]

PARALLAX = 0.15
_SHAPE = {"blob": 0, "bokeh": 2}

# --- spawn behaviour (tunable) ---------------------------------------------
_SPAWN_MIN_DT = 0.18     # throttle continuous (brush) spawns; discrete actions bypass it
_MAX_BLOBS = 200         # safety cap on live shapes
_OTHER_CHANCE = 0.05     # spawn in a sibling feature's colour instead of the active one
_INVERT_CHANCE = 0.01    # spawn in the inverted active colour
_HUE_JITTER = 0.04       # per-shape hue wander around its colour
_LIFE = (4.0, 9.0)       # base life in seconds (multiplied by the cosmetic's life_mul)
_FADE_IN = 0.5           # ease-in time (s) - the shape appears gradually
_FADE_OUT_FRAC = 0.35    # the last this-fraction of life eases back out

# --- bokeh cursor avoidance (a gentle, heavily damped push - calm) ---------
_AVOID_R = 200.0
_AVOID_R2 = _AVOID_R * _AVOID_R
_AVOID_ACCEL = 350.0
_AVOID_DAMP = 0.02       # per-second velocity retention (heavy)

# --- transient level-up wash (a brief whole-field tint) --------------------
_TAU_WASH = 6.0


@dataclass
class _Blob:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    color: Color
    angle: float
    spin: float
    breathe: float
    phase: float
    age: float
    life: float


_blobs: List[_Blob] = []
_cur_style = ""
_screen_w = 1
_screen_h = 1
_last_spawn_t = 0.0
_bt = 0.0                 # time accumulator for the aurora "breathe"
_wash = 0.0
_wash_color: Color = (1.0, 1.0, 1.0)


def _soft(v: float) -> float:
    return v + (1.0 - v) * 0.12


def _soft_color(c) -> Color:
    return (_soft(float(c[0])), _soft(float(c[1])), _soft(float(c[2])))


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _hue_jitter(c: Color, amp: float = _HUE_JITTER) -> Color:
    h, s, v = colorsys.rgb_to_hsv(*c)
    h = (h + random.uniform(-amp, amp)) % 1.0
    s = _clamp(s + random.uniform(-0.06, 0.06), 0.0, 1.0)
    v = _clamp(v + random.uniform(-0.05, 0.05), 0.0, 1.0)
    return colorsys.hsv_to_rgb(h, s, v)


def _params() -> dict:
    return cosmetics.params(cosmetics.BACKGROUND)


def _siblings(active: Color) -> List[Color]:
    # The OTHER features in the current tomogram (annotation context). We never
    # reach into the model tab here - spawns come from annotation actions.
    frame_ = getattr(cfg, "se_active_frame", None)
    if frame_ is None:
        return []
    out: List[Color] = []
    for f in getattr(frame_, "features", []) or []:
        c = getattr(f, "colour", None)
        if c is None:
            continue
        c = (float(c[0]), float(c[1]), float(c[2]))
        if c != active:
            out.append(c)
    return out


def spawn(active_colour, throttle: bool = True) -> None:
    """A user action drops one new shape. Colour is mostly the active tool's, but
    ~_OTHER_CHANCE of the time a sibling feature's colour and ~_INVERT_CHANCE of
    the time the inverted colour. No-op unless a background is equipped. The
    continuous brush passes throttle=True; discrete clicks pass throttle=False."""
    global _last_spawn_t
    prm = _params()
    if not prm.get("enabled", False) or active_colour is None:
        return
    now = time.monotonic()
    if throttle and (now - _last_spawn_t) < _SPAWN_MIN_DT:
        return
    _last_spawn_t = now

    active = (float(active_colour[0]), float(active_colour[1]), float(active_colour[2]))
    rr = random.random()
    if rr < _INVERT_CHANCE:
        base = (1.0 - active[0], 1.0 - active[1], 1.0 - active[2])
    elif rr < _INVERT_CHANCE + _OTHER_CHANCE:
        sib = _siblings(active)
        base = random.choice(sib) if sib else active
    else:
        base = active
    col = _hue_jitter(_soft_color(base))

    style = prm.get("style", "bokeh")
    lifecycle = style == "bokeh"
    rmin = prm.get("rmin", 30.0)
    rmax = prm.get("rmax", 130.0)
    life_mul = prm.get("life_mul", 1.0)
    vlim = 6.0 if lifecycle else 16.0
    _blobs.append(_Blob(
        x=random.uniform(0.0, _screen_w),
        y=random.uniform(0.0, _screen_h),
        vx=random.uniform(-vlim, vlim),
        vy=random.uniform(-vlim, vlim),
        r=random.uniform(rmin, rmax),
        color=col,
        angle=random.uniform(-0.6, 0.6),
        spin=random.uniform(-0.25, 0.25),
        breathe=random.uniform(0.15, 0.4),
        phase=random.uniform(0.0, 6.28),
        age=0.0,
        life=random.uniform(*_LIFE) * life_mul,
    ))
    if len(_blobs) > _MAX_BLOBS:
        del _blobs[: len(_blobs) - _MAX_BLOBS]


def pulse(kind: str, colour) -> None:
    # Discrete-action hook. A level-up briefly washes the whole field; box / copy
    # just drop a shape (unthrottled) in that colour.
    global _wash, _wash_color
    if colour is None:
        return
    if kind == "levelup":
        _wash = 1.0
        _wash_color = _soft_color(colour)
    else:
        spawn(colour, throttle=False)


def notify_levelup(ev) -> None:
    pulse("levelup", ev.color)


def _alpha(age: float, life: float) -> float:
    fade_out = _FADE_OUT_FRAC * life
    if age < _FADE_IN:
        return age / _FADE_IN
    if age > life - fade_out:
        return max(0.0, (life - age) / max(fade_out, 1e-4))
    return 1.0


def frame(dt: float, w: int, h: int, camera, cursor=None):
    """Advance the field and return (blobs, base_colour, intensity, shape) for the
    GL renderer, or None when disabled. blobs = [(x, y, radius, rgb, angle, alpha)].
    cursor is the mouse position in screen px (bokeh discs avoid it)."""
    global _screen_w, _screen_h, _cur_style, _bt, _wash
    _screen_w, _screen_h = w, h
    prm = _params()
    if not prm.get("enabled", False):
        if _blobs:
            _blobs.clear()
        return None
    style = prm.get("style", "bokeh")
    if style != _cur_style:      # switching modes: don't carry shapes across
        _blobs.clear()
        _cur_style = style
    lifecycle = style == "bokeh"
    if dt > 0.1:
        dt = 0.1
    _bt += dt
    _wash *= math.exp(-dt / _TAU_WASH)

    px = camera.position[0] * camera.zoom * PARALLAX
    py = -camera.position[1] * camera.zoom * PARALLAX
    cursor_bs = (cursor[0] - px, cursor[1] - py) if (cursor is not None and lifecycle) else None
    damp = _AVOID_DAMP ** dt

    alive: List[_Blob] = []
    for b in _blobs:
        b.age += dt
        if b.age >= b.life:
            continue
        if cursor_bs is not None:
            dx, dy = b.x - cursor_bs[0], b.y - cursor_bs[1]
            d2 = dx * dx + dy * dy
            if 1.0 < d2 < _AVOID_R2:
                d = math.sqrt(d2)
                push = (1.0 - d / _AVOID_R) * _AVOID_ACCEL
                b.vx += (dx / d) * push * dt
                b.vy += (dy / d) * push * dt
        b.x += b.vx * dt
        b.y += b.vy * dt
        if lifecycle:
            b.vx *= damp
            b.vy *= damp
        b.angle += b.spin * dt
        alive.append(b)
    _blobs[:] = alive

    intensity = prm.get("intensity", 0.4) + 0.12 * _wash
    wmix = 0.22 * _wash
    amp = 0.08

    out = []
    for b in _blobs:
        a = _alpha(b.age, b.life)
        if a <= 0.0:
            continue
        if lifecycle:
            rad = b.r
        else:
            rad = b.r * (1.0 + amp * math.sin(_bt * b.breathe + b.phase))
        col = (b.color[0] + (_wash_color[0] - b.color[0]) * wmix,
               b.color[1] + (_wash_color[1] - b.color[1]) * wmix,
               b.color[2] + (_wash_color[2] - b.color[2]) * wmix)
        out.append((b.x + px, b.y + py, rad, col, b.angle, a))
    base = tuple(cfg.COLOUR_WINDOW_BACKGROUND[:3])
    return out, base, intensity, _SHAPE.get(style, 0)


def clear() -> None:
    global _cur_style, _bt, _wash
    _blobs.clear()
    _cur_style = ""
    _bt = 0.0
    _wash = 0.0
