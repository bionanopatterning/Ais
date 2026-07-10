# Living background: feature-colour shapes rendered in a GL pre-pass BEHIND the
# tomogram (fills the empty papery space; the data draws over it). Mostly screen-
# space with a slight camera parallax. Opt-in via the Background cosmetic.
#
# Styles (from the equipped cosmetic's params):
#   "blob"        - soft drifting gaussian washes (Aurora)
#   "bokeh"       - crisp discs that fade in/out in place and gently avoid the cursor
#   "brushstroke" - large blurred rotated wisps spawned along your annotations, then fade
#
# It starts EMPTY (like plain paper) and fills in as you annotate: a "wake"
# counter grows with activity, revealing shapes one by one. Colour follows the
# ACTIVE feature tightly (small hue jitter), with only a faint trace of others.
# One master "arousal" scalar rises fast while you work and decays slowly when
# idle, modulating intensity/motion within bounded multipliers; discrete events
# add small whole-field swells. All responses are whole-field (the pass is
# occluded by the data, so they only show in the papery margins).
#
# The GL draw lives in Renderer.render_background; this module hands it, each
# frame, a list of (x, y, radius, colour, angle, alpha) + base colour + intensity
# + a shape id.
from __future__ import annotations

import colorsys
import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import Ais.core.config as cfg
from . import cosmetics
from . import events
from . import profile as _profile


Color = Tuple[float, float, float]

PARALLAX = 0.15
FILL_UP = 3.2            # how fast shapes wake with activity (scaled by energy)
FILL_DOWN = 0.4         # how fast they recede when idle
_SHAPE = {"blob": 0, "brushstroke": 1, "bokeh": 2}

# Bokeh cursor avoidance: a gentle push away, heavily damped so blobs drift a
# little then quickly come to rest (peaceful & calm).
_AVOID_R = 200.0
_AVOID_R2 = _AVOID_R * _AVOID_R
_AVOID_ACCEL = 350.0
_AVOID_DAMP = 0.02       # per-second velocity retention (heavy)

# Brushstroke mode: large blurred wisps spawned along the user's annotations.
_STROKE_MIN_DT = 0.10    # throttle spawns while painting
_STROKE_MAX = 40

A_MAX = 0.70
DRIVE_GRACE = 0.5
DRIVE_RAMP = 3.5
TAU_RISE = 1.5
TAU_FALL = 12.0
TAU_EVENT = 2.5
TAU_WASH = 6.0
TAU_COLOR = 1.6         # how fast shapes migrate to a new active-feature colour
_KICK = {"box": 0.12, "copy": 0.28, "levelup": 0.28}


@dataclass
class _Blob:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    color: Color
    color_target: Color
    phase: float
    breathe: float
    angle: float = 0.0
    spin: float = 0.0
    alpha: float = 1.0
    life_age: float = 0.0
    life_span: float = 1.0


_blobs: List[_Blob] = []
_cur_style = ""
_cur_n = 0
_cur_w = 0
_cur_h = 0
_bt = 0.0
_recolor_accum = 0.0
_energy = 0.0
_event = 0.0
_wash = 0.0
_wash_color: Color = (1.0, 1.0, 1.0)
_E = 0.0
_awake = 0.0             # how many shapes have woken up (grows with activity)
_pending: List[Tuple[str, Color]] = []

_strokes: List[_Blob] = []           # brushstroke mode: event-spawned, fading wisps
_last_stroke_pt: Optional[Tuple[float, float]] = None
_last_stroke_t = 0.0


def _soft(v: float) -> float:
    return v + (1.0 - v) * 0.12


def _soft_color(c) -> Color:
    return (_soft(float(c[0])), _soft(float(c[1])), _soft(float(c[2])))


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _hue_jitter(c: Color, amp: float = 0.035) -> Color:
    h, s, v = colorsys.rgb_to_hsv(*c)
    h = (h + random.uniform(-amp, amp)) % 1.0
    s = _clamp(s + random.uniform(-0.06, 0.06), 0.0, 1.0)
    v = _clamp(v + random.uniform(-0.05, 0.05), 0.0, 1.0)
    return colorsys.hsv_to_rgb(h, s, v)


def _default_palette() -> List[Color]:
    return [(0.42, 0.52, 0.90), (0.90, 0.52, 0.62), (0.48, 0.80, 0.66), (0.92, 0.78, 0.42)]


def _palette() -> List[Color]:
    p = _profile.get_profile()
    cols = [_soft_color(c) for n, c in p.colors.items() if p.skill_xp(n) > 0]
    return cols or [_soft_color(c) for c in _default_palette()]


def _target_color() -> Color:
    # Strongly favour the ACTIVE feature; only a faint trace of others. Small hue
    # jitter keeps it lively without wandering off the active feature's colour.
    p = _profile.get_profile()
    active = None
    if cfg.se_active_frame is not None and getattr(cfg.se_active_frame, "active_feature", None) is not None:
        active = cfg.se_active_frame.active_feature.title
    weighted: List[Tuple[float, Color]] = []
    for name, c in p.colors.items():
        if p.skill_xp(name) <= 0:
            continue
        if name == active:
            w = 14.0
        else:
            # only very recently-touched other features get any weight at all
            w = 0.004 + 0.25 * math.exp(-events.time_since_feature(name) / 5.0)
        weighted.append((w, _soft_color(c)))
    if not weighted:
        return _hue_jitter(random.choice(_palette()))
    total = sum(w for w, _ in weighted)
    r = random.uniform(0.0, total)
    acc = 0.0
    chosen = weighted[-1][1]
    for w, col in weighted:
        acc += w
        if r <= acc:
            chosen = col
            break
    return _hue_jitter(chosen)


def pulse(kind: str, color) -> None:
    if kind in _KICK:
        _pending.append((kind, _soft_color(color)))


def notify_levelup(ev) -> None:
    _pending.append(("levelup", _soft_color(ev.color)))


def stroke(sx: float, sy: float, colour) -> None:
    # Brushstroke mode: spawn a big blurred wisp at the annotation, oriented along
    # the gesture but rotated a little, that grows and fades away. Throttled; a
    # no-op unless the Brushstroke background is equipped.
    global _last_stroke_pt, _last_stroke_t
    prm = _params()
    if prm.get("style") != "brushstroke" or not prm.get("enabled", False):
        _last_stroke_pt = None
        return
    now = time.monotonic()
    if now - _last_stroke_t < _STROKE_MIN_DT:
        return
    if _last_stroke_pt is not None and (now - _last_stroke_t) < 0.5:
        dx, dy = sx - _last_stroke_pt[0], sy - _last_stroke_pt[1]
        base_ang = math.atan2(dy, dx) if (dx * dx + dy * dy) > 4.0 else random.uniform(0.0, 6.28)
    else:
        base_ang = random.uniform(0.0, 6.28)
    _last_stroke_pt, _last_stroke_t = (sx, sy), now
    ang = base_ang + random.uniform(-0.5, 0.5)   # rotated relative to the gesture
    col = _hue_jitter(_soft_color(colour))
    r = random.uniform(prm.get("rmin", 240.0), prm.get("rmax", 520.0))
    _strokes.append(_Blob(
        x=float(sx), y=float(sy),
        vx=math.cos(ang) * 6.0, vy=math.sin(ang) * 6.0,
        r=r, color=col, color_target=col,
        phase=0.0, breathe=0.0,
        angle=ang, spin=random.uniform(-0.1, 0.1),
        alpha=0.0, life_age=0.0,
        life_span=random.uniform(2.2, 4.0),
    ))
    if len(_strokes) > _STROKE_MAX:
        del _strokes[: len(_strokes) - _STROKE_MAX]


def _tick_strokes(dt: float) -> None:
    if dt > 0.1:
        dt = 0.1
    _pending.clear()   # pulses aren't used in this mode; don't let them accumulate
    alive: List[_Blob] = []
    for b in _strokes:
        b.life_age += dt
        frac = b.life_age / b.life_span
        if frac >= 1.0:
            continue
        b.alpha = _life_alpha(frac)
        b.x += b.vx * dt
        b.y += b.vy * dt
        b.angle += b.spin * dt
        b.r += b.r * 0.12 * dt          # slowly grow as it dissipates
        b.vx *= (1.0 - 0.9 * dt)        # ease the drift to a halt
        b.vy *= (1.0 - 0.9 * dt)
        alive.append(b)
    _strokes[:] = alive


def _params() -> dict:
    return cosmetics.params(cosmetics.BACKGROUND)


def _spawn(w: int, h: int, rmin: float, rmax: float, style: str, life_mul: float = 1.0) -> _Blob:
    c = _target_color()
    lifecycle = style == "bokeh"
    if lifecycle:
        vx, vy = random.uniform(-6, 6), random.uniform(-6, 6)
    else:
        vx, vy = random.uniform(-16, 16), random.uniform(-16, 16)
    return _Blob(
        x=random.uniform(0, w), y=random.uniform(0, h),
        vx=vx, vy=vy,
        r=random.uniform(rmin, rmax),
        color=c, color_target=c,
        phase=random.uniform(0, 6.28),
        breathe=random.uniform(0.15, 0.4),
        angle=random.uniform(-0.6, 0.6),
        spin=random.uniform(-0.25, 0.25),
        alpha=0.0 if lifecycle else 1.0,
        life_age=0.0,
        life_span=random.uniform(4.0, 9.0) * life_mul,
    )


def _ensure(w: int, h: int, style: str, n: int, rmin: float, rmax: float, life_mul: float) -> None:
    global _cur_style, _cur_n, _cur_w, _cur_h
    if _blobs and _cur_style == style and _cur_n == n:
        if _cur_w > 0 and (_cur_w != w or _cur_h != h):
            sx, sy = w / _cur_w, h / _cur_h    # follow window resizes
            for b in _blobs:
                b.x *= sx
                b.y *= sy
        _cur_w, _cur_h = w, h
        return
    _blobs.clear()
    for _ in range(n):
        _blobs.append(_spawn(w, h, rmin, rmax, style, life_mul))
    _cur_style, _cur_n, _cur_w, _cur_h = style, n, w, h


def _life_alpha(frac: float) -> float:
    if frac < 0.18:
        return frac / 0.18
    if frac > 0.72:
        return max(0.0, (1.0 - frac) / 0.28)
    return 1.0


def _tick(dt: float, w: int, h: int, style: str, rmin: float, rmax: float, life_mul: float,
          cursor_bs: Optional[Tuple[float, float]] = None) -> None:
    global _bt, _recolor_accum, _energy, _event, _wash, _wash_color, _E, _awake
    if dt > 0.1:
        dt = 0.1
    lifecycle = style == "bokeh"

    while _pending:
        kind, col = _pending.pop(0)
        _event += _KICK[kind] * (1.0 - _clamp(_energy + _event, 0.0, 1.0))
        # the active feature's colour is boss: on any annotation the whole field
        # retargets to it and migrates there via the colour lerp below. Bokeh is
        # exempt - each disc keeps the colour it was born with.
        if not lifecycle:
            for b in _blobs:
                b.color_target = _hue_jitter(col)
        if kind == "levelup":
            _wash = 1.0
            _wash_color = col

    idle = events.time_since_last_award()
    a = A_MAX * _clamp(1.0 - (idle - DRIVE_GRACE) / DRIVE_RAMP, 0.0, 1.0)
    tau = TAU_RISE if a > _energy else TAU_FALL
    _energy += (a - _energy) * (1.0 - math.exp(-dt / tau))
    _energy = min(_energy, math.exp(-max(0.0, idle - 2.0) / 12.0))
    _event *= math.exp(-dt / TAU_EVENT)
    _wash *= math.exp(-dt / TAU_WASH)
    _E = _clamp(_energy + _event, 0.0, 1.0)
    _bt += (1.0 + 0.3 * _E) * dt

    # wake shapes in proportional to how much the user is doing: a lot of
    # activity reveals many, a little reveals few, and they recede when idle.
    _awake = _clamp(_awake + dt * (FILL_UP * _E - FILL_DOWN), 0.0, float(len(_blobs)))

    if lifecycle:
        rate = 1.0 + 0.5 * _E
        damp = _AVOID_DAMP ** dt
        for b in _blobs:
            b.life_age += dt * rate
            if b.life_age >= b.life_span:
                # respawn in the current active colour, then keep it for life
                nb = _spawn(w, h, rmin, rmax, style, life_mul)
                b.x, b.y, b.vx, b.vy, b.r = nb.x, nb.y, nb.vx, nb.vy, nb.r
                b.color = nb.color
                b.angle, b.spin = nb.angle, nb.spin
                b.life_age, b.life_span = 0.0, nb.life_span
            b.alpha = _life_alpha(b.life_age / b.life_span)
            # gently avoid the cursor, heavily damped so they come to rest quickly
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
            b.vx *= damp
            b.vy *= damp
            b.angle += b.spin * dt
    else:
        kc = 1.0 - math.exp(-dt / TAU_COLOR)
        spd = 0.5 + 0.9 * _E
        for b in _blobs:
            b.color = (b.color[0] + (b.color_target[0] - b.color[0]) * kc,
                       b.color[1] + (b.color_target[1] - b.color[1]) * kc,
                       b.color[2] + (b.color_target[2] - b.color[2]) * kc)
            b.x += b.vx * spd * dt
            b.y += b.vy * spd * dt
            m = b.r
            if b.x < -m: b.x = w + m
            elif b.x > w + m: b.x = -m
            if b.y < -m: b.y = h + m
            elif b.y > h + m: b.y = -m

    # ambient colour drift: bias the field toward the active feature over a few
    # seconds even without discrete events (e.g. while brushing). Bokeh is exempt.
    if not lifecycle:
        _recolor_accum += dt
        if _blobs and _recolor_accum >= (8.0 - 5.0 * _E):
            _recolor_accum = 0.0
            random.choice(_blobs).color_target = _target_color()


def frame(dt: float, w: int, h: int, camera, cursor=None):
    """Advance the field and return (blobs, base_colour, intensity, shape) for the
    GL renderer, or None when disabled. blobs = [(x, y, radius, rgb, angle, alpha)].
    cursor is the mouse position in screen px (bokeh avoids it)."""
    prm = _params()
    if not prm.get("enabled", False):
        if _pending:
            _pending.clear()   # nothing consumes events while disabled; don't accumulate
        return None
    style = prm.get("style", "blob")
    px = camera.position[0] * camera.zoom * PARALLAX
    py = -camera.position[1] * camera.zoom * PARALLAX

    if style == "brushstroke":
        if _blobs:
            _blobs.clear()          # the ambient pool isn't used in this mode
        _tick_strokes(dt)
        out = [(b.x + px, b.y + py, b.r, b.color, b.angle, b.alpha) for b in _strokes]
        base = tuple(cfg.COLOUR_WINDOW_BACKGROUND[:3])
        return out, base, prm.get("intensity", 0.6), _SHAPE["brushstroke"]

    if _strokes:
        _strokes.clear()            # left over from a previous mode
    n = int(prm.get("n", 8))
    rmin = prm.get("rmin", 320.0)
    rmax = prm.get("rmax", 720.0)
    life_mul = prm.get("life_mul", 1.0)
    _ensure(w, h, style, n, rmin, rmax, life_mul)
    cursor_bs = (cursor[0] - px, cursor[1] - py) if (cursor is not None and style == "bokeh") else None
    _tick(dt, w, h, style, rmin, rmax, life_mul, cursor_bs)

    i0 = prm.get("intensity", 0.45)
    intensity = i0 * (0.55 + 0.45 * _E) + 0.12 * _wash
    size_mul = 0.94 + 0.10 * _E
    amp = 0.07 + 0.06 * _E
    wmix = 0.22 * _wash
    lifecycle = style == "bokeh"

    out = []
    for i, b in enumerate(_blobs):
        wake = _clamp(_awake - i, 0.0, 1.0)   # empty at launch; shapes fade in with use
        if wake <= 0.0:
            continue
        if lifecycle:
            rad = b.r * size_mul
        else:
            rad = b.r * size_mul * (1.0 + amp * math.sin(_bt * b.breathe + b.phase))
        col = (b.color[0] + (_wash_color[0] - b.color[0]) * wmix,
               b.color[1] + (_wash_color[1] - b.color[1]) * wmix,
               b.color[2] + (_wash_color[2] - b.color[2]) * wmix)
        out.append((b.x + px, b.y + py, rad, col, b.angle, b.alpha * wake))
    base = tuple(cfg.COLOUR_WINDOW_BACKGROUND[:3])
    return out, base, intensity, _SHAPE.get(style, 0)


def clear() -> None:
    global _bt, _energy, _event, _wash, _cur_style, _cur_n, _cur_w, _cur_h, _awake, _last_stroke_pt
    _blobs.clear()
    _strokes.clear()
    _bt = 0.0
    _energy = 0.0
    _event = 0.0
    _wash = 0.0
    _cur_style = ""
    _cur_n = 0
    _cur_w = 0
    _cur_h = 0
    _awake = 0.0
    _last_stroke_pt = None
    _pending.clear()
