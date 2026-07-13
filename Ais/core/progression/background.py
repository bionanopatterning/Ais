# Living background: feature-colour shapes rendered in a GL pre-pass BEHIND the
# tomogram (fills the empty papery space; the data draws over it). Mostly screen-
# space with a slight camera parallax. Opt-in via the Background cosmetic.
#
# Styles (from the equipped cosmetic's params):
#   "blob"    - soft drifting gaussian washes (Aurora)
#   "bokeh"   - crisp discs that fade in/out in place and gently avoid the cursor
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
_SHAPE = {"blob": 0, "bokeh": 2}

# Bokeh cursor avoidance: a gentle push away, heavily damped so blobs drift a
# little then quickly come to rest (peaceful & calm).
_AVOID_R = 200.0
_AVOID_R2 = _AVOID_R * _AVOID_R
_AVOID_ACCEL = 350.0
_AVOID_DAMP = 0.02       # per-second velocity retention (heavy)

# Bokeh blob turnover is tied to recent action: near-frozen when idle (so no new
# discs appear while you do nothing), normal speed while you annotate. Turnover
# ramps to the idle floor within _BOKEH_TURN_GRACE seconds of the last award.
_BOKEH_TURN_IDLE = 0.08
_BOKEH_TURN_ACTIVE = 1.2
_BOKEH_TURN_GRACE = 1.5

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
    woken: bool = False       # aurora: has this wash faded in yet (and been coloured)?


_blobs: List[_Blob] = []
_cur_style = ""
_cur_n = 0
_cur_w = 0
_cur_h = 0
_bt = 0.0
_energy = 0.0
_event = 0.0
_wash = 0.0
_wash_color: Color = (1.0, 1.0, 1.0)
_E = 0.0
_awake = 0.0             # how many shapes have woken up (grows with activity)
_pending: List[Tuple[str, Color]] = []


def _soft(v: float) -> float:
    return v + (1.0 - v) * 0.12


def _soft_color(c) -> Color:
    return (_soft(float(c[0])), _soft(float(c[1])), _soft(float(c[2])))


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _hue_jitter(c: Color, amp: float = 0.055) -> Color:
    h, s, v = colorsys.rgb_to_hsv(*c)
    h = (h + random.uniform(-amp, amp)) % 1.0
    s = _clamp(s + random.uniform(-0.09, 0.09), 0.0, 1.0)
    v = _clamp(v + random.uniform(-0.07, 0.07), 0.0, 1.0)
    return colorsys.hsv_to_rgb(h, s, v)


def _default_palette() -> List[Color]:
    return [(0.42, 0.52, 0.90), (0.90, 0.52, 0.62), (0.48, 0.80, 0.66), (0.92, 0.78, 0.42)]


def _palette() -> List[Color]:
    p = _profile.get_profile()
    cols = [_soft_color(c) for n, c in p.colors.items() if p.skill_xp(n) > 0]
    return cols or [_soft_color(c) for c in _default_palette()]


def _active_feature():
    frame_ = getattr(cfg, "se_active_frame", None)
    if frame_ is None:
        return None
    return getattr(frame_, "active_feature", None)


def _active_key() -> Optional[str]:
    af = _active_feature()
    return af.title if af is not None else None


def _active_color() -> Optional[Color]:
    # The colour of the feature currently being drawn, straight from the live
    # feature object - available immediately, before it has earned any XP or been
    # stored on the profile. This is what the field should follow.
    af = _active_feature()
    if af is None or getattr(af, "colour", None) is None:
        return None
    try:
        return _soft_color(af.colour)
    except Exception:
        return None


def _target_color() -> Color:
    # Strongly favour the ACTIVE feature; only a faint trace of others. Small hue
    # jitter keeps it lively without wandering off the active feature's colour.
    p = _profile.get_profile()
    active = _active_key()
    active_col = _active_color()
    weighted: List[Tuple[float, Color]] = []
    seen_active = False
    for name, c in p.colors.items():
        if p.skill_xp(name) <= 0:
            continue
        if name == active:
            seen_active = True
            # use the live colour so a re-coloured feature updates instantly
            weighted.append((14.0, active_col or _soft_color(c)))
        else:
            # only very recently-touched other features get any weight at all
            w = 0.004 + 0.25 * math.exp(-events.time_since_feature(name) / 5.0)
            weighted.append((w, _soft_color(c)))
    # the feature you're drawing drives the colour even before it earns XP or is
    # persisted - without this the field falls back to the default palette (blue)
    # for the first stretch of a fresh feature.
    if active_col is not None and not seen_active:
        weighted.append((14.0, active_col))
    if not weighted:
        return _hue_jitter(active_col if active_col is not None else random.choice(_palette()))
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
    global _cur_style, _cur_n, _cur_w, _cur_h, _awake
    if _blobs and _cur_style == style and _cur_n == n:
        if _cur_w > 0 and (_cur_w != w or _cur_h != h):
            sx, sy = w / _cur_w, h / _cur_h    # follow window resizes
            for b in _blobs:
                b.x *= sx
                b.y *= sy
        _cur_w, _cur_h = w, h
        return
    # fresh pool (first use or a mode/count change): start EMPTY and let it wake
    # up with activity, rather than inheriting the previous mode's wake level.
    _blobs.clear()
    _awake = 0.0
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
    global _bt, _energy, _event, _wash, _wash_color, _E, _awake
    if dt > 0.1:
        dt = 0.1
    lifecycle = style == "bokeh"

    while _pending:
        kind, col = _pending.pop(0)
        _event += _KICK[kind] * (1.0 - _clamp(_energy + _event, 0.0, 1.0))
        # shapes keep the colour they were born with; only the transient level-up
        # wash briefly tints the whole field.
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

    # NB: shapes keep the colour they were BORN with - the field is a short-term
    # memory of what you've been doing, so a blob spawned in one feature's colour
    # stays that colour for life (no migration to the current active feature).
    # New/respawned shapes pick up the current colour via _spawn -> _target_color.

    if lifecycle:
        # turnover follows RECENT ACTION (time since last award), not the slow
        # arousal - so within _BOKEH_TURN_GRACE seconds of you stopping, the
        # lifecycle freezes and no new discs appear until you act again.
        act = _clamp(1.0 - idle / _BOKEH_TURN_GRACE, 0.0, 1.0)
        rate = _BOKEH_TURN_IDLE + (_BOKEH_TURN_ACTIVE - _BOKEH_TURN_IDLE) * act
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
        spd = 0.5 + 0.9 * _E
        for b in _blobs:
            b.x += b.vx * spd * dt
            b.y += b.vy * spd * dt
            m = b.r
            if b.x < -m: b.x = w + m
            elif b.x > w + m: b.x = -m
            if b.y < -m: b.y = h + m
            elif b.y > h + m: b.y = -m


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
            if not lifecycle:
                b.woken = False   # aurora wash slept; recolour it when it fades in again
            continue
        if not lifecycle and not b.woken:
            # Aurora washes take their colour when they FIRST fade in - and your
            # activity is what fades them in - so the field reflects the feature
            # you're currently working, not whatever was active when the pool was
            # built. Set once per wake; a visible wash never changes colour.
            b.color = _target_color()
            b.woken = True
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
    global _bt, _energy, _event, _wash, _cur_style, _cur_n, _cur_w, _cur_h, _awake
    _blobs.clear()
    _bt = 0.0
    _energy = 0.0
    _event = 0.0
    _wash = 0.0
    _cur_style = ""
    _cur_n = 0
    _cur_w = 0
    _cur_h = 0
    _awake = 0.0
    _pending.clear()
