# Living background: feature-colour shapes rendered in a GL pre-pass BEHIND the
# tomogram (fills the empty papery space; the data draws over it). Mostly screen-
# space with a slight camera parallax. Opt-in via the Background cosmetic.
#
# Styles (from the equipped cosmetic's params):
#   "blob"    - soft drifting gaussian washes (Aurora)
#   "bokeh"   - crisp discs that fade in/out in place and gently avoid the cursor
#   "mosaic"  - a coarse grid of tiles; annotating lights one in the feature
#               colour, and lit tiles fade back out over time
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
_SHAPE = {"blob": 0, "brushstroke": 1, "bokeh": 2, "mosaic": 3}

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

# Mosaic mode: a fine grid of exactly-touching tiles behind the data. Each
# annotation interaction lights one random tile in the feature colour; lit tiles
# fade IN slowly, hold, then fade back out (the time-gated turn-off), so the
# mosaic fills up under sustained annotation and empties when you stop. Tiles
# have no outline. At most _MOSAIC_MAX_LIT are lit at once (<= shader budget).
_MOSAIC_CELL_PX = 60.0       # target tile size in px (smaller -> finer mosaic)
_MOSAIC_OVERLAP = 0.0        # tiles fill their cell exactly; hard shader edges -> perfect touch
_MOSAIC_MAX_LIT = 90         # max simultaneously-lit tiles (< shader MAXB=96)
_MOSAIC_FADE_IN = 1.8        # s to fade in to peak (slow appearance)
_MOSAIC_HOLD = 22.0          # s a lit tile holds at peak before fading
_MOSAIC_FADE_OUT = 9.0       # s to fade back to invisible (the time-gated turn-off)
_MOSAIC_PEAK = 0.5           # peak opacity of a tile (higher transparency than solid)
_MOSAIC_BRUSH_CHANCE = 0.10  # the continuous brush only occasionally lights a tile

# Brushstroke mode: the user's click-hold-drag gesture is captured, then its
# SHAPE is echoed into the background - rotated, moved elsewhere, and left to
# linger. Each echo is a chain of big soft wisps tracing the (transformed) path.
_STROKE_SEG = 7.0            # min gesture point spacing while capturing (px)
_STROKE_DOT_R = 100.0        # wisp radius - big & bold (~8x the old value)
_STROKE_MAX_PTS = 22         # wisps per echoed gesture (after downsampling)
_STROKE_MAX = 480            # total wisp budget (<= shader MAXB); many gestures linger
_STROKE_SCALE = (0.7, 1.2)   # echo size relative to the drawn gesture (was 0.4-0.7)
_STROKE_LIFE = (220.0, 450.0)  # linger a very long time
_STROKE_FADE_IN = 1.5
_STROKE_FADE_OUT = 4.0
_STROKE_HUE_JITTER = 0.0     # per-stroke hue wander (0 = faithful feature colour)
_GESTURE_END_S = 0.2         # a gap this long ends the current gesture

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

@dataclass
class _Tile:
    cx: float
    cy: float
    hw: float                  # half-width  (fills the cell -> tiles exactly touch)
    hh: float                  # half-height
    color: Color
    alpha: float = 0.0
    age: float = 0.0
    lit: bool = False
    fading: bool = False       # forced off early by an erase


_mosaic: List[_Tile] = []
_mosaic_cols = 0
_mosaic_rows = 0
_mosaic_w = 0
_mosaic_h = 0
_mosaic_q: List[Tuple[str, Color]] = []   # pending touches: (kind, colour)

_strokes: List[_Blob] = []               # brushstroke mode: echoed-gesture wisps
_gesture: List[Tuple[float, float]] = []  # the gesture being captured (screen px)
_gesture_col: Color = (1.0, 1.0, 1.0)
_gesture_last_t = 0.0                      # monotonic time of the last captured point


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


def stroke(sx: float, sy: float, colour) -> None:
    # Brushstroke mode: accumulate the current click-hold-drag gesture. Points are
    # captured whenever the cursor moves far enough (so we record the SHAPE, not
    # the timing); the gesture is echoed into the background when it ends (see
    # _finalize_gesture). A no-op unless the Brushstroke background is equipped.
    global _gesture_col, _gesture_last_t
    prm = _params()
    if prm.get("style") != "brushstroke" or not prm.get("enabled", False):
        _gesture.clear()
        return
    if _gesture:
        dx, dy = sx - _gesture[-1][0], sy - _gesture[-1][1]
        if dx * dx + dy * dy < _STROKE_SEG * _STROKE_SEG:
            return
    _gesture.append((float(sx), float(sy)))
    _gesture_col = _soft_color(colour)
    _gesture_last_t = time.monotonic()


def _finalize_gesture(w: int, h: int) -> None:
    # Echo the captured gesture's shape into the background: recentre it, rotate
    # by a random angle, scale it, and drop it somewhere else on screen as a chain
    # of big soft wisps that linger. Colour is the faithful feature colour (no
    # hue jitter unless _STROKE_HUE_JITTER is raised). Then clear the gesture.
    pts = list(_gesture)
    _gesture.clear()
    if len(pts) < 2:
        return
    # downsample to at most _STROKE_MAX_PTS, keeping endpoints
    if len(pts) > _STROKE_MAX_PTS:
        step = len(pts) / float(_STROKE_MAX_PTS)
        pts = [pts[min(len(pts) - 1, int(i * step))] for i in range(_STROKE_MAX_PTS)]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    theta = random.uniform(0.0, 2.0 * math.pi)
    ct, st = math.cos(theta), math.sin(theta)
    scale = random.uniform(*_STROKE_SCALE)
    margin = 120.0
    tx = random.uniform(margin, max(margin + 1.0, w - margin))
    ty = random.uniform(margin, max(margin + 1.0, h - margin))
    life = random.uniform(*_STROKE_LIFE)
    col = _hue_jitter(_gesture_col, amp=_STROKE_HUE_JITTER) if _STROKE_HUE_JITTER > 0.0 else _gesture_col
    placed = []
    for px, py in pts:
        rx, ry = (px - cx) * scale, (py - cy) * scale
        placed.append((tx + rx * ct - ry * st, ty + rx * st + ry * ct))
    for i, (x, y) in enumerate(placed):
        j = min(i + 1, len(placed) - 1)
        k = max(i - 1, 0)
        ang = math.atan2(placed[j][1] - placed[k][1], placed[j][0] - placed[k][0])
        _strokes.append(_Blob(
            x=x, y=y, vx=0.0, vy=0.0,
            r=_STROKE_DOT_R, color=col, color_target=col,
            phase=0.0, breathe=0.0,
            angle=ang, spin=0.0,
            alpha=0.0, life_age=0.0, life_span=life,
        ))
    if len(_strokes) > _STROKE_MAX:
        del _strokes[: len(_strokes) - _STROKE_MAX]


def _stroke_alpha(age: float, life: float) -> float:
    if age < _STROKE_FADE_IN:
        return age / _STROKE_FADE_IN
    if age > life - _STROKE_FADE_OUT:
        return max(0.0, (life - age) / _STROKE_FADE_OUT)
    return 1.0


def _tick_strokes(dt: float, w: int, h: int) -> None:
    if dt > 0.1:
        dt = 0.1
    _pending.clear()   # ambient pulses aren't used in this mode; don't accumulate
    if _gesture and (time.monotonic() - _gesture_last_t) > _GESTURE_END_S:
        _finalize_gesture(w, h)
    alive: List[_Blob] = []
    for b in _strokes:
        b.life_age += dt
        if b.life_age >= b.life_span:
            continue
        b.alpha = _stroke_alpha(b.life_age, b.life_span)
        alive.append(b)
    _strokes[:] = alive


def touch(colour, kind: str = "box") -> None:
    # Mosaic mode: queue a tile change from an annotation interaction. kind is
    # "box"/"brush" (light a tile in the feature colour) or "erase" (clear a lit
    # tile). A no-op unless the Mosaic background is equipped.
    prm = _params()
    if prm.get("style") != "mosaic" or not prm.get("enabled", False):
        return
    col = _soft_color(colour) if colour is not None else (1.0, 1.0, 1.0)
    _mosaic_q.append((kind, col))


def _ensure_mosaic(w: int, h: int) -> None:
    # Build (or rescale) the invisible tile grid: cells ~_MOSAIC_CELL_PX across,
    # each tile filling its cell exactly (+_MOSAIC_OVERLAP) so neighbours touch.
    global _mosaic_cols, _mosaic_rows, _mosaic_w, _mosaic_h
    if w <= 0 or h <= 0:
        return
    cols = max(1, int(round(w / _MOSAIC_CELL_PX)))
    rows = max(1, int(round(h / _MOSAIC_CELL_PX)))
    if _mosaic and _mosaic_cols == cols and _mosaic_rows == rows and _mosaic_w == w and _mosaic_h == h:
        return
    cw, ch = w / cols, h / rows
    hw, hh = 0.5 * cw + _MOSAIC_OVERLAP, 0.5 * ch + _MOSAIC_OVERLAP
    _mosaic.clear()
    for r in range(rows):
        for c in range(cols):
            _mosaic.append(_Tile(cx=(c + 0.5) * cw, cy=(r + 0.5) * ch, hw=hw, hh=hh, color=(1.0, 1.0, 1.0)))
    _mosaic_cols, _mosaic_rows, _mosaic_w, _mosaic_h = cols, rows, w, h


def _mosaic_light(col: Color) -> None:
    # Prefer an unlit tile so the field fills out. Cap the number lit at once at
    # _MOSAIC_MAX_LIT (<= shader budget): once at the cap, refresh a random lit
    # tile instead so the upload count never overflows.
    lit = [t for t in _mosaic if t.lit]
    if len(lit) >= _MOSAIC_MAX_LIT:
        t = random.choice(lit) if lit else None
    else:
        unlit = [t for t in _mosaic if not t.lit]
        t = random.choice(unlit) if unlit else (random.choice(_mosaic) if _mosaic else None)
    if t is None:
        return
    t.color = col
    t.age = 0.0
    t.lit = True
    t.fading = False


def _mosaic_clear() -> None:
    # Erase: start fading a random currently-solid tile back to invisible.
    lit = [t for t in _mosaic if t.lit and not t.fading]
    if lit:
        random.choice(lit).fading = True


def _tick_mosaic(dt: float, w: int, h: int) -> None:
    if dt > 0.1:
        dt = 0.1
    _pending.clear()   # ambient pulses aren't used in this mode; don't accumulate
    _ensure_mosaic(w, h)
    while _mosaic_q:
        kind, col = _mosaic_q.pop(0)
        if kind == "brush" and random.random() > _MOSAIC_BRUSH_CHANCE:
            continue
        if kind == "erase":
            _mosaic_clear()
        else:
            _mosaic_light(col)
    peak = _MOSAIC_PEAK
    for t in _mosaic:
        if not t.lit:
            continue
        t.age += dt
        if t.fading:
            t.alpha -= (dt / _MOSAIC_FADE_OUT) * peak
        elif t.age < _MOSAIC_FADE_IN:
            t.alpha = max(t.alpha, (t.age / _MOSAIC_FADE_IN) * peak)   # only ramp up (no flicker on refresh)
        elif t.age > _MOSAIC_HOLD:
            t.alpha = (1.0 - (t.age - _MOSAIC_HOLD) / _MOSAIC_FADE_OUT) * peak
        else:
            t.alpha = peak
        if t.alpha <= 0.0:
            t.alpha = 0.0
            t.lit = False
            t.fading = False


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
        _mosaic_q.clear()
        _gesture.clear()
        return None
    style = prm.get("style", "blob")
    px = camera.position[0] * camera.zoom * PARALLAX
    py = -camera.position[1] * camera.zoom * PARALLAX

    if style == "brushstroke":
        if _blobs:
            _blobs.clear()          # the ambient pool isn't used in this mode
        if _mosaic:
            _mosaic.clear()
        _tick_strokes(dt, w, h)
        out = [(b.x + px, b.y + py, b.r, b.color, b.angle, b.alpha) for b in _strokes]
        base = tuple(cfg.COLOUR_WINDOW_BACKGROUND[:3])
        return out, base, prm.get("intensity", 0.7), _SHAPE["brushstroke"]

    if style == "mosaic":
        if _blobs:
            _blobs.clear()          # the ambient pool isn't used in this mode
        if _strokes:
            _strokes.clear()
        _gesture.clear()
        _tick_mosaic(dt, w, h)
        # radius slot carries half-width, angle slot carries half-height (the
        # mosaic shader reads them as a rectangle so tiles fill their cells)
        out = [(t.cx + px, t.cy + py, t.hw, t.color, t.hh, t.alpha)
               for t in _mosaic if t.alpha > 0.001]
        base = tuple(cfg.COLOUR_WINDOW_BACKGROUND[:3])
        return out, base, prm.get("intensity", 0.9), _SHAPE["mosaic"]

    # ambient modes (aurora / bokeh): drop any leftover special-mode state
    if _mosaic:
        _mosaic.clear()
    _mosaic_q.clear()
    if _strokes:
        _strokes.clear()
    _gesture.clear()
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
    global _mosaic_cols, _mosaic_rows, _mosaic_w, _mosaic_h
    _blobs.clear()
    _mosaic.clear()
    _mosaic_q.clear()
    _mosaic_cols = _mosaic_rows = _mosaic_w = _mosaic_h = 0
    _strokes.clear()
    _gesture.clear()
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
