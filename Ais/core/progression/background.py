# Living background: soft feature-colour blobs rendered in a GL pre-pass BEHIND
# the tomogram (fills the empty papery space; the data draws over it). Mostly
# screen-space, with a slight camera parallax. Opt-in via the Background cosmetic.
#
# It is RESPONSIVE to annotation via one master "arousal" scalar (_energy) that
# rises fast while you work and decays slowly when idle, plus a recency-weighted
# feature palette and small transient kicks from discrete events (box/copy/level
# -up). Every channel is a bounded multiplier on the calm baseline, so even at
# full energy it stays subtle. Because the pass is occluded by the data, all
# responses are whole-field (they only show in the papery margins).
#
# This module holds the state and hands the renderer a list of (x, y, radius,
# colour) plus the papery base colour and an intensity each frame; the GL draw
# lives in Renderer.render_background.
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import Ais.core.config as cfg
from . import cosmetics
from . import events
from . import profile as _profile


Color = Tuple[float, float, float]

N_BLOBS = 8
PARALLAX = 0.08            # fraction of the camera pan the background follows

A_MAX = 0.70              # ceiling of the continuous arousal drive
DRIVE_GRACE = 0.5        # seconds after an award that drive stays maxed
DRIVE_RAMP = 3.5         # drive ramps to 0 over this many more seconds
TAU_RISE = 1.5           # fast attack on _energy
TAU_FALL = 12.0          # slow release on _energy
TAU_EVENT = 2.5          # decay of the discrete-event transient
TAU_WASH = 6.0           # decay of the level-up colour wash
TAU_COLOR = 4.0          # per-blob colour crossfade
_KICK = {"box": 0.12, "copy": 0.28, "levelup": 0.28}
_RETARGET = {"box": 1, "copy": 2, "levelup": 2}


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


_blobs: List[_Blob] = []
_bt = 0.0                 # breath clock (accumulates at an energy-scaled rate)
_recolor_accum = 0.0
_energy = 0.0
_event = 0.0
_wash = 0.0
_wash_color: Color = (1.0, 1.0, 1.0)
_E = 0.0                  # last E_out, shared between _tick and frame
_pending: List[Tuple[str, Color]] = []   # (kind, softened colour) from event hooks


def _soft(v: float) -> float:
    return v + (1.0 - v) * 0.12


def _soft_color(c) -> Color:
    return (_soft(float(c[0])), _soft(float(c[1])), _soft(float(c[2])))


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _default_palette() -> List[Color]:
    return [(0.42, 0.52, 0.90), (0.90, 0.52, 0.62), (0.48, 0.80, 0.66), (0.92, 0.78, 0.42)]


def _palette() -> List[Color]:
    p = _profile.get_profile()
    cols = [_soft_color(c) for n, c in p.colors.items() if p.skill_xp(n) > 0]
    return cols or [_soft_color(c) for c in _default_palette()]


def _weighted_target() -> Color:
    # Pick a target colour weighted toward recently/actively annotated features;
    # the whole palette stays faintly present via a floor weight.
    p = _profile.get_profile()
    active = None
    if cfg.se_active_frame is not None and getattr(cfg.se_active_frame, "active_feature", None) is not None:
        active = cfg.se_active_frame.active_feature.title
    weighted: List[Tuple[float, Color]] = []
    for name, c in p.colors.items():
        if p.skill_xp(name) <= 0:
            continue
        w = 0.05 + math.exp(-events.time_since_feature(name) / 12.0)
        if name == active:
            w *= 1.6
        weighted.append((w, _soft_color(c)))
    if not weighted:
        return random.choice(_palette())
    total = sum(w for w, _ in weighted)
    r = random.uniform(0.0, total)
    acc = 0.0
    for w, col in weighted:
        acc += w
        if r <= acc:
            return col
    return weighted[-1][1]


def pulse(kind: str, color) -> None:
    # Called from the box/copy award sites. Queues a whole-field swell + palette
    # bias; drained on the next _tick (sole owner of the blob list).
    if kind in _KICK:
        _pending.append((kind, _soft_color(color)))


def notify_levelup(ev) -> None:
    _pending.append(("levelup", _soft_color(ev.color)))


def _ensure(w: int, h: int) -> None:
    if _blobs:
        return
    pal = _palette()
    for _ in range(N_BLOBS):
        c = random.choice(pal)
        _blobs.append(_Blob(
            x=random.uniform(0, w), y=random.uniform(0, h),
            vx=random.uniform(-16, 16), vy=random.uniform(-16, 16),
            r=random.uniform(320, 720),
            color=c, color_target=c,
            phase=random.uniform(0, 6.28),
            breathe=random.uniform(0.15, 0.4),
        ))


def _tick(dt: float, w: int, h: int) -> None:
    global _bt, _recolor_accum, _energy, _event, _wash, _wash_color, _E
    if dt > 0.1:
        dt = 0.1

    # drain discrete-event hooks: saturating kick to _event, palette bias, wash
    while _pending:
        kind, col = _pending.pop(0)
        _event += _KICK[kind] * (1.0 - _clamp(_energy + _event, 0.0, 1.0))
        for b in random.sample(_blobs, min(_RETARGET[kind], len(_blobs))):
            b.color_target = col
        if kind == "levelup":
            _wash = 1.0
            _wash_color = col

    # continuous drive from time-since-any-award: fast attack, slow release
    idle = events.time_since_last_award()
    a = A_MAX * _clamp(1.0 - (idle - DRIVE_GRACE) / DRIVE_RAMP, 0.0, 1.0)
    tau = TAU_RISE if a > _energy else TAU_FALL
    _energy += (a - _energy) * (1.0 - math.exp(-dt / tau))
    # idle-envelope guard: can never stay bright once genuinely away
    _energy = min(_energy, math.exp(-max(0.0, idle - 2.0) / 12.0))

    _event *= math.exp(-dt / TAU_EVENT)
    _wash *= math.exp(-dt / TAU_WASH)
    _E = _clamp(_energy + _event, 0.0, 1.0)

    # per-blob colour crossfade (no hard swaps)
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

    _bt += (1.0 + 0.3 * _E) * dt

    # palette churn: livelier while working, holds still when away
    _recolor_accum += dt
    if _recolor_accum >= (12.0 - 7.0 * _E) and _blobs:
        _recolor_accum = 0.0
        random.choice(_blobs).color_target = _weighted_target()


def frame(dt: float, w: int, h: int, camera) -> Optional[Tuple[List[Tuple[float, float, float, Color]], Color, float]]:
    """Advance the field and return (blobs, base_colour, intensity) for the GL
    renderer, or None when disabled. blobs = [(x, y, radius, rgb)]."""
    prm = cosmetics.params(cosmetics.BACKGROUND)
    if not prm.get("enabled", False):
        return None
    _ensure(w, h)
    _tick(dt, w, h)

    i0 = prm.get("intensity", 0.45)
    intensity = i0 * (0.55 + 0.45 * _E) + 0.12 * _wash
    size_mul = 0.94 + 0.10 * _E
    amp = 0.07 + 0.06 * _E
    wmix = 0.22 * _wash
    px = camera.position[0] * camera.zoom * PARALLAX
    py = -camera.position[1] * camera.zoom * PARALLAX

    out: List[Tuple[float, float, float, Color]] = []
    for b in _blobs:
        rad = b.r * size_mul * (1.0 + amp * math.sin(_bt * b.breathe + b.phase))
        col = (b.color[0] + (_wash_color[0] - b.color[0]) * wmix,
               b.color[1] + (_wash_color[1] - b.color[1]) * wmix,
               b.color[2] + (_wash_color[2] - b.color[2]) * wmix)
        out.append((b.x + px, b.y + py, rad, col))
    base = tuple(cfg.COLOUR_WINDOW_BACKGROUND[:3])
    return out, base, intensity


def clear() -> None:
    global _bt, _energy, _event, _wash
    _blobs.clear()
    _bt = 0.0
    _energy = 0.0
    _event = 0.0
    _wash = 0.0
    _pending.clear()
