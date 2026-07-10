# XP orbs: on earning XP, a few small orbs pop at the interaction point and then
# home into that feature's row in the top-right XP HUD (Minecraft-style), where
# the bar flashes as they land. Pure screen-space UI effect.
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import imgui

from . import cosmetics
from . import particles


Color = Tuple[float, float, float]

MAX_ORBS = 300
DRIFT_S = 0.16          # brief outward scatter before homing
ARRIVE_PX = 14.0
BASE_SPEED = 260.0
ACCEL = 3000.0          # px/s^2 ramp while homing
MAX_SPEED = 1500.0
STRANDED_S = 1.6        # give up if the target row never appears
PULSE_S = 0.35          # bar-flash duration after an orb lands
STAGGER_S = (0.02, 0.06)   # per-orb birth spacing so a burst trickles out


@dataclass
class _Orb:
    x: float
    y: float
    vx: float
    vy: float
    color: Color
    size: float
    feature: str
    spawn_at: float = 0.0   # monotonic time this orb becomes active
    age: float = 0.0
    homing_t: float = 0.0


_orbs: List[_Orb] = []
_targets: Dict[str, Tuple[float, float]] = {}
_impacts: Dict[str, float] = {}


def emit(sx: float, sy: float, feature_color: Color, n: int, feature: str) -> None:
    prm = cosmetics.params(cosmetics.ORB)
    palette = prm.get("palette", "feature")
    size_mul = prm.get("size_mul", 1.0)
    t = time.monotonic()   # stagger births so a multi-orb gain trickles out
    for _ in range(n):
        ang = random.uniform(0.0, 2.0 * math.pi)
        spd = random.uniform(40.0, 120.0)
        _orbs.append(_Orb(
            x=sx + random.uniform(-4, 4),
            y=sy + random.uniform(-4, 4),
            vx=math.cos(ang) * spd,
            vy=math.sin(ang) * spd - 40.0,   # slight upward pop
            color=particles.palette_color(palette, feature_color),
            size=random.uniform(2.8, 3.8) * size_mul,
            feature=feature,
            spawn_at=t,
        ))
        t += random.uniform(*STAGGER_S)
    if len(_orbs) > MAX_ORBS:
        del _orbs[: len(_orbs) - MAX_ORBS]


def set_targets(targets: Dict[str, Tuple[float, float]]) -> None:
    global _targets
    _targets = targets


def impact_pulse(feature: str) -> float:
    t = _impacts.get(feature)
    if t is None:
        return 0.0
    return max(0.0, 1.0 - (time.monotonic() - t) / PULSE_S)


def tick(dt: float) -> None:
    if not _orbs:
        return
    if dt > 0.05:
        dt = 0.05
    now = time.monotonic()
    survivors: List[_Orb] = []
    for o in _orbs:
        if o.spawn_at > now:
            survivors.append(o)   # not born yet
            continue
        o.age += dt
        tgt = _targets.get(o.feature)
        if o.age < DRIFT_S or tgt is None:
            # scatter phase, or waiting for the row to exist
            o.vy += 220.0 * dt
            o.vx *= (1.0 - 1.8 * dt)
            o.vy *= (1.0 - 1.8 * dt)
            o.x += o.vx * dt
            o.y += o.vy * dt
            if tgt is None and o.age > STRANDED_S:
                continue   # target never showed up; drop it
            survivors.append(o)
            continue
        # homing phase: steer straight at the row, ramping speed
        o.homing_t += dt
        tx, ty = tgt
        dx, dy = tx - o.x, ty - o.y
        dist = math.hypot(dx, dy)
        if dist < ARRIVE_PX:
            _impacts[o.feature] = time.monotonic()
            continue
        speed = min(MAX_SPEED, BASE_SPEED + ACCEL * o.homing_t)
        step = min(dist, speed * dt)
        o.x += dx / dist * step
        o.y += dy / dist * step
        survivors.append(o)
    _orbs[:] = survivors


def draw() -> None:
    if not _orbs:
        return
    now = time.monotonic()
    dl = imgui.get_foreground_draw_list()
    for o in _orbs:
        if o.spawn_at > now:
            continue   # not born yet
        r, g, b = o.color
        # homing orbs stretch slightly toward the bar via a faint trailing glow
        dl.add_circle_filled(o.x, o.y, o.size * 2.0, imgui.get_color_u32_rgba(r, g, b, 0.18), 12)
        dl.add_circle_filled(o.x, o.y, o.size, imgui.get_color_u32_rgba(r, g, b, 0.95), 12)
        dl.add_circle_filled(o.x - o.size * 0.25, o.y - o.size * 0.25, o.size * 0.4,
                             imgui.get_color_u32_rgba(min(1.0, r + 0.3), min(1.0, g + 0.3), min(1.0, b + 0.3), 0.95), 8)


def clear() -> None:
    _orbs.clear()
    _impacts.clear()
