# Living background: soft, slowly drifting + breathing blobs coloured from the
# user's feature palette, drawn on the ImGui background draw list (behind panels).
# Opt-in via the "Background" cosmetic; calmer while a dataset is loaded.
# Note: the viewport is GL-rendered under ImGui, so this tints the viewport
# rather than sitting strictly behind the tomogram.
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import imgui

import Ais.core.config as cfg
from . import cosmetics
from . import profile as _profile


Color = Tuple[float, float, float]

N_BLOBS = 9
_RECOLOR_S = 6.0
# soft radial falloff: (radius_fraction, alpha_fraction) from rim to core
_RINGS = ((1.0, 0.08), (0.74, 0.18), (0.52, 0.34), (0.34, 0.55), (0.18, 0.85))


@dataclass
class _Blob:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    color: Color
    phase: float
    breathe: float   # radians/s of the size oscillation


_blobs: List[_Blob] = []
_t = 0.0
_recolor_accum = 0.0


def _palette() -> List[Color]:
    p = _profile.get_profile()
    cols = [c for n, c in p.colors.items() if p.skill_xp(n) > 0]
    if not cols:
        cols = [(0.45, 0.55, 0.90), (0.90, 0.55, 0.65), (0.50, 0.80, 0.68), (0.92, 0.78, 0.45)]
    return [(_soft(c[0]), _soft(c[1]), _soft(c[2])) for c in cols]


def _soft(v: float) -> float:
    # nudge toward the papery background so blobs read rich but not garish
    return v + (1.0 - v) * 0.18


def _ensure(w: int, h: int) -> None:
    if _blobs:
        return
    pal = _palette()
    for _ in range(N_BLOBS):
        _blobs.append(_Blob(
            x=random.uniform(0, w), y=random.uniform(0, h),
            vx=random.uniform(-16, 16), vy=random.uniform(-16, 16),
            r=random.uniform(300, 680),
            color=random.choice(pal),
            phase=random.uniform(0, 6.28),
            breathe=random.uniform(0.15, 0.4),
        ))


def tick(dt: float, w: int, h: int) -> None:
    if not cosmetics.params(cosmetics.BACKGROUND).get("enabled", False):
        return
    _ensure(w, h)
    if dt > 0.1:
        dt = 0.1
    global _t, _recolor_accum
    _t += dt
    for b in _blobs:
        b.x += b.vx * dt
        b.y += b.vy * dt
        m = b.r
        if b.x < -m: b.x = w + m
        elif b.x > w + m: b.x = -m
        if b.y < -m: b.y = h + m
        elif b.y > h + m: b.y = -m
    _recolor_accum += dt
    if _recolor_accum >= _RECOLOR_S and _blobs:
        _recolor_accum = 0.0
        random.choice(_blobs).color = random.choice(_palette())


def draw(w: int, h: int) -> None:
    prm = cosmetics.params(cosmetics.BACKGROUND)
    if not prm.get("enabled", False) or not _blobs:
        return
    intensity = prm.get("intensity", 0.35)
    scale = 1.0 if cfg.se_active_frame is None else prm.get("dim", 0.55)
    dl = imgui.get_background_draw_list()
    for b in _blobs:
        r, g, bl = b.color
        rad = b.r * (1.0 + 0.12 * math.sin(_t * b.breathe + b.phase))
        for rf, af in _RINGS:
            dl.add_circle_filled(b.x, b.y, rad * rf,
                                 imgui.get_color_u32_rgba(r, g, bl, intensity * af * scale), 32)


def clear() -> None:
    global _t
    _blobs.clear()
    _t = 0.0
