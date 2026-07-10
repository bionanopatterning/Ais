# Living background: soft, slowly drifting blobs coloured from the user's feature
# palette, drawn on the ImGui background draw list (behind panels). Opt-in via the
# "Background" cosmetic; dimmed while a dataset is loaded so it never fights the
# data. Note: the viewport is GL-rendered under ImGui, so this tints the viewport
# rather than sitting strictly behind the tomogram.
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import imgui

import Ais.core.config as cfg
from . import cosmetics
from . import profile as _profile


Color = Tuple[float, float, float]

N_BLOBS = 6
_RECOLOR_S = 8.0


@dataclass
class _Blob:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    color: Color


_blobs: List[_Blob] = []
_recolor_accum = 0.0


def _palette() -> List[Color]:
    p = _profile.get_profile()
    cols = [c for n, c in p.colors.items() if p.skill_xp(n) > 0]
    if not cols:
        cols = [(0.55, 0.60, 0.85), (0.85, 0.62, 0.70), (0.60, 0.80, 0.72), (0.90, 0.80, 0.55)]
    # soften toward the papery background so it reads as a tint, not paint
    return [(_l(c[0]), _l(c[1]), _l(c[2])) for c in cols]


def _l(v: float) -> float:
    return v + (1.0 - v) * 0.35


def _ensure(w: int, h: int) -> None:
    if _blobs:
        return
    pal = _palette()
    for _ in range(N_BLOBS):
        _blobs.append(_Blob(
            x=random.uniform(0, w), y=random.uniform(0, h),
            vx=random.uniform(-14, 14), vy=random.uniform(-14, 14),
            r=random.uniform(220, 440),
            color=random.choice(pal),
        ))


def tick(dt: float, w: int, h: int) -> None:
    if cosmetics.params(cosmetics.BACKGROUND).get("enabled", False) is False:
        return
    _ensure(w, h)
    if dt > 0.1:
        dt = 0.1
    for b in _blobs:
        b.x += b.vx * dt
        b.y += b.vy * dt
        m = b.r
        if b.x < -m: b.x = w + m
        if b.x > w + m: b.x = -m
        if b.y < -m: b.y = h + m
        if b.y > h + m: b.y = -m
    global _recolor_accum
    _recolor_accum += dt
    if _recolor_accum >= _RECOLOR_S and _blobs:
        _recolor_accum = 0.0
        random.choice(_blobs).color = random.choice(_palette())


def draw(w: int, h: int) -> None:
    prm = cosmetics.params(cosmetics.BACKGROUND)
    if not prm.get("enabled", False) or not _blobs:
        return
    intensity = prm.get("intensity", 0.14)
    # calmer over a loaded dataset so it never fights the data
    scale = 1.0 if cfg.se_active_frame is None else prm.get("dim", 0.4)
    dl = imgui.get_background_draw_list()
    for b in _blobs:
        r, g, bl = b.color
        for rf, af in ((1.0, 0.10), (0.66, 0.17), (0.36, 0.26)):
            dl.add_circle_filled(b.x, b.y, b.r * rf,
                                 imgui.get_color_u32_rgba(r, g, bl, intensity * af * scale), 28)


def clear() -> None:
    _blobs.clear()
