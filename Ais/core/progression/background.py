# Living background: soft feature-colour blobs rendered in a GL pre-pass BEHIND
# the tomogram (fills the empty papery space; the data draws over it). Mostly
# screen-space, with a slight camera parallax. Opt-in via the Background cosmetic.
#
# This module holds the blob state and, each frame, hands the renderer a list of
# (x, y, radius, colour) plus the papery base colour and an intensity. The GL
# draw itself lives in Renderer.render_background.
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import Ais.core.config as cfg
from . import cosmetics
from . import profile as _profile


Color = Tuple[float, float, float]

N_BLOBS = 8
_RECOLOR_S = 6.0
PARALLAX = 0.08          # fraction of the camera pan the background follows
_BREATHE_AMP = 0.12


@dataclass
class _Blob:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    color: Color
    phase: float
    breathe: float


_blobs: List[_Blob] = []
_t = 0.0
_recolor_accum = 0.0


def _palette() -> List[Color]:
    p = _profile.get_profile()
    cols = [c for n, c in p.colors.items() if p.skill_xp(n) > 0]
    if not cols:
        cols = [(0.42, 0.52, 0.90), (0.90, 0.52, 0.62), (0.48, 0.80, 0.66), (0.92, 0.78, 0.42)]
    return [(_soft(c[0]), _soft(c[1]), _soft(c[2])) for c in cols]


def _soft(v: float) -> float:
    return v + (1.0 - v) * 0.12


def _ensure(w: int, h: int) -> None:
    if _blobs:
        return
    pal = _palette()
    for _ in range(N_BLOBS):
        _blobs.append(_Blob(
            x=random.uniform(0, w), y=random.uniform(0, h),
            vx=random.uniform(-16, 16), vy=random.uniform(-16, 16),
            r=random.uniform(320, 720),
            color=random.choice(pal),
            phase=random.uniform(0, 6.28),
            breathe=random.uniform(0.15, 0.4),
        ))


def _tick(dt: float, w: int, h: int) -> None:
    global _t, _recolor_accum
    if dt > 0.1:
        dt = 0.1
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


def frame(dt: float, w: int, h: int, camera) -> Optional[Tuple[List[Tuple[float, float, float, Color]], Color, float]]:
    """Advance the blobs and return (blobs, base_colour, intensity) for the GL
    renderer, or None when the effect is disabled. blobs = [(x, y, radius, rgb)]."""
    prm = cosmetics.params(cosmetics.BACKGROUND)
    if not prm.get("enabled", False):
        return None
    _ensure(w, h)
    _tick(dt, w, h)
    # slight parallax: the background follows a small fraction of the camera pan
    px = camera.position[0] * camera.zoom * PARALLAX
    py = -camera.position[1] * camera.zoom * PARALLAX
    out: List[Tuple[float, float, float, Color]] = []
    for b in _blobs:
        rad = b.r * (1.0 + _BREATHE_AMP * math.sin(_t * b.breathe + b.phase))
        out.append((b.x + px, b.y + py, rad, b.color))
    base = tuple(cfg.COLOUR_WINDOW_BACKGROUND[:3])
    return out, base, prm.get("intensity", 0.45)


def clear() -> None:
    global _t
    _blobs.clear()
    _t = 0.0
