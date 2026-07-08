# award() hook + level-up queue. Level boundary crossings are pushed onto a
# queue the toast UI drains; the feature colour is persisted onto the profile.
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from . import profile as _profile


Color = Tuple[float, float, float]


@dataclass
class LevelUp:
    name: str
    old_level: int
    new_level: int
    color: Color
    timestamp: float


_queue: Deque[LevelUp] = deque(maxlen=16)
_queue_lock = threading.Lock()

_last_award_t: Dict[str, float] = {}    # for rate-limited award sites
_RATE_LIMIT_S = 0.10

_global_last_award_t: float = 0.0       # monotonic seconds, any feature
_feature_last_seen_t: Dict[str, float] = {}   # monotonic seconds, per feature
_feature_last_gain: Dict[str, int] = {}        # last delta xp per feature, for hud float

_last_award_cursor: Optional[Tuple[float, float]] = None  # cursor pos of the last successful award
_CURSOR_EPS_PX = 0.5


def time_since_last_award() -> float:
    if _global_last_award_t == 0.0:
        return 1e9
    return time.monotonic() - _global_last_award_t


def time_since_feature(name: str) -> float:
    t = _feature_last_seen_t.get(name)
    if t is None:
        return 1e9
    return time.monotonic() - t


def recent_features(window_s: float = 12.0) -> List[Tuple[str, float]]:
    # [(name, seconds_ago)] for features touched within window_s, recent first
    now = time.monotonic()
    out: List[Tuple[str, float]] = []
    for name, t in _feature_last_seen_t.items():
        dt = now - t
        if dt <= window_s:
            out.append((name, dt))
    out.sort(key=lambda kv: kv[1])
    return out


def last_gain(name: str) -> int:
    return _feature_last_gain.get(name, 0)


def award(
    skill: str,
    xp: int,
    color: Optional[Color] = None,
    rate_limit: bool = False,
    cursor_pos: Optional[Tuple[float, float]] = None,
) -> None:
    if xp <= 0 or not skill:
        return

    # names containing '+' earn 10x XP
    if "+" in skill:
        xp *= 10

    # skip if the cursor hasn't moved since the last award; discrete events
    # (box placement) pass no cursor_pos and bypass this gate
    global _last_award_cursor
    if cursor_pos is not None and _last_award_cursor is not None:
        if (abs(cursor_pos[0] - _last_award_cursor[0]) < _CURSOR_EPS_PX
                and abs(cursor_pos[1] - _last_award_cursor[1]) < _CURSOR_EPS_PX):
            return

    if rate_limit:
        now = time.monotonic()
        if now - _last_award_t.get(skill, 0.0) < _RATE_LIMIT_S:
            return
        _last_award_t[skill] = now

    if cursor_pos is not None:
        _last_award_cursor = (float(cursor_pos[0]), float(cursor_pos[1]))

    p = _profile.get_profile()
    before = p.skill_level(skill)
    p.skills[skill] = p.skills.get(skill, 0) + xp
    after = p.skill_level(skill)
    if color is not None:
        p.colors[skill] = (float(color[0]), float(color[1]), float(color[2]))
    # earning XP un-hides a skill
    if skill in p.hidden:
        p.hidden.discard(skill)

    if after > before:
        _push(LevelUp(
            name=skill,
            old_level=before,
            new_level=after,
            color=tuple(color) if color is not None else p.skill_color(skill),
            timestamp=time.time(),
        ))

    now_mono = time.monotonic()
    global _global_last_award_t
    _global_last_award_t = now_mono
    _feature_last_seen_t[skill] = now_mono
    _feature_last_gain[skill] = xp
    _profile.mark_dirty()


def _push(ev: LevelUp) -> None:
    with _queue_lock:
        _queue.append(ev)


def pop_level_up() -> Optional[LevelUp]:
    with _queue_lock:
        if _queue:
            return _queue.popleft()
    return None


def peek_pending() -> int:
    with _queue_lock:
        return len(_queue)
