# Profile data, XP curve, and persistence. Levels are derived from XP and never
# stored; per-level cost = 75 * 1.20**L, capped at LEVEL_CAP.
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


PROFILE_DIR = Path(os.path.expanduser("~")) / ".Ais"
PROFILE_PATH = PROFILE_DIR / "profile.json"
SCHEMA_VERSION = 2
LEVEL_CAP = 99
XP_COST_SCALE = 1.5   # higher -> slower levelling

_CUM_XP = [0, 0]  # _CUM_XP[L] = total XP required to be at level L; L=0 placeholder, L=1 free
for _L in range(2, LEVEL_CAP + 1):
    _CUM_XP.append(_CUM_XP[-1] + int(75 * XP_COST_SCALE * (1.20 ** (_L - 1))))


def level_for_xp(xp: int) -> int:
    if xp <= 0:
        return 1
    for L in range(LEVEL_CAP, 0, -1):
        if xp >= _CUM_XP[L]:
            return L
    return 1


def xp_into_level(xp: int) -> Tuple[int, int, int]:
    # (level, xp_into_current_level, xp_needed_for_next); needed is 0 at the cap
    L = level_for_xp(xp)
    if L >= LEVEL_CAP:
        return (LEVEL_CAP, 0, 0)
    base = _CUM_XP[L]
    nxt = _CUM_XP[L + 1]
    return (L, xp - base, nxt - base)


Color = Tuple[float, float, float]
_DEFAULT_COLOR: Color = (0.55, 0.55, 0.60)


def is_placeholder_skill(name: str) -> bool:
    # Placeholder feature/model names ("Unnamed feature 1", "Unnamed model") are
    # not real features - a user just hasn't named them yet. They are never
    # tracked as progression skills.
    return (not name) or ("unnamed" in name.lower())


@dataclass
class Profile:
    skills: Dict[str, int] = field(default_factory=dict)
    colors: Dict[str, Color] = field(default_factory=dict)
    hidden: Set[str] = field(default_factory=set)
    equipped: Dict[str, str] = field(default_factory=dict)   # category -> cosmetic id
    created_at: float = 0.0
    schema_version: int = SCHEMA_VERSION

    def skill_xp(self, name: str) -> int:
        return self.skills.get(name, 0)

    def skill_level(self, name: str) -> int:
        return level_for_xp(self.skills.get(name, 0))

    def skill_color(self, name: str) -> Color:
        c = self.colors.get(name)
        if c is None:
            return _DEFAULT_COLOR
        return (float(c[0]), float(c[1]), float(c[2]))

    def is_hidden(self, name: str) -> bool:
        return name in self.hidden

    def hide(self, name: str) -> None:
        if name in self.skills:
            self.hidden.add(name)

    def unhide(self, name: str) -> None:
        self.hidden.discard(name)

    def visible_skills(self) -> Dict[str, int]:
        return {k: v for k, v in self.skills.items() if k not in self.hidden}

    def hidden_skills(self) -> Dict[str, int]:
        return {k: v for k, v in self.skills.items() if k in self.hidden}

    def overall_level(self) -> int:
        # Sum of levels over visible skills only; hidden skills do not count.
        return sum(level_for_xp(xp) for k, xp in self.skills.items() if k not in self.hidden)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "skills": dict(self.skills),
            "colors": {k: [float(v[0]), float(v[1]), float(v[2])] for k, v in self.colors.items()},
            "hidden": sorted(self.hidden),
            "equipped": dict(self.equipped),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        raw_colors = data.get("colors", {}) or {}
        colors: Dict[str, Color] = {}
        for k, v in raw_colors.items():
            try:
                colors[k] = (float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                pass
        hidden_list = data.get("hidden", []) or []
        return cls(
            skills=dict(data.get("skills", {})),
            colors=colors,
            hidden=set(hidden_list),
            equipped=dict(data.get("equipped", {}) or {}),
            created_at=float(data.get("created_at", time.time())),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


_profile: Optional[Profile] = None
_lock = threading.Lock()
_dirty = False
_last_save = 0.0
_SAVE_THROTTLE_S = 5.0


def get_profile() -> Profile:
    global _profile
    if _profile is None:
        with _lock:
            if _profile is None:
                _profile = _load()
    return _profile


def _load() -> Profile:
    try:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        if PROFILE_PATH.exists():
            with open(PROFILE_PATH, "r") as f:
                return Profile.from_dict(json.load(f))
    except Exception:
        pass
    return Profile(created_at=time.time())


def mark_dirty() -> None:
    global _dirty
    _dirty = True


def maybe_save(force: bool = False) -> None:
    global _dirty, _last_save
    if not _dirty:
        return
    now = time.time()
    if not force and (now - _last_save) < _SAVE_THROTTLE_S:
        return
    try:
        with _lock:
            p = get_profile()
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = PROFILE_PATH.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(p.to_dict(), f, indent=2)
            os.replace(tmp, PROFILE_PATH)
            _dirty = False
            _last_save = now
    except Exception:
        pass
