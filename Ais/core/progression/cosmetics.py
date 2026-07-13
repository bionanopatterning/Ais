# Cosmetic catalogue + equip logic. Cosmetics are pure visual parameter packs
# applied to the everyday effects (cursor trail, XP orbs, box burst) and the
# living background. Everything is free and directly selectable; there is no
# shop or currency. One item is equipped per category.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import profile as _profile


# categories (one equipped item per category)
CURSOR = "cursor_trail"
ORB = "xp_orb"
BURST = "box_burst"
BACKGROUND = "background"

CATEGORY_LABELS = {
    CURSOR: "Cursor trail",
    ORB: "XP orbs",
    BURST: "Box burst",
    BACKGROUND: "Background",
}


@dataclass(frozen=True)
class Cosmetic:
    id: str
    category: str
    name: str
    params: dict


def _c(id, category, name, **params):
    return Cosmetic(id, category, name, params)


CATALOG: Dict[str, List[Cosmetic]] = {
    CURSOR: [
        # Particle colour always follows the feature; level-based hue jitter is
        # handled by the perk tiers, so there are no colour swaps here.
        _c("cursor.spark",  CURSOR, "Sparks",  palette="feature", size_mul=1.0),
    ],
    ORB: [
        _c("orb.default",   ORB, "Orbs",      palette="feature", size_mul=1.0),
    ],
    BURST: [
        _c("burst.default", BURST, "Burst",   palette="feature", size_mul=1.0),
    ],
    BACKGROUND: [
        _c("bg.paper",    BACKGROUND, "Basic",       enabled=False),
        _c("bg.aurora",   BACKGROUND, "Aurora",      enabled=True, style="blob",        n=40, rmin=340, rmax=760, intensity=0.05),
        _c("bg.bokeh",    BACKGROUND, "Bokeh",       enabled=True, style="bokeh",       n=44, rmin=30,  rmax=130, intensity=0.34, life_mul=3.0),
        _c("bg.mosaic",   BACKGROUND, "Mosaic",      enabled=True, style="mosaic",       intensity=0.34),
        _c("bg.brush",    BACKGROUND, "Brushstroke", enabled=True, style="brushstroke", intensity=0.05),
    ],
}


def default_id(category: str) -> str:
    return CATALOG[category][0].id


def get(item_id: str) -> Optional[Cosmetic]:
    for items in CATALOG.values():
        for it in items:
            if it.id == item_id:
                return it
    return None


def equipped_id(p: "_profile.Profile", category: str) -> str:
    chosen = p.equipped.get(category)
    it = get(chosen) if chosen else None
    if it is not None and it.category == category:
        return chosen
    return default_id(category)   # falls back if the equipped id was removed


def equipped_item(p: "_profile.Profile", category: str) -> Cosmetic:
    return get(equipped_id(p, category))


def params(category: str) -> dict:
    """Equipped item's params for the global profile — the renderer entry point."""
    return equipped_item(_profile.get_profile(), category).params


def equip(p: "_profile.Profile", item: Cosmetic) -> bool:
    p.equipped[item.category] = item.id
    _profile.mark_dirty()
    return True


# --- Background selection (surfaced directly in the Party-mode menu) ---

def background_choices() -> List[Tuple[str, str]]:
    return [(it.id, it.name) for it in CATALOG[BACKGROUND]]


def equipped_background_id() -> str:
    return equipped_id(_profile.get_profile(), BACKGROUND)


def equip_background(item_id: str) -> None:
    it = get(item_id)
    if it is not None and it.category == BACKGROUND:
        equip(_profile.get_profile(), it)
