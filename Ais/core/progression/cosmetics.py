# Deterministic cosmetic catalogue + purchase/equip logic. Cosmetics are pure
# visual parameter packs applied to the everyday effects (cursor trail, XP orbs,
# box burst, confetti, HUD), so buying one changes the daily feel, not only the
# rare level-up. Earned with coins, or level-gated for prestige. No randomness.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from . import profile as _profile


# categories (one equipped item per category)
CURSOR = "cursor_trail"
ORB = "xp_orb"
BURST = "box_burst"
CONFETTI = "confetti"
HUD = "hud_theme"

CATEGORY_LABELS = {
    CURSOR: "Cursor trail",
    ORB: "XP orbs",
    BURST: "Box burst",
    CONFETTI: "Confetti",
    HUD: "HUD theme",
}


@dataclass(frozen=True)
class Cosmetic:
    id: str
    category: str
    name: str
    price: int          # coins (0 for defaults and level-unlocks)
    min_level: int      # total-level gate (0 = none)
    params: dict


def _c(id, category, name, price, min_level, **params):
    return Cosmetic(id, category, name, price, min_level, params)


CATALOG: Dict[str, List[Cosmetic]] = {
    CURSOR: [
        _c("cursor.spark",  CURSOR, "Sparks",  0,   0,  palette="feature", size_mul=1.0),
        _c("cursor.ember",  CURSOR, "Embers",  60,  0,  palette="warm",    size_mul=1.15),
        _c("cursor.frost",  CURSOR, "Frost",   60,  0,  palette="cool",    size_mul=1.0),
        _c("cursor.gild",   CURSOR, "Gilded",  120, 0,  palette="gold",    size_mul=1.1),
        _c("cursor.prism",  CURSOR, "Prism",   0,   15, palette="prism",   size_mul=1.1),
    ],
    ORB: [
        _c("orb.default",   ORB, "Orbs",      0,   0,  palette="feature", size_mul=1.0),
        _c("orb.gold",      ORB, "Gold",      40,  0,  palette="gold",    size_mul=1.0),
        _c("orb.star",      ORB, "Starlight", 40,  0,  palette="mono",    size_mul=1.15),
        _c("orb.prism",     ORB, "Prism",     0,   10, palette="prism",   size_mul=1.0),
    ],
    BURST: [
        _c("burst.default", BURST, "Burst",   0,   0,  palette="feature", size_mul=1.0),
        _c("burst.ember",   BURST, "Embers",  50,  0,  palette="warm",    size_mul=1.15),
        _c("burst.prism",   BURST, "Prism",   90,  0,  palette="prism",   size_mul=1.0),
    ],
    CONFETTI: [
        _c("conf.default",  CONFETTI, "Confetti",  0,  0,  palette="feature", shape="rect", size_mul=1.0),
        _c("conf.party",    CONFETTI, "Party",     50, 0,  palette="prism",   shape="rect", size_mul=1.0),
        _c("conf.bubbles",  CONFETTI, "Bubbles",   50, 0,  palette="feature", shape="dot",  size_mul=1.15),
        _c("conf.gold",     CONFETTI, "Gold Rain", 0,  20, palette="gold",    shape="rect", size_mul=1.0),
    ],
    HUD: [
        _c("hud.slate",  HUD, "Slate",  0,  0,  track=(0.18, 0.18, 0.20), track_alpha=0.9,  backdrop=0.55),
        _c("hud.cream",  HUD, "Cream",  70, 0,  track=(0.80, 0.80, 0.74), track_alpha=0.85, backdrop=0.25),
        _c("hud.mint",   HUD, "Mint",   70, 0,  track=(0.16, 0.34, 0.30), track_alpha=0.9,  backdrop=0.5),
        _c("hud.ink",    HUD, "Ink",    0,  10, track=(0.05, 0.05, 0.07), track_alpha=0.95, backdrop=0.7),
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


def is_default(item: Cosmetic) -> bool:
    return item.price == 0 and item.min_level == 0


def is_owned(p: "_profile.Profile", item: Cosmetic) -> bool:
    return is_default(item) or item.id in p.owned


def is_unlocked(p: "_profile.Profile", item: Cosmetic) -> bool:
    return p.overall_level() >= item.min_level


def equipped_id(p: "_profile.Profile", category: str) -> str:
    chosen = p.equipped.get(category)
    it = get(chosen) if chosen else None
    if it is not None and it.category == category and is_owned(p, it):
        return chosen
    return default_id(category)


def equipped_item(p: "_profile.Profile", category: str) -> Cosmetic:
    return get(equipped_id(p, category))


def params(category: str) -> dict:
    """Equipped item's params for the global profile — the renderer entry point."""
    return equipped_item(_profile.get_profile(), category).params


def equip(p: "_profile.Profile", item: Cosmetic) -> bool:
    if not is_owned(p, item) or not is_unlocked(p, item):
        return False
    p.equipped[item.category] = item.id
    _profile.mark_dirty()
    return True


def buy(p: "_profile.Profile", item: Cosmetic) -> bool:
    # Acquire a non-default item: needs its level gate met and coins for its price
    # (price 0 level-unlocks are simply claimed). Auto-equips on acquire.
    if is_owned(p, item) or not is_unlocked(p, item):
        return False
    if not p.spend(item.price):
        return False
    p.owned.add(item.id)
    p.equipped[item.category] = item.id
    _profile.mark_dirty()
    return True
