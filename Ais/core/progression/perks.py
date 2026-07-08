# Visual-only perk tiers; each carries the numeric params the emitters read.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PerkTier:
    min_level: int
    name: str
    description: str
    box_burst_n: int       # 0 -> no box-outline burst at this level
    cursor_n: int          # 0 -> no cursor sparkles at this level
    hue_amp: float         # max hue jitter on particles (capped at 0.15)
    border_anim: bool      # panel-tile border pulses at this tier and above


PERK_TIERS: Tuple[PerkTier, ...] = (
    PerkTier( 5, "Spark",   "Box-placement burst",        box_burst_n=24, cursor_n=0, hue_amp=0.00, border_anim=False),
    PerkTier(10, "Glow",    "+ Cursor sparkles",          box_burst_n=36, cursor_n=1, hue_amp=0.00, border_anim=False),
    PerkTier(15, "Sparkle", "+ Denser particles",         box_burst_n=48, cursor_n=2, hue_amp=0.05, border_anim=False),
    PerkTier(20, "Shimmer", "+ Hue jitter, pulse border", box_burst_n=56, cursor_n=2, hue_amp=0.10, border_anim=True),
    PerkTier(25, "Aurora",  "+ More particles",           box_burst_n=64, cursor_n=3, hue_amp=0.12, border_anim=True),
    PerkTier(30, "Mastery", "+ Maxed out",                box_burst_n=72, cursor_n=4, hue_amp=0.15, border_anim=True),
)


def perk_for_level(level: int) -> Optional[PerkTier]:
    # highest tier unlocked at this level, or None below tier 1
    best = None
    for p in PERK_TIERS:
        if level >= p.min_level:
            best = p
        else:
            break
    return best


def next_perk_for_level(level: int) -> Optional[PerkTier]:
    for p in PERK_TIERS:
        if level < p.min_level:
            return p
    return None
