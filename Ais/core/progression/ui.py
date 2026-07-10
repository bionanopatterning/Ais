# ImGui helpers: XP HUD, profile panel, the centre level-up celebration, and
# the perk-gated cursor/box particle emitters. The editor's loop drives these.
from __future__ import annotations

import colorsys
import math
import os
import time
from typing import List, Optional, Tuple

import imgui

import Ais.core.config as cfg
from . import background
from . import cosmetics
from . import events
from . import orbs
from . import particles
from . import perks
from . import profile as _profile


Color = Tuple[float, float, float]


# Skill level at which the visual "perks" (cursor sparkles, etc.) unlock.
PERK_UNLOCK_LEVEL = 5

# 8-direction outline offsets for big-text rendering.
_OUTLINE_OFFSETS_8 = ((-2, -2), (0, -2), (2, -2), (-2, 0), (2, 0), (-2, 2), (0, 2), (2, 2))

# Editor passes the panel logo texture handle on startup.
_panel_icon_renderer_id: Optional[int] = None

# Large font baked by the editor for crisp level-up text; None -> upscale the
# default font (blurry fallback). See big_font_path().
_big_font = None
BIG_FONT_PX = 64


def set_panel_icon(renderer_id: int) -> None:
    global _panel_icon_renderer_id
    _panel_icon_renderer_id = renderer_id


def set_big_font(font) -> None:
    global _big_font
    _big_font = font


def big_font_path() -> Optional[str]:
    # matplotlib bundles DejaVuSans on every platform; fall back to a system font
    try:
        import matplotlib
        p = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    for p in ("C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(p):
            return p
    return None


_active_levelup: Optional[events.LevelUp] = None
_levelup_started_at: float = 0.0
LEVELUP_DURATION_S = 3.2

# Firework pops queued at (fire_time, x, y, color, n) and drained each frame so
# they go off in sequence rather than all at once.
_pending_bursts: List[Tuple[float, float, float, Color, int]] = []

_panel_open: bool = False


def toggle_panel() -> None:
    global _panel_open
    _panel_open = not _panel_open


def is_panel_open() -> bool:
    return _panel_open


def tick_particles(dt: float) -> None:
    particles.tick(dt)
    orbs.tick(dt)


def draw_particles(camera=None, screen_h: float = 0.0) -> None:
    particles.draw(camera, screen_h)
    orbs.draw()


def emit_brush_trail(cx: float, cy: float, radius_px: float, color: Color, skill_level: int) -> None:
    if not cfg.settings.get("PERK_CURSOR", True):
        return
    tier = perks.perk_for_level(skill_level)
    if tier is None or tier.cursor_n <= 0:
        return
    prm = cosmetics.params(cosmetics.CURSOR)
    particles.emit_brush_ring(cx, cy, radius_px, color, n=tier.cursor_n, h_amp=tier.hue_amp, world=True,
                              palette=prm.get("palette", "feature"), size_mul=prm.get("size_mul", 1.0))


def emit_box_burst(cx: float, cy: float, size_px: float, color: Color, skill_level: int) -> None:
    if not cfg.settings.get("PERK_BOX_BURST", True):
        return
    tier = perks.perk_for_level(skill_level)
    if tier is None or tier.box_burst_n <= 0:
        return
    prm = cosmetics.params(cosmetics.BURST)
    particles.emit_box_outline_burst(cx, cy, size_px, color, n=max(1, tier.box_burst_n // 2), h_amp=tier.hue_amp, world=True,
                                     palette=prm.get("palette", "feature"), size_mul=prm.get("size_mul", 1.0))


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _luminance(c: Color) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _readable_on(bg: Color) -> Tuple[Color, Color]:
    if _luminance(bg) > 0.55:
        return ((0.06, 0.06, 0.10), (1.0, 1.0, 1.0))
    return ((0.97, 0.97, 0.98), (0.0, 0.0, 0.0))


def _draw_text_outlined(dl, x: float, y: float, text: str, fg: Color, outline: Color, outline_alpha: float = 0.7) -> None:
    out_u = imgui.get_color_u32_rgba(outline[0], outline[1], outline[2], outline_alpha)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dl.add_text(x + dx, y + dy, out_u, text)
    dl.add_text(x, y, imgui.get_color_u32_rgba(fg[0], fg[1], fg[2], 1.0), text)


def _tinted_badge_bg(base: Color, mix: float = 0.55) -> Color:
    return (
        _lerp(0.86, base[0], mix),
        _lerp(0.86, base[1], mix),
        _lerp(0.78, base[2], mix),
    )


def _draw_swatch(dl, x: float, y: float, size: float, color: Color) -> None:
    dl.add_rect_filled(x, y, x + size, y + size, imgui.get_color_u32_rgba(color[0], color[1], color[2], 1.0), size * 0.30)
    dl.add_rect(x, y, x + size, y + size, imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.30), size * 0.30, 0, 1.0)


def _draw_track(
    dl, x: float, y: float, w: float, h: float, frac: float,
    bg: Color, fill: Color, overlay: str = "",
) -> None:
    dl.add_rect_filled(x, y, x + w, y + h, imgui.get_color_u32_rgba(bg[0], bg[1], bg[2], 1.0), h * 0.5)
    fw = max(h, w * max(0.0, min(1.0, frac)))
    if fw > 0:
        dl.add_rect_filled(x, y, x + fw, y + h, imgui.get_color_u32_rgba(fill[0], fill[1], fill[2], 1.0), h * 0.5)
    if overlay:
        tw, th = imgui.calc_text_size(overlay)
        tx = x + (w - tw) * 0.5
        ty = y + (h - th) * 0.5 - 1
        blend_color = fill if frac >= 0.45 else bg
        fg, ol = _readable_on(blend_color)
        _draw_text_outlined(dl, tx, ty, overlay, fg, ol, outline_alpha=0.8)


def _truncate_to_width(text: str, max_w: float) -> str:
    tw, _ = imgui.calc_text_size(text)
    if tw <= max_w:
        return text
    ellip = "..."
    while text and imgui.calc_text_size(text + ellip)[0] > max_w:
        text = text[:-1]
    return text + ellip if text else ""


_XP_HUD_W = 280
_XP_HUD_ROW_H = 28
_XP_HUD_RECENT_WINDOW_S = 12.0
_XP_HUD_MAX_ROWS = 10


def render_xp_hud(window_width: int, window_height: int, hidden: bool = False) -> None:
    if hidden:
        return

    p = _profile.get_profile()

    active_names: List[str] = []
    color_overrides: dict = {}
    if cfg.se_active_frame is not None:
        seen = set()
        for f in cfg.se_active_frame.features:
            if f.title in seen:
                continue
            seen.add(f.title)
            active_names.append(f.title)
            color_overrides[f.title] = tuple(f.colour)

    recent = events.recent_features(_XP_HUD_RECENT_WINDOW_S)

    rows: List[Tuple[str, float, bool]] = []
    placed = set()
    for name, dt in recent:
        if p.is_hidden(name):
            continue
        rows.append((name, dt, name in active_names))
        placed.add(name)
    for name in active_names:
        if name in placed:
            continue
        if p.is_hidden(name):
            continue
        rows.append((name, 1e9, True))
        placed.add(name)

    rows = rows[:_XP_HUD_MAX_ROWS]
    if not rows:
        return

    n = len(rows)
    body_h = n * _XP_HUD_ROW_H + 8
    pad = 14
    win_x = window_width - _XP_HUD_W - pad
    win_y = pad + 22

    flags = (
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_NO_SCROLLBAR
        | imgui.WINDOW_NO_SAVED_SETTINGS
        | imgui.WINDOW_NO_FOCUS_ON_APPEARING
        | imgui.WINDOW_NO_NAV
        | imgui.WINDOW_NO_INPUTS
        | imgui.WINDOW_NO_BACKGROUND
    )

    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
    imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)

    imgui.set_next_window_position(win_x, win_y, imgui.ALWAYS)
    imgui.set_next_window_size(_XP_HUD_W, body_h)
    imgui.begin("##progression_xp_hud", False, flags)

    dl = imgui.get_window_draw_list()
    win_pos = imgui.get_window_position()

    theme = cosmetics.params(cosmetics.HUD)
    track_col = theme.get("track", (0.18, 0.18, 0.20))
    track_alpha = theme.get("track_alpha", 0.9)
    backdrop_a = theme.get("backdrop", 0.55)
    orb_targets = {}

    for i, (name, dt, in_frame) in enumerate(rows):
        row_top = win_pos[1] + 4 + i * _XP_HUD_ROW_H
        row_x = win_pos[0]

        if in_frame:
            row_alpha = 1.0
        else:
            fade_start = _XP_HUD_RECENT_WINDOW_S - 3.0
            row_alpha = 1.0 if dt < fade_start else max(0.0, 1.0 - (dt - fade_start) / 3.0)
        if row_alpha <= 0.0:
            continue

        gain_pulse = max(0.0, 1.0 - dt / 1.2) if dt < 1.2 else 0.0

        color = color_overrides.get(name) or p.skill_color(name)
        xp = p.skill_xp(name)
        L, into, needed = _profile.xp_into_level(xp)

        # colour box: square, spans the full row (name + bar), black border
        sw = 22.0
        sw_x = row_x
        sw_y = row_top + 3
        dl.add_rect_filled(sw_x, sw_y, sw_x + sw, sw_y + sw,
                           imgui.get_color_u32_rgba(color[0], color[1], color[2], row_alpha), 0.0)
        dl.add_rect(sw_x, sw_y, sw_x + sw, sw_y + sw,
                    imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.85 * row_alpha), 0.0, 0, 1.5)

        content_x = sw_x + sw + 8

        lv_text = f"level {L}"
        lv_tw, lv_th = imgui.calc_text_size(lv_text)
        lv_x = row_x + _XP_HUD_W - 4 - lv_tw

        name_x = content_x
        name_max_w = lv_x - name_x - 8
        name_text = _truncate_to_width(name, name_max_w)
        name_fg = (0.0, 0.0, 0.0)
        if gain_pulse > 0:
            name_fg = (
                _lerp(0.0, color[0], 0.7 * gain_pulse),
                _lerp(0.0, color[1], 0.7 * gain_pulse),
                _lerp(0.0, color[2], 0.7 * gain_pulse),
            )
        _draw_text_outlined(dl, name_x, row_top + 3, name_text, name_fg, (1.0, 1.0, 1.0), outline_alpha=0.85 * row_alpha)
        _draw_text_outlined(dl, lv_x, row_top + 3, lv_text, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), outline_alpha=0.85 * row_alpha)

        # progress bar: square, themed track over a black backdrop for contrast
        bar_x = content_x
        bar_y = row_top + 20
        bar_w = row_x + _XP_HUD_W - bar_x - 4
        bar_h = 6.0
        frac = (into / needed) if needed else 1.0
        orb_targets[name] = (bar_x + 6.0, bar_y + bar_h * 0.5)
        flash = max(gain_pulse, orbs.impact_pulse(name))   # brighten on gain and on orb landing
        dl.add_rect_filled(bar_x - 1, bar_y - 1, bar_x + bar_w + 1, bar_y + bar_h + 1,
                           imgui.get_color_u32_rgba(0.0, 0.0, 0.0, backdrop_a * row_alpha), 0.0)
        dl.add_rect_filled(bar_x, bar_y, bar_x + bar_w, bar_y + bar_h,
                           imgui.get_color_u32_rgba(track_col[0], track_col[1], track_col[2], track_alpha * row_alpha), 0.0)
        fill_w = max(bar_h, bar_w * max(0.0, min(1.0, frac)))
        if fill_w > 0:
            fill = color
            if flash > 0:
                fill = (
                    _lerp(color[0], 1.0, 0.40 * flash),
                    _lerp(color[1], 1.0, 0.40 * flash),
                    _lerp(color[2], 1.0, 0.40 * flash),
                )
            dl.add_rect_filled(bar_x, bar_y, bar_x + fill_w, bar_y + bar_h,
                               imgui.get_color_u32_rgba(fill[0], fill[1], fill[2], row_alpha), 0.0)
        if flash > 0:
            dl.add_rect_filled(bar_x, bar_y, bar_x + bar_w, bar_y + bar_h,
                               imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.30 * flash), 0.0)

    orbs.set_targets(orb_targets)
    imgui.end()
    imgui.pop_style_var(2)


def render_level_up(window_width: int, window_height: int, hidden: bool = False) -> None:
    global _active_levelup, _levelup_started_at
    if hidden:
        return
    now = time.time()
    # fire any firework pops whose scheduled time has arrived
    if _pending_bursts:
        for (_t, _bx, _by, _bc, _bn) in [b for b in _pending_bursts if b[0] <= now]:
            particles.emit_burst(_bx, _by, _bc, _bn)
        _pending_bursts[:] = [b for b in _pending_bursts if b[0] > now]
    if _active_levelup is None:
        ev = events.pop_level_up()
        if ev is None:
            return
        background.notify_levelup(ev)   # whole-field colour swell in the background
        if cfg.settings.get("PERK_CONFETTI", True):
            _cp = cosmetics.params(cosmetics.CONFETTI)
            particles.emit_confetti(window_width, ev.color, n=70,
                                    palette=_cp.get("palette", "feature"),
                                    shape=_cp.get("shape", "rect"),
                                    size_mul=_cp.get("size_mul", 1.0))
        # confetti fires regardless; the centre text is opt-out via PERK_MILESTONE
        if not cfg.settings.get("PERK_MILESTONE", True):
            return
        _active_levelup = ev
        _levelup_started_at = now
        # a sequence of firework pops around the text, ~140 ms apart
        if cfg.settings.get("PERK_CONFETTI", True):
            _bx, _by = window_width * 0.5, window_height * 0.5
            _spread = window_width * 0.11
            _plan = ((0.0, 0.0, 48), (-1.0, -0.02, 32), (1.0, -0.02, 32),
                     (-0.5, 0.06, 26), (0.5, 0.06, 26))
            for _i, (_ox, _oy, _bn) in enumerate(_plan):
                _pending_bursts.append((now + _i * 0.14,
                                        _bx + _spread * _ox,
                                        _by + window_height * _oy,
                                        ev.color, _bn))

    elapsed = now - _levelup_started_at
    if elapsed > LEVELUP_DURATION_S:
        _active_levelup = None
        return

    ev = _active_levelup

    # phases:
    # 0.00-0.45: scale 0 → 1.25 (overshoot)  with elastic-ish curve, alpha 0 → 1
    # 0.45-2.20: hold at scale 1.0, alpha 1
    # 2.20-3.20: alpha 1 → 0, scale 1.0 → 1.10
    if elapsed < 0.45:
        t = elapsed / 0.45
        if t < 0.7:
            scale = 1.25 * _ease_out_cubic(t / 0.7)
        else:
            u = (t - 0.7) / 0.3
            scale = 1.25 - 0.25 * _ease_out_cubic(u)
        alpha = min(1.0, t * 2.0)
    elif elapsed < 2.20:
        scale = 1.0
        alpha = 1.0
    else:
        t = (elapsed - 2.20) / 1.00
        scale = 1.0 + 0.10 * t
        alpha = max(0.0, 1.0 - t)

    alpha = max(0.0, min(1.0, alpha))

    text = f"LEVEL {ev.new_level}"
    use_big = _big_font is not None
    if use_big:
        imgui.push_font(_big_font)
    # imgui asserts scale > 0 (first frame is 0 from the ease-in). With the baked
    # font cur_scale rests at 1.0 for crisp 1:1 text; else upscale the default.
    if use_big:
        cur_scale = max(0.01, scale)
        sub_scale = 0.42
    else:
        cur_scale = max(0.01, 4.2 * scale)
        sub_scale = 1.6

    base_tw, base_th = imgui.calc_text_size(text)
    big_tw = base_tw * cur_scale
    big_th = base_th * cur_scale

    sub_text = _truncate_to_width(ev.name, 460)
    sub_base_tw, sub_base_th = imgui.calc_text_size(sub_text)
    sub_tw = sub_base_tw * sub_scale
    sub_th = sub_base_th * sub_scale

    win_w = max(big_tw, sub_tw) + 80
    win_h = big_th + sub_th + 40
    win_x = (window_width - win_w) / 2
    win_y = (window_height - win_h) / 2

    flags = (
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_NO_SCROLLBAR
        | imgui.WINDOW_NO_SAVED_SETTINGS
        | imgui.WINDOW_NO_FOCUS_ON_APPEARING
        | imgui.WINDOW_NO_NAV
        | imgui.WINDOW_NO_INPUTS
        | imgui.WINDOW_NO_BACKGROUND
    )
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
    imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)
    imgui.set_next_window_position(int(win_x), int(win_y), imgui.ALWAYS)
    imgui.set_next_window_size(int(win_w), int(win_h))
    imgui.begin("##progression_milestone", False, flags)

    imgui.set_window_font_scale(cur_scale)
    text_offset_x = (win_w - big_tw) / 2
    text_offset_y = 16
    for ox, oy in _OUTLINE_OFFSETS_8:
        imgui.set_cursor_pos((text_offset_x + ox, text_offset_y + oy))
        imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 0.0, 0.0, alpha * 0.55)
        imgui.text(text)
        imgui.pop_style_color(1)
    halo_color = (
        min(1.0, ev.color[0] + 0.4),
        min(1.0, ev.color[1] + 0.4),
        min(1.0, ev.color[2] + 0.4),
    )
    for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        imgui.set_cursor_pos((text_offset_x + ox, text_offset_y + oy))
        imgui.push_style_color(imgui.COLOR_TEXT, halo_color[0], halo_color[1], halo_color[2], alpha * 0.35)
        imgui.text(text)
        imgui.pop_style_color(1)
    imgui.set_cursor_pos((text_offset_x, text_offset_y))
    imgui.push_style_color(imgui.COLOR_TEXT, ev.color[0], ev.color[1], ev.color[2], alpha)
    imgui.text(text)
    imgui.pop_style_color(1)

    imgui.set_window_font_scale(sub_scale)
    sub_offset_x = (win_w - sub_tw) / 2
    sub_offset_y = text_offset_y + big_th + 8
    for ox, oy in ((1, 1), (-1, -1)):
        imgui.set_cursor_pos((sub_offset_x + ox, sub_offset_y + oy))
        imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 0.0, 0.0, alpha * 0.5)
        imgui.text(sub_text)
        imgui.pop_style_color(1)
    imgui.set_cursor_pos((sub_offset_x, sub_offset_y))
    imgui.push_style_color(imgui.COLOR_TEXT, 0.20, 0.20, 0.24, alpha)
    imgui.text(sub_text)
    imgui.pop_style_color(1)

    imgui.set_window_font_scale(1.0)
    imgui.end()
    imgui.pop_style_var(2)
    if use_big:
        imgui.pop_font()


def render_profile_panel() -> None:
    global _panel_open
    if not _panel_open:
        return

    imgui.set_next_window_size(560, 640, imgui.FIRST_USE_EVER)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (16, 14))
    imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, cfg.WINDOW_ROUNDING)
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *cfg.COLOUR_WINDOW_BACKGROUND)
    imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *cfg.COLOUR_TITLE_BACKGROUND)
    imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *cfg.COLOUR_TITLE_BACKGROUND)
    imgui.push_style_color(imgui.COLOR_BORDER, *cfg.COLOUR_FRAME_EXTRA_DARK[:3], 0.85)
    imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT)

    expanded, opened = imgui.begin(
        "Progression##progression_panel",
        True,
        imgui.WINDOW_NO_COLLAPSE,
    )
    if not opened:
        _panel_open = False
    if not expanded:
        imgui.end()
        imgui.pop_style_color(5)
        imgui.pop_style_var(2)
        return

    p = _profile.get_profile()
    _render_header_card(p)
    imgui.dummy(0, 10)

    visible = p.visible_skills()
    _section_label("SKILLS")
    if not visible:
        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT_DISABLED)
        imgui.text("  Annotate a feature to begin a skill.")
        imgui.pop_style_color(1)
    else:
        _render_skill_grid(visible, p)

    imgui.dummy(0, 14)
    _render_shop(p)

    imgui.dummy(0, 14)
    _render_perk_toggles()

    imgui.end()
    imgui.pop_style_color(5)
    imgui.pop_style_var(2)


def _render_shop_item(p: _profile.Profile, it, equipped_id: str) -> None:
    is_eq = (it.id == equipped_id)
    owned = cosmetics.is_owned(p, it)
    unlocked = cosmetics.is_unlocked(p, it)
    affordable = p.can_afford(it.price)

    if is_eq:
        label, btn, txt = f"* {it.name}", (0.20, 0.48, 0.32), (1.0, 1.0, 1.0)
    elif owned:
        label, btn, txt = it.name, cfg.COLOUR_FRAME_DARK[:3], (0.10, 0.10, 0.12)
    elif not unlocked:
        label, btn, txt = f"{it.name}  Lv{it.min_level}", (0.86, 0.86, 0.84), (0.55, 0.55, 0.58)
    elif it.price > 0 and not affordable:
        label, btn, txt = f"{it.name}  {it.price}c", (0.86, 0.86, 0.84), (0.55, 0.55, 0.58)
    elif it.price > 0:
        label, btn, txt = f"{it.name}  {it.price}c", (0.85, 0.72, 0.30), (0.15, 0.12, 0.05)
    else:
        label, btn, txt = f"{it.name}  claim", (0.55, 0.72, 0.85), (0.10, 0.12, 0.16)

    imgui.push_style_color(imgui.COLOR_BUTTON, btn[0], btn[1], btn[2], 1.0)
    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, min(1.0, btn[0] + 0.08), min(1.0, btn[1] + 0.08), min(1.0, btn[2] + 0.08), 1.0)
    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, btn[0], btn[1], btn[2], 1.0)
    imgui.push_style_color(imgui.COLOR_TEXT, txt[0], txt[1], txt[2], 1.0)
    if imgui.button(f"{label}##shop_{it.id}"):
        if owned:
            cosmetics.equip(p, it)
        elif unlocked and (it.price == 0 or affordable):
            cosmetics.buy(p, it)
    imgui.pop_style_color(4)


def _render_shop(p: _profile.Profile) -> None:
    _section_label("SHOP")
    imgui.push_style_color(imgui.COLOR_TEXT, 0.62, 0.50, 0.16, 1.0)
    imgui.text(f"{p.coins} coins")
    imgui.pop_style_color(1)
    imgui.spacing()
    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 3.0)
    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (7, 3))
    for cat in (cosmetics.BACKGROUND, cosmetics.CURSOR, cosmetics.ORB, cosmetics.BURST, cosmetics.CONFETTI, cosmetics.HUD):
        if len(cosmetics.CATALOG[cat]) <= 1:
            continue   # nothing to choose (only the feature-coloured default)
        imgui.push_style_color(imgui.COLOR_TEXT, 0.40, 0.38, 0.34, 1.0)
        imgui.text(cosmetics.CATEGORY_LABELS[cat])
        imgui.pop_style_color(1)
        eq = cosmetics.equipped_id(p, cat)
        for j, it in enumerate(cosmetics.CATALOG[cat]):
            if j > 0:
                imgui.same_line()
            _render_shop_item(p, it, eq)
        imgui.spacing()
    imgui.pop_style_var(2)


def _render_perk_toggles() -> None:
    _section_label("PERKS")
    items = (
        ("PERK_CURSOR",     "Cursor sparkles"),
        ("PERK_XP_ORBS",    "XP orbs"),
        ("PERK_BOX_BURST",  "Box-placement burst"),
        ("PERK_CONFETTI",   "Level-up confetti"),
        ("PERK_MILESTONE",  "Level-up celebration"),
        ("PERK_COLOR_ANIM", "Color animations"),
    )
    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 2.0)
    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (4, 3))
    imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (12, 9))
    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_FRAME_DARK)
    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_FRAME_EXTRA_DARK)
    imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.10, 0.12, 0.16, 1.0)
    col_w = imgui.get_content_region_available_width() * 0.5
    for i, (key, label) in enumerate(items):
        cur = bool(cfg.settings.get(key, True))
        changed, new_val = imgui.checkbox(f"{label}##{key}", cur)
        if changed:
            cfg.edit_setting(key, bool(new_val))
        if i % 2 == 0 and i != len(items) - 1:
            imgui.same_line(col_w)
    imgui.pop_style_color(3)
    imgui.pop_style_var(3)


def _render_header_card(p: _profile.Profile) -> None:
    dl = imgui.get_window_draw_list()
    avail_w = imgui.get_content_region_available_width()
    x0, y0 = imgui.get_cursor_screen_pos()
    card_h = 64.0
    dl.add_rect_filled(
        x0, y0, x0 + avail_w, y0 + card_h,
        imgui.get_color_u32_rgba(*cfg.COLOUR_TITLE_BACKGROUND[:3], 1.0),
        cfg.WINDOW_ROUNDING,
    )
    dl.add_rect(
        x0, y0, x0 + avail_w, y0 + card_h,
        imgui.get_color_u32_rgba(*cfg.COLOUR_FRAME_EXTRA_DARK[:3], 1.0),
        cfg.WINDOW_ROUNDING, 0, 1.0,
    )

    icon_size = 40
    icon_x = x0 + 12
    icon_y = y0 + (card_h - icon_size) * 0.5
    if _panel_icon_renderer_id is not None:
        dl.add_image(
            _panel_icon_renderer_id,
            (icon_x, icon_y), (icon_x + icon_size, icon_y + icon_size),
            (0.0, 0.0), (1.0, 1.0),
            imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0),
        )
    name_x = icon_x + icon_size + 12
    dl.add_text(name_x, y0 + 22, imgui.get_color_u32_rgba(0.10, 0.18, 0.45, 1.0), "Ais")

    overall = p.overall_level()
    big = f"Total Level   {overall}"
    big_tw, _ = imgui.calc_text_size(big)
    dl.add_text(x0 + avail_w - big_tw - 18, y0 + 16, imgui.get_color_u32_rgba(0.16, 0.20, 0.36, 1.0), big)

    coins_text = f"{p.coins} coins"
    ct_w, _ = imgui.calc_text_size(coins_text)
    dl.add_text(x0 + avail_w - ct_w - 18, y0 + 36, imgui.get_color_u32_rgba(0.62, 0.50, 0.16, 1.0), coins_text)

    imgui.dummy(avail_w, card_h)


def _section_label(label: str) -> None:
    imgui.push_style_color(imgui.COLOR_TEXT, 0.36, 0.34, 0.30, 1.0)
    imgui.text(label)
    imgui.pop_style_color(1)
    imgui.separator()
    imgui.spacing()


_TILE_W = 128
_TILE_H = 116
_TILE_SPACING = 10


def _render_skill_grid(visible: dict, p: _profile.Profile) -> None:
    dl = imgui.get_window_draw_list()
    avail_w = imgui.get_content_region_available_width()
    cols = max(1, int((avail_w + _TILE_SPACING) // (_TILE_W + _TILE_SPACING)))
    items = sorted(visible.items(), key=lambda kv: (-_profile.level_for_xp(kv[1]), kv[0]))
    n = len(items)
    rows = (n + cols - 1) // cols
    grid_h = rows * (_TILE_H + _TILE_SPACING) - _TILE_SPACING if rows else 0

    base_x, base_y = imgui.get_cursor_screen_pos()
    grid_x_offset = (avail_w - (cols * _TILE_W + (cols - 1) * _TILE_SPACING)) * 0.5

    for i, (name, xp) in enumerate(items):
        col = i % cols
        row = i // cols
        tx = base_x + grid_x_offset + col * (_TILE_W + _TILE_SPACING)
        ty = base_y + row * (_TILE_H + _TILE_SPACING)
        _draw_skill_tile(dl, tx, ty, name, xp, p.skill_color(name))
        imgui.set_cursor_screen_pos((tx + _TILE_W - 22, ty + 4))
        if imgui.invisible_button(f"##hide_tile_{name}", 18, 18):
            p.hide(name)
            _profile.mark_dirty()
        if imgui.is_item_hovered():
            with imgui.begin_tooltip():
                imgui.text("Hide this skill")

    imgui.dummy(avail_w, grid_h)


def _draw_skill_tile(dl, tx: float, ty: float, name: str, xp: int, base_color: Color) -> None:
    L, into, needed = _profile.xp_into_level(xp)
    tier = perks.perk_for_level(L)
    border_anim = bool(tier and tier.border_anim and cfg.settings.get("PERK_COLOR_ANIM", True))

    # Tile background — cream. Border is in the feature color, optionally
    # animated (alpha pulse + subtle hue cycle within the perk's hue_amp) at
    # higher skill tiers.
    bg_u = imgui.get_color_u32_rgba(*cfg.COLOUR_TITLE_BACKGROUND[:3], 1.0)
    if border_anim:
        t = time.time()
        alpha = 0.65 + 0.30 * (0.5 + 0.5 * math.sin(t * 1.6))
        h_amp = min(0.15, tier.hue_amp) if tier else 0.0
        h, s, v = colorsys.rgb_to_hsv(*base_color)
        h = (h + h_amp * math.sin(t * 1.2)) % 1.0
        br, bg, bb = colorsys.hsv_to_rgb(h, s, v)
    else:
        alpha = 0.85
        br, bg, bb = base_color
    border_u = imgui.get_color_u32_rgba(br, bg, bb, alpha)
    dl.add_rect_filled(tx, ty, tx + _TILE_W, ty + _TILE_H, bg_u, 8.0)
    dl.add_rect(tx, ty, tx + _TILE_W, ty + _TILE_H, border_u, 8.0, 0, 2.0)

    dl.add_rect_filled(tx + 8, ty + 6, tx + _TILE_W - 8, ty + 9, border_u, 1.5)

    _draw_swatch(dl, tx + 8, ty + 16, 12.0, base_color)

    name_max_w = _TILE_W - 30
    name_text = _truncate_to_width(name, name_max_w)
    name_color = (
        max(0.0, min(0.55, base_color[0] * 0.65)),
        max(0.0, min(0.55, base_color[1] * 0.65)),
        max(0.0, min(0.55, base_color[2] * 0.65)),
    )
    _, name_th = imgui.calc_text_size(name_text)
    dl.add_text(tx + 24, ty + 16 + (12 - name_th) * 0.5, imgui.get_color_u32_rgba(name_color[0], name_color[1], name_color[2], 1.0), name_text)

    glyph = "x"
    g_tw, g_th = imgui.calc_text_size(glyph)
    dl.add_text(tx + _TILE_W - 22 + (18 - g_tw) * 0.5, ty + 4 + (18 - g_th) * 0.5, imgui.get_color_u32_rgba(0.50, 0.50, 0.54, 0.85), glyph)

    level_text = f"Level {L}"
    lvl_tw, lvl_th = imgui.calc_text_size(level_text)
    lvl_x = tx + (_TILE_W - lvl_tw) * 0.5
    lvl_y = ty + 40
    out_u = imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.30)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dl.add_text(lvl_x + dx, lvl_y + dy, out_u, level_text)
    dl.add_text(lvl_x, lvl_y, imgui.get_color_u32_rgba(base_color[0] * 0.45, base_color[1] * 0.45, base_color[2] * 0.45, 1.0), level_text)

    bar_x = tx + 10
    bar_y = ty + _TILE_H - 22
    bar_w = _TILE_W - 20
    bar_h = 8.0
    if needed == 0:
        overlay = "MAX"
        frac = 1.0
    else:
        overlay = f"{into:,} / {needed:,}"
        frac = into / needed
    _draw_track(dl, bar_x, bar_y, bar_w, bar_h, frac, bg=(0.80, 0.80, 0.74), fill=base_color, overlay=overlay)
