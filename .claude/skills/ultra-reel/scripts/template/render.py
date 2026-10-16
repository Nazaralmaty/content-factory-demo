"""Picture for the ICON testimonial reel (1080x1920, 30 fps), driven by plan.py.

python render.py frames <out_dir> <t0> <t1> [...]     -> PNG stills for QA
python render.py video <out.mp4> <audio.wav> [workers=4]
"""
import math
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
import skia

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ULTRA_ENGINE", os.path.join(HERE, "..")))  # <skill>/scripts
sys.path.insert(0, HERE)

from engine import gfx as G  # noqa: E402
from engine.gfx import (hexc, prog, ease_out_back, ease_out_expo, ease_in_out, ease_out_cubic,  # noqa: E402
                        ease_in_cubic, spring, pulse, clamp01, draw_text, paint)
import plan as PL  # noqa: E402

W, H, FPS = PL.W, PL.H, PL.FPS
FONT_DIR = os.environ.get("REEL_FONTS", os.path.expanduser("~/.cache/ultra-reel/fonts"))
F = G.Fonts(FONT_DIR, {
    "head": "Oswald-700.ttf", "head6": "Oswald-600.ttf", "punch": "DelaGothicOne-400.ttf",
    "wide": "Unbounded-900.ttf", "serif": "PlayfairDisplay-700i.ttf", "mono": "JetBrainsMono-800.ttf",
    "mono5": "JetBrainsMono-500.ttf", "cap": "Montserrat-900.ttf", "cap7": "Montserrat-700.ttf",
    "body": "Montserrat-500.ttf",
})

PAPER, CREAM, INK, RED = hexc("#EEE7DA"), hexc("#F4EBDC"), hexc("#141414"), hexc("#E3342F")
WHITE, GREY, PAPER_DK = hexc("#FFFFFF"), hexc("#8B8478"), hexc("#D9D0C1")
WORD = [w["s"] for w in PL.WORDS]
SPK = np.load(os.path.join(HERE, "speaker.npy"), mmap_mode="r")
SRC_INSET = (4, 4, 396, 396)
FACE = (196, 168)  # face centre in source pixels (for close-ups)

# ------------------------------------------------------------------ shared timing helpers
KICKS = []
for n in range(1, 22):
    sec = next((k for k, (a, b) in PL.SECTIONS.items() if a <= n < b), None)
    if sec in ("intro", "lift", "verse", "level", "b", "drop", "final", "build"):
        for b in range(4):
            t = PL.bar(n, b)
            if (n == 13 and t >= PL.TAPE_STOP - 0.01) or (n == 21 and t >= PL.FINAL_STOP - 0.01):
                continue
            if n == 8 and b in (1, 3):
                continue
            KICKS.append(t)
KICKS = np.array(KICKS)
IMPACTS = [(0.0, 26), (0.26, 22), (0.50, 30), (4.36, 10), (4.93, 14), (6.04, 24), (11.12, 10),
           (15.83, 14), (WORD[69], 34), (27.68, 40), (WORD[83], 12), (36.02, 16), (37.295, 26),
           (PL.TOP, 46)]


def kick_env(t, decay=0.14):
    i = np.searchsorted(KICKS, t, side="right") - 1
    if i < 0:
        return 0.0
    return math.exp(-(t - KICKS[i]) / decay)


def shake(t):
    amp = 0.0
    for t0, a in IMPACTS:
        if 0 <= t - t0 < 0.6:
            amp += a * math.exp(-(t - t0) / 0.12)
    if amp < 0.3:
        return 0.0, 0.0, 0.0
    k = t * 61.0
    return (amp * math.sin(k * 1.7 + 0.3) * 0.9, amp * math.cos(k * 2.3 + 1.1) * 0.7,
            amp * 0.04 * math.sin(k * 1.1))


def chroma_amount(t):
    a = 0.0
    for t0, s in ((0.0, 6), (6.04, 8), (14.62, 10), (WORD[69], 8), (27.68, 14), (37.295, 10), (PL.TOP, 16)):
        if t >= t0:
            a = max(a, s * math.exp(-(t - t0) / 0.16))
    return a


# ------------------------------------------------------------------ assets
_BG = {}


def bg_image(kind):
    if kind not in _BG:
        base = {"paper": (238, 231, 218), "cream": (244, 235, 220), "ink": (22, 21, 20), "red": (227, 52, 47)}[kind]
        _BG[kind] = G.paper_texture(H, W, base, seed=hash(kind) % 100, strength=1.0 if kind != "ink" else 1.6)
    return _BG[kind]


def bg(c, kind):
    c.drawImage(bg_image(kind), 0, 0)


@lru_cache(maxsize=8)
def spk_img(idx):
    return G.image(np.asarray(SPK[idx]))


def frame_idx(t):
    return int(min(len(SPK) - 1, max(0, round(t * FPS))))


def speaker(c, t, cx, cy, r, zoom=1.0, face=False, alpha=1.0, cf=None, ring=True, ring_col=None,
            misreg=True, shadow=True, pump=0.0):
    r = r * (1 + pump * kick_env(t) * 0.025)
    img = spk_img(frame_idx(t))
    if shadow:
        c.drawCircle(cx + 6, cy + 16, r, paint(hexc("#000000"), alpha=0.22 * alpha,
                                               filt=skia.ImageFilters.Blur(22, 22)))
    if misreg:
        c.drawCircle(cx + 12, cy + 10, r + 10, paint(ring_col or RED, stroke=7, alpha=alpha))
    off = (0, 0)
    if face:
        sc = 2 * r * zoom / (SRC_INSET[2] - SRC_INSET[0])
        off = (-(FACE[0] - 200) * sc * 0.85, -(FACE[1] - 200) * sc * 0.85)
    G.draw_circle_image(c, img, cx, cy, r, src=SRC_INSET, alpha=alpha, color_filter=cf, zoom=zoom, offset=off)
    if ring:
        c.drawCircle(cx, cy, r + 10, paint(INK, stroke=5, alpha=alpha))


@lru_cache(maxsize=4)
def halftone_img(idx, size, inv):
    f = np.asarray(SPK[idx]).astype(np.float32) / 255
    gray = f[..., 0] * 0.3 + f[..., 1] * 0.59 + f[..., 2] * 0.11
    gray = np.clip((gray - 0.12) / 0.8, 0, 1) ** 1.15
    if inv:
        rgb = G.halftone(gray, size, cell=12, ink=(22, 21, 20), paper=(238, 231, 218))
    else:
        rgb = G.halftone(gray, size, cell=12, ink=(20, 20, 20), paper=(238, 231, 218))
    return G.image(rgb)


def speaker_halftone(c, t, cx, cy, r, alpha=1.0):
    img = halftone_img(frame_idx(t), int(2 * r), False)
    c.save()
    c.clipPath(G.circle_path(cx, cy, r), doAntiAlias=True)
    p = skia.Paint(AntiAlias=True)
    p.setAlphaf(alpha)
    c.drawImageRect(img, skia.Rect.MakeLTRB(cx - r, cy - r, cx + r, cy + r), G.CUBIC, p)
    c.restore()


# ------------------------------------------------------------------ reusable graphic atoms
def label(c, text, x, y, size, fg, bgc, pad=(22, 12), rot=0.0, font="mono", alpha=1.0, align="left",
          radius=10, tracking=0.04, shadow=True):
    f = F(font, size)
    w = G.text_width(text, f, tracking)
    ch = size * 0.74
    bw, bh = w + pad[0] * 2, ch + pad[1] * 2
    x0 = x - bw / 2 if align == "center" else x
    c.save()
    c.translate(x0 + bw / 2, y)
    c.rotate(rot)
    filt = G.shadow(0, 8, 12, 0.22) if shadow else None
    G.rrect(c, -bw / 2, -bh / 2, bw, bh, radius, bgc, alpha=alpha, filt=filt)
    draw_text(c, text, -w / 2, ch / 2, f, fg, tracking=tracking, alpha=alpha)
    c.restore()
    return bw


def slam_text(c, t, t0, text, x, y, font, color, align="center", from_scale=1.45, dur=0.14, tracking=0.0,
              alpha=1.0, rot=0.0):
    k = prog(t, t0, dur)
    if t < t0:
        return
    s = from_scale + (1 - from_scale) * ease_out_expo(k)
    w = G.text_width(text, font, tracking)
    ax = x if align == "center" else (x + w / 2 if align == "left" else x - w / 2)
    c.save()
    c.translate(ax, y - font.getSize() * 0.35)
    c.rotate(rot * (1 - ease_out_expo(k)))
    c.scale(s, s)
    draw_text(c, text, 0, font.getSize() * 0.35, font, color, align="center", tracking=tracking,
              alpha=alpha * clamp01(k * 4))
    c.restore()


def kinetic(c, t, t0, text, x, y, font, color, mode="rise", align="left", stagger=0.03, dur=0.3, tracking=0.0,
            alpha=1.0, seed=0):
    if t < t0:
        return 0
    return G.draw_glyphs_kinetic(c, text, x, y, font, color, t - t0, stagger=stagger, dur=dur, mode=mode,
                                 tracking=tracking, align=align, alpha=alpha, seed=seed)


def marker_underline(c, t, t0, x0, x1, y, color=RED, width=16, dur=0.3, seed=4):
    k = ease_out_cubic(prog(t, t0, dur))
    if k > 0:
        G.stroke_path(c, G.scribble_path(x0, x1, y, amp=7, waves=1.6, seed=seed), color, width, progress=k,
                      alpha=0.95)


def header_rule(c, t, t0, y, color=INK):
    k = ease_out_expo(prog(t, t0, 0.4))
    c.drawLine(80, y, 80 + 920 * k, y, paint(color, stroke=3))


def info_bar(c, t, dark=False, alpha=1.0):
    """Persistent top strip: format tag + names, like an editorial running head."""
    col = PAPER if dark else INK
    f = F("mono", 26)
    draw_text(c, "ОТЗЫВ", 80, 196, f, PAPER if dark else RED, tracking=0.12, alpha=alpha)
    draw_text(c, "CONVERTMONSTER → ICON", 1000, 196, f, col, align="right", tracking=0.06, alpha=alpha * 0.85)
    c.drawLine(80, 214, 1000, 214, paint(col, stroke=2, alpha=0.5 * alpha))


def rings_pulse(c, t, cx, cy, r0, color, times, life=0.7, spread=260, width=4):
    for t0 in times:
        if 0 <= t - t0 < life:
            k = (t - t0) / life
            c.drawCircle(cx, cy, r0 + spread * ease_out_cubic(k), paint(color, stroke=width * (1 - k) + 0.5,
                                                                        alpha=(1 - k) * 0.9))


def icon_logo(c, t, cx, cy, size, color, t_in, speaker_in_o=True, o_scale=1.0, pump=0.0, dot=0.0):
    """IC◯N wordmark; the O is the live speaker circle (dot>0 morphs it into a solid red dot)."""
    f = F("wide", size)
    w_ic = G.text_width("IC", f)
    w_n = G.text_width("N", f)
    capH = size * 0.72
    r = capH * 0.56 * o_scale
    gap = size * 0.07
    total = w_ic + gap + 2 * r + gap + w_n
    x0 = cx - total / 2
    base = cy + capH / 2
    k = spring(prog(t, t_in, 0.9) * 0.9, 2.2, 6.5) if t >= t_in else 0
    if k <= 0:
        return
    c.save()
    s = (0.75 + 0.25 * k) * (1 + pump * kick_env(t) * 0.035)
    c.translate(cx, cy)
    c.scale(s, s)
    c.translate(-cx, -cy)
    draw_text(c, "IC", x0, base, f, color)
    ox = x0 + w_ic + gap + r
    if speaker_in_o:
        speaker(c, t, ox, cy, r, ring=False, misreg=False, shadow=False)
    if dot > 0:
        c.drawCircle(ox, cy, r * (0.15 + 0.85 * ease_out_back(dot, 1.4)), paint(RED, alpha=clamp01(dot * 2)))
    c.drawCircle(ox, cy, r, paint(color, stroke=size * 0.085))
    draw_text(c, "N", ox + r + gap, base, f, color)
    c.restore()
    return ox, cy, r


# ------------------------------------------------------------------ pictograms for the tools grid
def picto(c, kind, cx, cy, s, col):
    p = paint(col, stroke=s * 0.07, cap=skia.Paint.kRound_Cap, join=skia.Paint.kRound_Join)
    if kind == "target":
        for rr in (0.42, 0.27, 0.12):
            c.drawCircle(cx, cy, s * rr, p)
        c.drawLine(cx, cy, cx + s * 0.42, cy - s * 0.42, p)
    elif kind == "chart":
        for i, hh in enumerate((0.3, 0.55, 0.42, 0.8)):
            x = cx - s * 0.36 + i * s * 0.24
            c.drawLine(x, cy + s * 0.38, x, cy + s * 0.38 - hh * s * 0.8, paint(col, stroke=s * 0.11))
    elif kind == "cursor":
        path = skia.Path()
        path.moveTo(cx - s * 0.25, cy - s * 0.38)
        path.lineTo(cx + s * 0.3, cy + s * 0.08)
        path.lineTo(cx + s * 0.04, cy + s * 0.12)
        path.lineTo(cx + s * 0.18, cy + s * 0.4)
        path.lineTo(cx + s * 0.06, cy + s * 0.45)
        path.lineTo(cx - s * 0.08, cy + s * 0.17)
        path.lineTo(cx - s * 0.25, cy + s * 0.3)
        path.close()
        c.drawPath(path, paint(col))
    elif kind == "gear":
        c.drawCircle(cx, cy, s * 0.17, p)
        for i in range(8):
            a = i * math.pi / 4
            c.drawLine(cx + math.cos(a) * s * 0.26, cy + math.sin(a) * s * 0.26,
                       cx + math.cos(a) * s * 0.4, cy + math.sin(a) * s * 0.4, paint(col, stroke=s * 0.1))
        c.drawCircle(cx, cy, s * 0.3, p)
    elif kind == "link":
        for dx in (-0.15, 0.15):
            G.rrect(c, cx + dx * s - s * 0.22, cy - s * 0.12 + dx * s * 0.6, s * 0.44, s * 0.24, s * 0.12, col,
                    stroke=s * 0.07)
    elif kind == "mega":
        path = skia.Path()
        path.moveTo(cx - s * 0.36, cy - s * 0.1)
        path.lineTo(cx + s * 0.2, cy - s * 0.36)
        path.lineTo(cx + s * 0.2, cy + s * 0.36)
        path.lineTo(cx - s * 0.36, cy + s * 0.1)
        path.close()
        c.drawPath(path, p)
        c.drawLine(cx + s * 0.32, cy - s * 0.12, cx + s * 0.42, cy - s * 0.2, p)
        c.drawLine(cx + s * 0.32, cy + s * 0.12, cx + s * 0.42, cy + s * 0.2, p)


# ================================================================== scenes
HOOK_O = (82 + 0, 1215, 128)  # (x unused) cy, r of the red O in the hook


def hook_o_center():
    fp = F("punch", 330)
    return 82 + G.text_width("Т", fp) + 14 + HOOK_O[2], HOOK_O[1]


def s_hook(c, t):
    bg(c, "paper")
    info_bar(c, t, alpha=clamp01(t / 0.1 + 0.6))
    fo = F("punch", 250)
    slam_text(c, t, -0.05, "ОНИ", 82, 700, fo, INK, align="left", rot=-4)
    ft = G.fit_font(F, "punch", "ТОЧНО", 920, 250)
    slam_text(c, t, 0.21, "ТОЧНО", 82, 980, ft, INK, align="left", rot=3)
    fp = F("punch", 330)
    if t >= 0.45:
        k = ease_out_expo(prog(t, 0.45, 0.14))
        s = 1.45 - 0.45 * k
        wt, wp = G.text_width("Т", fp), G.text_width("П", fp)
        r = HOOK_O[2]
        x0 = 82
        c.save()
        c.translate(540, 1215)
        c.scale(s, s)
        c.translate(-540, -1215)
        draw_text(c, "Т", x0, 1340, fp, INK, alpha=clamp01(k * 4))
        c.drawCircle(x0 + wt + 14 + r, 1215, r * ease_out_back(prog(t, 0.47, 0.2), 2.0), paint(RED))
        draw_text(c, "П", x0 + wt + 28 + 2 * r, 1340, fp, INK, alpha=clamp01(k * 4))
        c.restore()
    draw_text(c, "«", 60, 520, F("serif", 260), RED, alpha=clamp01(t * 10))


def s_intro(c, t):
    bg(c, "paper")
    info_bar(c, t)
    # header: the hook, now small
    k = ease_out_expo(prog(t, 0.62, 0.45))
    if k > 0:
        f = F("head", 66)
        draw_text(c, "«ОНИ ТОЧНО ТОП»", 540, 300 - (1 - k) * 60, f, INK, align="center", alpha=k, tracking=0.02)
    push = 1 + 0.07 * ease_in_out(prog(t, 3.18, 1.2))
    cx, cy, r = 540, 930, 390 * push
    speaker(c, t, cx, cy, r, pump=1.0)
    # rotating dashed orbit
    c.save()
    c.translate(cx, cy)
    c.rotate(t * 18)
    dash = skia.DashPathEffect.Make([18, 22], 0)
    c.drawCircle(0, 0, r + 46, paint(INK, stroke=3, alpha=0.55, effect=dash))
    c.restore()
    # name + company stickers
    out_k = ease_in_cubic(prog(t, 3.1, 0.25))
    k1 = spring(prog(t, WORD[3] - 0.05, 1.0), 2.0, 7.0)
    if k1 > 0:
        x = -700 + 790 * k1 - out_k * 900
        label(c, "АНТОН ПЕТРАЧЕНКОВ", x, 1390, 58, PAPER, INK, font="head", rot=-3, pad=(26, 18), tracking=0.02)
    k2 = spring(prog(t, WORD[6] - 0.05, 1.0), 2.0, 7.0)
    if k2 > 0:
        x = 1200 - 950 * k2 + out_k * 900
        label(c, "CONVERTMONSTER", x, 1490, 44, WHITE, RED, rot=2.5, pad=(22, 16), tracking=0.06)
    # "лично" — handwritten-feel kicker
    if t >= WORD[11]:
        f = F("serif", 96)
        kk = ease_out_back(prog(t, WORD[11], 0.3))
        c.save()
        c.translate(800, 1420)
        c.rotate(-8)
        c.scale(kk, kk)
        draw_text(c, "лично", 0, 0, f, RED, align="center")
        c.restore()
        marker_underline(c, t, WORD[11] + 0.12, 680, 930, 1440, RED, 9, 0.25)


def s_thank(c, t):
    bg(c, "paper")
    info_bar(c, t)
    speaker(c, t, 540, 1080, 340, pump=1.0)
    f = G.fit_font(F, "head", "ПОБЛАГОДАРИТЬ", 940, 170)
    slam_text(c, t, 4.33, "ПОБЛАГОДАРИТЬ", 540, 610, f, INK, from_scale=1.6)
    k = ease_out_expo(prog(t, 4.36, 0.3))
    fs = F("serif", 64)
    draw_text(c, "хотел бы лично", 540, 400, fs, RED, align="center", alpha=k)
    c.drawRect(skia.Rect.MakeXYWH(540 - 470 * k, 660, 940 * k, 14), paint(RED))


def s_two(c, t):
    bg(c, "paper")
    info_bar(c, t)
    t0 = 4.84
    k = ease_out_expo(prog(t, t0, 0.25))
    f2 = F("punch", 1000)
    c.save()
    c.translate(540, 900)
    c.rotate(-6 * (1 - k))
    s = 0.55 + 0.45 * k
    c.scale(s, s)
    draw_text(c, "2", 0, 360, f2, RED, align="center", alpha=clamp01(k * 3))
    c.restore()
    fh = G.fit_font(F, "head", "МАРКЕТОЛОГА", 900, 150)
    kinetic(c, t, WORD[14] - 0.04, "МАРКЕТОЛОГА", 540, 1340, fh, INK, mode="rise", align="center", stagger=0.025)
    if t >= WORD[15]:
        fs = F("serif", 70)
        draw_text(c, "из компании", 540, 1460, fs, INK, align="center", alpha=ease_out_cubic(prog(t, WORD[15], 0.2)))
    speaker(c, t, 860, 420, 135, pump=1.0)


def s_icon1(c, t):
    bg(c, "red")
    info_bar(c, t, dark=True)
    rings_pulse(c, t, 540, 960, 140, PAPER, [6.04, 6.28], life=0.6, spread=420, width=6)
    icon_logo(c, t, 540, 960, 262, PAPER, 6.0)


def s_names(c, t):
    bg(c, "paper")
    info_bar(c, t)
    k_up = ease_out_expo(prog(t, 6.46, 0.35))
    cy = 960 - (960 - 520) * k_up
    r = 125 + (230 - 125) * k_up
    speaker(c, t, 540, cy, r, pump=1.0)
    cards = [(WORD[19], "01", "А", "АБУБАКИР", 300, -3.0), (WORD[21], "02", "Е", "ЕРКЕБУЛАН", 780, 3.0)]
    for i, (t0, num, ini, name, x, rot) in enumerate(cards):
        if t < t0 - 0.05:
            continue
        # connector from the speaker to the card
        kl = ease_out_cubic(prog(t, t0 - 0.05, 0.35))
        p = skia.Path()
        p.moveTo(540, cy + r + 20)
        p.cubicTo(540, cy + r + 140, x, 760, x, 880)
        G.stroke_path(c, p, INK, 4, progress=kl, alpha=0.8)
        k = spring(prog(t, t0 - 0.05, 1.0), 2.4, 6.5)
        c.save()
        c.translate(x, 1190 + (1 - k) * 260 + 7 * math.sin(t * 2.2 + i * 1.7))
        c.rotate(rot * k + 0.8 * math.sin(t * 1.6 + i))
        sc = 0.7 + 0.3 * k
        c.scale(sc, sc)
        G.rrect(c, -205, -300, 410, 600, 26, WHITE, filt=G.shadow(0, 18, 26, 0.25))
        draw_text(c, num, -170, -240, F("mono", 34), RED, tracking=0.1)
        c.drawCircle(0, -100, 118, paint(RED if i == 0 else INK))
        draw_text(c, ini, 0, -58, F("punch", 130), PAPER, align="center")
        fn = G.fit_font(F, "head", name, 360, 76)
        draw_text(c, name, 0, 110, fn, INK, align="center")
        draw_text(c, "МАРКЕТОЛОГ · ICON", 0, 170, F("mono5", 26), GREY, align="center", tracking=0.06)
        c.drawLine(-160, 215, 160, 215, paint(INK, stroke=2, alpha=0.25))
        G.stroke_path(c, G.check_path(0, 255, 60), RED, 9, progress=prog(t, t0 + 0.25, 0.25))
        c.restore()


def s_helped(c, t):
    bg(c, "paper")
    info_bar(c, t)
    speaker(c, t, 540, 1080, 380, pump=1.0)
    f1 = F("head", 70)
    x = 540
    words = [(WORD[25], "ОНИ"), (WORD[26], "МНЕ"), (WORD[27], "ОЧЕНЬ"), (WORD[28], "СИЛЬНО")]
    line = " ".join(wd for _, wd in words)
    lw = G.text_width(line, f1)
    xx = x - lw / 2
    for t0, wd in words:
        kinetic(c, t, t0 - 0.03, wd, xx, 330, f1, INK, mode="rise", stagger=0.02)
        xx += G.text_width(wd + " ", f1)
    fp = G.fit_font(F, "punch", "ПОМОГЛИ", 940, 175)
    slam_text(c, t, WORD[29] - 0.03, "ПОМОГЛИ", 540, 560, fp, RED, from_scale=1.7)
    marker_underline(c, t, WORD[29] + 0.08, 130, 950, 615, INK, 12, 0.3, seed=7)


def s_course(c, t):
    bg(c, "paper")
    info_bar(c, t)
    k = ease_out_expo(prog(t, 11.45, 0.4))
    c.save()
    c.translate(540, 950 + (1 - k) * 180)
    c.rotate(-1.5)
    G.rrect(c, -450, -400, 900, 800, 30, WHITE, filt=G.shadow(0, 24, 30, 0.25))
    G.rrect(c, -450, -400, 900, 92, 30, INK)
    c.drawRect(skia.Rect.MakeXYWH(-450, -340, 900, 32), paint(INK))
    for i, col in enumerate((RED, PAPER, PAPER)):
        c.drawCircle(-400 + i * 36, -354, 11, paint(col))
    draw_text(c, "АКАДЕМИЯ · НОВЫЙ КУРС", 410, -342, F("mono", 26), PAPER, align="right", tracking=0.08)
    ft = F("head", 104)
    if t >= WORD[32] - 0.05:
        kinetic(c, t, WORD[32] - 0.05, "КУРС", -400, -180, ft, INK, mode="drop", stagger=0.03)
    if t >= WORD[33] - 0.05:
        kinetic(c, t, WORD[33] - 0.05, "ПО ТРАФИКУ", -400 + G.text_width("КУРС ", ft), -180, ft, RED,
                mode="drop", stagger=0.025)
    rows = ["МОДУЛЬ 01", "МОДУЛЬ 02", "МОДУЛЬ 03"]
    ticks = [PL.bar(7, 0), PL.bar(7, 1), PL.bar(7, 2)]
    for i, (row, tk) in enumerate(zip(rows, ticks)):
        ka = ease_out_expo(prog(t, 11.6 + i * 0.12, 0.35))
        if ka <= 0:
            continue
        y = -60 + i * 105
        G.rrect(c, -400, y - 40, 800 * ka, 84, 14, hexc("#F1ECE3"))
        draw_text(c, row, -360, y + 14, F("mono", 36), INK, tracking=0.05, alpha=ka)
        c.drawRect(skia.Rect.MakeXYWH(300, y - 22, 48, 48), paint(INK, stroke=4, alpha=ka))
        G.stroke_path(c, G.check_path(324, y + 2, 52), RED, 10, progress=prog(t, tk, 0.18))
    # progress bar
    pk = ease_in_out(prog(t, WORD[31], WORD[38] + 0.05 - WORD[31]))
    G.rrect(c, -400, 300, 800, 34, 17, hexc("#E4DDD0"))
    G.rrect(c, -400, 300, 800 * pk, 34, 17, RED)
    draw_text(c, f"{int(pk * 100)}%", 400, 280, F("mono", 34), INK, align="right")
    if t >= WORD[38] + 0.05:
        ka = ease_out_cubic(prog(t, WORD[38] + 0.05, 0.15))
        bw = label(c, "ГОТОВО", -300, 262, 30, PAPER, INK, align="center", radius=8, shadow=False, alpha=ka)
        G.stroke_path(c, G.check_path(-300 + bw / 2 + 34, 262, 40), RED, 8, progress=prog(t, WORD[38] + 0.1, 0.15))
    c.restore()
    speaker(c, t, 200, 430, 150, pump=1.0)
    if t >= WORD[35] - 0.05:
        f = F("serif", 66)
        draw_text(c, "в рамках нашей академии", 540, 1480, f, INK, align="center",
                  alpha=ease_out_cubic(prog(t, WORD[35] - 0.05, 0.25)))


def s_level(c, t):
    bg(c, "ink")
    info_bar(c, t, dark=True)
    cx, cy, r = 420, 900, 330
    c.drawCircle(cx + 14, cy + 12, r + 12, paint(RED, stroke=8))
    speaker_halftone(c, t, cx, cy, r)
    c.drawCircle(cx, cy, r + 12, paint(PAPER, stroke=5))
    # level meter, one bar per arpeggio note
    k0 = int(math.ceil(PL.step16(14.62)))
    steps = [PL.BAR1 + (k0 + i) * PL.BEAT / 4 for i in range(10)]
    lit = sum(1 for s in steps if t >= s)
    draw_text(c, "LVL", 900, 1290, F("mono", 34), PAPER, align="center", tracking=0.1)
    for i in range(10):
        y = 1230 - i * 62
        on = i < lit
        col = RED if on else hexc("#3A3734")
        G.rrect(c, 820, y - 50, 160, 50, 8, col)
    if t >= 15.83:
        kk = ease_out_back(prog(t, 15.83, 0.25), 2.5)
        c.save()
        c.translate(900, 590)
        c.scale(kk, kk)
        label(c, "MAX", 0, 0, 40, INK, PAPER, align="center", shadow=False)
        c.restore()
    fk = F("mono", 44)
    if t >= WORD[40] - 0.04:
        kinetic(c, t, WORD[40] - 0.04, "ОЧЕНЬ ВЫСОКОГО УРОВНЯ", 540, 1330, fk, RED, mode="rise", align="center",
                stagger=0.015, tracking=0.04)
    fh = G.fit_font(F, "head", "ПРОФЕССИОНАЛЫ", 960, 160)
    slam_text(c, t, WORD[43] - 0.02, "ПРОФЕССИОНАЛЫ", 540, 1490, fh, PAPER, from_scale=1.5)


def s_ifyou(c, t):
    bg(c, "paper")
    info_bar(c, t)
    k = ease_out_expo(prog(t, 16.59, 0.4))
    r = 340 + 90 * k
    speaker(c, t, 540, 900, r, zoom=1.0 + 0.32 * k, face=True, pump=1.0)
    f = F("head", 128)
    words = [(WORD[45], "ЕСЛИ"), (WORD[46], "ВАМ"), (WORD[47], "НУЖЕН")]
    line = " ".join(w for _, w in words)
    xx = 540 - G.text_width(line, f) / 2
    for t0, wd in words:
        kinetic(c, t, t0 - 0.04, wd, xx, 420, f, INK if wd != "НУЖЕН" else RED, mode="drop", stagger=0.025)
        xx += G.text_width(wd + " ", f)


def s_search(c, t):
    bg(c, "paper")
    info_bar(c, t)
    k = ease_out_expo(prog(t, 18.78, 0.35))
    draw_text(c, "ПОИСК ПОДРЯДЧИКА", 80, 640, F("mono", 34), RED, tracking=0.12, alpha=k)
    c.save()
    c.translate(0, (1 - k) * 80)
    G.rrect(c, 70, 700, 940, 150, 75, WHITE, filt=G.shadow(0, 16, 24, 0.22))
    G.rrect(c, 70, 700, 940, 150, 75, INK, stroke=5)
    c.drawCircle(160, 768, 30, paint(INK, stroke=7))
    c.drawLine(181, 789, 205, 813, paint(INK, stroke=9, cap=skia.Paint.kRound_Cap))
    times = PL.typing_times()
    n = sum(1 for x in times if t >= x)
    typed = PL.SEARCH_TEXT[:n]
    ff = F("cap7", 46)
    tw = draw_text(c, typed, 240, 792, ff, INK)
    if (int(t * 2.5) % 2 == 0) or n < len(times):
        c.drawRect(skia.Rect.MakeXYWH(244 + tw, 742, 5, 64), paint(RED))
    # button
    press = 1 - 0.1 * math.exp(-max(0, t - 20.10) / 0.08) if t >= 20.10 else 1.0
    c.save()
    c.translate(540, 1000)
    c.scale(press, press)
    G.rrect(c, -170, -62, 340, 124, 62, RED, filt=G.shadow(0, 10, 16, 0.25))
    draw_text(c, "НАЙТИ", 0, 22, F("head", 64), WHITE, align="center", tracking=0.06)
    c.restore()
    c.restore()
    # cursor flies to the button
    kc = ease_in_out(prog(t, 19.4, 0.65))
    cx, cy = 900 - (900 - 600) * kc, 1400 - (1400 - 1030) * kc
    picto(c, "cursor", cx, cy, 120, INK)
    speaker(c, t, 220, 1330, 150, pump=1.0)


TOOLS = ["target", "chart", None, "cursor", "gear", "link", "mega"]


def s_tools(c, t):
    bg(c, "paper")
    info_bar(c, t)
    f = F("head", 124)
    if t >= WORD[55] - 0.04:
        draw_text(c, "который", 80, 290, F("serif", 64), RED, alpha=ease_out_cubic(prog(t, WORD[55] - 0.04, 0.2)))
    kinetic(c, t, WORD[56] - 0.04, "УМЕЕТ", 80, 420, f, INK, mode="rise", stagger=0.025)
    kinetic(c, t, WORD[57] - 0.04, "РАБОТАТЬ", 80 + G.text_width("УМЕЕТ ", f), 420, f, RED, mode="rise",
            stagger=0.025)
    if t >= WORD[58] - 0.05:
        draw_text(c, "со всеми этими", 80, 520, F("serif", 64), INK, alpha=ease_out_cubic(prog(t, WORD[58] - 0.05, 0.25)))
    cells = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    kinds = ["target", "chart", "speaker", "cursor", "gear", "link"]
    order = [0, 1, 2, 3, 4, 5]
    for i, ((gx, gy), kind) in enumerate(zip(cells, kinds)):
        t0 = 20.2 + order[i] * 0.12 if kind != "speaker" else 20.16
        kk = spring(prog(t, t0, 1.0), 2.6, 6.0) if t >= t0 else 0
        if kk <= 0:
            continue
        x = 220 + gx * 320
        y = 790 + gy * 330
        bump = 1 + 0.05 * kick_env(t)
        c.save()
        c.translate(x, y)
        c.scale(kk * bump, kk * bump)
        c.rotate((i % 2 * 2 - 1) * 2.5)
        if kind == "speaker":
            speaker(c, t, 0, 0, 135, shadow=True)
        else:
            G.rrect(c, -135, -135, 270, 270, 28, WHITE if i % 3 else INK, filt=G.shadow(0, 12, 18, 0.2))
            picto(c, kind, 0, 0, 150, INK if i % 3 else PAPER)
        c.restore()


def globe_lines(c, t, cx, cy, r, col, alpha=0.75):
    c.save()
    c.clipPath(G.circle_path(cx, cy, r), doAntiAlias=True)
    p = paint(col, stroke=3, alpha=alpha)
    for k in range(6):
        ph = (t * 0.35 + k / 6) % 1.0
        rx = abs(math.cos(ph * math.pi)) * r
        c.drawOval(skia.Rect.MakeLTRB(cx - rx, cy - r, cx + rx, cy + r), p)
    for k in range(1, 6):
        yy = cy - r + k * 2 * r / 6
        hw = math.sqrt(max(0, r * r - (yy - cy) ** 2))
        c.drawOval(skia.Rect.MakeLTRB(cx - hw, yy - hw * 0.08, cx + hw, yy + hw * 0.08), p)
    c.restore()


def s_globe(c, t):
    bg(c, "paper")
    info_bar(c, t)
    cx, cy, r = 540, 1000, 380
    speaker(c, t, cx, cy, r, pump=1.0)
    globe_lines(c, t, cx, cy, r, PAPER, 0.7)
    f = F("head", 120)
    kinetic(c, t, WORD[61] - 0.04, "ЗАРУБЕЖНЫЕ", 540, 360, f, INK, mode="rise", align="center", stagger=0.02)
    kinetic(c, t, WORD[62] - 0.04, "ИНСТРУМЕНТЫ", 540, 490, f, RED, mode="rise", align="center", stagger=0.02)
    chips = ["ADS", "PIXEL", "API", "CRM", "BI", "AI"]
    for i, ch in enumerate(chips):
        t0 = 22.9 + i * 0.16
        if t < t0:
            continue
        a = t * 0.5 + i * math.pi / 3
        x = cx + math.cos(a) * (r + 70)
        y = cy + math.sin(a) * (r + 40) * 0.92
        kk = ease_out_back(prog(t, t0, 0.3), 2.0)
        c.save()
        c.translate(x, y)
        c.scale(kk, kk)
        label(c, ch, 0, 0, 30, PAPER if i % 2 else INK, INK if i % 2 else WHITE, align="center", pad=(16, 10))
        c.restore()
    t_pin = 24.80
    if t >= t_pin - 0.1:
        k = ease_out_back(prog(t, t_pin - 0.1, 0.3), 2.2)
        px, py = cx + 150, cy - 140
        c.save()
        c.translate(px, py - (1 - k) * 200)
        path = skia.Path()
        path.moveTo(0, 0)
        path.cubicTo(-46, -60, -46, -110, 0, -110)
        path.cubicTo(46, -110, 46, -60, 0, 0)
        c.drawPath(path, paint(RED, filt=G.shadow(0, 8, 10, 0.3)))
        c.drawCircle(0, -78, 16, paint(PAPER))
        c.restore()
        rings_pulse(c, t, px, py, 6, RED, [t_pin], life=0.8, spread=120, width=5)
    if t >= t_pin:
        kk = spring(prog(t, t_pin, 1.0), 2.4, 6.5)
        c.save()
        c.translate(540, 1480)
        c.scale(kk, kk)
        c.rotate(-2)
        bw = label(c, "РАБОТАЮТ В СНГ    ", 0, 0, 62, WHITE, RED, font="head", align="center", pad=(30, 20))
        G.stroke_path(c, G.check_path(bw / 2 - 62, -2, 54), WHITE, 11, progress=prog(t, t_pin + 0.15, 0.2))
        c.restore()


def s_stop(c, t):
    bg(c, "ink")
    info_bar(c, t, dark=True)
    z = 1 + 0.12 * ease_in_cubic(prog(t, 25.22, 27.68 - 25.22))
    c.save()
    c.translate(540, 960)
    c.scale(z, z)
    c.translate(-540, -960)
    speaker(c, t, 540, 900, 350, cf=G.saturation_filter(0.55, 1.08), ring_col=RED)
    f = F("head", 96)
    kinetic(c, t, WORD[67] - 0.03, "Я", 400, 420, f, PAPER, mode="rise")
    kinetic(c, t, WORD[68] - 0.03, "ТОЧНО", 400 + G.text_width("Я ", f), 420, f, PAPER, mode="rise", stagger=0.03)
    # the stamp
    ts = WORD[69] - 0.02
    if t >= ts:
        k = ease_out_expo(prog(t, ts, 0.1))
        s = 2.3 - 1.3 * k
        c.save()
        c.translate(540, 1300)
        c.rotate(-8)
        c.scale(s, s)
        fs = F("head", 124)
        w = G.text_width("РЕКОМЕНДУЮ", fs, 0.02)
        eff = skia.DiscretePathEffect.Make(10, 2.5, 7)
        rect = skia.Path()
        rect.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-w / 2 - 36, -112, w + 72, 190), 12, 12))
        c.drawPath(rect, paint(RED, stroke=12, alpha=clamp01(k * 3), effect=eff))
        draw_text(c, "РЕКОМЕНДУЮ", 0, 50, fs, RED, align="center", tracking=0.02, alpha=clamp01(k * 3))
        c.restore()
    if t >= WORD[70] - 0.03:
        txt = "ВАМ ОБРАЩАТЬСЯ В АГЕНТСТВО"
        n = int(len(txt) * prog(t, WORD[70] - 0.03, WORD[73] + 0.35 - WORD[70]))
        draw_text(c, txt[:n], 540, 1490, F("mono", 38), PAPER, align="center", tracking=0.06)
    c.restore()
    # tension strobe on the roll (low intensity, <3 big flashes/s)
    if t >= PL.bar(14, 3):
        ph = ((t - PL.bar(14, 3)) / (PL.BEAT / 4)) % 1
        c.drawRect(skia.Rect.MakeWH(W, H), paint(PAPER, alpha=0.07 * (1 - ph)))


def s_drop1(c, t):
    bg(c, "red")
    info_bar(c, t, dark=True)
    rings_pulse(c, t, 540, 900, 150, PAPER, [k for k in KICKS if 27.6 <= k < 29.3], life=0.55, spread=520, width=6)
    icon_logo(c, t, 540, 900, 262, PAPER, 27.64, pump=1.0)
    k = ease_out_expo(prog(t, 27.9, 0.4))
    draw_text(c, "АГЕНТСТВО", 540, 1200, F("mono", 54), PAPER, align="center", tracking=0.3, alpha=k)
    words = [(WORD[75], "В ОБЩЕМ,"), (WORD[77], "САМ,"), (WORD[78], "ПОЛУЧАЕТСЯ...")]
    cur = [wd for t0, wd in words if t >= t0]
    if cur:
        draw_text(c, cur[-1], 540, 1480, F("serif", 72), PAPER, align="center", alpha=0.95)


def s_contractors(c, t):
    bg(c, "paper")
    info_bar(c, t)
    speaker(c, t, 720, 1010, 300, pump=1.0)
    f = F("head", 108)
    kinetic(c, t, WORD[79] - 0.04, "ОНИ У МЕНЯ —", 80, 420, f, INK, mode="rise", stagger=0.02)
    fp = G.fit_font(F, "punch", "ПОДРЯДЧИКИ", 930, 150)
    slam_text(c, t, WORD[83] - 0.03, "ПОДРЯДЧИКИ", 80, 600, fp, RED, align="left", from_scale=1.5)
    # seal
    if t >= WORD[83] + 0.1:
        k = ease_out_back(prog(t, WORD[83] + 0.1, 0.3), 2.0)
        c.save()
        c.translate(300, 1180)
        c.rotate(-14 + 6 * k)
        c.scale(k, k)
        c.drawCircle(0, 0, 150, paint(INK))
        c.drawCircle(0, 0, 132, paint(PAPER, stroke=3))
        G.stroke_path(c, G.check_path(0, -14, 120), RED, 18)
        draw_text(c, "ПРОВЕРЕНО", 0, 72, F("mono", 26), PAPER, align="center", tracking=0.08)
        draw_text(c, "ЛИЧНО", 0, 104, F("mono", 26), PAPER, align="center", tracking=0.2)
        c.restore()
    # rhythm grid
    beats = [k for k in KICKS if 29.17 <= k < 31.5]
    for i in range(12):
        gx, gy = i % 4, i // 4
        lit = i < sum(1 for b in beats if t >= b) * 2
        x, y = 90 + gx * 44, 1440 + gy * 44
        c.drawRect(skia.Rect.MakeXYWH(x, y, 34, 34), paint(RED if lit else INK, stroke=None if lit else 3,
                                                            alpha=1.0 if lit else 0.5))


SPARK = np.random.default_rng(12).uniform(0, 1, (len(PL.PLUCK_WORDS), 3))


def s_thanks(c, t):
    bg(c, "cream")
    info_bar(c, t)
    # soft moving light
    lx = 540 + 200 * math.sin(t * 0.6)
    shader = skia.GradientShader.MakeRadial(skia.Point(lx, 800), 900,
                                            [hexc("#FFFFFF", 0.55).toColor(), hexc("#FFFFFF", 0.0).toColor()])
    c.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Shader=shader))
    t0 = WORD[87] - 0.06
    if t >= t0:
        f = G.fit_font(F, "punch", "СПАСИБО", 1000, 230)
        drift = (t - t0) * 14
        kinetic(c, t, t0, "СПАСИБО", 540 - drift, 590, f, RED, mode="rise", align="center", stagger=0.05,
                dur=0.45)
    fy = 4 * math.sin(t * 1.3)
    speaker(c, t, 540, 1030 + fy, 400)
    if t >= WORD[85] - 0.04:
        draw_text(c, "ребят,", 540, 330, F("serif", 76), INK, align="center",
                  alpha=ease_out_cubic(prog(t, WORD[85] - 0.04, 0.3)))
    lines = [(WORD[89] - 0.04, WORD[92], "ЗА ВАШ ТРУД"), (WORD[92], WORD[97], "ОЧЕНЬ СИЛЬНО БЛАГОДАРЕН"),
             (WORD[97], 99, "ЭТО РЕАЛЬНО")]
    for a, b, txt in lines:
        if a <= t < b:
            f = G.fit_font(F, "head", txt, 940, 92)
            kinetic(c, t, a, txt, 540, 1530, f, INK, mode="rise", align="center", stagger=0.02)
    for i, wi in enumerate(PL.PLUCK_WORDS):
        t1 = WORD[wi]
        if 0 <= t - t1 < 0.6:
            k = (t - t1) / 0.6
            ang = SPARK[i, 0] * 2 * math.pi
            rr = 470 + SPARK[i, 1] * 60
            x, y = 540 + math.cos(ang) * rr, 1030 + math.sin(ang) * rr * 0.95
            s = 34 * math.sin(math.pi * k) * (0.7 + 0.6 * SPARK[i, 2])
            c.drawPath(G.star_path(x, y, s, s * 0.32, 4), paint(RED if i % 2 else INK, alpha=1 - k * 0.5))


def seal(c, t, x, y, r, t0, text="КАЧЕСТВО • КАЧЕСТВО • "):
    k = ease_out_back(prog(t, t0, 0.25), 2.2)
    if k <= 0:
        return
    c.save()
    c.translate(x, y)
    c.scale(k, k)
    c.rotate(-12 + (t - t0) * 40)
    c.drawCircle(0, 0, r, paint(RED))
    c.drawCircle(0, 0, r - 16, paint(PAPER, stroke=3))
    f = F("mono", 30)
    n = len(text)
    for i, ch in enumerate(text):
        a = i / n * 2 * math.pi
        c.save()
        c.rotate(math.degrees(a))
        draw_text(c, ch, 0, -r + 52, f, PAPER, align="center")
        c.restore()
    c.drawPath(G.star_path(0, 0, r * 0.38, r * 0.16, 5), paint(PAPER))
    c.restore()


def s_quality(c, t):
    bg(c, "paper")
    info_bar(c, t)
    z = 1 + 0.08 * ease_in_cubic(prog(t, 35.37, PL.BAR))
    c.save()
    c.translate(540, 960)
    c.scale(z, z)
    c.translate(-540, -960)
    speaker(c, t, 400, 1120, 300, pump=1.0)
    fh = G.fit_font(F, "head", "ВЫСОКОКАЧЕСТВЕННАЯ", 960, 140)
    kinetic(c, t, WORD[100] - 0.04, "ВЫСОКОКАЧЕСТВЕННАЯ", 540, 420, fh, INK, mode="rise", align="center",
            stagger=0.012)
    fp = G.fit_font(F, "punch", "РАБОТА", 940, 210)
    slam_text(c, t, WORD[101] - 0.03, "РАБОТА", 540, 660, fp, RED, from_scale=1.6)
    seal(c, t, 800, 1260, 170, WORD[101] + 0.05)
    if t >= WORD[102] - 0.03:
        draw_text(c, "желаю вам всего...", 540, 1520, F("serif", 64), INK, align="center",
                  alpha=ease_out_cubic(prog(t, WORD[102] - 0.03, 0.25)))
    c.restore()
    if t >= PL.bar(19, 2):
        ph = ((t - PL.bar(19, 2)) / (PL.BEAT / 2)) % 1
        c.drawRect(skia.Rect.MakeWH(W, H), paint(INK, alpha=0.05 * (1 - ph)))


def s_final(c, t):
    b = [PL.bar(20, i) for i in range(8)]
    if t < b[1]:
        bg(c, "red")
        info_bar(c, t, dark=True)
        speaker(c, t, 540, 1050, 400, pump=1.0, ring_col=PAPER)
        f = G.fit_font(F, "punch", "КЛАССНОГО!", 960, 170)
        slam_text(c, t, 37.27, "КЛАССНОГО!", 540, 480, f, PAPER, from_scale=1.5)
        draw_text(c, "желаю вам всего", 540, 300, F("serif", 64), PAPER, align="center")
    elif t < b[3]:
        bg(c, "paper")
        info_bar(c, t)
        speaker(c, t, 720, 1180, 300, pump=1.0)
        f = F("head", 150)
        kinetic(c, t, WORD[107] - 0.03, "ЕСЛИ", 80, 470, f, INK, mode="drop", stagger=0.02)
        kinetic(c, t, WORD[108] - 0.03, "ВАМ", 80, 640, f, INK, mode="drop", stagger=0.02)
        kinetic(c, t, WORD[109] - 0.03, "НУЖНЫ", 80, 810, f, RED, mode="drop", stagger=0.02)
    elif t < b[5]:
        bg(c, "ink")
        info_bar(c, t, dark=True)
        rings_pulse(c, t, 540, 1220, 160, PAPER, [k for k in KICKS if b[3] - 0.01 <= k < b[5]], life=0.5,
                    spread=300, width=4)
        speaker(c, t, 540, 1220, 230, pump=1.0, ring_col=RED)
        f1 = G.fit_font(F, "head", "КЛАССНЫЕ", 940, 190)
        kinetic(c, t, WORD[110] - 0.03, "КЛАССНЫЕ", 540, 470, f1, PAPER, mode="scale", align="center", stagger=0.02)
        f2 = G.fit_font(F, "punch", "ПОДРЯДЧИКИ", 960, 160)
        slam_text(c, t, WORD[111] - 0.03, "ПОДРЯДЧИКИ", 540, 700, f2, RED, from_scale=1.5)
    else:
        bg(c, "red")
        info_bar(c, t, dark=True)
        f = G.fit_font(F, "punch", "ИДИТЕ К НИМ", 960, 150)
        slam_text(c, t, WORD[112] - 0.03, "ИДИТЕ К НИМ", 540, 520, f, PAPER, from_scale=1.4)
        k = ease_out_expo(prog(t, WORD[112], 0.35))
        x1 = 140 + 560 * k
        G.stroke_path(c, G.arrow_path(140, 1000, x1 + 120, 1000, 70), PAPER, 22)
        icon_logo(c, t, 540, 1330, 190, PAPER, WORD[112] + 0.05, pump=1.0)


def s_pause(c, t):
    bg(c, "ink")
    z = 1 + 0.05 * prog(t, 40.65, 0.5)
    speaker(c, t, 540, 900, 330 * z, cf=G.saturation_filter(0.4, 1.05), ring_col=RED)
    f = F("head", 120)
    kinetic(c, t, WORD[115] - 0.03, "ОНИ", 540 - G.text_width("ОНИ ТОЧНО", f) / 2, 1480, f, PAPER, mode="rise")
    kinetic(c, t, WORD[116] - 0.03, "ТОЧНО", 540 - G.text_width("ОНИ ТОЧНО", f) / 2 + G.text_width("ОНИ ", f),
            1480, f, PAPER, mode="rise")


def s_top(c, t):
    bg(c, "red")
    t0 = PL.TOP - 0.03
    k = ease_out_expo(prog(t, t0, 0.12))
    s = 2.2 - 1.2 * k
    f = F("punch", 350)
    wt = G.text_width("Т", f)
    wp = G.text_width("П", f)
    r = 148
    gap = 16
    total = wt + gap + 2 * r + gap + wp
    x0 = 540 - total / 2
    c.save()
    c.translate(540, 900)
    c.scale(s, s)
    c.translate(-540, -900)
    base = 900 + 126
    draw_text(c, "Т", x0, base, f, PAPER, alpha=clamp01(k * 3))
    ox = x0 + wt + gap + r
    speaker(c, t, ox, 900, r, ring=False, misreg=False, shadow=False, alpha=clamp01(k * 3))
    c.drawCircle(ox, 900, r, paint(PAPER, stroke=34, alpha=clamp01(k * 3)))
    draw_text(c, "П", ox + r + gap, base, f, PAPER, alpha=clamp01(k * 3))
    c.restore()
    rings_pulse(c, t, ox, 900, r, PAPER, [PL.TOP], life=0.9, spread=700, width=8)
    q = F("serif", 160)
    ka = ease_out_cubic(prog(t, PL.TOP + 0.25, 0.3))
    draw_text(c, "«", 120, 640, q, PAPER, alpha=ka)
    draw_text(c, "»", 960, 1260, q, PAPER, alpha=ka)
    if t >= PL.TOP + 0.35:
        kk = ease_out_expo(prog(t, PL.TOP + 0.35, 0.4))
        draw_text(c, "АНТОН ПЕТРАЧЕНКОВ", 540, 1420, F("head", 64), PAPER, align="center", alpha=kk, tracking=0.03)
        draw_text(c, "CONVERTMONSTER", 540, 1490, F("mono", 34), INK, align="center", alpha=kk, tracking=0.18)


def s_logo(c, t):
    bg(c, "paper")
    t0 = 42.40
    dot = ease_in_out(prog(t, 43.55, 0.35))
    icon_logo(c, t, 540, 880, 250, INK, t0 - 0.1, dot=dot)
    k = ease_out_expo(prog(t, t0 + 0.2, 0.5))
    c.drawLine(540 - 300 * k, 1060, 540 + 300 * k, 1060, paint(RED, stroke=6))
    draw_text(c, "ПОДРЯДЧИК ПО ТРАФИКУ", 540, 1140, F("mono", 40), INK, align="center", tracking=0.14, alpha=k)
    k2 = ease_out_expo(prog(t, t0 + 0.5, 0.5))
    draw_text(c, "отзыв: Антон Петраченков, ConvertMonster", 540, 1330, F("body", 34), GREY, align="center",
              alpha=k2)


SCENE_FN = {
    "hook": s_hook, "intro": s_intro, "thank": s_thank, "two": s_two, "icon1": s_icon1, "names": s_names,
    "helped": s_helped, "course": s_course, "level": s_level, "ifyou": s_ifyou, "search": s_search,
    "tools": s_tools, "globe": s_globe, "stop": s_stop, "drop1": s_drop1, "contractors": s_contractors,
    "thanks": s_thanks, "quality": s_quality, "final": s_final, "pause": s_pause, "top": s_top, "logo": s_logo,
}

# ------------------------------------------------------------------ captions (only where the picture does not
# already carry the words)
CAPTION_SCENES = {"intro", "ifyou"}
CAPTION_WINDOWS = {"intro": (WORD[7] - 0.05, WORD[11] - 0.05), "ifyou": (WORD[48] - 0.05, 99)}
GROUPS = PL.caption_groups()


def captions(c, t, scene):
    if scene not in CAPTION_SCENES:
        return
    lo, hi = CAPTION_WINDOWS[scene]
    if not lo <= t < hi:
        return
    g = next((g for g in GROUPS if PL.WORDS[g[0]]["s"] - 0.05 <= t <= PL.WORDS[g[-1]]["e"] + 0.25), None)
    if g is None:
        return
    f = F("cap", 50)
    toks = []
    for j in g:
        wd = PL.WORDS[j]["w"].upper()
        if wd.startswith("-") and toks:
            toks[-1] = (toks[-1][0] + wd, toks[-1][1])
        else:
            toks.append((wd, j))
    space = G.text_width(" ", f)
    widths = [G.text_width(wd, f) for wd, _ in toks]
    total = sum(widths) + space * (len(toks) - 1)
    x = 540 - total / 2
    y = 1490
    k = ease_out_expo(prog(t, PL.WORDS[g[0]]["s"] - 0.05, 0.2))
    for (wd, j), wdt in zip(toks, widths):
        active = PL.WORDS[j]["s"] - 0.03 <= t
        cur = active and (t <= PL.WORDS[j]["e"] + 0.05 or j == g[-1])
        if cur:
            G.rrect(c, x - 10, y - 50, wdt + 20, 64, 10, RED)
        col = WHITE if cur else (INK if active else with_alpha(INK, 0.35))
        draw_text(c, wd, x, y + (1 - k) * 30, f, col)
        x += wdt + space


def with_alpha(col, a):
    return G.with_alpha(col, a)


# ------------------------------------------------------------------ transitions & frame assembly
def scene_at(t):
    for i, (a, b, name, tr) in enumerate(PL.SCENES):
        if a <= t < b:
            return i
    return len(PL.SCENES) - 1


TR_WIN = {"iris": (0.0, 0.3), "whip_left": (0.12, 0.14), "whip_right": (0.12, 0.14), "slide_up": (0.05, 0.3),
          "zoom": (0.14, 0.2), "glitch": (0.07, 0.1), "dissolve": (0.2, 0.25), "slam": (0.0, 0.15),
          "flash": (0.0, 0.14)}


NO_KICK_CAM = {"hook", "thanks", "stop", "pause", "top", "logo"}


def render_scene(i, t):
    surf = G.surface(W, H)
    c = surf.getCanvas()
    a, b_, name, _ = PL.SCENES[i]
    # virtual camera: slow push through every scene + a small bump on each kick
    s = 1 + 0.03 * ease_in_out(clamp01((t - a) / max(b_ - a, 0.1)))
    if name not in NO_KICK_CAM:
        s *= 1 + 0.012 * kick_env(t)
    c.translate(540, 960)
    c.scale(s, s)
    c.translate(-540, -960)
    SCENE_FN[name](c, t)
    captions(c, t, name)
    return surf


def compose(t):
    i = scene_at(t)
    a, b, name, tr = PL.SCENES[i]
    pre, post = TR_WIN.get(tr, (0, 0))
    # pre-roll: before the boundary, the outgoing scene may already be leaving
    nxt = i + 1 if i + 1 < len(PL.SCENES) else None
    if nxt is not None:
        na, _, _, ntr = PL.SCENES[nxt]
        npre, npost = TR_WIN.get(ntr, (0, 0))
        if na - npre <= t < na:
            return transition(ntr, i, nxt, t, (t - (na - npre)) / max(npre + npost, 1e-6), na)
    if i > 0 and t < a + post and tr in TR_WIN:
        return transition(tr, i - 1, i, t, (t - (a - pre)) / max(pre + post, 1e-6), a)
    return np.asarray(G.snapshot(render_scene(i, t)))[..., :3].astype(np.float32)


def rgb(surf):
    return np.asarray(G.snapshot(surf))[..., :3].astype(np.float32)


def transition(kind, ia, ib, t, k, boundary):
    k = clamp01(k)
    if kind in ("whip_left", "whip_right"):
        d = -1 if kind == "whip_left" else 1
        e = ease_in_out(k)
        A = rgb(render_scene(ia, min(t, boundary - 1e-3)))
        B = rgb(render_scene(ib, max(t, boundary)))
        shift = int(e * W)
        blur = 120 * math.sin(math.pi * k)
        out = np.empty_like(A)
        if d < 0:
            out[:, :W - shift] = A[:, shift:]
            out[:, W - shift:] = B[:, :shift]
        else:
            out[:, shift:] = A[:, :W - shift]
            out[:, :shift] = B[:, W - shift:]
        return G.directional_blur(out, blur, axis=1)
    if kind == "zoom":
        A = render_scene(ia, min(t, boundary - 1e-3))
        B = render_scene(ib, max(t, boundary))
        surf = G.surface(W, H)
        c = surf.getCanvas()
        e = ease_in_out(k)
        c.drawImage(B.makeImageSnapshot(), 0, 0)
        if e < 1:
            s = 1 + 1.6 * e
            c.save()
            c.translate(540, 960)
            c.scale(s, s)
            c.translate(-540, -960)
            p = skia.Paint()
            p.setAlphaf(1 - e)
            p.setImageFilter(skia.ImageFilters.Blur(10 * e, 10 * e))
            c.drawImage(A.makeImageSnapshot(), 0, 0, G.LINEAR, p)
            c.restore()
        return rgb(surf)
    if kind == "slide_up":
        A = rgb(render_scene(ia, min(t, boundary - 1e-3)))
        B = rgb(render_scene(ib, max(t, boundary)))
        e = ease_out_expo(k)
        sh = int(e * H)
        out = np.empty_like(A)
        out[:H - sh] = A[sh:]
        out[H - sh:] = B[:sh]
        return G.directional_blur(out, 60 * math.sin(math.pi * k), axis=0)
    if kind == "iris":
        A = render_scene(ia, min(t, boundary - 1e-3))
        B = render_scene(ib, max(t, boundary))
        surf = G.surface(W, H)
        c = surf.getCanvas()
        c.drawImage(A.makeImageSnapshot(), 0, 0)
        cx, cy = hook_o_center() if ia == 0 else (540, 900)
        r = ease_in_out(k) * 2300
        c.save()
        c.clipPath(G.circle_path(cx, cy, r), doAntiAlias=True)
        c.drawImage(B.makeImageSnapshot(), 0, 0)
        c.restore()
        if 0 < k < 1:
            c.drawCircle(cx, cy, r, paint(RED, stroke=26 * (1 - k) + 4))
        return rgb(surf)
    if kind == "glitch":
        A = rgb(render_scene(ia, min(t, boundary - 1e-3)))
        B = rgb(render_scene(ib, max(t, boundary)))
        rng = np.random.default_rng(int(t * 1000))
        out = B.copy() if k > 0.5 else A.copy()
        for _ in range(14):
            y0 = int(rng.integers(0, H - 40))
            hh = int(rng.integers(12, 120))
            dx = int(rng.integers(-160, 160))
            src = B if rng.random() < k else A
            out[y0:y0 + hh] = np.roll(src[y0:y0 + hh], dx, axis=1)
        return G.chroma_shift(out, 14)
    if kind == "dissolve":
        A = rgb(render_scene(ia, min(t, boundary - 1e-3)))
        B = rgb(render_scene(ib, max(t, boundary)))
        e = ease_in_out(k)
        return A * (1 - e) + B * e
    if kind == "slam":
        B = render_scene(ib, t)
        surf = G.surface(W, H)
        c = surf.getCanvas()
        s = 1.12 - 0.12 * ease_out_expo(k)
        c.translate(540, 960)
        c.scale(s, s)
        c.translate(-540, -960)
        c.drawImage(B.makeImageSnapshot(), 0, 0, G.LINEAR, None)
        return rgb(surf)
    if kind == "flash":
        B = rgb(render_scene(ib, t))
        a = (1 - k) ** 2 * 0.85
        return B * (1 - a) + 255 * a
    return rgb(render_scene(ib, t))


GRAIN = None
VIG = None


def finish(frame, t, k):
    global GRAIN, VIG
    if GRAIN is None:
        GRAIN = G.Grain(H, W, n=8, sigma=5.5)
        VIG = G.vignette_mask(H, W, 0.16)
    dx, dy, rot = shake(t)
    if abs(dx) > 0.5 or abs(dy) > 0.5:
        frame = np.roll(frame, (int(dy), int(dx)), axis=(0, 1))
    ca = chroma_amount(t)
    if ca > 0.6:
        frame = G.chroma_shift(frame, ca)
    frame = GRAIN.apply(frame, k, 1.0) * VIG
    return np.clip(frame, 0, 255).astype(np.uint8)


def frame_at(k):
    t = k / FPS
    return finish(compose(t), t, k)


# ------------------------------------------------------------------ CLI
def render_range(k0, k1, out_path):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-x264-params", "keyint=60", out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(k0, k1):
        p.stdin.write(frame_at(k).tobytes())
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "frames":
        from PIL import Image
        out = sys.argv[2]
        os.makedirs(out, exist_ok=True)
        for ts in sys.argv[3:]:
            k = int(round(float(ts) * FPS))
            Image.fromarray(frame_at(k)).save(os.path.join(out, f"f_{float(ts):06.2f}.png"))
    elif mode == "range":
        render_range(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif mode == "video":
        out, audio = sys.argv[2], sys.argv[3]
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        total = int(round(PL.DURATION * FPS))
        tmp = out + ".parts"
        os.makedirs(tmp, exist_ok=True)
        bounds = np.linspace(0, total, workers + 1).astype(int)
        procs = []
        for w_ in range(workers):
            part = os.path.join(tmp, f"part{w_}.mp4")
            procs.append(subprocess.Popen([sys.executable, __file__, "range", str(bounds[w_]), str(bounds[w_ + 1]), part]))
        codes = [p.wait() for p in procs]
        assert all(c_ == 0 for c_ in codes), codes
        with open(os.path.join(tmp, "list.txt"), "w") as fl:
            for w_ in range(workers):
                fl.write(f"file 'part{w_}.mp4'\n")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", os.path.join(tmp, "list.txt"),
                        "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        "-ar", "48000", "-shortest", "-movflags", "+faststart", out], check=True)
        print("wrote", out)
