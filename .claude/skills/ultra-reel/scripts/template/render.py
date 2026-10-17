"""Painterly picture ("холст") with ink-and-wash illustrations for the ICON testimonial, driven by plan.py.

One continuous canvas, no cuts: the speaker node glides between layouts and shrinks into a live card when a
painted world takes the middle; illustrations land on their spoken words, words rise or are written by a brush,
watercolor washes change the mood. Nothing is drawn that the speaker did not say.

python render.py frames <out_dir> <t> [<t> ...]     -> PNG stills for QA
python render.py range <k0> <k1> <out.mp4>          -> one worker's segment
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
sys.path.insert(0, os.environ.get("ULTRA_ENGINE", os.path.join(HERE, "..", "ultra")))
sys.path.insert(0, HERE)

from engine import gfx as G  # noqa: E402
from engine import paint as P  # noqa: E402
from engine import illo as I  # noqa: E402
from engine.gfx import prog, clamp01, ease_in_out, ease_out_cubic  # noqa: E402
import plan as PL  # noqa: E402

W, H, FPS = PL.W, PL.H, PL.FPS
FONT_DIR = os.environ.get("REEL_FONTS", os.path.join(HERE, "..", "fonts"))
F = G.Fonts(FONT_DIR, {
    "serif": "CormorantGaramond-600.ttf", "serif_i": "CormorantGaramond-600i.ttf",
    "serif_wide": "CormorantGaramond-600.ttf", "serif_small": "CormorantGaramond-500.ttf",
    "brush": "ComforterBrush-400.ttf",
})
RGB = {"canvas": (241, 234, 223), "ink": (42, 39, 36), "terra": (185, 87, 62), "ochre": (201, 154, 62),
       "dusk": (46, 59, 78), "cream": (243, 236, 224), "grey": (139, 132, 120), "slate": (70, 92, 120)}
COL = {k: skia.Color4f(r / 255, g / 255, b / 255, 1) for k, (r, g, b) in RGB.items()}
SPK = np.load(os.path.join(HERE, "speaker.npy"), mmap_mode="r")
SRC = (4, 4, 396, 396)
FACE = (196, 168)
TRACK = {"serif_wide": 0.16}
MAX_W = 940

# ------------------------------------------------------------------ assets (deterministic, built once per worker)
_A = {}


def asset(name):
    if name not in _A:
        if name == "canvas":
            _A[name] = P.canvas_texture(H + 160, W + 160, RGB["canvas"], seed=7)
        elif name == "ink":
            _A[name] = P.ink_texture(seed=4)
        elif name == "mask":
            _A[name] = P.brush_circle_mask(1000, seed=2)
        elif name == "halo":
            _A[name] = P.watercolor(900, 900, RGB["terra"], seed=4, spread=0.42)
        elif name == "halo_ochre":
            _A[name] = P.watercolor(900, 900, RGB["ochre"], seed=6, spread=0.42)
        elif name == "fill":
            _A[name] = P.watercolor(800, 800, RGB["terra"], seed=9, spread=0.47, density=0.85, rim=0.4)
        elif name.startswith("wash"):
            i = int(name[4:])
            t_in, t_full, o0, o1, col, a, seed, ww, hh, _ = PL.WASHES[i]
            _A[name] = P.watercolor(ww, hh, RGB[col], seed=seed, spread=0.44, edge_rough=0.26, density=0.9, rim=0.35)
        elif name.startswith("dot"):
            k = int(name[3:])
            _A[name] = P.watercolor(150, 150, RGB["terra" if k % 2 else "ochre"], seed=60 + k, spread=0.36,
                                    density=0.8, rim=0.6)
    return _A[name]


# ------------------------------------------------------------------ rhythm helpers
KICKS = []
for n in range(1, 22):
    sec = next((k for k, (a, b) in PL.SECTIONS.items() if a <= n < b), None)
    if sec in ("intro", "lift", "verse", "level", "b", "drop", "final", "build"):
        for b in range(4):
            t = PL.bar(n, b)
            if (n == 13 and t >= PL.TAPE_STOP - 0.01) or (n == 21 and t >= PL.FINAL_STOP - 0.01) or (n == 8 and b in (1, 3)):
                continue
            KICKS.append(t)
KICKS = np.array(KICKS)


def kick_env(t, decay=0.16):
    i = np.searchsorted(KICKS, t, side="right") - 1
    return 0.0 if i < 0 else math.exp(-(t - KICKS[i]) / decay)


# ------------------------------------------------------------------ the speaker node (persistent, tweened)
def scene_index(t):
    i = 0
    for k, (s, _, _) in enumerate(PL.SCENES):
        if t >= s:
            i = k
    return i


def layout_at(t):
    """Interpolated (cx, cy, r, zoom) — every layout change is a smooth glide centred on the boundary."""
    i = scene_index(t + PL.GLIDE / 2)
    cur = PL.L[PL.SCENES[i][1]]
    if i == 0:
        return cur
    prev = PL.L[PL.SCENES[i - 1][1]]
    b = PL.SCENES[i][0]
    k = prog(t, b - PL.GLIDE / 2, PL.GLIDE)
    if PL.SCENES[i - 1][1] == "hidden":  # the very first reveal: paint spreading open
        k = prog(t, b, 0.55)
        e = ease_out_cubic(k)
        return (cur[0], cur[1], cur[2] * e, cur[3])
    e = ease_in_out(k)
    return tuple(p + (c - p) * e for p, c in zip(prev, cur))


@lru_cache(maxsize=6)
def frame_image(idx):
    return G.image(np.asarray(SPK[idx]))


def draw_speaker(c, t):
    cx, cy, r, zoom = layout_at(t)
    if r < 2:
        return cx, cy, r
    i = scene_index(t)
    groove = i > 0 and PL.SCENES[i][2] != "ochre" and not (PL.TAPE_STOP < t < 27.68) and \
        not (PL.FINAL_STOP <= t < PL.TOP)
    r *= 1 + (0.007 * kick_env(t) if groove else 0.0)
    idx = int(min(len(SPK) - 1, max(0, round(t * FPS))))
    img = frame_image(idx)

    def content(cc):
        rr = r * zoom
        ox = -(FACE[0] - 200) / 392 * 2 * rr * 0.85 if zoom > 1 else 0
        oy = -(FACE[1] - 200) / 392 * 2 * rr * 0.85 if zoom > 1 else 0
        # the logo ending: the portrait is slowly painted over with terracotta
        k = prog(t, 43.55, 0.7)
        vp = skia.Paint()
        vp.setAlphaf(1 - ease_in_out(prog(t, 43.65, 0.6)))
        cc.drawImageRect(img, skia.Rect.MakeLTRB(*SRC),
                         skia.Rect.MakeLTRB(cx - rr + ox, cy - rr + oy, cx + rr + ox, cy + rr + oy), G.CUBIC, vp)
        if k > 0:
            P.draw_bloom(cc, asset("fill"), cx, cy, k, scale=2 * r / 800 * 1.15, alpha=0.95)

    P.draw_masked(c, asset("mask"), cx, cy, r, content)
    return cx, cy, r


# ------------------------------------------------------------------ washes
def wash_alpha(t, w_):
    t_in, t_full, o0, o1 = w_[:4]
    if t < t_in or t > o1:
        return 0.0, 0.0
    grow = prog(t, t_in, t_full - t_in)
    fade = 1 - ease_in_out(prog(t, o0, o1 - o0))
    return grow, fade


def draw_washes(c, t, sx, sy):
    for i, w_ in enumerate(PL.WASHES):
        grow, fade = wash_alpha(t, w_)
        if grow <= 0 or fade <= 0:
            continue
        cx_, cy_ = (sx, sy + 60) if w_[9] == "speaker" else w_[9]
        P.draw_bloom(c, asset(f"wash{i}"), cx_, cy_, grow, alpha=w_[5] * fade)


def mood_dark(t):
    """0..1: how much a dark/strong wash covers the canvas now (decides cream vs ink)."""
    m = 0.0
    for w_ in PL.WASHES:
        if w_[4] == "ochre" or w_[7] < 1000:
            continue
        grow, fade = wash_alpha(t, w_)
        m = max(m, min(1.0, grow * 1.6) * fade)
    return m


def draw_halo(c, t, cx, cy, r):
    if r < 2:
        return
    a = 0.5 * (1 - mood_dark(t))
    ochre = clamp01(prog(t, 31.45, 0.8) - prog(t, 35.1, 0.7))
    s = 2 * r * 1.32 / 900
    k = prog(t, 0.6, 0.9)
    if a * (1 - ochre) > 0.01:
        P.draw_bloom(c, asset("halo"), cx + 14, cy + 18, k, scale=s, alpha=a * (1 - ochre))
    if ochre > 0.01:
        P.draw_bloom(c, asset("halo_ochre"), cx + 14, cy + 18, 1.0, scale=s * 1.1, alpha=0.55 * ochre)


# ------------------------------------------------------------------ strokes
_STROKES = {}


def stroke_obj(i, cx=None, cy=None, r=None):
    t_in, dur, t_out, kind, p, col = PL.STROKES[i]
    key = i
    if key not in _STROKES:
        if kind in ("enso", "enso2"):
            _STROKES[key] = P.make_enso(400, p["width"], seed=p["seed"], start_deg=-62 if kind == "enso" else 120)
        elif kind == "line":
            pts = [(p["x0"], p["y"]), ((p["x0"] + p["x1"]) / 2, p["y"] - 9), (p["x1"], p["y"] + 6)]
            _STROKES[key] = P.Stroke(pts, p["w"], seed=p["seed"], dryness=0.6, taper=(0.08, 0.4))
        elif kind == "under":
            x0, x1, base, size = text_geom(find_text(p["text"], t_in))
            y = base + 0.17 * size
            pts = [(x0 - 14, y + 4), ((x0 + x1) / 2, y - 5), (x1 + 18, y + 3)]
            _STROKES[key] = P.Stroke(pts, max(13, size * 0.16), seed=p["seed"], dryness=0.6, taper=(0.08, 0.4))
        elif kind == "ring":
            x0, x1, base, size = text_geom(find_text(p["text"], t_in))
            cx, cy = (x0 + x1) / 2, base - 0.3 * size
            rx, ry = (x1 - x0) / 2 + 44, 0.52 * size + 30
            rng = np.random.default_rng(p["seed"])
            a = np.linspace(math.radians(-205), math.radians(170), 90)
            k = np.linspace(1.04, 0.95, len(a)) * (1 + 0.012 * rng.standard_normal(len(a)).cumsum() / 6)
            pts = list(zip(cx + rx * k * np.cos(a), cy + ry * k * np.sin(a) - np.linspace(0, 10, len(a))))
            _STROKES[key] = P.Stroke(pts, 15, seed=p["seed"], dryness=0.55, taper=(0.06, 0.3), bristles=30)
        elif kind == "arrow":
            pts = p["pts"]
            body = P.Stroke(pts, p["w"], seed=p["seed"], dryness=0.25, taper=(0.1, 0.12), depletion=0.1, core=0.85)
            (xa, ya), (xb, yb) = pts[-2], pts[-1]
            ang = math.atan2(yb - ya, xb - xa)
            heads = []
            for d, s_ in ((2.5, 1), (-2.5, 2)):
                hx, hy = xb + 115 * math.cos(ang + d), yb + 115 * math.sin(ang + d)
                heads.append(P.Stroke([(xb, yb), ((hx + xb) / 2, (hy + yb) / 2), (hx, hy)], p["w"] * 0.85,
                                      seed=p["seed"] + s_, dryness=0.2, taper=(0.05, 0.3), depletion=0.1, core=0.85))
            _STROKES[key] = (body, heads)
    return _STROKES[key]


def draw_strokes(c, t, cx, cy, r):
    dark = mood_dark(t)
    for i, (t_in, dur, t_out, kind, p, col) in enumerate(PL.STROKES):
        if t < t_in or t > t_out:
            continue
        k = ease_in_out(prog(t, t_in, dur))
        fade = 1 - prog(t, t_out - 0.2, 0.2)  # leaves together with its word
        st = stroke_obj(i)
        if kind in ("enso", "enso2"):
            if r < 2:
                continue
            s = (r + p["r_pad"]) / 400
            c.save()
            c.translate(cx, cy)
            c.scale(s, s)
            if kind == "enso":  # persistent: ink on light canvas, cream over dark washes
                st.draw(c, COL["ink"], k, alpha=(1 - dark) * fade)
                if dark > 0.01:
                    st.draw(c, COL["cream"], k, alpha=dark * fade)
            else:
                st.draw(c, COL[col], k, alpha=fade)
            c.restore()
        elif kind == "arrow":
            body, heads = st
            kk = prog(t, t_in, dur * 1.35)
            body.draw(c, COL[col], clamp01(kk / 0.78), alpha=fade)
            for j, hd in enumerate(heads):
                hd.draw(c, COL[col], ease_in_out(prog(kk, 0.76, 0.2)), alpha=fade)
        else:
            st.draw(c, COL[col], k, alpha=fade)


# ------------------------------------------------------------------ illustrations
SIGN_FONT = lambda s: F("serif_wide", s)  # noqa: E731
BOOK_FONT = lambda s: F("serif", s)  # noqa: E731
OBJECTS = {
    "heart": lambda w: I.heart(w),
    "fig_terra": lambda w: I.figure_one(w, "terra", seed=5),
    "fig_terra_wave": lambda w: I.figure_one(w, "terra", seed=5, arm_l="wave"),
    "fig_terra_wave_out": lambda w: I.figure_one(w, "terra", seed=5, arm_l="wave"),
    "fig_terra_point": lambda w: I.figure_one(w, "terra", seed=5, arm_r="point"),
    "fig_slate": lambda w: I.figure_one(w, "slate", seed=6, hair="side"),
    "fig_slate_wave": lambda w: I.figure_one(w, "slate", seed=6, hair="side", arm_r="wave"),
    "fig_slate_wave_out": lambda w: I.figure_one(w, "slate", seed=6, hair="side", arm_r="wave"),
    "fig_slate_point": lambda w: I.figure_one(w, "slate", seed=6, hair="side", arm_l="point"),
    "duo": lambda w: I.duo(w),
    "duo_wave": lambda w: I.duo(w, ((None, "wave"), ("wave", None))),
    "duo_pedestal": lambda w: I.duo_on_pedestal(w),
    "sign_icon": lambda w: I.sign(w, "ICON", SIGN_FONT),
    "book_blank": lambda w: I.book_open(w),
    "book_written": lambda w: I.book_open(w, True, "Трафик", lambda s: F("serif_i", s)),
    "book_plain": lambda w: I.book_closed(w, "Курс\nпо трафику", BOOK_FONT),
    "book_sealed": lambda w: I.book_closed(w, "Курс\nпо трафику", BOOK_FONT, seal=True),
    "cap": lambda w: I.cap(w), "stairs": lambda w: I.stairs(w), "sparkle": lambda w: I.sparkle(w),
    "magnifier": lambda w: I.magnifier(w), "magnifier_duo": lambda w: I.magnifier_duo(w),
    "toolbox": lambda w: I.toolbox(w), "wrench": lambda w: I.wrench(w), "gear": lambda w: I.gear(w),
    "magnet": lambda w: I.magnet(w), "globe": lambda w: I.globe(w), "chart": lambda w: I.chart(w),
    "pin": lambda w: I.pin(w),
}
_OBJ = {}


def obj(key, w, rim):
    """(image, bottom_px): the object painted once per worker; bottom_px = lowest painted row (feet)."""
    k = (key, w, rim)
    if k not in _OBJ:
        img = OBJECTS[key](w)
        if rim:
            img = I.with_rim(img, r=max(5, int(w * 0.018)))
        a = np.asarray(G.snapshot(img) if hasattr(img, "getCanvas") else img.toarray())[..., 3]
        rows = np.where(a.max(axis=1) > 40)[0]
        _OBJ[k] = (img, float(rows[-1]) if len(rows) else img.height())
    return _OBJ[k]


HOP = 0.24


def world_push(t):
    for a, b in PL.WORLDS:
        if a <= t < b:
            return 1 + 0.04 * ease_in_out(prog(t, a, b - a))
    return 1.0


def prop_pose(p, t):
    """Position, scale, squash, rotation and alpha of a prop at time t (None = not on screen)."""
    lead = 0.22 if p["anim"] == "drop" else 0.0
    t0 = p["t"] - lead
    if t < t0 or t > p["out"]:
        return None
    x, y = p["x"], p["y"]
    s, sx, sy, rot, alpha, blur = 1.0, 1.0, 1.0, p["rot"], 1.0, 0.0
    land = p["t"]
    if p["hops"]:
        kf = [(p["t"], p["x"], p["y"])] + p["hops"]
        for (ta, xa, ya), (tb, xb, yb) in zip(kf, kf[1:]):
            if t >= tb:
                x, y, land = xb, yb, tb
            elif t > tb - HOP:
                u = (t - (tb - HOP)) / HOP
                x = xa + (xb - xa) * ease_in_out(u)
                y = ya + (yb - ya) * ease_in_out(u) - 90 * math.sin(math.pi * u)
                sy, sx = 1.04, 0.97
                break
            else:
                break
    anim = p["anim"]
    if anim == "pop":
        k = prog(t, t0, 0.36)
        s = 0.55 + 0.45 * G.ease_out_back(k)
        alpha = clamp01(k * 3.5)
        rot += -7 * (1 - ease_out_cubic(k))
    elif anim == "drop":
        u = prog(t, t0, lead)
        y -= 320 * (1 - u * u)
        alpha = clamp01(u * 4)
    elif anim == "rise":
        k = prog(t, t0, 0.5)
        y += 90 * (1 - ease_out_cubic(k))
        alpha = clamp01(k * 2)
        blur = 9 * (1 - ease_out_cubic(k))
    if anim in ("drop",) or p["hops"]:
        dt = t - land
        if 0 <= dt < 0.3:  # squash on landing
            q = 0.09 * math.exp(-dt / 0.07) * math.cos(dt * 38)
            sy *= 1 - q
            sx *= 1 + q * 0.6
    out = prog(t, p["out"] - 0.2, 0.2)
    alpha *= 1 - out
    s *= 1 - 0.06 * out
    return x, y, s, sx, sy, rot, alpha, blur


def draw_props(c, t):
    z = world_push(t)
    for i, p in enumerate(PL.PROPS):
        pose = prop_pose(p, t)
        if pose is None:
            continue
        x, y, s, sx, sy, rot, alpha, blur = pose
        if alpha <= 0.003:
            continue
        # idle life: a slow bob and sway, never still
        y += 2.5 * math.sin(2 * math.pi * 0.45 * t + i * 1.7)
        rot += 0.7 * math.sin(2 * math.pi * 0.27 * t + i)
        x, y, s = 540 + (x - 540) * z, 960 + (y - 960) * z, s * z
        layers = [(p["key"], 1.0)]
        pulse = 1.0
        for ts, key in p["states"]:
            if t >= ts:
                u = prog(t, ts, 0.14)
                layers = [(layers[-1][0], 1 - u), (key, u)] if u < 1 else [(key, 1.0)]
                pulse = 1 + 0.08 * math.sin(math.pi * prog(t, ts, 0.26))
        for key, la in layers:
            img, bottom = obj(key, p["w"], p["rim"])
            iw, ih = img.width(), img.height()
            ay = bottom if p["anchor"] == "bottom" else ih / 2
            c.save()
            c.translate(x, y)
            if p["shadow"] and p["anchor"] == "bottom" and la == layers[-1][1]:
                sp = G.paint(skia.Color4f(0.35, 0.27, 0.2, 0.17 * alpha))
                sp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 9))
                c.drawOval(skia.Rect.MakeLTRB(-iw * 0.36 * s, -12, iw * 0.36 * s, 12), sp)
            c.rotate(rot)
            c.scale(s * sx * pulse, s * sy * pulse)
            lp = skia.Paint()
            lp.setAlphaf(alpha * la)
            if blur > 0.3:
                lp.setImageFilter(skia.ImageFilters.Blur(blur, blur))
            c.saveLayer(None, lp)
            c.drawImage(img, -iw / 2, -ay, G.CUBIC)
            if p["anim"] == "paint":  # the wash is laid in left to right with a soft wet front
                k = ease_in_out(prog(t, p["t"], 0.55))
                if k < 1:
                    fe = iw * 0.22
                    front = -iw / 2 - fe + (iw + 2 * fe) * k
                    sh = skia.GradientShader.MakeLinear([skia.Point(front - fe, 0), skia.Point(front, 0)],
                                                        [skia.ColorWHITE, skia.ColorTRANSPARENT])
                    mp = skia.Paint(Shader=sh)
                    mp.setBlendMode(skia.BlendMode.kDstIn)
                    c.drawRect(skia.Rect.MakeLTRB(-iw, -ih * 2, iw, ih * 2), mp)
            c.restore()
            c.restore()


# ------------------------------------------------------------------ text
def text_color(name, t):
    if name == "auto":
        return G.mixc(COL["ink"], COL["cream"], mood_dark(t))
    return COL[name]


def fitted(font_name, size, text):
    f = F(font_name, size)
    w = G.text_width(text, f, TRACK.get(font_name, 0.0))
    return F(font_name, size * MAX_W / w) if w > MAX_W else f


def line_groups():
    groups = {}
    for i, it in enumerate(PL.TEXT):
        if it[6] < 0:
            groups.setdefault(it[7], []).append((-it[6], i))
    pos = {}
    for y, items in groups.items():
        items.sort()
        fonts = [fitted(PL.TEXT[i][4], PL.TEXT[i][5], PL.TEXT[i][3]) for _, i in items]
        words = [PL.TEXT[i][3] for _, i in items]
        space = G.text_width(" ", fonts[0])
        total = sum(G.text_width(wd, f) for wd, f in zip(words, fonts)) + space * (len(words) - 1)
        x = W / 2 - total / 2
        for (_, i), wd, f in zip(items, words, fonts):
            pos[i] = x
            x += G.text_width(wd, f) + space
    return pos


LINE_POS = line_groups()


LABEL_W = 600


def text_font(i):
    t_in, t_out, kind, text, font, size, x, y, col = PL.TEXT[i]
    if kind == "label":
        f = F(font, size)
        w = G.text_width(text, f, TRACK.get(font, 0.0))
        return F(font, size * LABEL_W / w) if w > LABEL_W else f
    return fitted(font, size, text)


def text_geom(i):
    """(x0, x1, baseline, size) of a TEXT item as drawn."""
    t_in, t_out, kind, text, font, size, x, y, col = PL.TEXT[i]
    f = text_font(i)
    w = G.text_width(text, f, TRACK.get(font, 0.0))
    if i in LINE_POS:
        x0 = LINE_POS[i]
    elif kind == "label" or x < 0 or (kind == "brush" and x != 540):
        x0 = x
    else:
        x0 = x - w / 2
    return x0, x0 + w, y, f.getSize()


def find_text(s, t):
    cands = [i for i, it in enumerate(PL.TEXT) if it[3] == s and it[0] <= t + 1.0 and it[1] >= t]
    assert cands, s
    return cands[0]


def draw_texts(c, t):
    for i, (t_in, t_out, kind, text, font, size, x, y, col) in enumerate(PL.TEXT):
        if t < t_in or t > t_out:
            continue
        f = text_font(i)
        color = text_color(col, t)
        fo = 0.2
        out = prog(t, t_out - fo, fo)
        align = "left" if x < 0 or kind == "label" or (kind == "brush" and x != 540) else "center"
        xx = LINE_POS.get(i, x)
        if kind in ("rise", "label"):
            k = prog(t, t_in, 0.55)
            c.save()
            c.translate(0, -16 * ease_in_out(out))
            P.rise_text(c, text, xx, y, f, color, k, align=align, alpha=1 - out, tracking=TRACK.get(font, 0.0))
            c.restore()
        else:
            dur = (0.06 if t_in < 0 else 0.2) if t_in < 1 else min(0.75, 0.12 + 0.055 * len(text))
            k = prog(t, t_in, dur)
            P.written_text(c, text, xx, y, f, color, k, ink=asset("ink"), align=align, seed=i, alpha=1 - out,
                           jitter=1.0, bold=f.getSize() * 0.012)


DOT_POS = np.random.default_rng(12).uniform(0, 1, (len(PL.PLUCK_WORDS), 3))


def draw_dots(c, t, cx, cy, r):
    if not 31.4 < t < 35.8:
        return
    fade = 1 - prog(t, 35.15, 0.5)
    for k, wi in enumerate(PL.PLUCK_WORDS):
        t0 = PL.WORDS[wi]["s"]
        if t < t0:
            continue
        ang = -math.pi * 1.12 + DOT_POS[k, 0] * math.pi * 1.24
        rr = r + 70 + DOT_POS[k, 1] * 70
        x, y = cx + math.cos(ang) * rr, cy + math.sin(ang) * rr * 0.95
        s = 0.35 + 0.45 * DOT_POS[k, 2]
        P.draw_bloom(c, asset(f"dot{k % 4}"), x, y, prog(t, t0, 0.35), scale=s, alpha=0.85 * fade)


# ------------------------------------------------------------------ frame
def draw_frame(t):
    surf = G.surface(W, H)
    c = surf.getCanvas()
    # camera: the canvas drifts slowly under a fixed frame (Motion Canvas-style camera move)
    ox = -80 + 46 * math.sin(t * 0.13 + 0.6)
    oy = -80 + 52 * math.cos(t * 0.09)
    c.drawImage(asset("canvas"), ox, oy)
    cx, cy, r, _ = layout_at(t)
    draw_washes(c, t, cx, cy)
    draw_halo(c, t, cx, cy, r)
    draw_dots(c, t, cx, cy, r)
    draw_props(c, t)
    cx, cy, r = draw_speaker(c, t)
    draw_strokes(c, t, cx, cy, r)
    draw_texts(c, t)
    return surf


GRAIN = None
VIG = None


def finish(rgb, k):
    global GRAIN, VIG
    if GRAIN is None:
        GRAIN = G.Grain(H, W, n=6, sigma=3.2, seed=11)
        VIG = G.vignette_mask(H, W, 0.11, 2.4)
    out = GRAIN.apply(rgb, k, 1.0) * VIG
    return np.clip(out, 0, 255).astype(np.uint8)


def frame_at(k):
    t = k / FPS
    rgb = np.asarray(G.snapshot(draw_frame(t)))[..., :3].astype(np.float32)
    return finish(rgb, k)


def render_range(k0, k1, out_path):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
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
            Image.fromarray(frame_at(int(round(float(ts) * FPS)))).save(os.path.join(out, f"f_{float(ts):06.2f}.png"))
    elif mode == "range":
        render_range(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif mode == "video":
        out, audio = sys.argv[2], sys.argv[3]
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        total = int(round(PL.DURATION * FPS))
        tmp = out + ".parts"
        os.makedirs(tmp, exist_ok=True)
        bounds = np.linspace(0, total, workers + 1).astype(int)
        procs = [subprocess.Popen([sys.executable, __file__, "range", str(bounds[i]), str(bounds[i + 1]),
                                   os.path.join(tmp, f"part{i}.mp4")]) for i in range(workers)]
        codes = [p.wait() for p in procs]
        assert all(c_ == 0 for c_ in codes), codes
        with open(os.path.join(tmp, "list.txt"), "w") as fl:
            for i in range(workers):
                fl.write(f"file 'part{i}.mp4'\n")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", os.path.join(tmp, "list.txt"),
                        "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        "-ar", "48000", "-shortest", "-movflags", "+faststart", out], check=True)
        print("wrote", out)
