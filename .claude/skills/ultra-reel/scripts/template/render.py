"""Painterly picture ("холст") for the ICON testimonial reel, driven by plan.py.

One continuous canvas, no cuts: the speaker node glides between layouts, words rise or are written
by a brush, watercolor washes change the mood. Motion is eased and calm; nothing is drawn that the
speaker did not say.

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
sys.path.insert(0, os.environ.get("ULTRA_ENGINE", os.path.join(HERE, "..")))  # <skill>/scripts
sys.path.insert(0, HERE)

from engine import gfx as G  # noqa: E402
from engine import paint as P  # noqa: E402
from engine.gfx import prog, clamp01, ease_in_out, ease_out_cubic  # noqa: E402
import plan as PL  # noqa: E402

W, H, FPS = PL.W, PL.H, PL.FPS
FONT_DIR = os.environ.get("REEL_FONTS", os.path.expanduser("~/.cache/ultra-reel/fonts"))
F = G.Fonts(FONT_DIR, {
    "serif": "CormorantGaramond-600.ttf", "serif_i": "CormorantGaramond-600i.ttf",
    "serif_wide": "CormorantGaramond-600.ttf", "serif_small": "CormorantGaramond-500.ttf",
    "brush": "ComforterBrush-400.ttf",
})
RGB = {"canvas": (241, 234, 223), "ink": (42, 39, 36), "terra": (185, 87, 62), "ochre": (201, 154, 62),
       "dusk": (46, 59, 78), "cream": (243, 236, 224), "grey": (139, 132, 120)}
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
    groove = scene_index(t) not in (0, 15, 19) and not (PL.TAPE_STOP < t < 27.68)
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
        if w_[4] == "ochre" or w_[9] != "speaker":
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
        elif kind == "check":
            x, y, s = p["x"], p["y"], p["s"]
            pts = [(x - 0.45 * s, y), (x - 0.12 * s, y + 0.34 * s), (x + 0.5 * s, y - 0.45 * s)]
            _STROKES[key] = P.Stroke(pts, s * 0.2, seed=p["seed"], dryness=0.5, taper=(0.1, 0.35))
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
        else:
            st.draw(c, COL[col], k, alpha=fade)


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


def draw_texts(c, t):
    for i, (t_in, t_out, kind, text, font, size, x, y, col) in enumerate(PL.TEXT):
        if t < t_in or t > t_out:
            continue
        f = fitted(font, size, text)
        color = text_color(col, t)
        fo = 0.2
        out = prog(t, t_out - fo, fo)
        align = "left" if x < 0 or (kind == "brush" and x != 540) else "center"
        xx = LINE_POS.get(i, x)
        if kind == "rise":
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
        ang = -math.pi * 0.95 + DOT_POS[k, 0] * math.pi * 1.9
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
