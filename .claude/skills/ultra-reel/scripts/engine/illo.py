"""Ink-and-wash illustrations for the canvas style: objects drawn like a sketchbook — a wobbly ink line over a
watercolor fill that is slightly off-register, granulated, darker at its drying edge.

Every object is a function returning a skia.Image (straight alpha) at a given pixel size; the shapes are built on a
fixed design grid and seeded, so an object looks the same in every frame and every render worker. Animation happens
outside (pop, drop, painted reveal, state swap), never by re-drawing the object differently.
"""
import math

import numpy as np
import skia

from . import gfx as G

INK = (42, 39, 36)
PAL = {
    "terra": (185, 87, 62), "slate": (70, 92, 120), "ochre": (201, 154, 62), "sage": (122, 140, 104),
    "kraft": (196, 160, 112), "cream": (243, 236, 224), "skin": (226, 190, 156), "paper": (250, 246, 238),
    "ink": INK, "red": (176, 64, 48), "gold": (214, 168, 74),
}


def c4(rgb, a=1.0):
    return skia.Color4f(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, a)


_TEX = {}


def grain_image(size=512, seed=1):
    """Alpha texture: watercolor granulation + pale blooms (used to texture fills via DstIn)."""
    if seed not in _TEX:
        rng = np.random.default_rng(seed)
        a = G.value_noise(size, size, 2.2, rng) * 0.45 + G.value_noise(size, size, 9, rng) * 0.3 + G.value_noise(size, size, 60, rng) * 0.25
        a = 0.62 + 0.38 * a
        rgba = np.zeros((size, size, 4), np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        _TEX[seed] = G.image(rgba)
    return _TEX[seed]


class Sketch:
    """Collects (path, fill, ink width) items in design units and paints them at any scale."""

    def __init__(self, w, h, seed=0):
        self.w, self.h, self.seed = w, h, seed
        self.items = []

    def add(self, path, fill=None, ink=4.0, ink_rgb=INK, wash_alpha=0.88, off=(5, 4), edge=True):
        self.items.append((path, fill, ink, ink_rgb, wash_alpha, off, edge))
        return self

    def text(self, s, x, y, size, font, rgb=INK, align="center"):
        self.items.append(("text", s, x, y, size, font, rgb, align))
        return self

    def render(self, px_w, min_ink_px=2.4):
        k = px_w / self.w
        W, H = int(math.ceil(self.w * k)) + 24, int(math.ceil(self.h * k)) + 24
        surf = G.surface(W, H)
        c = surf.getCanvas()
        c.translate(12, 12)
        c.scale(k, k)
        rng = np.random.default_rng(self.seed)
        tex = grain_image(seed=self.seed % 5 + 1)
        for it in self.items:
            if it[0] == "text":
                _, s, x, y, size, font, rgb, align = it
                G.draw_text(c, s, x, y, font(size), c4(rgb), align=align)
                continue
            path, fill, ink, ink_rgb, wa, off, edge = it
            if fill is not None:
                rgb = PAL[fill] if isinstance(fill, str) else fill
                c.saveLayer(None, None)
                c.save()
                c.translate(off[0] * rng.uniform(0.4, 1.0), off[1] * rng.uniform(0.4, 1.0))
                c.drawPath(path, G.paint(c4(rgb, wa)))
                if edge:  # pigment pools at the drying edge
                    ep = G.paint(c4(tuple(int(v * 0.78) for v in rgb), 0.45), stroke=7 / k)
                    ep.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 2.5 / k))
                    c.save()
                    c.clipPath(path, doAntiAlias=True)
                    c.drawPath(path, ep)
                    c.restore()
                c.restore()
                tp = skia.Paint()
                tp.setBlendMode(skia.BlendMode.kDstIn)
                m = skia.Matrix.Scale(1 / k, 1 / k)
                tp.setShader(tex.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat, G.LINEAR, m))
                c.drawPaint(tp)
                c.restore()
            if ink and ink > 0:
                ink = max(ink, min_ink_px / k)
                eff = skia.DiscretePathEffect.Make(16 / k, 0.9 / k, int(rng.integers(1, 1 << 30)))
                p1 = G.paint(c4(ink_rgb, 0.92), stroke=ink, cap=skia.Paint.kRound_Cap, join=skia.Paint.kRound_Join,
                             effect=eff)
                c.drawPath(path, p1)
                # a second, thinner pass slightly off the first: the line breathes like a real nib
                c.save()
                c.translate(0.9 / k, 0.6 / k)
                p2 = G.paint(c4(ink_rgb, 0.45), stroke=ink * 0.45, cap=skia.Paint.kRound_Cap,
                             effect=skia.DiscretePathEffect.Make(11 / k, 1.2 / k, int(rng.integers(1, 1 << 30))))
                c.drawPath(path, p2)
                c.restore()
        return surf.makeImageSnapshot()


# ------------------------------------------------------------------ path helpers (design units)
def rrect(x, y, w, h, r):
    p = skia.Path()
    p.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r))
    return p


def ellipse(cx, cy, rx, ry):
    p = skia.Path()
    p.addOval(skia.Rect.MakeLTRB(cx - rx, cy - ry, cx + rx, cy + ry))
    return p


def poly(pts, close=True):
    p = skia.Path()
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    if close:
        p.close()
    return p


def curve(pts):
    """Smooth open curve through points (Catmull-Rom -> cubic)."""
    p = skia.Path()
    p.moveTo(*pts[0])
    for i in range(len(pts) - 1):
        p0 = pts[max(0, i - 1)]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[min(len(pts) - 1, i + 2)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        p.cubicTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    return p


# ------------------------------------------------------------------ objects
def limb(pts, width):
    """Closed outline of a thick polyline (sleeve, leg) with rounded ends."""
    p = skia.Path()
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    paint = skia.Paint(Style=skia.Paint.kStroke_Style, StrokeWidth=width, StrokeCap=skia.Paint.kRound_Cap,
                       StrokeJoin=skia.Paint.kRound_Join)
    out = skia.Path()
    paint.getFillPath(p, out)
    return skia.Op(out, skia.Path(), skia.PathOp.kUnion_PathOp) or out


ARMS = {  # (shoulder->elbow->hand) offsets for the right arm, mirrored for the left
    None: [(56, -252), (70, -192), (66, -134)],
    "wave": [(56, -252), (96, -296), (104, -352)],
    "point": [(56, -252), (104, -228), (156, -246)],
    "hold": [(56, -252), (80, -196), (122, -186)],
    "cheer": [(56, -252), (92, -300), (88, -362)],
}
HAIR = {
    "short": [(-36, -322), (-30, -350), (-6, -358), (22, -354), (38, -334), (36, -318), (18, -332), (-12, -334)],
    "side": [(-38, -316), (-34, -348), (-10, -360), (24, -356), (40, -330), (38, -312), (26, -334), (-4, -340), (-24, -330)],
}


def figure(sk, x, y, color, s=1.0, arm_r=None, arm_l=None, hair="short", legs=(78, 72, 70)):
    """A faceless person (never a portrait). (x, y) = between the feet, height ~370*s."""
    def P(dx, dy):
        return (x + dx * s, y + dy * s)

    body = skia.Path()
    body.moveTo(*P(-50, -118))
    body.lineTo(*P(-58, -228))
    body.cubicTo(*P(-60, -258), *P(-42, -270), *P(-16, -272))
    body.lineTo(*P(16, -272))
    body.cubicTo(*P(42, -270), *P(60, -258), *P(58, -228))
    body.lineTo(*P(50, -118))
    body.cubicTo(*P(20, -110), *P(-20, -110), *P(-50, -118))
    body.close()
    head = ellipse(*P(0, -318), 34 * s, 38 * s)
    for side in (-1, 1):  # legs + shoes, cut at the hem so no line shows through the wash
        leg = limb([P(side * 22, -128), P(side * 24, -64), P(side * 24, -14)], 30 * s)
        sk.add(skia.Op(leg, body, skia.PathOp.kDifference_PathOp), legs, 3.8)
        sk.add(ellipse(*P(side * 30, -8), 24 * s, 11 * s), "ink", 3.0, wash_alpha=0.85)
    neck = skia.Op(rrect(*P(-12, -292), 24 * s, 26 * s, 6 * s), head, skia.PathOp.kDifference_PathOp)
    sk.add(skia.Op(neck, body, skia.PathOp.kDifference_PathOp), "skin", 2.6)
    arms = []
    for side, arm in ((1, arm_r), (-1, arm_l)):
        pts = [P(side * dx, dy) for dx, dy in ARMS[arm]]
        arms.append((limb(pts, 26 * s), pts[-1]))
    torso = body
    for a, _ in arms:  # the sleeve is in front: no torso line under it
        torso = skia.Op(torso, a, skia.PathOp.kDifference_PathOp)
    sk.add(torso, color, 4.0)
    for a, hand in arms:
        sk.add(a, color, 3.6)
        sk.add(ellipse(*hand, 13 * s, 13 * s), "skin", 3.0)
    sk.add(head, "skin", 3.8)
    sk.add(poly([P(dx, dy) for dx, dy in HAIR[hair]]), (58, 50, 46), 3.0, wash_alpha=0.9)


def duo(px, arms=((None, None), (None, None)), seed=3, gap=200):
    """The two marketers: terracotta + slate. arms = ((right, left) of A, (right, left) of B)."""
    w = 260 + gap
    sk = Sketch(w, 400, seed)
    figure(sk, 130, 392, "terra", 1.0, arms[0][0], arms[0][1], hair="short")
    figure(sk, 130 + gap, 392, "slate", 1.04, arms[1][0], arms[1][1], hair="side")
    return sk.render(px)


def figure_one(px, color="terra", seed=5, arm_r=None, arm_l=None, hair="short"):
    sk = Sketch(260, 400, seed)
    figure(sk, 130, 392, color, 1.0, arm_r, arm_l, hair)
    return sk.render(px)


def sign(px, text, font, seed=7):
    sk = Sketch(560, 300, seed)
    sk.add(poly([(140, 20), (100, 110)], close=False), None, 5)
    sk.add(poly([(420, 20), (460, 110)], close=False), None, 5)
    sk.add(ellipse(280, 18, 10, 10), "ink", 3)
    sk.add(poly([(140, 20), (280, 18), (420, 20)], close=False), None, 4)
    sk.add(rrect(40, 100, 480, 170, 18), "paper", 5)
    sk.text(text, 280, 222, 104, font, INK)
    return sk.render(px)


def book_open(px, written=False, title=None, font=None, seed=11):
    sk = Sketch(760, 460, seed)
    left = curve([(380, 80), (300, 50), (160, 46), (40, 70)])
    pl = poly([(380, 80), (300, 50), (160, 46), (40, 70), (40, 420), (160, 400), (300, 404), (380, 430)])
    pr = poly([(380, 80), (460, 50), (600, 46), (720, 70), (720, 420), (600, 400), (460, 404), (380, 430)])
    sk.add(poly([(30, 90), (380, 110), (730, 90), (730, 440), (380, 456), (30, 440)]), "kraft", 4.5)
    sk.add(pl, "paper", 4.0, wash_alpha=0.96)
    sk.add(pr, "paper", 4.0, wash_alpha=0.96)
    sk.add(poly([(380, 80), (380, 430)], close=False), None, 4)
    for i in range(6):
        y = 140 + i * 44
        a = 1.0 if written else 0.35
        rgb = INK if written else (180, 172, 160)
        if i == 0 and title is not None and written:
            sk.text(title, 210, y + 12, 50, font, PAL["terra"])
        else:
            ln = 230 - (i * 31 % 70)
            sk.add(curve([(90, y), (90 + ln * 0.5, y - 3), (90 + ln, y + 2)]), None, 5 if written else 3, ink_rgb=rgb)
        if written:
            # right page: little traffic arrows, the course is about traffic
            if i in (1, 3):
                sk.add(curve([(440, y), (520, y - 16), (600, y + 4)]), None, 5, ink_rgb=PAL["terra"])
                sk.add(poly([(600, y + 4), (582, y - 12), (586, y + 16)]), "terra", 3, ink_rgb=PAL["terra"])
            else:
                ln = 200 - (i * 37 % 80)
                sk.add(curve([(440, y), (440 + ln * 0.5, y + 3), (440 + ln, y - 1)]), None, 5)
        else:
            ln = 200 - (i * 37 % 80)
            sk.add(curve([(440, y), (440 + ln * 0.5, y + 3), (440 + ln, y - 1)]), None, 3, ink_rgb=(180, 172, 160))
    return sk.render(px)


def book_closed(px, title, font, seal=False, seed=13):
    sk = Sketch(520, 640, seed)
    sk.add(rrect(70, 40, 400, 560, 14), "slate", 5)
    sk.add(rrect(70, 40, 46, 560, 10), (52, 70, 94), 4)
    sk.add(rrect(150, 140, 280, 150, 10), "paper", 4)
    lines = title.split("\n")
    for i, ln in enumerate(lines):
        sk.text(ln, 290, 200 + i * 56, 46, font, INK)
    sk.add(poly([(150, 600), (470, 600), (470, 612), (150, 612)]), "paper", 3)
    if seal:
        blob = []
        rng = np.random.default_rng(seed)
        for k in range(18):
            a = k / 18 * 2 * math.pi
            r = 88 * (1 + 0.08 * rng.standard_normal())
            blob.append((360 + r * math.cos(a), 470 + r * math.sin(a)))
        sk.add(poly(blob), "red", 4)
        sk.add(ellipse(360, 470, 58, 58), None, 4, ink_rgb=(120, 40, 30))
        star = G.star_path(360, 470, 40, 16, 5)
        sk.add(star, (150, 52, 38), 0)
    return sk.render(px)


def cap(px, seed=17):
    sk = Sketch(420, 300, seed)
    sk.add(poly([(110, 120), (310, 120), (300, 230), (120, 230)]), "ink", 4, wash_alpha=0.9)
    sk.add(poly([(210, 30), (400, 100), (210, 170), (20, 100)]), "ink", 4.5, wash_alpha=0.95)
    sk.add(curve([(210, 100), (330, 120), (350, 200)]), None, 5, ink_rgb=PAL["gold"])
    sk.add(ellipse(352, 214, 14, 20), "gold", 3)
    return sk.render(px)


def stairs(px, seed=19):
    sk = Sketch(820, 520, seed)
    pts = [(20, 500), (20, 400), (220, 400), (220, 300), (420, 300), (420, 200), (620, 200), (620, 100), (800, 100), (800, 500)]
    sk.add(poly(pts), "kraft", 5)
    for x, y in ((120, 450), (320, 350), (520, 250), (710, 150)):
        sk.add(poly([(x - 60, y), (x + 60, y)], close=False), None, 3, ink_rgb=(150, 120, 80))
    return sk.render(px)


def toolbox(px, seed=23):
    sk = Sketch(620, 420, seed)
    sk.add(poly([(40, 170), (580, 170), (560, 400), (60, 400)]), "terra", 5)
    sk.add(poly([(40, 170), (580, 170), (600, 120), (20, 120)]), (160, 72, 52), 5)
    sk.add(curve([(220, 120), (230, 60), (390, 60), (400, 120)]), None, 12)
    sk.add(rrect(270, 230, 80, 40, 8), "gold", 4)
    return sk.render(px)


def wrench(px, seed=29):
    sk = Sketch(300, 300, seed)
    head = ellipse(150, 70, 54, 54)
    handle = rrect(130, 90, 40, 190, 18)
    jaw = rrect(134, -10, 32, 76, 6)
    p = skia.Op(head, handle, skia.PathOp.kUnion_PathOp)
    p = skia.Op(p, jaw, skia.PathOp.kDifference_PathOp)
    p.transform(skia.Matrix.RotateDeg(38, skia.Point(150, 160)))
    sk.add(p, (150, 156, 168), 4.5)
    return sk.render(px)


def gear(px, seed=31):
    sk = Sketch(300, 300, seed)
    pts = []
    for k in range(32):
        a = k / 32 * 2 * math.pi
        r = 130 if (k // 2) % 2 == 0 else 104
        pts.append((150 + r * math.cos(a), 150 + r * math.sin(a)))
    sk.add(poly(pts), "ochre", 4.5)
    sk.add(ellipse(150, 150, 42, 42), "paper", 4.5)
    return sk.render(px)


def magnet(px, seed=37):
    sk = Sketch(300, 300, seed)
    outer = skia.Path()
    outer.moveTo(60, 40)
    outer.lineTo(60, 160)
    outer.cubicTo(60, 260, 240, 260, 240, 160)
    outer.lineTo(240, 40)
    outer.lineTo(176, 40)
    outer.lineTo(176, 160)
    outer.cubicTo(176, 196, 124, 196, 124, 160)
    outer.lineTo(124, 40)
    outer.close()
    sk.add(outer, "red", 4.5)
    sk.add(rrect(60, 40, 64, 46, 4), "paper", 4)
    sk.add(rrect(176, 40, 64, 46, 4), "paper", 4)
    return sk.render(px)


def chart(px, seed=41):
    sk = Sketch(300, 300, seed)
    sk.add(rrect(20, 30, 260, 240, 14), "paper", 4.5)
    for i, hgt in enumerate((60, 100, 80, 150)):
        sk.add(rrect(50 + i * 56, 240 - hgt, 36, hgt, 4), "slate" if i < 3 else "terra", 3.5)
    return sk.render(px)


def globe(px, seed=43):
    sk = Sketch(440, 440, seed)
    sk.add(ellipse(220, 220, 196, 196), "slate", 5, wash_alpha=0.55)
    for rx in (196, 120, 46):
        sk.add(ellipse(220, 220, rx, 196), None, 3)
    for dy in (-110, 0, 110):
        hw = math.sqrt(196 ** 2 - dy ** 2)
        sk.add(poly([(220 - hw, 220 + dy), (220 + hw, 220 + dy)], close=False), None, 3)
    land = [(250, 100), (330, 92), (372, 140), (350, 190), (300, 196), (262, 160)]
    sk.add(curve(land + [land[0]]), "sage", 3)
    return sk.render(px)


def pin(px, seed=47):
    sk = Sketch(160, 220, seed)
    p = skia.Path()
    p.moveTo(80, 210)
    p.cubicTo(20, 130, 14, 96, 14, 80)
    p.cubicTo(14, 40, 44, 12, 80, 12)
    p.cubicTo(116, 12, 146, 40, 146, 80)
    p.cubicTo(146, 96, 140, 130, 80, 210)
    p.close()
    sk.add(p, "terra", 5)
    sk.add(ellipse(80, 78, 26, 26), "paper", 4)
    return sk.render(px)


def heart(px, seed=53, color="terra"):
    sk = Sketch(240, 220, seed)
    p = skia.Path()
    p.moveTo(120, 200)
    p.cubicTo(20, 130, 10, 60, 60, 36)
    p.cubicTo(96, 20, 116, 44, 120, 64)
    p.cubicTo(124, 44, 144, 20, 180, 36)
    p.cubicTo(230, 60, 220, 130, 120, 200)
    p.close()
    sk.add(p, color, 4.5)
    return sk.render(px)


def magnifier(px, seed=59):
    sk = Sketch(300, 300, seed)
    sk.add(poly([(186, 186), (268, 268)], close=False), None, 26, ink_rgb=(120, 90, 60))
    sk.add(ellipse(120, 120, 86, 86), (210, 226, 236), 9, wash_alpha=0.6)
    return sk.render(px)


def pedestal(sk, x, y, w, h):
    """Gold plinth with a painted star; (x, y) = top centre."""
    sk.add(poly([(x - w / 2 - 14, y + h), (x - w / 2, y), (x + w / 2, y), (x + w / 2 + 14, y + h)]), "gold", 5)
    sk.add(poly([(x - w / 2 - 22, y + 6), (x + w / 2 + 22, y + 6), (x + w / 2 + 22, y - 16), (x - w / 2 - 22, y - 16)]),
           (190, 146, 60), 4.5)
    sk.add(G.star_path(x, y + h * 0.55, h * 0.3, h * 0.13, 5), "terra", 3)


def duo_on_pedestal(px, seed=61):
    sk = Sketch(560, 620, seed)
    pedestal(sk, 280, 440, 420, 170)
    figure(sk, 190, 424, "terra", 1.0, "cheer", None, hair="short")
    figure(sk, 380, 424, "slate", 1.04, None, "cheer", hair="side")
    return sk.render(px)


def footstep(px, seed=67, left=True):
    sk = Sketch(120, 200, seed)
    sk.add(ellipse(60, 120, 34, 62), "ink", 0, wash_alpha=0.75)
    for k in range(5):
        x = 30 + k * 15
        sk.add(ellipse(x, 40 - (k == 2) * 6, 9, 12), "ink", 0, wash_alpha=0.75)
    return sk.render(px)


def sparkle(px, seed=71, color="gold"):
    sk = Sketch(200, 200, seed)
    sk.add(G.star_path(100, 100, 92, 22, 4), color, 3)
    return sk.render(px)


def with_rim(img, r=7, rgb=PAL["paper"], soft=1.5):
    """The object as a paper cut-out: its silhouette dilated in paper colour under it."""
    w, h = img.width() + 4 * r, img.height() + 4 * r
    surf = G.surface(w, h)
    c = surf.getCanvas()
    rim = skia.Paint()
    filt = skia.ImageFilters.Dilate(r, r)
    filt = skia.ImageFilters.Blur(soft, soft, skia.TileMode.kDecal, filt)
    cf = skia.ColorFilters.Matrix([0, 0, 0, 0, rgb[0] / 255, 0, 0, 0, 0, rgb[1] / 255, 0, 0, 0, 0, rgb[2] / 255,
                                   0, 0, 0, 1.4, 0])
    rim.setImageFilter(skia.ImageFilters.ColorFilter(cf, filt))
    c.drawImage(img, 2 * r, 2 * r, G.LINEAR, rim)
    c.drawImage(img, 2 * r, 2 * r)
    return surf.makeImageSnapshot()


def magnifier_duo(px, seed=59):
    """The magnifier after the search: the duo found inside its lens."""
    sk = Sketch(300, 300, seed)
    sk.add(poly([(186, 186), (268, 268)], close=False), None, 26, ink_rgb=(120, 90, 60))
    sk.add(ellipse(120, 120, 86, 86), (210, 226, 236), 0, wash_alpha=0.6)
    figure(sk, 94, 176, "terra", 0.26, hair="short")
    figure(sk, 146, 176, "slate", 0.27, hair="side")
    sk.add(ellipse(120, 120, 86, 86), None, 9)
    return sk.render(px)
