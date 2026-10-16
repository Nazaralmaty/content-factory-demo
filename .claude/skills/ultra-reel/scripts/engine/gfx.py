"""2D drawing kit on top of skia-python: RGBA surfaces, shaped text, kinetic glyphs, shapes,
circle-masked video, halftone, paper/grain textures and frame post-processing."""
import math
import os

import numpy as np
import skia

RGBA = skia.kRGBA_8888_ColorType
CUBIC = skia.SamplingOptions(skia.CubicResampler.Mitchell())
LINEAR = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)


# ------------------------------------------------------------------ colours & easing
def hexc(h, a=1.0):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return skia.Color4f(r / 255, g / 255, b / 255, a)


def mixc(c1, c2, t):
    return skia.Color4f(c1.fR + (c2.fR - c1.fR) * t, c1.fG + (c2.fG - c1.fG) * t,
                        c1.fB + (c2.fB - c1.fB) * t, c1.fA + (c2.fA - c1.fA) * t)


def with_alpha(c, a):
    return skia.Color4f(c.fR, c.fG, c.fB, c.fA * a)


def clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def prog(t, t0, dur):
    return clamp01((t - t0) / dur) if dur > 0 else float(t >= t0)


def ease_out_cubic(x):
    return 1 - (1 - x) ** 3


def ease_in_cubic(x):
    return x ** 3


def ease_in_out(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_expo(x):
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ease_in_expo(x):
    return 0.0 if x <= 0 else 2 ** (10 * x - 10)


def ease_out_back(x, s=1.70158):
    x = x - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def spring(x, freq=4.5, damp=5.0):
    """0 -> 1 with an overshooting, decaying wobble (x in seconds-normalised units)."""
    if x <= 0:
        return 0.0
    return 1 - math.exp(-damp * x) * math.cos(2 * math.pi * freq * x)


def pulse(t, t0, decay=0.18):
    """1 at t0 decaying exponentially afterwards (0 before)."""
    if t < t0:
        return 0.0
    return math.exp(-(t - t0) / decay)


# ------------------------------------------------------------------ surfaces
def surface(w, h):
    return skia.Surface.MakeRaster(skia.ImageInfo.Make(w, h, RGBA, skia.kPremul_AlphaType))


def snapshot(surf):
    return surf.makeImageSnapshot().toarray(colorType=RGBA)


def image(arr):
    """HxWx4 (straight alpha) or HxWx3 uint8 -> skia.Image."""
    if arr.shape[2] == 3:
        arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
    return skia.Image.fromarray(np.ascontiguousarray(arr), colorType=RGBA, alphaType=skia.kUnpremul_AlphaType)


def paint(color=None, stroke=None, aa=True, alpha=None, blend=None, filt=None, cap=None, join=None, effect=None):
    p = skia.Paint(AntiAlias=aa)
    if color is not None:
        p.setColor4f(color if alpha is None else with_alpha(color, alpha))
    if stroke is not None:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
    if blend is not None:
        p.setBlendMode(blend)
    if filt is not None:
        p.setImageFilter(filt)
    if cap is not None:
        p.setStrokeCap(cap)
    if join is not None:
        p.setStrokeJoin(join)
    if effect is not None:
        p.setPathEffect(effect)
    return p


def shadow(dx=0, dy=14, blur=22, alpha=0.28, color="#000000"):
    return skia.ImageFilters.DropShadow(dx, dy, blur, blur, hexc(color, alpha).toColor())


# ------------------------------------------------------------------ fonts & text
class Fonts:
    def __init__(self, font_dir, files):
        self.dir = font_dir
        self.files = files
        self._tf = {}
        self._f = {}

    def tf(self, name):
        if name not in self._tf:
            self._tf[name] = skia.Typeface.MakeFromFile(os.path.join(self.dir, self.files[name]))
        return self._tf[name]

    def __call__(self, name, size):
        key = (name, round(size, 1))
        if key not in self._f:
            f = skia.Font(self.tf(name), size)
            f.setSubpixel(True)
            f.setEdging(skia.Font.Edging.kAntiAlias)
            self._f[key] = f
        return self._f[key]


def glyph_layout(text, font, tracking=0.0):
    """Glyph ids and x offsets (with tracking in em fraction) for a single line."""
    glyphs = font.textToGlyphs(text)
    widths = font.getWidths(glyphs)
    xs, x = [], 0.0
    track = tracking * font.getSize()
    for wdt in widths:
        xs.append(x)
        x += wdt + track
    return glyphs, xs, (x - track if len(widths) else 0.0)


def text_width(text, font, tracking=0.0):
    if tracking == 0:
        return font.measureText(text)
    return glyph_layout(text, font, tracking)[2]


def cap_height(font):
    m = font.getMetrics()
    return -m.fAscent * 0.72 if not m.fCapHeight else m.fCapHeight


def draw_text(c, text, x, y, font, color, align="left", tracking=0.0, alpha=1.0, stroke=None, filt=None):
    """y is the baseline. Uses HarfBuzz shaping when no tracking is requested."""
    if not text:
        return 0.0
    w = text_width(text, font, tracking)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    p = paint(color, stroke=stroke, alpha=alpha, filt=filt)
    if tracking == 0:
        blob = skia.TextBlob.MakeFromShapedText(text, font)
        y += font.getMetrics().fAscent  # shaped blobs are laid out from the line top, not the baseline
    else:
        glyphs, xs, _ = glyph_layout(text, font, tracking)
        blob = skia.TextBlob.MakeFromPosTextH(text, xs, 0, font)
    if blob is not None:
        c.drawTextBlob(blob, x, y, p)
    return w


def draw_glyphs_kinetic(c, text, x, y, font, color, t_local, stagger=0.035, dur=0.32, mode="rise",
                        tracking=0.0, align="left", alpha=1.0, seed=0):
    """Per-letter entrance animation. t_local is seconds since the word started."""
    glyphs, xs, w = glyph_layout(text, font, tracking)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    size = font.getSize()
    rng = np.random.default_rng(seed)
    for i, ch in enumerate(text):
        k = prog(t_local, i * stagger, dur)
        if k <= 0:
            continue
        c.save()
        gx, gy = x + xs[i], y
        if mode == "rise":
            e = ease_out_back(k, 2.2)
            c.translate(gx, gy + (1 - e) * size * 0.55)
            a = clamp01(k * 3)
        elif mode == "drop":
            e = spring(k * 0.9, 3.2, 6.0)
            c.translate(gx, gy - (1 - e) * size * 0.9)
            a = clamp01(k * 4)
        elif mode == "scale":
            e = ease_out_back(k, 3.0)
            cw = font.getWidths([glyphs[i]])[0]
            c.translate(gx + cw / 2, gy - size * 0.35)
            c.scale(max(e, 0.001), max(e, 0.001))
            c.translate(-cw / 2, size * 0.35)
            a = clamp01(k * 3)
        elif mode == "jitter":
            j = (1 - k) * size * 0.25
            c.translate(gx + rng.uniform(-j, j), gy + rng.uniform(-j, j))
            c.rotate(float(rng.uniform(-12, 12) * (1 - k)))
            a = clamp01(k * 3)
        else:
            c.translate(gx, gy)
            a = k
        blob = skia.TextBlob.MakeFromText(ch, font)
        if blob is not None:
            c.drawTextBlob(blob, 0, 0, paint(color, alpha=a * alpha))
        c.restore()
    return w


def fit_font(fonts, name, text, max_w, size, min_size=20, tracking=0.0):
    f = fonts(name, size)
    w = text_width(text, f, tracking)
    if w > max_w:
        f = fonts(name, max(min_size, size * max_w / w))
    return f


# ------------------------------------------------------------------ shapes
def circle_path(cx, cy, r):
    p = skia.Path()
    p.addCircle(cx, cy, r)
    return p


def rrect(c, x, y, w, h, r, color, alpha=1.0, stroke=None, filt=None):
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r),
                paint(color, stroke=stroke, alpha=alpha, filt=filt))


def trimmed(path, t0, t1):
    return skia.TrimPathEffect.Make(t0, t1)


def stroke_path(c, path, color, width, progress=1.0, alpha=1.0, cap=skia.Paint.kRound_Cap, rough=0.0, seed=1):
    if progress <= 0:
        return
    eff = skia.TrimPathEffect.Make(0, clamp01(progress)) if progress < 1 else None
    if rough > 0:
        disc = skia.DiscretePathEffect.Make(8, rough, seed)
        eff = disc if eff is None else skia.PathEffect.MakeCompose(eff, disc)
    c.drawPath(path, paint(color, stroke=width, alpha=alpha, cap=cap, join=skia.Paint.kRound_Join, effect=eff))


def check_path(x, y, s):
    p = skia.Path()
    p.moveTo(x - 0.42 * s, y + 0.02 * s)
    p.lineTo(x - 0.12 * s, y + 0.32 * s)
    p.lineTo(x + 0.46 * s, y - 0.34 * s)
    return p


def arrow_path(x0, y0, x1, y1, head=40):
    p = skia.Path()
    p.moveTo(x0, y0)
    p.lineTo(x1, y1)
    ang = math.atan2(y1 - y0, x1 - x0)
    for s in (-1, 1):
        p.moveTo(x1, y1)
        p.lineTo(x1 - head * math.cos(ang + s * 0.5), y1 - head * math.sin(ang + s * 0.5))
    return p


def scribble_path(x0, x1, y, amp=10, waves=2.5, seed=3):
    rng = np.random.default_rng(seed)
    p = skia.Path()
    n = 28
    for i in range(n + 1):
        u = i / n
        xx = x0 + (x1 - x0) * u
        yy = y + math.sin(u * math.pi * 2 * waves + seed) * amp * (0.6 + 0.4 * rng.random()) + u * amp * 0.6
        if i == 0:
            p.moveTo(xx, yy)
        else:
            p.lineTo(xx, yy)
    return p


def star_path(cx, cy, r_out, r_in, points=5, rot=-90):
    p = skia.Path()
    for i in range(points * 2):
        r = r_out if i % 2 == 0 else r_in
        a = math.radians(rot + i * 180 / points)
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        if i == 0:
            p.moveTo(x, y)
        else:
            p.lineTo(x, y)
    p.close()
    return p


# ------------------------------------------------------------------ video in a circle
def draw_circle_image(c, img, cx, cy, r, src=None, alpha=1.0, color_filter=None, zoom=1.0, offset=(0, 0)):
    c.save()
    c.clipPath(circle_path(cx, cy, r), doAntiAlias=True)
    p = skia.Paint(AntiAlias=True)
    p.setAlphaf(alpha)
    if color_filter is not None:
        p.setColorFilter(color_filter)
    rr = r * zoom
    dst = skia.Rect.MakeLTRB(cx - rr + offset[0], cy - rr + offset[1], cx + rr + offset[0], cy + rr + offset[1])
    if src is None:
        c.drawImageRect(img, dst, CUBIC, p)
    else:
        c.drawImageRect(img, skia.Rect.MakeLTRB(*src), dst, CUBIC, p)
    c.restore()


def saturation_filter(s=0.0, contrast=1.0, tint=(1.0, 1.0, 1.0), lift=0.0):
    lr, lg, lb = 0.2126, 0.7152, 0.0722
    m = np.array([
        [lr * (1 - s) + s, lg * (1 - s), lb * (1 - s)],
        [lr * (1 - s), lg * (1 - s) + s, lb * (1 - s)],
        [lr * (1 - s), lg * (1 - s), lb * (1 - s) + s],
    ]) * contrast
    off = 0.5 * (1 - contrast) + lift
    mat = []
    for i in range(3):
        mat += [m[i, 0] * tint[i], m[i, 1] * tint[i], m[i, 2] * tint[i], 0, off * tint[i]]
    mat += [0, 0, 0, 1, 0]
    return skia.ColorFilters.Matrix(mat)


def halftone(gray, size, cell=11.0, angle=45.0, ink=(20, 20, 20), paper=(238, 231, 218), gamma=1.0):
    """gray: HxW float 0..1 (1 = white). Returns size x size RGB uint8 halftone."""
    from PIL import Image
    g = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
                   dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    a = math.radians(angle)
    u = xx * math.cos(a) + yy * math.sin(a)
    v = -xx * math.sin(a) + yy * math.cos(a)
    du = (u % cell) - cell / 2
    dv = (v % cell) - cell / 2
    d = np.sqrt(du * du + dv * dv)
    dark = np.clip(1 - g, 0, 1) ** gamma
    rad = cell * 0.72 * np.sqrt(dark)
    m = np.clip(rad - d + 0.6, 0, 1)
    ink = np.array(ink, np.float32)
    paper = np.array(paper, np.float32)
    out = paper[None, None, :] * (1 - m[..., None]) + ink[None, None, :] * m[..., None]
    return out.astype(np.uint8)


# ------------------------------------------------------------------ textures
def value_noise(h, w, scale, rng):
    from PIL import Image
    small = rng.random((max(2, int(h / scale)), max(2, int(w / scale)))).astype(np.float32)
    return np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                      dtype=np.float32) / 255.0


def paper_texture(h, w, base, seed=5, fibers=900, strength=1.0):
    rng = np.random.default_rng(seed)
    n = (value_noise(h, w, 260, rng) * 0.5 + value_noise(h, w, 60, rng) * 0.3 + value_noise(h, w, 9, rng) * 0.2)
    n = (n - n.mean()) * 2
    base = np.array(base, np.float32)
    img = base[None, None, :] * (1 + n[..., None] * 0.045 * strength)
    img += rng.normal(0, 2.2 * strength, (h, w, 1)).astype(np.float32)
    surf = surface(w, h)
    c = surf.getCanvas()
    c.drawImage(image(np.clip(img, 0, 255).astype(np.uint8)), 0, 0)
    for _ in range(fibers):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ln, ang = rng.uniform(8, 40), rng.uniform(0, math.pi)
        dark = rng.random() < 0.5
        col = skia.Color4f(0, 0, 0, 0.05 * strength) if dark else skia.Color4f(1, 1, 1, 0.08 * strength)
        c.drawLine(x, y, x + ln * math.cos(ang), y + ln * math.sin(ang), paint(col, stroke=rng.uniform(0.6, 1.4)))
    return surf.makeImageSnapshot()


class Grain:
    def __init__(self, h, w, n=6, sigma=10.0, seed=9):
        rng = np.random.default_rng(seed)
        self.tiles = [rng.normal(0, sigma, (h, w, 1)).astype(np.float32) for _ in range(n)]

    def apply(self, rgb, k, amount=1.0):
        out = rgb.astype(np.float32) + self.tiles[k % len(self.tiles)] * amount
        return out


def vignette_mask(h, w, strength=0.22, power=2.2):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - w / 2) / (w / 2)
    dy = (yy - h / 2) / (h / 2)
    d = np.sqrt(dx * dx * 0.9 + dy * dy * 0.75)
    return (1 - strength * np.clip(d, 0, 1.4) ** power)[..., None].astype(np.float32)


def chroma_shift(rgb, px):
    if px < 0.5:
        return rgb
    p = int(round(px))
    out = rgb.copy()
    out[:, p:, 0] = rgb[:, :-p, 0]
    out[:, :-p, 2] = rgb[:, p:, 2]
    return out


def directional_blur(rgb, px, axis=1):
    """Box blur along one axis (whip-pan motion blur)."""
    k = int(px)
    if k < 2:
        return rgb
    a = rgb.astype(np.float32)
    cs = np.cumsum(a, axis=axis)
    pad = [(0, 0)] * a.ndim
    pad[axis] = (k, 0)
    cs = np.pad(cs, pad, mode="edge")
    sl_hi = [slice(None)] * a.ndim
    sl_lo = [slice(None)] * a.ndim
    sl_hi[axis] = slice(k, None)
    sl_lo[axis] = slice(0, -k)
    return (cs[tuple(sl_hi)] - cs[tuple(sl_lo)]) / k
