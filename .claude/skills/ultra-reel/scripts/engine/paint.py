"""Painterly kit: canvas weave, dry-brush strokes, ensō, watercolor washes, brush-edged masks, written-on text.

Everything is seeded, so a stroke looks the same on every frame and in every render worker; animation only
changes how much of it is revealed (progress), never its shape.
"""
import math

import numpy as np
import skia

from . import gfx as G


# ------------------------------------------------------------------ noise helpers
def smooth_noise_1d(n, scale, rng):
    """1-D value noise, n samples, feature size ~scale samples, range ~[-1, 1]."""
    k = max(2, int(n / scale) + 2)
    pts = rng.uniform(-1, 1, k)
    x = np.linspace(0, k - 1.001, n)
    i = x.astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    return pts[i] * (1 - f) + pts[i + 1] * f


# ------------------------------------------------------------------ canvas
def canvas_texture(h, w, base, seed=1, weave=1.0):
    """Primed cotton canvas: warm base, soft mottling, fine weave, a few fibres."""
    rng = np.random.default_rng(seed)
    base = np.array(base, np.float32)
    mott = G.value_noise(h, w, 340, rng) * 0.6 + G.value_noise(h, w, 90, rng) * 0.4
    mott = (mott - mott.mean()) * 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # weave: two thread directions with irregular thickness
    tx = np.sin(xx * 2 * math.pi / 3.1 + smooth_noise_1d(w, 40, rng)[None, :] * 2.0)
    ty = np.sin(yy * 2 * math.pi / 3.3 + smooth_noise_1d(h, 40, rng)[:, None] * 2.0)
    thread_v = G.value_noise(h, w, 6, rng) - 0.5
    weave_map = (tx * 0.5 + ty * 0.5) * (0.6 + thread_v)
    img = base[None, None, :] * (1 + mott[..., None] * 0.035) + weave_map[..., None] * 3.2 * weave
    img += rng.normal(0, 1.6, (h, w, 1)).astype(np.float32)
    return G.image(np.clip(img, 0, 255).astype(np.uint8))


def ink_texture(size=1024, seed=3, holes=0.0):
    """Alpha texture for dry ink: mostly opaque with streaky gaps (used with DstIn)."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    a = G.value_noise(size, size, 3, rng) * 0.5 + G.value_noise(size, size, 14, rng) * 0.5
    small = (rng.random((size // 2, max(2, size // 24))) * 255).astype(np.uint8)
    streak = np.asarray(Image.fromarray(small).resize((size, size), Image.BICUBIC), np.float32) / 255
    m = np.clip((a * 0.7 + streak * 0.3 - holes * 2.2 - 0.25) * 5 + 0.5, 0, 1)
    m = 0.78 + 0.22 * m
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (m * 255).astype(np.uint8)
    return G.image(rgba)


# ------------------------------------------------------------------ brush strokes
def resample(pts, step=2.0):
    pts = np.asarray(pts, np.float64)
    d = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    n = max(2, int(s[-1] / step))
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], axis=1), s[-1]


class Stroke:
    """A brush stroke along a centreline: a wet core that runs out early, plus separate dry bristles whose
    gaps show the canvas — heavier at the start, broken into streaks toward the end."""

    def __init__(self, pts, width, seed=0, bristles=46, dryness=0.5, taper=(0.12, 0.35), wobble=0.12,
                 pressure=None, depletion=0.35, core=0.62):
        rng = np.random.default_rng(seed)
        P, length = resample(pts, 2.0)
        n = len(P)
        tang = np.gradient(P, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        u = np.linspace(0, 1, n)
        t0, t1 = taper
        prof = np.clip(u / max(t0, 1e-3), 0, 1) ** 0.6 * np.clip((1 - u) / max(t1, 1e-3), 0, 1) ** 0.8
        prof = 0.3 + 0.7 * prof
        if pressure is not None:
            prof = prof * pressure(u)
        prof *= 1 + 0.14 * smooth_noise_1d(n, 70, rng) + 0.05 * smooth_noise_1d(n, 12, rng)
        self.u = u
        self.paths = []  # (u_start, u_end, xy, width, alpha)
        # wet core: three overlapping tapered bodies of different length -> paint thins out gradually
        self.cores = []
        for k in range(3):
            end = int(n * np.clip(core * (0.75 + 0.25 * k) + rng.uniform(-0.05, 0.05), 0.15, 1.0))
            uu = np.linspace(0, 1, end)
            half = width * prof[:end] * (0.42 - 0.08 * k) * (1 - uu ** 1.8)
            half *= 1 + 0.06 * smooth_noise_1d(end, 20, rng)
            sh = rng.uniform(-0.06, 0.06) * width
            left = P[:end] + nrm[:end] * (half + sh)[:, None]
            right = P[:end] - nrm[:end] * (half - sh)[:, None]
            self.cores.append((u[end - 1], left, right, 0.55 - 0.12 * k))
        spacing = width / bristles
        for b_ in range(bristles):
            off = (b_ / (bristles - 1) - 0.5) + rng.normal(0, 0.01)
            drift = smooth_noise_1d(n, 90, rng) * wobble * 0.06
            o = (off + drift) * width * prof
            xy = P + nrm * o[:, None]
            edge = abs(off) * 2
            thr = dryness * (u ** 1.5) + 0.25 * edge ** 3 + rng.uniform(-0.06, 0.06)
            alive = smooth_noise_1d(n, rng.uniform(25, 90), rng) * 0.5 + 0.5 > thr
            alpha = rng.uniform(0.45, 1.0) * (1 - 0.3 * edge)
            bwb = spacing * rng.uniform(1.0, 1.9)
            i = 0
            while i < n:
                if alive[i]:
                    j = i
                    while j < n and alive[j]:
                        j += 1
                    if j - i > 3:
                        self.paths.append((u[i], u[j - 1], xy[i:j], bwb, alpha))
                    i = j
                else:
                    i += 1
        self.length = length
        self.depletion = depletion
        self.p0, self.p1 = P[0], P[-1]
        self.center = None  # set for arcs (ensō) so paint fades around the circle

    def _load_shader(self, color):
        hi = skia.Color4f(color.fR, color.fG, color.fB, 1.0).toColor()
        lo = skia.Color4f(color.fR, color.fG, color.fB, max(0.0, 1 - self.depletion)).toColor()
        if self.center is not None:
            cx, cy, a0, sweep = self.center
            m = skia.Matrix.RotateDeg(a0, skia.Point(cx, cy))
            return skia.GradientShader.MakeSweep(cx, cy, [hi, lo, lo], [0.0, sweep / 360.0, 1.0], localMatrix=m)
        return skia.GradientShader.MakeLinear([skia.Point(*self.p0), skia.Point(*self.p1)], [hi, lo])

    def draw(self, c, color, progress=1.0, alpha=1.0, blur=0.45):
        if progress <= 0:
            return
        if blur > 0:
            p = skia.Paint()
            p.setImageFilter(skia.ImageFilters.Blur(blur, blur))
            c.saveLayer(None, p)
        shader = self._load_shader(color) if self.depletion > 0 else None
        for ue, left, right, a in self.cores:
            m = len(left) if ue <= progress else max(2, int(len(left) * progress / max(ue, 1e-6)))
            path = skia.Path()
            path.moveTo(*left[0])
            for x, y in left[1:m:2]:
                path.lineTo(x, y)
            for x, y in right[m - 1::-2]:
                path.lineTo(x, y)
            path.close()
            pp = G.paint(color, alpha=a * alpha)
            if shader is not None:
                pp.setShader(shader)
                pp.setAlphaf(a * alpha)
            c.drawPath(path, pp)
        for us, ue, xy, bw, a in self.paths:
            if us > progress:
                continue
            m = xy
            if ue > progress:
                frac = (progress - us) / max(ue - us, 1e-6)
                m = xy[: max(2, int(len(xy) * frac))]
            path = skia.Path()
            path.moveTo(*m[0])
            for x, y in m[1::2]:
                path.lineTo(x, y)
            pp = G.paint(color, stroke=bw, alpha=a * alpha, cap=skia.Paint.kRound_Cap, join=skia.Paint.kRound_Join)
            if shader is not None:
                pp.setShader(shader)
                pp.setAlphaf(a * alpha)
            c.drawPath(path, pp)
        if blur > 0:
            c.restore()


def enso_points(cx, cy, r, start_deg=-60, sweep=330, seed=0, wobble=0.03):
    rng = np.random.default_rng(seed)
    n = 260
    a = np.radians(start_deg + np.linspace(0, sweep, n))
    # a hand never closes a perfect circle: slow wobble, slight ellipse, spiral outward at the end
    rr = r * (1 + wobble * smooth_noise_1d(n, 70, rng) + 0.02 * np.cos(2 * a + rng.uniform(0, 6)) + np.linspace(0, 0.045, n))
    return np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], axis=1)


def make_enso(r, width, seed=0, start_deg=-60, sweep=328):
    pts = enso_points(0, 0, r, start_deg, sweep, seed)
    st = Stroke(pts, width, seed=seed, bristles=54, dryness=0.7, taper=(0.03, 0.4), wobble=0.25,
                pressure=lambda u: 1.2 - 0.5 * u, depletion=0.25, core=0.8)
    st.center = (0.0, 0.0, start_deg, sweep)
    return st


# ------------------------------------------------------------------ watercolor (layered deformed polygons)
def _deform(poly, depth, var, rng, decay=0.62):
    for lvl in range(depth):
        var_l = var * decay ** lvl
        out = []
        m = len(poly)
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            ln = math.hypot(b[0] - a[0], b[1] - a[1])
            mid = ((a[0] + b[0]) / 2 + rng.normal(0, var_l * ln), (a[1] + b[1]) / 2 + rng.normal(0, var_l * ln))
            out += [a, mid]
        poly = out
    return poly


def _fbm(h, w, rng, scales=(260, 110, 45, 18), weights=(0.5, 0.3, 0.15, 0.05)):
    f = np.zeros((h, w), np.float32)
    for sc, wt in zip(scales, weights):
        f += (G.value_noise(h, w, sc, rng) - 0.5) * 2 * wt
    return f


def watercolor(w, h, color, seed=0, shape="blob", spread=0.42, edge_rough=0.32, density=0.62, rim=0.55,
               bloom=0.35, granulate=0.3):
    """Noise-field watercolor wash: fractal contour, uneven pale interior, dried pigment rim, granulation,
    a faint bleed halo. color: (r,g,b) 0-255. Returns a straight-alpha skia image."""
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2) / (w * spread)
    ny = (yy - h / 2) / (h * spread * (1.0 if shape == "blob" else 0.42))
    d = np.sqrt(nx * nx + ny * ny)
    field = 1 - d + edge_rough * _fbm(h, w, rng)
    core = np.clip(field / 0.035, 0, 1)

    def blur(a, r):
        return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)),
                          np.float32) / 255

    interior = density * (0.55 + 0.45 * (G.value_noise(h, w, 140, rng) * 0.6 + G.value_noise(h, w, 40, rng) * 0.4))
    # cauliflower bloom: a paler patch with a darker frontier somewhere inside
    bl = np.clip((G.value_noise(h, w, 180, rng) - 0.62) * 6, 0, 1)
    bl_edge = np.clip(bl - blur(bl, 4), 0, 1)
    A = core * interior * (1 - bloom * bl) + bl_edge * bloom * core * 0.9
    edge = np.clip(core - blur(core, 3.5), 0, 1)
    A = A + edge * rim
    A = A + blur(core, 10) * 0.05  # bleed into the paper
    A = A * (1 - granulate / 2 + granulate * (G.value_noise(h, w, 1.6, rng) * 0.6 + G.value_noise(h, w, 5, rng) * 0.4))
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = color
    out[..., 3] = (np.clip(A, 0, 1) * 255).astype(np.uint8)
    return G.image(out)


def draw_bloom(c, img, cx, cy, progress, scale=1.0, alpha=1.0, seed=0):
    """Wet bloom: the wash spreads out from its centre with a soft, irregular front."""
    if progress <= 0:
        return
    w, h = img.width() * scale, img.height() * scale
    rmax = math.hypot(w, h) / 2
    r = rmax * G.ease_out_cubic(min(1.0, progress))
    p = skia.Paint()
    p.setAlphaf(alpha)
    c.saveLayer(None, p)
    c.drawImageRect(img, skia.Rect.MakeXYWH(cx - w / 2, cy - h / 2, w, h), G.LINEAR)
    if progress < 1:
        shader = skia.GradientShader.MakeRadial(skia.Point(cx, cy), max(r, 1.0),
                                                [skia.ColorWHITE, skia.ColorWHITE, skia.ColorTRANSPARENT],
                                                [0.0, 0.8, 1.0])
        mp = skia.Paint(Shader=shader)
        mp.setBlendMode(skia.BlendMode.kDstIn)
        c.drawRect(skia.Rect.MakeXYWH(cx - w, cy - h, 2 * w, 2 * h), mp)
    c.restore()


# ------------------------------------------------------------------ brush-edged circle (the speaker's frame)
def brush_circle_mask(size, seed=0, rough=0.012, feather=2.2):
    """Alpha mask of a hand-painted disc: gently irregular rim, a few dry gaps, soft edge."""
    rng = np.random.default_rng(seed)
    R = size / 2 - size * 0.04
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = xx - size / 2, yy - size / 2
    rad = np.sqrt(dx * dx + dy * dy)
    th = np.arctan2(dy, dx)
    n = 4096
    idx = ((th + math.pi) / (2 * math.pi) * (n - 1)).astype(int)
    edge = R * (1 + rough * smooth_noise_1d(n, 600, rng)[idx] + rough * 0.5 * smooth_noise_1d(n, 120, rng)[idx]
                + rough * 0.18 * smooth_noise_1d(n, 25, rng)[idx])
    a = np.clip((edge - rad) / feather + 0.5, 0, 1)
    # dry brush: in a few sectors the rim breaks into fine bristle gaps
    sector = np.clip(smooth_noise_1d(n, 500, rng) * 1.6 - 0.4, 0, 1)[idx]
    bristle = smooth_noise_1d(n, 4, rng)[idx] * 0.5 + 0.5
    depth = size * 0.022 * (0.4 + 0.6 * smooth_noise_1d(n, 30, rng)[idx] ** 2)
    rim = np.clip(1 - (edge - rad) / np.maximum(depth, 1), 0, 1)
    a *= 1 - sector * rim * (bristle < 0.3) * 0.7
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (a * 255).astype(np.uint8)
    return G.image(rgba)


def draw_masked(c, mask, cx, cy, r, draw_content):
    """Draw content (a callable taking the canvas) clipped by a soft image mask scaled to radius r."""
    c.saveLayer(skia.Rect.MakeLTRB(cx - r * 1.1, cy - r * 1.1, cx + r * 1.1, cy + r * 1.1), None)
    c.clipRect(skia.Rect.MakeLTRB(cx - r * 1.08, cy - r * 1.08, cx + r * 1.08, cy + r * 1.08))
    draw_content(c)
    mp = skia.Paint()
    mp.setBlendMode(skia.BlendMode.kDstIn)
    s = r / (mask.width() / 2 - mask.width() * 0.04)
    hw = mask.width() / 2 * s
    c.drawImageRect(mask, skia.Rect.MakeLTRB(cx - hw, cy - hw, cx + hw, cy + hw), G.LINEAR, mp)
    c.restore()


# ------------------------------------------------------------------ text that is painted, not typeset
def written_text(c, text, x, y, font, color, progress, ink=None, align="left", jitter=1.2, seed=0, alpha=1.0,
                 feather=0.18, bold=0.0):
    """Write-on reveal left->right with a soft, irregular front; slight per-glyph hand jitter."""
    if progress <= 0 or not text:
        return 0.0
    glyphs, xs, w = G.glyph_layout(text, font)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    rng = np.random.default_rng(seed)
    m = font.getMetrics()
    top, bot = y + m.fAscent * 1.1, y + m.fDescent * 1.4
    c.saveLayer(skia.Rect.MakeLTRB(x - 60, top - 40, x + w + 80, bot + 40), None)
    for i, ch in enumerate(text):
        dy = rng.normal(0, jitter)
        rot = rng.normal(0, jitter * 0.6)
        blob = skia.TextBlob.MakeFromText(ch, font)
        if blob is None:
            continue
        c.save()
        c.translate(x + xs[i], y + dy)
        c.rotate(rot)
        gp = G.paint(color, alpha=alpha)
        if bold > 0:  # a loaded brush: fatter strokes than the font's dry pen
            gp.setStyle(skia.Paint.kStrokeAndFill_Style)
            gp.setStrokeWidth(bold)
            gp.setStrokeJoin(skia.Paint.kRound_Join)
        c.drawTextBlob(blob, 0, 0, gp)
        c.restore()
    if ink is not None:
        ip = skia.Paint()
        ip.setBlendMode(skia.BlendMode.kDstIn)
        ip.setShader(ink.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat, G.LINEAR))
        c.drawRect(skia.Rect.MakeLTRB(x - 60, top - 40, x + w + 80, bot + 40), ip)
    if progress < 1:
        front = x - w * feather + (w * (1 + 2 * feather)) * G.ease_in_out(progress)
        shader = skia.GradientShader.MakeLinear([skia.Point(front - w * feather, 0), skia.Point(front, 0)],
                                                [skia.ColorWHITE, skia.ColorTRANSPARENT])
        mp = skia.Paint(Shader=shader)
        mp.setBlendMode(skia.BlendMode.kDstIn)
        c.drawRect(skia.Rect.MakeLTRB(x - 60, top - 40, x + w + 80, bot + 40), mp)
    c.restore()
    return w


def rise_text(c, text, x, y, font, color, k, align="left", alpha=1.0, dist=28, blur=7.0, tracking=0.0):
    """Motion-Canvas-like entrance: up from below, blur to sharp, fade in. k in [0,1]."""
    if k <= 0 or not text:
        return 0.0
    e = G.ease_out_cubic(k)
    p = skia.Paint()
    b = blur * (1 - e)
    if b > 0.3:
        p.setImageFilter(skia.ImageFilters.Blur(b, b))
    p.setAlphaf(min(1.0, k * 1.6) * alpha)
    c.saveLayer(None, p)
    w = G.draw_text(c, text, x, y + dist * (1 - e), font, color, align=align, tracking=tracking)
    c.restore()
    return w
