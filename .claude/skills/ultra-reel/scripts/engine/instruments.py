"""Procedural instruments and sound-design FX. Each returns a mono or stereo numpy array."""
import numpy as np
from .dsp import (SR, n_samples, mtof, saw, square, sine, tri, noise, env_exp, env_adsr,
                  exp_ramp, ramp, svf, butter, eq_peak, softclip, reverb, to_stereo, pan,
                  haas, widen, RNG)


# ======================================================================= drums

def kick(punch=1.0, tone=48.0, decay=0.38, click=0.6, drive=1.6):
    n = n_samples(decay * 2.2)
    t = np.arange(n) / SR
    f = tone + (tone * 3.4 - tone) * np.exp(-t / 0.032) + 260 * np.exp(-t / 0.004) * punch
    body = sine(f, n) * env_exp(n, decay, 0.0008)
    clk = butter(noise(n_samples(0.006)), 3500, "hp", 2) * np.linspace(1, 0, n_samples(0.006)) * click
    body[: len(clk)] += clk
    return softclip(body * drive, 1.2) * 0.95


def sub_hit(note=33, dur=0.9, glide_from=None, drive=2.2):
    n = n_samples(dur)
    t = np.arange(n) / SR
    f_end = mtof(note)
    f = f_end if glide_from is None else f_end + (mtof(glide_from) - f_end) * np.exp(-t / 0.05)
    s = sine(f, n) + 0.18 * sine(np.asarray(f) * 2, n)
    e = env_adsr(n, 0.002, dur * 0.5, 0.6, min(0.12, dur * 0.3))
    return softclip(s * e * drive, 1.0) * 0.8


def snare(tone=190, snap=1.0, decay=0.16, body=0.55):
    n = n_samples(decay * 3)
    t = np.arange(n) / SR
    b = sine(tone * (1 + 0.6 * np.exp(-t / 0.01)), n) * env_exp(n, decay * 0.5) * body
    nz = noise(n)
    nz = svf(nz, 5200, 0.9, "bp") * 1.6 + butter(nz, 7000, "hp", 2) * 0.7
    nz *= env_exp(n, decay, 0.0005) * snap
    return softclip((b + nz) * 1.4, 1.1) * 0.8


def clap(decay=0.22, spread=0.011, bursts=4):
    n = n_samples(decay * 3 + spread * bursts)
    out = np.zeros(n)
    for k in range(bursts):
        off = n_samples(spread * k * (0.85 + 0.3 * RNG.random()))
        ln = n_samples(0.012 if k < bursts - 1 else decay * 2.5)
        e = env_exp(ln, 0.006 if k < bursts - 1 else decay, 0.0003)
        out[off:off + ln] += noise(ln) * e
    out = svf(out, 1300, 1.4, "bp") * 1.3 + butter(out, 2500, "hp", 2) * 0.5
    return softclip(out * 2.2, 1.3) * 0.7


def hat(open_=False, decay=None, bright=1.0):
    d = decay if decay is not None else (0.32 if open_ else 0.045)
    n = n_samples(d * 3)
    freqs = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * 1.7
    m = sum(square(f, n) for f in freqs) / 6
    m = svf(m, 8800 * bright, 1.2, "bp") + butter(noise(n), 9000, "hp", 2) * 0.35
    return butter(m, 6500, "hp", 2) * env_exp(n, d, 0.0004) * 0.55


def shaker(decay=0.07):
    n = n_samples(decay * 2.5)
    e = env_adsr(n, decay * 0.35, decay * 0.4, 0.0, decay * 0.3)
    return butter(noise(n), 6000, "hp", 2) * e * 0.4


def rim(tone=1700):
    n = n_samples(0.06)
    return (sine(tone, n) * 0.6 + svf(noise(n), 3200, 3.0, "bp")) * env_exp(n, 0.012) * 0.7


def tom(note=50, decay=0.35):
    n = n_samples(decay * 2)
    t = np.arange(n) / SR
    f = mtof(note) * (1 + 0.5 * np.exp(-t / 0.03))
    return softclip(sine(f, n) * env_exp(n, decay) * 1.6, 1.1) * 0.7


def perc_blip(note=84, decay=0.08, fm=2.0):
    n = n_samples(decay * 3)
    f = mtof(note)
    mod = sine(f * fm, n) * f * 1.5 * env_exp(n, decay * 0.4)
    return sine(f + mod, n) * env_exp(n, decay) * 0.5


# ======================================================================= tonal

def supersaw(notes, dur, voices=5, detune=0.18, cutoff=(3000, 900), q=0.9, attack=0.01,
             release=0.25, sustain=0.8, decay=0.4, wide=1.4):
    n = n_samples(dur + release)
    L = np.zeros(n)
    R = np.zeros(n)
    for note in np.atleast_1d(notes):
        f0 = float(mtof(note))
        for v in range(voices):
            dt = (v - (voices - 1) / 2) / max(1, (voices - 1) / 2) * detune
            f = f0 * 2 ** (dt / 12)
            s = saw(f, n, RNG.random())
            p = (v / max(1, voices - 1)) * 2 - 1
            L += s * np.cos((p * wide + 1) * np.pi / 4)
            R += s * np.sin((p * wide + 1) * np.pi / 4)
    c0, c1 = cutoff
    cut = exp_ramp(n, c0, c1)
    L = svf(L, cut, q, "lp")
    R = svf(R, cut, q, "lp")
    e = env_adsr(n, attack, decay, sustain, release, gate_s=dur)
    norm = 1.0 / (len(np.atleast_1d(notes)) * voices) ** 0.6
    return np.stack([L * e, R * e], axis=1) * norm


def stab(notes, dur=0.22, bright=5200, decay=0.12, organ=0.35):
    n = n_samples(dur + 0.2)
    out = np.zeros(n)
    for note in np.atleast_1d(notes):
        f = float(mtof(note))
        out += saw(f, n) * 0.6 + square(f * 1.003, n) * 0.4 + sine(f * 2, n) * organ
    e = env_adsr(n, 0.002, decay, 0.25, 0.08, gate_s=dur)
    cut = 300 + bright * env_exp(n, decay * 0.9)
    out = svf(out, cut, 1.6, "lp") * e
    return out / (len(np.atleast_1d(notes)) ** 0.7) * 0.8


def pluck(note, dur=0.5, bright=0.6, decay=0.35):
    """Karplus-Strong-ish via filtered saw with fast filter envelope."""
    n = n_samples(dur + 0.1)
    f = float(mtof(note))
    s = saw(f, n) * 0.7 + square(f, n, 0.3) * 0.3
    cut = f * 1.5 + 9000 * bright * env_exp(n, decay * 0.25)
    return svf(s, cut, 1.2, "lp") * env_exp(n, decay, 0.001) * 0.7


def bell(note, dur=1.6, ratio=3.5, index=2.0):
    n = n_samples(dur)
    f = float(mtof(note))
    mod = sine(f * ratio, n) * f * index * env_exp(n, dur * 0.25)
    return sine(f + mod, n) * env_exp(n, dur * 0.45, 0.001) * 0.45


def reese(note, dur, detune=0.22, cutoff=700, drive=1.8):
    n = n_samples(dur + 0.08)
    f = float(mtof(note))
    s = saw(f * 2 ** (detune / 12), n) + saw(f * 2 ** (-detune / 12), n, 0.37)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * np.arange(n) / SR)
    s = svf(s, cutoff * (0.7 + 0.6 * lfo), 1.1, "lp")
    s += sine(f / 2, n) * 0.8
    e = env_adsr(n, 0.004, 0.2, 0.85, 0.06, gate_s=dur)
    return softclip(s * e * drive, 1.0) * 0.55


def pad(notes, dur, attack=0.6, release=1.2, cutoff=1800, shimmer=True):
    n = n_samples(dur + release)
    out = np.zeros((n, 2))
    for note in np.atleast_1d(notes):
        f = float(mtof(note))
        for det, p in ((-0.09, -0.7), (0.0, 0.0), (0.09, 0.7)):
            s = saw(f * 2 ** (det / 12), n, RNG.random()) * 0.5 + tri(f * 2 ** (det / 12), n) * 0.5
            out += pan(s, p)
        if shimmer:
            out += pan(sine(f * 4, n) * 0.08, 0.0)
    lfo = 1 + 0.25 * np.sin(2 * np.pi * 0.23 * np.arange(n) / SR)
    for c in range(2):
        out[:, c] = svf(out[:, c], cutoff * lfo, 0.8, "lp")
    e = env_adsr(n, attack, 1.0, 0.85, release, gate_s=dur)
    return out * e[:, None] / (len(np.atleast_1d(notes)) * 3) ** 0.6 * 0.9


def follow_lead(freq_curve, amp_curve, timbre="glass"):
    """Synth that plays a given per-sample pitch contour (speech-melody doubling)."""
    n = len(freq_curve)
    if timbre == "glass":
        s = sine(freq_curve, n) * 0.7 + sine(freq_curve * 2, n) * 0.18 + tri(freq_curve, n) * 0.25
    else:
        s = saw(freq_curve, n) * 0.5 + square(freq_curve * 0.5, n) * 0.25
        s = svf(s, np.clip(freq_curve * 4, 200, 9000), 1.0, "lp")
    return s * amp_curve


# ======================================================================= FX

def riser(dur, start_hz=300, end_hz=9000, tonal_note=None, amount=1.0):
    n = n_samples(dur)
    t = np.linspace(0, 1, n)
    nz = noise(n, "white")
    cut = exp_ramp(n, start_hz, end_hz)
    s = svf(nz, cut, 2.2, "bp") * 1.6
    if tonal_note is not None:
        f = mtof(tonal_note) * 2 ** (t * 2.0)  # two-octave sweep
        s += saw(f, n) * 0.18 + saw(f * 1.01, n) * 0.18
        s = svf(s, cut * 1.2, 0.9, "lp")
    e = t ** 2.2
    st = widen(pan(s * e, 0.0), 1.0 + t.mean())
    # stereo motion: slow auto-pan increasing in speed
    lfo = np.sin(2 * np.pi * np.cumsum(2 + 10 * t) / SR) * 0.6 * t
    st = st * np.stack([1 - lfo * 0.5, 1 + lfo * 0.5], axis=1)
    return st * amount * 0.6


def downlifter(dur=1.2, start_hz=7000, end_hz=200):
    n = n_samples(dur)
    t = np.linspace(0, 1, n)
    s = svf(noise(n), exp_ramp(n, start_hz, end_hz), 1.8, "bp") * (1 - t) ** 1.5
    return widen(pan(s, 0), 1.6) * 0.5


def impact(dur=2.0, sub_note=29, bright=1.0):
    n = n_samples(dur)
    t = np.arange(n) / SR
    sub = sine(mtof(sub_note) * (1 + 1.6 * np.exp(-t / 0.06)), n) * env_exp(n, 0.7)
    crack = butter(noise(n), 1200, "hp", 2) * env_exp(n, 0.05) * bright
    boom = svf(noise(n, "brown"), 180, 1.0, "lp") * env_exp(n, 0.4) * 1.5
    dry = softclip((sub * 1.2 + crack * 0.8 + boom) * 1.5, 1.2)
    return reverb(dry, 3.2, 0.55, 0.01, 4000, 150) * 0.85


def whoosh(dur=0.45, direction=1, low=400, high=5200):
    n = n_samples(dur)
    t = np.linspace(0, 1, n)
    bell_shape = np.sin(np.pi * t) ** 1.6
    peak = 0.6
    shape = np.where(t < peak, (t / peak) ** 2.4, ((1 - t) / (1 - peak)) ** 1.2)
    cut = low + (high - low) * shape
    s = svf(noise(n, "pink"), cut, 1.6, "bp") * shape * 1.8 + svf(noise(n), cut * 1.6, 3.0, "bp") * bell_shape * 0.3
    p = np.linspace(-0.85, 0.85, n) * direction
    return pan(s, p) * 0.55


def reverse_swell(dur=0.9, note=None):
    n = n_samples(dur)
    src = noise(n, "pink") * env_exp(n, dur * 0.35)
    if note is not None:
        tone = supersaw([note, note + 7], dur, cutoff=(4000, 2000), release=0.05)[:n, 0]
        tone = np.pad(tone, (0, n - len(tone)))
        src = src * 0.5 + tone * env_exp(n, dur * 0.3)
    wet = reverb(src, 2.0, 0.9)
    return wet[::-1] * 0.5


def tape_stop(x, start_s, dur_s, curve=2.0):
    """Slow the bus to a halt from start_s over dur_s (pitch and speed drop together)."""
    st = to_stereo(x).copy()
    i0 = n_samples(start_s)
    n = n_samples(dur_s)
    seg_len = min(len(st) - i0, n * 3)
    if seg_len <= 0:
        return st
    speed = np.clip(1 - np.linspace(0, 1, n) ** (1 / curve), 0, 1)
    pos = np.cumsum(speed)
    pos = pos[pos < seg_len - 1]
    seg = st[i0:i0 + seg_len]
    out = np.stack([np.interp(pos, np.arange(seg_len), seg[:, c]) for c in range(2)], axis=1)
    fade = np.linspace(1, 0, len(out)) ** 0.5
    st[i0:i0 + len(out)] = out * fade[:, None]
    st[i0 + len(out):i0 + n] = 0
    return st


def stutter(x, start_s, slice_s, repeats, decay=1.0):
    st = to_stereo(x).copy()
    i0 = n_samples(start_s)
    L = n_samples(slice_s)
    sl = st[i0:i0 + L].copy()
    w = np.ones(L)
    f = min(64, L // 4)
    w[:f] = np.linspace(0, 1, f)
    w[-f:] = np.linspace(1, 0, f)
    for r in range(repeats):
        a = i0 + r * L
        if a + L > len(st):
            break
        st[a:a + L] = sl * w[:, None] * (decay ** r)
    return st


def vinyl(dur, crackle=1.0):
    n = n_samples(dur)
    out = butter(noise(n, "brown"), 300, "lp", 1) * 0.15
    clicks = RNG.random(n) < (18 / SR) * crackle
    imp = np.zeros(n)
    imp[clicks] = RNG.standard_normal(clicks.sum()) * 2.0
    out += butter(imp, 1800, "hp", 2) * 0.5
    return pan(out, 0) * 0.25


def type_click(bright=1.0):
    n = n_samples(0.035)
    s = svf(noise(n), 2600 * bright, 2.5, "bp") * env_exp(n, 0.006) + sine(1900 * bright, n) * env_exp(n, 0.003) * 0.4
    return s * 0.8


def ui_pop(note=88, decay=0.07):
    n = n_samples(decay * 3)
    t = np.arange(n) / SR
    f = mtof(note) * (1 + 0.8 * np.exp(-t / 0.008))
    return sine(f, n) * env_exp(n, decay, 0.0008) * 0.5


def shutter():
    n = n_samples(0.12)
    a = svf(noise(n), 3000, 2, "bp") * env_exp(n, 0.008)
    b = np.zeros(n)
    k = n_samples(0.055)
    b[k:] = svf(noise(n - k), 2200, 2, "bp") * env_exp(n - k, 0.012)
    return (a + b) * 0.9


def stamp():
    n = n_samples(0.5)
    t = np.arange(n) / SR
    thud = sine(95 * (1 + np.exp(-t / 0.01)), n) * env_exp(n, 0.09)
    slap = svf(noise(n), 900, 1.2, "bp") * env_exp(n, 0.025)
    return softclip((thud * 1.3 + slap) * 1.5, 1.1) * 0.8


def glitch_burst(dur=0.18, seed=0):
    rng = np.random.default_rng(seed)
    n = n_samples(dur)
    out = np.zeros(n)
    i = 0
    while i < n:
        ln = int(rng.integers(n_samples(0.006), n_samples(0.03)))
        kind = rng.integers(0, 3)
        seg_n = min(ln, n - i)
        if kind == 0:
            out[i:i + seg_n] = square(rng.uniform(200, 3000), seg_n) * 0.4
        elif kind == 1:
            out[i:i + seg_n] = noise(seg_n) * 1.2
        else:
            out[i:i + seg_n] = np.sign(sine(rng.uniform(40, 120), seg_n)) * 0.3
        i += ln
    return butter(out, 120, "hp", 1) * 0.6


def heartbeat():
    out = np.zeros(n_samples(0.7))
    for off, g in ((0.0, 1.0), (0.22, 0.7)):
        k = kick(punch=0.2, tone=42, decay=0.12, click=0.0, drive=1.2) * g
        i = n_samples(off)
        out[i:i + len(k)] += k[: len(out) - i]
    return butter(out, 160, "lp", 2) * 1.4


def crash(decay=1.8, bright=1.0):
    n = n_samples(decay * 1.5)
    freqs = np.array([342.0, 517.3, 731.6, 1031.2, 1453.7, 2011.9]) * 1.9
    metal = sum(square(f, n) for f in freqs) / 6
    s = butter(metal, 3500 * bright, "hp", 2) * 0.5 + butter(noise(n), 5000, "hp", 2) * 0.8
    e = env_exp(n, decay, 0.001) * (0.6 + 0.4 * env_exp(n, 0.08))
    return widen(pan(s * e, 0), 1.7) * 0.45


def marker(dur=0.28):
    n = n_samples(dur)
    t = np.linspace(0, 1, n)
    grain = noise(n) * (0.6 + 0.4 * np.sin(2 * np.pi * 38 * t) ** 2)
    s = svf(grain, 2800 + 1800 * t, 1.4, "bp") * np.sin(np.pi * t) ** 0.6
    return s * 0.6


def mouse_click():
    out = np.zeros(n_samples(0.09))
    for off, f in ((0.0, 3100), (0.055, 2500)):
        c = svf(noise(n_samples(0.02)), f, 4.0, "bp") * env_exp(n_samples(0.02), 0.003)
        i = n_samples(off)
        out[i:i + len(c)] += c
    return out * 1.2


def shimmer(dur=1.4, notes=(76, 83, 88, 91)):
    n = n_samples(dur)
    out = np.zeros(n)
    for k, nt in enumerate(notes):
        b = bell(nt, dur * 0.9, ratio=2.0, index=0.8)
        i = n_samples(0.06 * k)
        out[i:i + len(b)] += b[: n - i] * (0.7 ** k)
    return reverb(out, 2.8, 0.6)


def hook_hit(note=40):
    k = kick(punch=1.2, tone=46, decay=0.3, click=1.0, drive=2.0)
    t_ = tom(note, 0.28)
    c = clap(0.12)
    n = max(len(k), len(t_), len(c))
    out = np.zeros(n)
    out[:len(k)] += k
    out[:len(t_)] += t_ * 0.8
    out[:len(c)] += c * 0.5
    return reverb(out, 1.2, 0.25)


def brush_swish(dur=0.4, bright=1.0, direction=1, seed=0):
    """A loaded brush dragged across canvas: breathy band noise with bristle grain and a soft release."""
    rng = np.random.default_rng(seed)
    n = n_samples(dur + 0.12)
    t = np.linspace(0, 1, n)
    env = np.clip(t * (dur + 0.12) / 0.05, 0, 1) * np.clip((1 - t) * (dur + 0.12) / 0.12, 0, 1)
    env *= 0.75 + 0.25 * np.sin(np.pi * np.clip(t * 1.1, 0, 1))
    grain = butter(np.abs(rng.standard_normal(n)), 40, "lp", 1)
    grain = 0.55 + 0.45 * grain / (np.abs(grain).max() + 1e-9)
    body = svf(noise(n, "pink"), 1700 * bright + 900 * t, 0.7, "bp") * 1.6
    hiss = butter(noise(n), 5200 * bright, "hp", 2) * 0.25
    crackle = np.zeros(n)
    hits = rng.random(n) < 90 / SR
    crackle[hits] = rng.standard_normal(hits.sum())
    crackle = svf(crackle, 3800, 2.0, "bp") * 0.5
    s = (body + hiss) * grain * env + crackle * env
    return pan(s, np.linspace(-0.35, 0.35, n) * direction) * 0.5


def air(dur=0.7):
    """Soft breath of air (a page turning slowly)."""
    n = n_samples(dur)
    t = np.linspace(0, 1, n)
    s = butter(noise(n, "pink"), 1400, "lp", 2) * np.sin(np.pi * t) ** 2
    return widen(pan(s, 0), 1.4) * 0.5


# ------------------------------------------------------------------ materials (abat: a sound by what lands)
def _modes(f0, ratios, decays, amps, dur):
    n = n_samples(dur)
    out = np.zeros(n)
    for r, d, a in zip(ratios, decays, amps):
        out += sine(f0 * r, n) * env_exp(n, d, 0.0004) * a
    return out


def wood(note=72, decay=0.16):
    """Marimba / wooden block: struck bar modes (1 : 3.9 : 9.2), a soft mallet click."""
    f = mtof(note)
    body = _modes(f, (1.0, 3.93, 9.24), (decay, decay * 0.35, decay * 0.12), (1.0, 0.32, 0.1), decay * 4)
    n = len(body)
    click = svf(noise(n_samples(0.012)), min(f * 6, 9000), 1.5, "bp") * env_exp(n_samples(0.012), 0.002)
    body[:len(click)] += click * 0.25
    return body * 0.6


def glass(note=91, decay=0.5):
    """A light tap on glass: thin inharmonic partials, long ring."""
    f = mtof(note)
    s = _modes(f, (1.0, 2.32, 4.25, 6.63), (decay, decay * 0.6, decay * 0.35, decay * 0.2), (1.0, 0.5, 0.25, 0.12),
               decay * 3)
    return reverb(s * 0.35, 1.4, 0.3)


def metal(note=84, decay=0.35):
    """A small tool set down: bright, slightly detuned partials and a hard click."""
    f = mtof(note)
    s = _modes(f, (1.0, 2.76, 5.40, 8.93), (decay, decay * 0.5, decay * 0.3, decay * 0.18), (1.0, 0.6, 0.35, 0.2),
               decay * 3)
    s += _modes(f * 1.012, (1.0, 2.71), (decay * 0.8, decay * 0.4), (0.5, 0.3), decay * 3)
    n = n_samples(0.01)
    s[:n] += svf(noise(n), 6000, 1.0, "bp") * env_exp(n, 0.0015) * 0.6
    return s * 0.35


def thunk(note=50):
    """Wood landing on wood (a sign, a cap): low knock + short boxy noise."""
    n = n_samples(0.35)
    t = np.arange(n) / SR
    f = mtof(note) * (1 + 0.6 * np.exp(-t / 0.012))
    knock = sine(f, n) * env_exp(n, 0.07)
    box = svf(noise(n), mtof(note) * 4, 2.0, "bp") * env_exp(n, 0.03)
    return softclip(knock * 1.2 + box * 0.8, 1.1) * 0.6


def clank(note=45):
    """A metal box set down: thud + loose rattle of what is inside."""
    out = thunk(note) * 1.1
    rng = np.random.default_rng(note)
    for k in range(4):
        m = metal(note + 36 + int(rng.integers(-3, 5)), 0.12)
        i = n_samples(0.015 + 0.03 * k + rng.uniform(0, 0.01))
        seg = m[: len(out) - i]
        out[i:i + len(seg)] += seg * (0.5 - 0.08 * k)
    return out


def paper(dur=0.32, seed=0):
    """A page lifted and laid: rustle bursts with flutter."""
    rng = np.random.default_rng(seed)
    n = n_samples(dur)
    t = np.linspace(0, 1, n)
    flutter = np.clip(np.sin(2 * np.pi * (14 + 6 * t) * t * dur) * 0.5 + 0.5, 0, 1) ** 2
    s = svf(noise(n), 3200 + 1600 * t, 0.9, "bp") * flutter * np.sin(np.pi * t) ** 0.8
    s += butter(noise(n), 6500, "hp", 2) * 0.15 * np.sin(np.pi * t)
    s *= 0.8 + 0.2 * rng.random()
    return pan(s, 0.2) * 0.55
