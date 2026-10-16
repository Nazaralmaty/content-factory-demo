"""Low-level DSP for the procedural score: oscillators, filters, dynamics, space.

Everything works on float64 numpy arrays at SR. Per-sample loops are numba-jitted.
"""
import numpy as np
from numba import njit
from scipy import signal

SR = 48000
RNG = np.random.default_rng(7)


def n_samples(sec):
    return max(1, int(round(sec * SR)))


def db(x):
    return 10 ** (x / 20.0)


def mtof(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12.0)


# ----------------------------------------------------------------- oscillators

def phase_from_freq(freq, n=None, phase0=0.0):
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,)) if n else np.asarray(freq, float)
    return (phase0 + np.cumsum(f) / SR) % 1.0


def _polyblep(t, dt):
    out = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt[m]
    out[m] = x + x - x * x - 1.0
    m2 = t > 1.0 - dt
    x = (t[m2] - 1.0) / dt[m2]
    out[m2] = x * x + x + x + 1.0
    return out


def saw(freq, n, phase0=0.0):
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,)).copy()
    ph = phase_from_freq(f, n, phase0)
    dt = np.clip(f / SR, 1e-6, 0.5)
    return 2.0 * ph - 1.0 - _polyblep(ph, dt)


def square(freq, n, pw=0.5, phase0=0.0):
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,)).copy()
    ph = phase_from_freq(f, n, phase0)
    dt = np.clip(f / SR, 1e-6, 0.5)
    sq = np.where(ph < pw, 1.0, -1.0)
    sq += _polyblep(ph, dt)
    sq -= _polyblep((ph + 1 - pw) % 1.0, dt)
    return sq


def sine(freq, n, phase0=0.0):
    ph = phase_from_freq(np.broadcast_to(np.asarray(freq, float), (n,)), n, phase0)
    return np.sin(2 * np.pi * ph)


def tri(freq, n, phase0=0.0):
    ph = phase_from_freq(np.broadcast_to(np.asarray(freq, float), (n,)), n, phase0)
    return 4 * np.abs(ph - 0.5) - 1


def noise(n, color="white"):
    w = RNG.standard_normal(n)
    if color == "white":
        return w / 3.0
    if color == "pink":
        b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
        p = signal.lfilter(b, a, w)
        return p / (np.std(p) * 3 + 1e-9)
    if color == "brown":
        p = np.cumsum(w)
        p = signal.lfilter([1, -1], [1, -0.995], p)
        return p / (np.std(p) * 3 + 1e-9)
    raise ValueError(color)


# ----------------------------------------------------------------- envelopes

def env_exp(n, decay_s, attack_s=0.0005):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack_s, 1e-6), 0, 1)
    return a * np.exp(-t / max(decay_s, 1e-5))


def env_adsr(n, a=0.005, d=0.1, s=0.7, r=0.1, gate_s=None):
    gate = n / SR - r if gate_s is None else gate_s
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-6), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-6)))
    rel = t > gate
    if np.any(rel):
        g_level = e[np.argmax(rel) - 1] if np.argmax(rel) > 0 else e[0]
        e = np.where(rel, g_level * np.exp(-(t - gate) / max(r / 4.0, 1e-6)), e)
    return e


def ramp(n, a, b, curve=1.0):
    x = np.linspace(0, 1, n) ** curve
    return a + (b - a) * x


def exp_ramp(n, a, b):
    return a * (b / a) ** np.linspace(0, 1, n)


# ----------------------------------------------------------------- filters

@njit(cache=True)
def _svf(x, cutoff, q, mode):
    # Zavalishin TPT state-variable filter, per-sample cutoff
    n = x.shape[0]
    y = np.empty(n)
    ic1 = 0.0
    ic2 = 0.0
    k = 1.0 / q
    for i in range(n):
        fc = cutoff[i]
        if fc < 10.0:
            fc = 10.0
        if fc > 0.45 * 48000.0:
            fc = 0.45 * 48000.0
        g = np.tan(np.pi * fc / 48000.0)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            y[i] = v2
        elif mode == 1:
            y[i] = x[i] - k * v1 - v2
        elif mode == 2:
            y[i] = v1
        else:
            y[i] = x[i] - k * v1
    return y


def svf(x, cutoff, q=0.707, mode="lp"):
    c = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape).copy()
    m = {"lp": 0, "hp": 1, "bp": 2, "notch": 3}[mode]
    return _svf(np.ascontiguousarray(x, dtype=float), c, float(q), m)


def butter(x, fc, kind="lp", order=2):
    btype = {"lp": "lowpass", "hp": "highpass", "bp": "bandpass"}[kind]
    sos = signal.butter(order, fc, btype=btype, fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def _biquad_peak(f0, gain_db, q):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
    return np.array(b) / a[0], np.array(a) / a[0]


def _biquad_shelf(f0, gain_db, high=True, s=1.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    cw = np.cos(w0)
    if high:
        b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    else:
        b = [A * ((A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha), 2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha, -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def eq_peak(x, f0, gain_db, q=1.0):
    b, a = _biquad_peak(f0, gain_db, q)
    return signal.lfilter(b, a, x, axis=0)


def eq_shelf(x, f0, gain_db, high=True):
    b, a = _biquad_shelf(f0, gain_db, high)
    return signal.lfilter(b, a, x, axis=0)


# ----------------------------------------------------------------- dynamics

@njit(cache=True)
def _env_follow(x, att, rel):
    n = x.shape[0]
    y = np.empty(n)
    e = 0.0
    ca = np.exp(-1.0 / (att * 48000.0))
    cr = np.exp(-1.0 / (rel * 48000.0))
    for i in range(n):
        v = abs(x[i])
        if v > e:
            e = ca * e + (1 - ca) * v
        else:
            e = cr * e + (1 - cr) * v
        y[i] = e
    return y


def env_follow(x, att=0.005, rel=0.1):
    mono = x if x.ndim == 1 else np.max(np.abs(x), axis=1)
    return _env_follow(np.ascontiguousarray(mono, dtype=float), att, rel)


@njit(cache=True)
def _gain_computer(env_db, thr, ratio, knee):
    n = env_db.shape[0]
    g = np.empty(n)
    for i in range(n):
        x = env_db[i]
        over = x - thr
        if 2 * over < -knee:
            gr = 0.0
        elif 2 * abs(over) <= knee:
            gr = (1 / ratio - 1) * (over + knee / 2) ** 2 / (2 * knee)
        else:
            gr = (1 / ratio - 1) * over
        g[i] = gr
    return g


def compress(x, thr_db=-18, ratio=3.0, att=0.005, rel=0.12, knee=6.0, makeup_db=0.0, sidechain=None):
    sc = x if sidechain is None else sidechain
    e = env_follow(sc, att, rel)
    e_db = 20 * np.log10(e + 1e-9)
    gr = _gain_computer(e_db, thr_db, ratio, knee)
    g = db(gr + makeup_db)
    return x * (g if x.ndim == 1 else g[:, None]), g


@njit(cache=True)
def _limiter_gain(peak, ceiling, rel_coef, la):
    n = peak.shape[0]
    target = np.empty(n)
    for i in range(n):
        p = peak[i]
        target[i] = ceiling / p if p > ceiling else 1.0
    # look-ahead min over window
    out = np.empty(n)
    g = 1.0
    for i in range(n):
        lo = i
        hi = i + la
        if hi > n:
            hi = n
        m = 1.0
        for j in range(lo, hi):
            if target[j] < m:
                m = target[j]
        if m < g:
            g = m
        else:
            g = rel_coef * g + (1 - rel_coef) * m
        out[i] = g
    return out


def limit(x, ceiling_db=-1.0, release=0.06, lookahead=0.004):
    ceiling = db(ceiling_db)
    peak = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    la = int(lookahead * SR)
    # decimate the look-ahead search for speed: block max
    g = _limiter_gain(np.ascontiguousarray(peak), ceiling, np.exp(-1 / (release * SR)), la)
    # smooth attack (a few samples) so gain moves before the peak
    k = np.hanning(la * 2 + 1)
    k /= k.sum()
    g = np.minimum(g, np.convolve(g, k, mode="same"))
    delayed = x  # look-ahead realised by min-window in gain, no delay needed for offline use
    y = delayed * (g[:, None] if x.ndim == 2 else g)
    return np.clip(y, -ceiling, ceiling)


def softclip(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


# ----------------------------------------------------------------- space

def make_ir(seconds=2.2, predelay=0.012, damp_hz=6500, width=1.0, seed=3):
    rng = np.random.default_rng(seed)
    n = n_samples(seconds)
    t = np.arange(n) / SR
    decay = np.exp(-6.9 * t / seconds)  # -60 dB at `seconds`
    ir = np.zeros((n + n_samples(predelay), 2))
    for ch in range(2):
        nz = rng.standard_normal(n) * decay
        # frequency-dependent damping: progressively darker tail
        early = butter(nz, damp_hz, "lp", 2)
        late = butter(nz, damp_hz / 3, "lp", 2)
        mix = np.clip(t / seconds * 2, 0, 1)
        ir[n_samples(predelay):, ch] = early * (1 - mix) + late * mix
    # decorrelate channels by width
    mid = (ir[:, 0] + ir[:, 1]) / 2
    side = (ir[:, 0] - ir[:, 1]) / 2 * width
    ir = np.stack([mid + side, mid - side], axis=1)
    ir /= np.sqrt(np.sum(ir ** 2) / 2) + 1e-9
    return ir


_IR_CACHE = {}


def reverb(x, seconds=2.2, wet=0.25, predelay=0.012, damp_hz=6500, hp=200):
    key = (round(seconds, 3), round(predelay, 4), damp_hz)
    if key not in _IR_CACHE:
        _IR_CACHE[key] = make_ir(seconds, predelay, damp_hz)
    ir = _IR_CACHE[key]
    st = to_stereo(x)
    src = butter(st, hp, "hp", 2)
    out = np.stack([signal.fftconvolve(src[:, c], ir[:, c])[: len(st)] for c in range(2)], axis=1)
    return st * (1 - wet) + out * wet * 0.35


def delay(x, time_s, feedback=0.35, wet=0.3, lp_hz=4500, pingpong=True):
    st = to_stereo(x)
    n = len(st)
    d = n_samples(time_s)
    out = np.zeros_like(st)
    tap = st.copy()
    gain = 1.0
    for k in range(1, 12):
        gain *= feedback
        if gain < 0.01 or k * d >= n:
            break
        tap = butter(tap, lp_hz, "lp", 1)
        side = (k % 2) if pingpong else 2
        seg = tap[: n - k * d] * gain
        if side == 2:
            out[k * d:] += seg
        else:
            out[k * d:, side] += seg[:, side] * 1.4
            out[k * d:, 1 - side] += seg[:, 1 - side] * 0.4
    return st + out * wet


def to_stereo(x):
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


def pan(x, p):
    """p in [-1, 1]; equal-power."""
    mono = x if x.ndim == 1 else x.mean(axis=1)
    p = np.broadcast_to(np.asarray(p, float), mono.shape)
    a = (p + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1) * np.sqrt(2)


def haas(x, ms=12, side=1):
    st = to_stereo(x).copy()
    d = n_samples(ms / 1000)
    ch = 1 if side > 0 else 0
    st[:, ch] = np.concatenate([np.zeros(d), st[:-d, ch]])
    return st


def widen(x, amount=1.3):
    st = to_stereo(x)
    m = (st[:, 0] + st[:, 1]) / 2
    s = (st[:, 0] - st[:, 1]) / 2 * amount
    return np.stack([m + s, m - s], axis=1)


# ----------------------------------------------------------------- buffers

class Bus:
    """Stereo accumulation buffer with time-addressed writes."""

    def __init__(self, seconds):
        self.buf = np.zeros((n_samples(seconds), 2))

    def add(self, t, x, gain=1.0, p=0.0):
        st = to_stereo(x) if p == 0.0 else pan(x, p)
        i = int(round(t * SR))
        if i < 0:
            st = st[-i:]
            i = 0
        j = min(len(self.buf), i + len(st))
        if j > i:
            self.buf[i:j] += st[: j - i] * gain

    def __len__(self):
        return len(self.buf)
