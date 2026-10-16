"""Voice chain, voice-aware ducking, kick pump and mastering."""
import numpy as np
import pyloudnorm as pyln
from scipy import signal

from .dsp import (SR, butter, eq_peak, eq_shelf, env_follow, compress, limit, db, to_stereo,
                  reverb, delay, n_samples)


def zero_phase(x, lo=None, hi=None, order=4):
    if lo and hi:
        sos = signal.butter(order, [lo, hi], btype="bandpass", fs=SR, output="sos")
    elif lo:
        sos = signal.butter(order, lo, btype="highpass", fs=SR, output="sos")
    else:
        sos = signal.butter(order, hi, btype="lowpass", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def lufs(x):
    return pyln.Meter(SR).integrated_loudness(to_stereo(x))


def process_voice(x, target_lufs=-16.0):
    """Phone-recorded speech -> broadcast-ish voice: HPF, EQ, expander, de-ess, compression."""
    x = butter(x, 65, "hp", 2)
    x = eq_peak(x, 300, -1.5, 1.0)
    x = eq_peak(x, 3400, 1.5, 0.8)
    x = eq_shelf(x, 9500, 2.0, high=True)
    # gentle downward expander to tame room tone between words
    env = env_follow(x, 0.002, 0.06)
    e_db = 20 * np.log10(env + 1e-9)
    floor = np.percentile(e_db, 10)
    thr = floor + 11
    gr = np.clip((e_db - thr) * 0.7, -9, 0)
    gr = signal.filtfilt(*signal.butter(1, 30, fs=SR), gr)
    x = x * db(gr)
    # de-esser on 5.5-10 kHz band
    hi = zero_phase(x, 5500, 10000)
    hi_c, _ = compress(hi, thr_db=-34, ratio=4, att=0.001, rel=0.04, knee=4)
    x = x - hi + hi_c
    x, _ = compress(x, thr_db=-24, ratio=3.2, att=0.004, rel=0.09, knee=6)
    x, _ = compress(x, thr_db=-14, ratio=6, att=0.001, rel=0.05, knee=3)
    gain = target_lufs - lufs(x)
    return x * db(gain)


def voice_activity(voice, att=0.008, rel=0.28):
    """0..1 curve of how much speech is present right now (for ducking)."""
    env = env_follow(voice, att, rel)
    ref = np.percentile(env[env > 1e-4], 90) if np.any(env > 1e-4) else 1.0
    return np.clip(env / ref, 0, 1) ** 0.6


def duck_under_voice(music, act, broad_db=-6.5, mid_db=-5.0):
    st = to_stereo(music)
    mid = zero_phase(st, 600, 4500, 2)
    rest = st - mid
    g_b = db(broad_db * act)[:, None]
    g_m = db(mid_db * act)[:, None]
    return rest * g_b + mid * g_b * g_m


def pump_curve(n, kick_times, depth_db=-9.0, release=0.22, attack=0.004):
    g = np.zeros(n)
    t = np.arange(int(release * SR * 1.6)) / SR
    shape = np.where(t < attack, t / attack, np.exp(-(t - attack) / (release / 3)))
    shape = 1 - np.clip(shape, 0, 1)
    shape = np.maximum(shape, 0)
    dip = 1 - shape  # 1 at the hit, falling to 0
    for kt in kick_times:
        i = int(kt * SR)
        if i >= n:
            continue
        j = min(n, i + len(dip))
        g[i:j] = np.maximum(g[i:j], dip[: j - i])
    return db(depth_db * g)


def master(x, target=-14.0, ceiling=-1.0, tp_max=-1.0):
    """Glue compression, loudness to `target` LUFS, sample-peak limiting and a true-peak guard."""
    st = to_stereo(x)
    st, _ = compress(st, thr_db=-14, ratio=2.0, att=0.012, rel=0.18, knee=8)
    lim_ceiling = ceiling
    for _ in range(6):
        gain = target - lufs(limit(st, lim_ceiling))
        st = st * db(gain)
        out = limit(st, lim_ceiling)
        tp = true_peak_db(out)
        if tp <= tp_max:
            break
        lim_ceiling -= (tp - tp_max) + 0.1
    return out, lufs(out)


def true_peak_db(x, oversample=4):
    st = to_stereo(x)
    up = signal.resample_poly(st, oversample, 1, axis=0)
    return 20 * np.log10(np.max(np.abs(up)) + 1e-12)
