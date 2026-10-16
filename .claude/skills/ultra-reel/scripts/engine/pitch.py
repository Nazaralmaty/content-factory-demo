"""YIN fundamental-frequency tracker (numba) for speech-melody features."""
import numpy as np
from numba import njit


@njit(cache=True)
def _yin(x, sr, hop, win, fmin, fmax, thresh):
    n_frames = (x.shape[0] - win) // hop
    tau_min = int(sr / fmax)
    tau_max = int(sr / fmin)
    f0 = np.zeros(n_frames)
    conf = np.zeros(n_frames)
    d = np.zeros(tau_max + 1)
    for fi in range(n_frames):
        s = fi * hop
        for tau in range(1, tau_max + 1):
            acc = 0.0
            for j in range(win - tau_max):
                diff = x[s + j] - x[s + j + tau]
                acc += diff * diff
            d[tau] = acc
        # cumulative mean normalised difference
        run = 0.0
        best_tau = -1
        cmnd_prev = 1.0
        cm = np.ones(tau_max + 1)
        for tau in range(1, tau_max + 1):
            run += d[tau]
            cm[tau] = d[tau] * tau / run if run > 0 else 1.0
        for tau in range(tau_min, tau_max):
            if cm[tau] < thresh:
                while tau + 1 < tau_max and cm[tau + 1] < cm[tau]:
                    tau += 1
                best_tau = tau
                break
        if best_tau > 0:
            # parabolic interpolation
            a = cm[best_tau - 1]
            b = cm[best_tau]
            c = cm[best_tau + 1]
            den = a - 2 * b + c
            shift = 0.5 * (a - c) / den if den != 0 else 0.0
            f0[fi] = sr / (best_tau + shift)
            conf[fi] = 1.0 - b
    return f0, conf


def track(x, sr, hop_s=0.01, fmin=65.0, fmax=400.0, thresh=0.15, target_sr=16000):
    """Return times, f0 (Hz, 0 where unvoiced), confidence."""
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g)
        sr = target_sr
    hop = int(sr * hop_s)
    win = int(sr / fmin * 2.2)
    f0, conf = _yin(np.ascontiguousarray(x, dtype=np.float64), sr, hop, win, fmin, fmax, thresh)
    t = (np.arange(len(f0)) * hop + win / 2) / sr
    return t, f0, conf
