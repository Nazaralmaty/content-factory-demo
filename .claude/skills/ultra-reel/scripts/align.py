"""Refine ASR word timings by aligning transcript vowels to detected syllable nuclei.

Whisper's word timestamps drift by up to ~0.5 s around names and pauses. Every Russian/Kazakh
syllable has exactly one vowel, so the vowel sequence of the transcript can be matched
monotonically (Needleman-Wunsch) against energy peaks of the vowel band, using the ASR times only
as a prior. Word boundaries are then placed in the energy valley between neighbouring words.

Usage: align.py voice.wav words.json out.json
"""
import json
import sys

import numpy as np
import soundfile as sf
from scipy import signal

VOWELS = set("аеёиоуыэюяәіөүұaeiouyАЕЁИОУЫЭЮЯӘІӨҮҰAEIOUY")
HOP = 0.01


def vowel_count(word):
    return sum(ch in VOWELS for ch in word)


def band_env(x, sr):
    sos = signal.butter(4, [300, 2500], btype="bandpass", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, x)
    hop, win = int(sr * HOP), int(sr * 0.03)
    n = (len(y) - win) // hop
    frames = np.lib.stride_tricks.sliding_window_view(y, win)[::hop][:n]
    e = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    k = np.hanning(5)
    return np.convolve(e, k / k.sum(), mode="same")


def nuclei(env, floor):
    peaks, props = signal.find_peaks(env, prominence=2.0, distance=7, height=floor + 8)
    return peaks * HOP, props["prominences"]


def align(words, env):
    floor = np.percentile(env, 8)
    pk_t, pk_p = nuclei(env, floor)
    tokens = []  # (word_index, prior_time)
    for i, w in enumerate(words):
        v = vowel_count(w["w"])
        for k in range(v):
            frac = (k + 0.5) / v
            tokens.append((i, w["s"] + (w["e"] - w["s"]) * frac))
    T, M = len(tokens), len(pk_t)
    sigma, skip_tok, skip_pk = 0.28, 1.6, 0.9
    D = np.full((T + 1, M + 1), np.inf)
    B = np.zeros((T + 1, M + 1), dtype=np.int8)
    D[0, :] = np.arange(M + 1) * skip_pk
    D[:, 0] = np.arange(T + 1) * skip_tok
    for a in range(1, T + 1):
        pa = tokens[a - 1][1]
        for b in range(1, M + 1):
            c_match = D[a - 1, b - 1] + ((pk_t[b - 1] - pa) / sigma) ** 2 * 0.5 - 0.15 * min(pk_p[b - 1], 8) / 8
            c_tok = D[a - 1, b] + skip_tok
            c_pk = D[a, b - 1] + skip_pk * (0.5 if pk_p[b - 1] < 4 else 1.0)
            best = min(c_match, c_tok, c_pk)
            D[a, b] = best
            B[a, b] = 0 if best == c_match else (1 if best == c_tok else 2)
    a, b = T, M
    tok_time = [None] * T
    while a > 0 and b > 0:
        if B[a, b] == 0:
            tok_time[a - 1] = pk_t[b - 1]
            a, b = a - 1, b - 1
        elif B[a, b] == 1:
            a -= 1
        else:
            b -= 1
    # fill unmatched tokens by interpolating prior offsets of matched neighbours
    prior = np.array([t[1] for t in tokens])
    known = np.array([t is not None for t in tok_time])
    off = np.zeros(T)
    if known.any():
        idx = np.where(known)[0]
        off_known = np.array([tok_time[i] - prior[i] for i in idx])
        off = np.interp(np.arange(T), idx, off_known)
    times = np.array([tok_time[i] if known[i] else prior[i] + off[i] for i in range(T)])
    times = np.maximum.accumulate(times)
    # per-word vowel times
    first, last = {}, {}
    for (wi, _), t in zip(tokens, times):
        first.setdefault(wi, t)
        last[wi] = t
    out = []
    n_frames = len(env)
    for i, w in enumerate(words):
        if i in first:
            f, l = first[i], last[i]
        else:  # vowelless word ("в", "к", "с"): glue to the next voiced word
            nxt = next((first[j] for j in range(i + 1, len(words)) if j in first), w["s"])
            f = l = nxt - 0.04
        out.append({**w, "v0": float(f), "v1": float(l)})
    # boundaries: valley of the envelope between last vowel of word i and first vowel of word i+1
    for i in range(len(out)):
        f = out[i]["v0"]
        lo = out[i - 1]["v1"] if i > 0 else max(0.0, f - 0.4)
        a_, b_ = int(lo / HOP) + 1, int(f / HOP)
        if b_ > a_:
            seg = env[a_:b_]
            vi = a_ + int(np.argmin(seg))
            start = vi * HOP
            # if the valley is real silence, start where energy rises again
            if env[vi] < floor + 6:
                j = vi
                while j < b_ and env[j] < floor + 8:
                    j += 1
                start = max(start, (j - 3) * HOP)
        else:
            start = f - 0.06
        out[i]["s"] = round(float(min(start, f - 0.02)), 3)
    for i in range(len(out)):
        nxt = out[i + 1]["s"] if i + 1 < len(out) else min(out[i]["v1"] + 0.35, n_frames * HOP)
        l = out[i]["v1"]
        a_, b_ = int(l / HOP), int(nxt / HOP)
        end = nxt
        if b_ > a_:
            j = a_
            while j < b_ and env[j] > floor + 7:
                j += 1
            end = min(nxt, (j + 2) * HOP)
        out[i]["e"] = round(float(max(end, out[i]["s"] + 0.06)), 3)
    return out, pk_t


if __name__ == "__main__":
    wav, wj, outp = sys.argv[1:4]
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x.mean(axis=1)
    data = json.load(open(wj))
    env = band_env(x, sr)
    refined, pk = align(data["words"], env)
    data["words_asr"] = data["words"]
    data["words"] = refined
    json.dump(data, open(outp, "w"), ensure_ascii=False, indent=1)
    for a, r in zip(data["words_asr"], refined):
        flag = " <<" if abs(a["s"] - r["s"]) > 0.15 else ""
        print(f"{r['w']:>20s}  asr {a['s']:6.2f}  ->  {r['s']:6.2f}-{r['e']:6.2f}{flag}")
