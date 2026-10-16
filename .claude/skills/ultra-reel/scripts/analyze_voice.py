"""Speaker pitch -> musical key, plus pauses. Usage: analyze_voice.py voice48k.wav WORK_DIR

Writes WORK_DIR/f0.npz (t, f0, conf on the WAV timeline) and prints median F0, the tonic note,
Krumhansl key candidates and every pause >= 180 ms (useful places for fills and stabs).
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import pitch  # noqa: E402

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

wav, work = sys.argv[1], sys.argv[2]
x, sr = sf.read(wav)
if x.ndim > 1:
    x = x.mean(axis=1)
t, f0, conf = pitch.track(x, sr)
np.savez(os.path.join(work, "f0.npz"), t=t, f0=f0, conf=conf)
v = f0[(f0 > 0) & (conf > 0.6)]

# cepstral cross-check (YIN can lock an octave low on deep voices)
hop, win = int(0.02 * sr), int(0.06 * sr)
cep = []
for i in range(0, len(x) - win, hop):
    seg = x[i:i + win] * np.hanning(win)
    if np.sqrt(np.mean(seg ** 2)) < 0.02:
        continue
    c = np.fft.irfft(np.log(np.abs(np.fft.rfft(seg, 8192)) + 1e-9))
    lo, hi = int(sr / 300), int(sr / 60)
    cep.append(sr / (lo + np.argmax(c[lo:hi])))
print(f"F0 median YIN {np.median(v):.1f} Hz, cepstrum {np.median(cep):.1f} Hz")
midi = 69 + 12 * np.log2(v / 440)
tonic = int(np.round(np.median(midi))) % 12
hist = np.bincount(np.round(midi).astype(int) % 12, minlength=12) / len(midi)
cands = sorted([(np.corrcoef(np.roll(MAJ, k), hist)[0, 1], NAMES[k] + " major") for k in range(12)] +
               [(np.corrcoef(np.roll(MIN, k), hist)[0, 1], NAMES[k] + " minor") for k in range(12)], reverse=True)
print(f"tonic from median pitch: {NAMES[tonic]}; key candidates: " + ", ".join(f"{n} ({c:.2f})" for c, n in cands[:4]))

# pauses
hop, win = int(sr * 0.01), int(sr * 0.025)
frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
floor = np.percentile(db, 8)
quiet = db < floor + 8
pauses, i = [], 0
while i < len(quiet):
    if quiet[i]:
        j = i
        while j < len(quiet) and quiet[j]:
            j += 1
        if (j - i) >= 18:
            pauses.append((i / 100, j / 100))
        i = j
    else:
        i += 1
print("pauses >=180ms:", ", ".join(f"{a:.2f}-{b:.2f}" for a, b in pauses))
