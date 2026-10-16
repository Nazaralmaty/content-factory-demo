"""Audio QA without ears: stems overview + the numbers that decide whether a reel mix works.

Usage: audio_qa.py AUDIO_DIR words.json out.png ['{"DROP": 27.68, ...}']
AUDIO_DIR must hold mix.wav, stem_music.wav, stem_voice.wav, stem_sfx.wav (stems are written at -6 dB).
"""
import json
import sys

import matplotlib
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d, words_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
marks = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
words = json.load(open(words_path))["words"]
mix, sr = sf.read(f"{d}/mix.wav")
mus, _ = sf.read(f"{d}/stem_music.wav")
voc, _ = sf.read(f"{d}/stem_voice.wav")
fx, _ = sf.read(f"{d}/stem_sfx.wav")
meter = pyln.Meter(sr)
mask = np.zeros(len(voc), bool)
for w in words:
    mask[int(w["s"] * sr):int(w["e"] * sr)] = True
phone = signal.butter(4, 250, "hp", fs=sr, output="sos")


def L(x, hp=False):
    return meter.integrated_loudness(signal.sosfiltfilt(phone, x, axis=0) if hp else x)


report = {
    "mix_LUFS": L(mix),
    "true_peak_dBTP": 20 * np.log10(np.abs(signal.resample_poly(mix, 4, 1, axis=0)).max() + 1e-12),
    "voice_over_music_LU_fullband": L(voc[mask]) - L(mus[mask]),
    "voice_over_music_LU_phone": L(voc[mask], True) - L(mus[mask], True),
    "mono_correlation_LR": float(np.corrcoef(mix[:, 0], mix[:, 1])[0, 1]),
    "mono_sum_loss_LU": L(mix) - L(np.stack([mix.mean(axis=1)] * 2, axis=1)),
    "dc_offset": float(np.abs(mix.mean(axis=0)).max()),
}
f, p = signal.welch(mus.mean(axis=1), sr, nperseg=8192)
bands = []
for lo in [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]:
    sel = (f >= lo / np.sqrt(2)) & (f < lo * np.sqrt(2))
    bands.append(10 * np.log10(p[sel].sum() + 1e-20))
bands = np.array(bands) - max(bands)
report["music_octave_bands_dB(31..16k)"] = [round(float(b), 1) for b in bands]
for k, v in report.items():
    print(f"{k:34s} {v if isinstance(v, list) else round(v, 3)}")
warn = []
if abs(report["mix_LUFS"] + 14) > 0.7:
    warn.append("master loudness off -14 LUFS")
if report["true_peak_dBTP"] > -0.9:
    warn.append("true peak above -1 dBTP")
if not 8 <= report["voice_over_music_LU_phone"] <= 13:
    warn.append("phone-weighted voice/music balance outside 8..13 LU")
if report["voice_over_music_LU_fullband"] < 4:
    warn.append("sub-bass too loud under speech (full-band < 4 LU)")
if bands[5] - bands[1] < -24:
    warn.append("music bed has no mids (1 kHz band >24 dB under 63 Hz)")
print("WARN: " + "; ".join(warn) if warn else "OK: all audio checks pass")

fig, ax = plt.subplots(4, 1, figsize=(22, 13), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 2.2]})
for a, (x, name, col) in zip(ax[:3], [(voc, "voice", "#1f6feb"), (mus, "music", "#d1242f"), (fx, "sfx", "#8250df")]):
    t = np.arange(len(x)) / sr
    a.plot(t[::40], x.mean(axis=1)[::40], color=col, lw=0.4)
    a.set_ylabel(name)
    a.set_ylim(-1, 1)
ax[3].specgram(mix.mean(axis=1), NFFT=2048, Fs=sr, noverlap=1536, cmap="magma", vmin=-120)
ax[3].set_ylim(0, 8000)
for k, v in marks.items():
    for a in ax:
        a.axvline(v, color="cyan" if a is not ax[3] else "w", lw=0.8, alpha=0.7)
    ax[0].text(v, 0.85, k, fontsize=8)
for wd in words[::3]:
    ax[0].text(wd["s"], -0.9, wd["w"], fontsize=6, rotation=90)
plt.tight_layout()
plt.savefig(out, dpi=70)
print("plot:", out)
