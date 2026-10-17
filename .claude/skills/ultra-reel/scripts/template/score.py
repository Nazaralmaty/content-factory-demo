"""Procedural score + sound design + voice mix for the ICON testimonial reel.

python score.py <voice48k.wav> <f0.npz> <out_dir>
"""
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ULTRA_ENGINE", os.path.join(HERE, "..")))  # <skill>/scripts
sys.path.insert(0, HERE)

from engine import dsp, instruments as I, mix as M  # noqa: E402
import plan as PL  # noqa: E402

SR = dsp.SR
N = dsp.n_samples(PL.DURATION)
rng = np.random.default_rng(2026)


# ------------------------------------------------------------------ helpers
def section_of(n):
    for name, (a, b) in PL.SECTIONS.items():
        if a <= n < b:
            return name
    return None


def chord_at(t):
    n = int(np.clip(np.floor((t - PL.BAR1) / PL.BAR) + 1, 1, 22))
    beat = (t - PL.bar(n)) / PL.BEAT
    name = PL.PROG[n][0][1]
    for off, nm in PL.PROG[n]:
        if beat >= off - 1e-6:
            name = nm
    return name


def speech_at(t0, t1, pad=0.03):
    return any(w["s"] - pad < t1 and w["e"] + pad > t0 for w in PL.WORDS)


def st16(n, step, swing=0.0):
    t = PL.bar(n) + step * PL.BEAT / 4
    if step % 2 == 1:
        t += swing
    return t


SWING = 0.024
drums, hats, bass, chords, lead, sfx, post = (dsp.Bus(PL.DURATION + 1) for _ in range(7))
kick_times = []

# ------------------------------------------------------------------ sound palette (cache one-shots)
KICK = I.kick(punch=1.2, tone=47, decay=0.36, click=1.3, drive=2.0)
KICK_SOFT = dsp.butter(I.kick(punch=0.4, tone=44, decay=0.3, click=0.0, drive=1.2), 220, "lp", 2)
CLAP = I.clap(0.2)
SNARE = I.snare(200, 1.0, 0.14)
GHOST = I.snare(230, 0.6, 0.06, body=0.2)
HAT = I.hat()
HAT_SOFT = I.hat(decay=0.03, bright=0.8)
OHAT = I.hat(open_=True, decay=0.28)
SHAKER = I.shaker()

# ------------------------------------------------------------------ drums
for n in range(1, 22):
    sec = section_of(n)
    if sec in ("intro", "lift", "verse", "level", "b", "drop", "final"):
        for b in range(4):
            t = PL.bar(n, b)
            if n == 13 and t >= PL.TAPE_STOP - 0.01:
                continue
            if n == 21 and t >= PL.FINAL_STOP - 0.01:
                continue
            if n == 8 and b in (1, 3):
                continue  # half-time feel under "высокого уровня", full kick returns on "профессионалы"
            drums.add(t, KICK, 0.95 if sec != "intro" else 0.8)
            kick_times.append(t)
        full = sec != "intro"
        for step in range(16):
            t = st16(n, step, SWING)
            if (n == 13 and t >= PL.TAPE_STOP) or (n == 21 and t >= PL.FINAL_STOP):
                continue
            if step in (2, 6, 10, 14):
                hats.add(t, HAT, 0.22 if full else 0.15, p=0.18 + rng.uniform(-0.12, 0.12))
            elif step % 2 == 1 and full:
                hats.add(t, HAT_SOFT, rng.uniform(0.04, 0.09), p=-0.3 + rng.uniform(-0.1, 0.1))
            if full and step in (4, 12):
                drums.add(t, CLAP, 0.55, p=0.02)
                drums.add(t, SNARE, 0.18)
            if sec in ("b", "drop", "final") and step in (7, 15) and rng.random() < 0.7:
                drums.add(t, GHOST, 0.14, p=-0.1)
            if sec in ("verse", "b", "drop", "final"):
                hats.add(t, SHAKER, 0.05 if step % 4 else 0.07, p=0.45)
            if sec in ("b", "drop", "final") and step == 14 and n % 2 == 0:
                hats.add(t, OHAT, 0.18, p=0.25)
        if sec in ("b", "drop", "final"):
            pent = [76, 79, 81, 83, 86]
            for step in (3, 11, 13) if n % 2 else (6, 9, 14):
                t = st16(n, step, SWING)
                if (n == 13 and t >= PL.TAPE_STOP) or (n == 21 and t >= PL.FINAL_STOP):
                    continue
                if not speech_at(t, t + 0.08):
                    drums.add(t, I.perc_blip(int(rng.choice(pent)) - 12, 0.09, 1.0), 0.16, p=rng.uniform(-0.6, 0.6))
    elif sec == "breakdown":
        drums.add(PL.bar(n), KICK_SOFT, 0.7)
        drums.add(PL.bar(n, 2.5), KICK_SOFT, 0.35)
        for step in range(16):
            hats.add(st16(n, step, SWING), SHAKER, 0.09 if step % 2 else 0.13, p=0.3)
    elif sec == "build":
        for b in range(4):
            drums.add(PL.bar(n, b), KICK, 0.9)
            kick_times.append(PL.bar(n, b))

# snare rolls: into the first drop (bar 14, beats 2-4) and the final drop (bar 19)
def roll(t0, t1, start_div, end_div, g0=0.15, g1=0.6, note_rise=True):
    t = t0
    while t < t1 - 1e-3:
        frac = (t - t0) / (t1 - t0)
        div = start_div if frac < 0.5 else (end_div if frac > 0.75 else (start_div + end_div) / 2)
        step = PL.BEAT / div
        g = g0 + (g1 - g0) * frac ** 1.5
        tone = 190 + (160 * frac if note_rise else 0)
        post.add(t, I.snare(tone, 1.0, 0.09), g, p=0.0)
        t += step


roll(PL.bar(14, 2), PL.bar(15), 4, 8)
roll(PL.bar(19), PL.bar(20), 2, 8, 0.12, 0.65)

# ------------------------------------------------------------------ bass
for n in range(1, 22):
    sec = section_of(n)
    if sec in ("verse", "level", "b", "drop", "final", "lift"):
        pattern = [(0, 5), (6, 2), (10, 3), (14, 2)]
        for step, ln in pattern:
            t = st16(n, step)
            if (n == 13 and t >= PL.TAPE_STOP) or (n == 21 and t >= PL.FINAL_STOP):
                continue
            root = PL.ROOT[chord_at(t + 0.01)]
            note = root + (12 if step == 14 else 0)
            dur = ln * PL.BEAT / 4 * 0.92
            bass.add(t, I.sub_hit(note, dur, glide_from=note + 5 if step == 0 and sec in ("drop", "final") else None), 0.52)
            cut_r = {"b": 650, "drop": 900, "final": 900}.get(sec, 520)
            g_r = {"b": 0.24, "drop": 0.3, "final": 0.3}.get(sec, 0.16)
            bass.add(t, I.reese(note + 12, dur, cutoff=cut_r), g_r)
    elif sec == "intro" and n == 2:
        bass.add(PL.bar(2), I.sub_hit(PL.ROOT["Cmaj9"], PL.BAR * 0.95), 0.5)
    elif sec == "breakdown":
        for off, name in PL.PROG[n]:
            ln = (4 - off) if len(PL.PROG[n]) == 1 else 2
            bass.add(PL.bar(n, off), I.sub_hit(PL.ROOT[name], ln * PL.BEAT * 0.95, drive=1.4), 0.42)
    elif sec == "build":
        for k in range(8):
            t = PL.bar(n, k * 0.5)
            root = PL.ROOT[chord_at(t + 0.01)]
            bass.add(t, I.sub_hit(root, PL.BEAT * 0.42), 0.5)

# ------------------------------------------------------------------ chords: pads, stabs, plucks, arp
for n in range(1, 22):
    sec = section_of(n)
    if sec in (None, "stop"):
        continue
    for k, (off, name) in enumerate(PL.PROG[n]):
        nxt = PL.PROG[n][k + 1][0] if k + 1 < len(PL.PROG[n]) else 4
        dur = (nxt - off) * PL.BEAT
        if n == 13 and PL.bar(n, off) >= PL.TAPE_STOP - 0.01:
            continue
        g = {"intro": 0.5, "breakdown": 0.85, "build": 0.55}.get(sec, 0.45)
        cut = {"breakdown": 2600, "intro": 1400}.get(sec, 1900)
        chords.add(PL.bar(n, off), I.pad(PL.CHORDS[name], dur, attack=0.25 if sec != "breakdown" else 0.6,
                                         release=0.5, cutoff=cut), g)
    if sec in ("lift", "verse", "b", "drop", "final", "level"):
        steps = (3, 10) if sec in ("lift", "verse", "level") else (3, 10, 14)
        for step in steps:
            t = st16(n, step)
            if (n == 13 and t >= PL.TAPE_STOP) or (n == 21 and t >= PL.FINAL_STOP):
                continue
            gap = not speech_at(t, t + 0.16)
            notes = [x + 12 for x in PL.CHORDS[chord_at(t + 0.01)][:4]]
            g = (0.75 if gap else 0.22) * (0.8 if sec == "intro" else 1.0)
            st_ = dsp.haas(I.stab(notes, 0.13, bright=4200 if gap else 2400), 9, 1 if step == 3 else -1)
            chords.add(t, st_, g)

# level meter arpeggio (16ths) rising into "профессионалы"
ARP = [64, 67, 71, 74, 78, 79, 83, 86, 88, 91]
k0 = int(np.ceil(PL.step16(14.62)))
for i, step in enumerate(range(k0, k0 + len(ARP))):
    t = PL.BAR1 + step * PL.BEAT / 4
    lead.add(t, I.pluck(ARP[i], 0.22, bright=0.5 + 0.05 * i, decay=0.18), 0.26 + 0.02 * i, p=-0.5 + i * 0.1)

# breakdown plucks: the speaker's stressed words play the melody
melody = [71, 74, 76, 79, 76, 74, 79, 81, 83, 81, 79, 76]
for i, wi in enumerate(PL.PLUCK_WORDS):
    t = PL.WORDS[wi]["s"] + 0.02
    note = melody[i % len(melody)]
    lead.add(t, I.pluck(note, 0.6, bright=0.35, decay=0.45), 0.34, p=(-0.4 if i % 2 else 0.4))
    lead.add(t, I.bell(note + 12, 1.0, ratio=2.0, index=0.6), 0.07)

# ------------------------------------------------------------------ speech-melody doubling
def speech_double(t0, t1, octaves=3, gain=0.2):
    f0d = np.load(sys.argv[2])
    tt, f0, conf = f0d["t"] + 0.020, f0d["f0"], f0d["conf"]
    m = (tt >= t0) & (tt <= t1)
    tt, f0, conf = tt[m], f0[m], conf[m]
    if len(tt) < 3:
        return
    voiced = (f0 > 0) & (conf > 0.45)
    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0, 1) / 440) + 12 * octaves, np.nan)
    scale = np.array([0, 3, 5, 7, 10])  # E minor pentatonic relative to E
    q = np.full_like(midi, np.nan)
    for i, v in enumerate(midi):
        if np.isnan(v):
            continue
        rel = v - 64
        octv = np.floor(rel / 12)
        cand = scale + 12 * octv
        cand = np.concatenate([cand, cand + 12])
        q[i] = 64 + cand[np.argmin(np.abs(cand - rel))]
    # hold last pitch through short gaps, glide between notes
    last = np.nan
    for i in range(len(q)):
        if np.isnan(q[i]):
            q[i] = last
        else:
            last = q[i]
    if np.all(np.isnan(q)):
        return
    first = q[~np.isnan(q)][0]
    q = np.where(np.isnan(q), first, q)
    n = dsp.n_samples(t1 - t0)
    ts = t0 + np.arange(n) / SR
    pitch_curve = np.interp(ts, tt, q)
    from scipy.ndimage import uniform_filter1d
    pitch_curve = uniform_filter1d(pitch_curve, size=int(0.03 * SR))
    amp = np.interp(ts, tt, voiced.astype(float))
    amp = uniform_filter1d(amp, size=int(0.05 * SR))
    tone = I.follow_lead(dsp.mtof(pitch_curve), amp, "glass")
    lead.add(t0, dsp.reverb(dsp.delay(tone, PL.BEAT * 0.75, 0.35, 0.35), 2.4, 0.45), gain)


speech_double(PL.WORDS[87]["s"], PL.WORDS[91]["e"] + 0.05, 3, 0.2)
speech_double(PL.WORDS[115]["s"], PL.WORDS[116]["e"], 3, 0.12)

# ------------------------------------------------------------------ sonic logo
def sonic_logo(t, short=True, final=False):
    notes = [(0.0, 71), (0.18, 76)] if short else [(0.0, 71), (0.12, 76), (0.24, 78), (0.36, 83)]
    for dt, nt in notes:
        g = 0.32 if short else 0.36
        sfx.add(t + dt, dsp.delay(I.bell(nt, 1.4, ratio=3.01, index=1.4), PL.BEAT * 0.75, 0.32, 0.3), g)
        sfx.add(t + dt, I.pluck(nt, 0.3, bright=0.7, decay=0.2), g * 0.6)
    if final:
        sfx.add(t + 0.48, dsp.reverb(I.bell(88, 2.6, ratio=2.0, index=0.9), 3.0, 0.5), 0.3)


# ------------------------------------------------------------------ sound-design cues
for cue in PL.SFX:
    if cue[0] == "typing":
        continue
    t, kind, prm = cue
    if kind == "hook_hit":
        sfx.add(t, I.hook_hit(prm["note"]), 0.9)
    elif kind == "reverse_swell":
        d = prm["until"] - t
        sw = I.reverse_swell(d + 0.25, prm.get("note"))
        sfx.add(prm["until"] - len(sw) / SR, sw, 0.55)
    elif kind == "swish":
        sfx.add(t - 0.08, I.whoosh(0.22, prm["dir"], 1500, 7000), 0.45)
    elif kind == "whoosh":
        d = prm.get("dur", 0.35)
        sfx.add(t - d * 0.6, I.whoosh(d, prm["dir"]), 0.6)
    elif kind == "zoom_whoosh":
        sfx.add(t - 0.3, I.whoosh(0.45, 1, 300, 6500), 0.55)
        sfx.add(t - 0.3, I.riser(0.3, 600, 6000, amount=0.5), 0.5)
    elif kind == "thud":
        sfx.add(t, I.kick(punch=0.5, tone=52, decay=0.18, click=0.4, drive=1.4), 0.45)
    elif kind == "stamp":
        sfx.add(t, I.stamp(), 0.55)
    elif kind == "stamp_big":
        sfx.add(t, I.stamp(), 0.85)
        sfx.add(t, I.sub_hit(28, 0.8, glide_from=40), 0.45)
        sfx.add(t, dsp.reverb(I.clap(0.15), 1.8, 0.5), 0.45)
    elif kind == "pop":
        sfx.add(t, I.ui_pop(prm["note"]), 0.45)
    elif kind == "card":
        sfx.add(t, I.ui_pop(prm["note"] + 12, 0.05), 0.4)
        sfx.add(t, I.bell(prm["note"], 0.9, ratio=2.0, index=0.8), 0.22)
        sfx.add(t - 0.05, I.whoosh(0.18, 1, 2000, 7000), 0.25)
    elif kind == "tick":
        sfx.add(t, I.ui_pop(prm["note"], 0.04), 0.35)
        sfx.add(t, I.rim(2400), 0.12)
    elif kind == "ding":
        sfx.add(t, dsp.reverb(I.bell(prm["note"], 1.2, ratio=3.5, index=1.2), 1.6, 0.35), 0.3)
    elif kind == "glitch":
        sfx.add(t - 0.04, I.glitch_burst(0.2, prm["seed"]), 0.45)
    elif kind == "marker":
        sfx.add(t, I.marker(0.3), 0.4)
    elif kind == "click":
        sfx.add(t, I.mouse_click(), 0.5)
    elif kind == "pin":
        sfx.add(t, I.ui_pop(88, 0.05), 0.4)
        sfx.add(t + 0.06, I.kick(punch=0.3, tone=60, decay=0.1, click=0.2), 0.3)
        sfx.add(t + 0.06, I.bell(83, 1.0), 0.2)
    elif kind == "shimmer":
        sfx.add(t, I.shimmer(1.6), 0.35)
    elif kind == "impact":
        sfx.add(t, I.impact(2.6, 28), 0.7)
        sfx.add(t, I.crash(2.2), 0.55)
    elif kind == "impact_s":
        sfx.add(t, I.impact(1.6, 33, 0.7), 0.4)
        sfx.add(t, I.crash(1.4, 0.9), 0.32)
    elif kind == "logo":
        sonic_logo(t, prm.get("short", True), prm.get("final", False))
    elif kind == "tape_stop":
        pass  # applied to the music bus below
    elif kind == "brush":
        sfx.add(t, I.brush_swish(prm.get("dur", 0.4), prm.get("bright", 1.0), prm.get("dir", 1), seed=abs(int(t * 100))), 0.5)
    elif kind == "brush_big":
        sfx.add(t, I.brush_swish(prm.get("dur", 0.5), 0.8, 1, seed=abs(int(t * 100))), 0.75)
        sfx.add(t + 0.02, I.sub_hit(28, 0.7, glide_from=36, drive=1.3), 0.35)
    elif kind == "air":
        sfx.add(t, I.air(prm.get("dur", 0.7)), 0.35)
    elif kind == "bell":
        sfx.add(t, dsp.reverb(I.bell(prm["note"], 1.3, ratio=2.0, index=0.7), 1.8, 0.4), 0.26)

# risers
sfx.add(PL.bar(3, 3) - 0.40, I.riser(0.44, 800, 8000, tonal_note=64, amount=0.6), 0.5)
sfx.add(PL.TAPE_STOP + 0.15, I.riser(PL.bar(15) - PL.TAPE_STOP - 0.15, 250, 11000, tonal_note=47), 0.75)
sfx.add(PL.bar(19), I.riser(PL.BAR, 300, 12000, tonal_note=52), 0.6)
sfx.add(PL.bar(20) - 0.6, I.reverse_swell(0.6), 0.5)

# final: chord swell + sub + outro tail
post.add(PL.TOP, I.pad(PL.CHORDS["Eadd9"], 1.2, attack=0.01, release=1.9, cutoff=2200), 0.3)
post.add(PL.TOP, I.sub_hit(28, 1.6, glide_from=40, drive=1.6), 0.45)
post.add(PL.TOP, I.stab([x + 12 for x in PL.CHORDS["Eadd9"]], 0.35, bright=5200), 0.32)
post.add(PL.TOP + 2.6, I.downlifter(1.6, 4000, 150), 0.25)

# ------------------------------------------------------------------ music bus assembly
chords.buf = dsp.reverb(chords.buf, 1.6, 0.28, 0.018, 5000, 300)
hats.buf = dsp.reverb(hats.buf, 0.9, 0.12, 0.008, 7000, 500)
pumped = bass.buf + chords.buf
pump = M.pump_curve(len(pumped), kick_times, depth_db=-8.5, release=0.24)
music = drums.buf * 1.0 + hats.buf * 0.9 + pumped * pump[:, None] + lead.buf

# section filter: intro opens up and snaps fully open on the first "ICON"
cut = np.full(len(music), 20000.0)
t = np.arange(len(music)) / SR
i_open = (t >= 0) & (t < PL.WORDS[17]["s"])
cut[i_open] = 650 * (3200 / 650) ** np.clip(t[i_open] / PL.WORDS[17]["s"], 0, 1)
bd = (t >= PL.bar(17)) & (t < PL.bar(19))
cut[bd] = 5200
bu = (t >= PL.bar(19)) & (t < PL.bar(20))
cut[bu] = 5200 * (20000 / 5200) ** ((t[bu] - PL.bar(19)) / PL.BAR)
for c in range(2):
    music[:, c] = dsp.svf(music[:, c], cut, 0.75, "lp")
# build: thin the low end (high-pass rising), restored on the drop
hp = np.full(len(music), 10.0)
hp[bu] = 20 * (320 / 20) ** ((t[bu] - PL.bar(19)) / PL.BAR)
for c in range(2):
    music[:, c] = dsp.svf(music[:, c], hp, 0.7, "hp")

music = dsp.eq_shelf(music, 11000, -2.5, high=True)
music = dsp.eq_shelf(music, 130, -9.5, high=False)

# tape stop on "я точно", then silence until the roll
music = I.tape_stop(music, PL.TAPE_STOP, 0.55, curve=1.6)
z0, z1 = int((PL.TAPE_STOP + 0.55) * SR), int(PL.bar(14, 2) * SR)
music[z0:z1] *= 0.0
# hard stop before the last word
f0_, f1_ = int(PL.FINAL_STOP * SR), int(PL.TOP * SR)
fade = int(0.012 * SR)
music[f0_:f0_ + fade] *= np.linspace(1, 0, fade)[:, None]
music[f0_ + fade:f1_] = 0
# outro fade
o0 = int((PL.TOP) * SR)
music[o0:] *= 0  # everything after "топ" comes from the post bus

music = music + post.buf

# ------------------------------------------------------------------ voice
vx, vsr = sf.read(sys.argv[1])
assert vsr == SR
if vx.ndim > 1:
    vx = vx.mean(axis=1)
voice = np.zeros(len(music))
v = M.process_voice(vx, target_lufs=-16.0)
off = int(0.020 * SR)
voice[off:off + len(v)] = v[: len(voice) - off]

# throws: echo on the last word, a whisper of one on "ICON."
def throw(word_i, gain, fb=0.45):
    s, e = PL.WORDS[word_i]["s"], PL.WORDS[word_i]["e"] + 0.05
    seg = voice[int(s * SR):int(e * SR)]
    seg = dsp.butter(seg, 350, "hp", 2)
    tail = np.concatenate([seg, np.zeros(int(2.6 * SR))])
    wet = dsp.delay(tail, PL.BEAT * 0.75, fb, 1.0) - dsp.to_stereo(tail)
    wet = dsp.reverb(wet, 2.2, 0.4)
    sfx.add(s, wet, gain)


throw(117, 0.55)
throw(74, 0.12, 0.3)

# ------------------------------------------------------------------ section dynamics
lvl = np.ones(len(music))
for name, g in {"intro": 0.7, "lift": 0.82, "verse": 0.9, "level": 0.86, "b": 1.0, "drop": 1.12,
                "breakdown": 0.85, "final": 1.15}.items():
    a, b = PL.SECTIONS[name]
    lvl[int(PL.bar(a) * SR):int(PL.bar(b) * SR)] = g
bu0, bu1 = int(PL.bar(19) * SR), int(PL.bar(20) * SR)
lvl[bu0:bu1] = np.linspace(0.8, 1.1, bu1 - bu0)
from scipy.ndimage import uniform_filter1d
lvl = uniform_filter1d(lvl, int(0.05 * SR))
music = music[:len(lvl)] * lvl[:, None]

# ------------------------------------------------------------------ mixdown (voice stays on top)
music, voice, sfxb = music[:N], voice[:N], sfx.buf[:N]
act = M.voice_activity(voice)
speech_mask = np.zeros(N, bool)
for wd in PL.WORDS:
    speech_mask[int(wd["s"] * SR):int(wd["e"] * SR)] = True


PHONE = signal.butter(4, 250, "hp", fs=SR, output="sos")


def speech_ratio(mg, phone=True):
    """Loudness gap (LU) between voice and ducked music while words are spoken.
    phone=True high-passes both at 250 Hz first: that is what a phone speaker actually plays."""
    md = M.duck_under_voice(music * mg, act, broad_db=-6.0, mid_db=-4.0)
    v, m = voice[speech_mask], md[speech_mask]
    if phone:
        v, m = signal.sosfiltfilt(PHONE, v, axis=0), signal.sosfiltfilt(PHONE, m, axis=0)
    return M.lufs(v) - M.lufs(m), md


TARGET_RATIO = float(os.environ.get("VOICE_OVER_MUSIC_LU", "10.0"))
mg = 1.0
for _ in range(6):
    r, music_d = speech_ratio(mg)
    mg *= dsp.db(r - TARGET_RATIO)
r, music_d = speech_ratio(mg)
r_full, _ = speech_ratio(mg, phone=False)
print(f"music gain {20*np.log10(mg):.1f} dB -> voice over music during speech: {r:.1f} LU phone-weighted, "
      f"{r_full:.1f} LU full-band")
sfx_d = M.duck_under_voice(sfxb * mg * 1.4, act, broad_db=-3.0, mid_db=-4.0)
mixbus = dsp.to_stereo(voice) + music_d + sfx_d
# short fade-out at the very end
fo = int(0.6 * SR)
mixbus[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 2
final, lu = M.master(mixbus, target=-14.0, ceiling=-1.2)

out = sys.argv[3]
os.makedirs(out, exist_ok=True)
sf.write(os.path.join(out, "mix.wav"), final.astype(np.float32), SR, subtype="FLOAT")
sf.write(os.path.join(out, "stem_music.wav"), (music_d * 0.5).astype(np.float32), SR, subtype="FLOAT")
sf.write(os.path.join(out, "stem_voice.wav"), (dsp.to_stereo(voice) * 0.5).astype(np.float32), SR, subtype="FLOAT")
sf.write(os.path.join(out, "stem_sfx.wav"), (sfx_d * 0.5).astype(np.float32), SR, subtype="FLOAT")
print(f"mix: {len(final)/SR:.2f}s  LUFS {lu:.2f}  true-peak {M.true_peak_db(final):.2f} dBTP")
