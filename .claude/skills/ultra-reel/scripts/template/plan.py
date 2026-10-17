"""Painterly cut ("холст") of the ICON testimonial: one timeline shared by score and picture.

All times are seconds on the VIDEO timeline (speech words already shifted by the AV offset).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORDS = json.load(open(os.path.join(HERE, "words.json")))["words"]
FPS = 30
W, H = 1080, 1920
DURATION = 44.6
SOURCE_END = 41.67

# ---------------------------------------------------------------- tempo grid
BPM = 124.8
BEAT = 60.0 / BPM
BAR = 4 * BEAT
BAR1 = 27.680 - 14 * BAR  # bar 15 downbeat is locked to the second "ICON" (the drop)


def bar(n, beat=0.0):
    """Start time of bar n (1-based) plus `beat` quarter notes."""
    return BAR1 + (n - 1) * BAR + beat * BEAT


def step16(t):
    """Index of the 16th-note step at time t (can be fractional)."""
    return (t - BAR1) / (BEAT / 4)


# chord per bar (bar number -> list of (beat_offset, chord_name))
CHORDS = {
    "Em9": [52, 55, 59, 62, 66], "Cmaj9": [48, 52, 55, 59, 62], "G69": [55, 59, 62, 64, 69],
    "D69": [54, 57, 62, 64, 71], "Am9": [57, 60, 64, 67, 71], "B7sus": [59, 64, 66, 69],
    "B7": [59, 63, 66, 69], "Eadd9": [52, 59, 64, 66, 67],
}
ROOT = {"Em9": 28, "Cmaj9": 36, "G69": 31, "D69": 38, "Am9": 33, "B7sus": 35, "B7": 35, "Eadd9": 28}
PROG = {
    1: [(0, "Em9")], 2: [(0, "Cmaj9")], 3: [(0, "Am9")], 4: [(0, "Em9")], 5: [(0, "Cmaj9")],
    6: [(0, "G69")], 7: [(0, "D69")], 8: [(0, "Em9")], 9: [(0, "Cmaj9")], 10: [(0, "Am9")],
    11: [(0, "Em9")], 12: [(0, "Cmaj9")], 13: [(0, "B7sus"), (2, "B7")], 14: [(0, "B7")],
    15: [(0, "Em9")], 16: [(0, "Cmaj9")], 17: [(0, "Cmaj9")], 18: [(0, "G69"), (2, "D69")],
    19: [(0, "Am9"), (2, "B7sus")], 20: [(0, "Em9")], 21: [(0, "Cmaj9")], 22: [(0, "Eadd9")],
}

# arrangement sections: name -> (start_bar, end_bar_exclusive)
SECTIONS = {
    "intro": (1, 3), "lift": (3, 4), "verse": (4, 8), "level": (8, 10), "b": (10, 14),
    "stop": (14, 15), "drop": (15, 17), "breakdown": (17, 19), "build": (19, 20), "final": (20, 22),
}
TAPE_STOP = bar(13, 3)       # "я" (25.22) — music grinds to a halt
FINAL_STOP = bar(21, 3)      # "они точно" — silence before the last word
TOP = bar(22)                # "топ." final hit (41.14)


def w(i):
    return WORDS[i]["s"]


# ---------------------------------------------------------------- picture scenes
# (start, end, scene_name, transition_in)
# ---------------------------------------------------------------- picture: one canvas, no cuts
# Speaker layouts (cx, cy, r, zoom_on_face). The speaker node glides between them (Motion Canvas style).
L = {
    "hidden": (540, 900, 0, 1.0), "intro": (540, 860, 340, 1.0), "up": (540, 760, 320, 1.0),
    "mid": (540, 720, 300, 1.0), "high": (540, 640, 260, 1.0), "center": (540, 820, 360, 1.0),
    "big": (540, 860, 440, 1.25), "logo": (540, 760, 250, 1.0),
}
# (start, layout, mood) — mood decides background washes and text colour
SCENES = [
    (0.00, "hidden", "canvas"), (0.85, "intro", "canvas"), (4.36, "up", "canvas"), (4.84, "mid", "canvas"),
    (6.46, "high", "canvas"), (9.78, "intro", "canvas"), (11.81, "high", "canvas"), (14.62, "center", "canvas"),
    (16.59, "big", "canvas"), (18.78, "mid", "canvas"), (20.16, "mid", "canvas"), (22.78, "mid", "canvas"),
    (25.22, "center", "dusk"), (27.68, "center", "terra"), (29.17, "mid", "canvas"), (31.47, "center", "ochre"),
    (35.37, "mid", "canvas"), (36.62, "mid", "canvas"), (37.78, "up", "terra"), (40.65, "center", "canvas"),
    (41.14, "center", "terra"), (42.40, "logo", "canvas"),
]
GLIDE = 0.62  # seconds for a layout change, centred on the boundary

# Big washes: (t_in, t_full, t_out_start, t_out_end, color, alpha, seed, w, h, centre)
WASHES = [
    (0.44, 0.70, 0.78, 1.05, "terra", 0.75, 20, 760, 460, (540, 1240)),
    (25.20, 25.95, 27.55, 28.10, "dusk", 0.93, 21, 1500, 2300, "speaker"),
    (27.64, 28.10, 29.00, 29.70, "terra", 0.86, 22, 1450, 2200, "speaker"),
    (31.45, 32.30, 35.10, 35.80, "ochre", 0.42, 23, 1300, 1900, "speaker"),
    (37.24, 37.70, 40.55, 40.80, "terra", 0.82, 24, 1450, 2200, "speaker"),
    (41.10, 41.42, 42.20, 42.75, "terra", 0.88, 25, 1450, 2200, "speaker"),
]

# Text items: (t_in, t_out, kind, text, font, size, x, y, color)
#   kind "rise" = serif up from below, blur->sharp;  "brush" = written on by a brush
#   color: ink / terra / cream / grey / auto (ink on light, cream on dark washes)
TEXT = [
    (-0.30, 1.00, "brush", "Они", "brush", 330, 540, 770, "ink"),
    (0.20, 1.00, "brush", "точно", "brush", 330, 540, 1020, "ink"),
    (0.44, 1.00, "brush", "топ.", "brush", 420, 540, 1320, "terra"),
    (1.05, 3.15, "rise", "Антон Петраченков", "serif", 78, 540, 1330, "ink"),
    (w(6) - 0.05, 3.15, "rise", "ConvertMonster", "serif_i", 62, 540, 1410, "terra"),
    (w(11) - 0.04, 4.82, "brush", "лично", "brush", 217, 540, 440, "terra"),
    (4.33, 4.82, "rise", "поблагодарить", "serif", 104, 540, 1290, "ink"),
    (w(13) - 0.04, 6.44, "rise", "двух маркетологов", "serif", 92, 540, 1210, "ink"),
    (w(15) - 0.03, 6.44, "rise", "из компании", "serif_i", 66, 540, 1300, "ink"),
    (w(17) - 0.04, 6.44, "rise", "ICON", "serif_wide", 150, 540, 1450, "ink"),
    (w(19) - 0.04, 9.76, "rise", "Абубакир", "serif_i", 110, 540, 1130, "ink"),
    (w(20) - 0.03, 9.76, "rise", "и", "serif_i", 69, 540, 1230, "grey"),
    (w(21) - 0.04, 9.76, "rise", "Еркебулан", "serif_i", 110, 540, 1350, "ink"),
    (w(27) - 0.04, 12.25, "rise", "очень сильно", "serif_i", 73, 540, 1300, "ink"),
    (w(29) - 0.04, 12.25, "brush", "помогли", "brush", 275, 540, 1470, "terra"),
    (11.81, 14.60, "rise", "в создании", "serif_i", 66, 540, 1090, "ink"),
    (w(32) - 0.04, 14.60, "rise", "курса по трафику", "serif", 108, 540, 1230, "ink"),
    (w(35) - 0.04, 14.60, "rise", "в рамках нашей академии", "serif_i", 62, 540, 1350, "ink"),
    (w(40) - 0.04, 16.57, "rise", "очень высокого уровня", "serif_i", 69, 540, 1290, "ink"),
    (w(43) - 0.04, 16.57, "rise", "профессионалы", "serif", 118, 540, 1430, "ink"),
    (w(45) - 0.04, 18.76, "rise", "если", "serif", 92, -1, 330, "ink"),
    (w(46) - 0.04, 18.76, "rise", "вам", "serif", 92, -2, 330, "ink"),
    (w(47) - 0.04, 18.76, "rise", "нужен", "serif", 92, -3, 330, "terra"),
    (w(51) - 0.04, 20.14, "rise", "классный подрядчик", "serif", 100, 540, 1210, "ink"),
    (w(53) - 0.04, 20.14, "brush", "по трафику", "brush", 217, 540, 1385, "terra"),
    (w(55) - 0.04, 22.76, "rise", "который умеет", "serif_i", 71, 540, 1150, "ink"),
    (w(57) - 0.04, 22.76, "rise", "работать", "serif", 120, 540, 1290, "ink"),
    (w(58) - 0.04, 22.76, "rise", "со всеми этими", "serif_i", 62, 540, 1390, "ink"),
    (w(61) - 0.04, 25.20, "rise", "зарубежными", "serif", 100, 540, 1180, "ink"),
    (w(62) - 0.04, 25.20, "rise", "инструментами,", "serif", 100, 540, 1290, "ink"),
    (w(63) - 0.04, 25.20, "rise", "которые работают в СНГ", "serif_i", 66, 540, 1400, "ink"),
    (w(67) - 0.03, 27.62, "rise", "я точно", "serif_i", 70, 540, 1290, "cream"),
    (w(69) - 0.05, 27.62, "brush", "рекомендую", "brush", 275, 540, 1465, "cream"),
    (w(71) - 0.04, 27.62, "rise", "обращаться в агентство", "serif_i", 59, 540, 330, "cream"),
    (27.62, 29.15, "rise", "ICON", "serif_wide", 170, 540, 1430, "ink"),
    (w(79) - 0.04, 31.45, "rise", "они у меня", "serif_i", 71, 540, 1170, "ink"),
    (w(83) - 0.04, 31.45, "rise", "подрядчики", "serif", 130, 540, 1320, "ink"),
    (w(86) - 0.04, 35.30, "rise", "ребят,", "serif_i", 71, 540, 330, "ink"),
    (w(87) - 0.05, 33.05, "brush", "спасибо", "brush", 333, 540, 1390, "terra"),
    (w(89) - 0.04, 33.05, "rise", "за ваш труд", "serif_i", 69, 540, 1490, "ink"),
    (w(92) - 0.04, 34.58, "rise", "я вам очень сильно благодарен", "serif", 66, 540, 1430, "ink"),
    (w(97) - 0.04, 35.35, "rise", "это реально", "serif_i", 71, 540, 1430, "ink"),
    (w(100) - 0.04, 36.60, "rise", "высококачественная", "serif", 108, 540, 1210, "ink"),
    (w(101) - 0.04, 36.60, "brush", "работа", "brush", 290, 540, 1390, "terra"),
    (w(102) - 0.04, 37.76, "rise", "желаю вам всего", "serif_i", 73, 540, 1210, "ink"),
    (w(105) - 0.04, 37.76, "brush", "классного", "brush", 290, 540, 1390, "ink"),
    (w(107) - 0.04, 39.90, "rise", "если вам нужны", "serif_i", 73, 540, 1250, "ink"),
    (w(110) - 0.04, 39.90, "rise", "классные подрядчики", "serif", 100, 540, 1370, "ink"),
    (w(112) - 0.05, 40.66, "brush", "идите к ним", "brush", 275, 540, 1340, "cream"),
    (w(115) - 0.04, 41.10, "rise", "они точно", "serif_i", 80, 540, 1400, "ink"),
    (TOP - 0.04, 42.38, "brush", "топ.", "brush", 420, 540, 1480, "cream"),
    (41.50, 42.38, "rise", "Антон Петраченков · ConvertMonster", "serif_i", 50, 540, 330, "ink"),
    (42.55, 99, "rise", "ICON", "serif_wide", 130, 540, 1180, "ink"),
    (42.75, 99, "rise", "подрядчик по трафику", "serif_i", 62, 540, 1265, "terra"),
    (42.95, 99, "rise", "отзыв · Антон Петраченков, ConvertMonster", "serif_small", 34, 540, 1370, "grey"),
]

# Brush strokes: (t_in, draw_dur, t_out, kind, params, color)
STROKES = [
    (0.75, 0.85, 99, "enso", {"r_pad": 26, "width": 34, "seed": 5}, "ink"),
    (4.45, 0.30, 4.82, "line", {"x0": 250, "x1": 830, "y": 1318, "w": 18, "seed": 31}, "ochre"),
    (6.08, 0.32, 6.44, "line", {"x0": 380, "x1": 700, "y": 1482, "w": 16, "seed": 32}, "terra"),
    (12.90, 0.40, 14.60, "line", {"x0": 470, "x1": 900, "y": 1262, "w": 18, "seed": 33}, "ochre"),
    (24.86, 0.28, 25.20, "check", {"x": 905, "y": 1385, "s": 70, "seed": 34}, "terra"),
    (27.70, 0.60, 29.15, "enso2", {"r_pad": 46, "width": 26, "seed": 41}, "cream"),
    (30.30, 0.40, 31.45, "line", {"x0": 230, "x1": 850, "y": 1350, "w": 20, "seed": 35}, "terra"),
    (36.10, 0.35, 36.60, "line", {"x0": 300, "x1": 780, "y": 1418, "w": 14, "seed": 36}, "ochre"),
    (41.16, 0.45, 42.38, "enso2", {"r_pad": 52, "width": 40, "seed": 42}, "cream"),
]

# breakdown: each stressed word leaves a small dab of paint (the same words play the plucks in the score)
PLUCK_WORDS = [85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98]

# ---------------------------------------------------------------- sound design cues (picture-locked)
SFX = [
    (0.00, "hook_hit", {"note": 40}), (0.26, "hook_hit", {"note": 43}), (0.50, "hook_hit", {"note": 47}),
    (0.21, "brush", {"dur": 0.24, "dir": 1}), (0.45, "brush", {"dur": 0.24, "dir": 1}),
    (0.30, "reverse_swell", {"until": bar(1), "note": 64}),
    (0.75, "brush", {"dur": 0.85, "dir": -1, "bright": 0.8}),
    (w(11) - 0.04, "brush", {"dur": 0.35, "dir": 1}),
    (4.45, "brush", {"dur": 0.3, "dir": 1, "bright": 1.2}),
    (w(17), "logo", {"short": True}), (w(17), "impact_s", {}), (6.08, "brush", {"dur": 0.32, "dir": 1}),
    (w(19), "bell", {"note": 76}), (w(21), "bell", {"note": 83}),
    (w(29) - 0.04, "brush", {"dur": 0.35, "dir": 1}),
    (12.90, "brush", {"dur": 0.4, "dir": 1, "bright": 1.1}),
    (w(53) - 0.04, "brush", {"dur": 0.32, "dir": 1}),
    (24.86, "brush", {"dur": 0.28, "dir": 1, "bright": 1.3}),
    (TAPE_STOP, "tape_stop", {}), (w(69) - 0.05, "brush_big", {"dur": 0.5}),
    (27.68, "impact", {}), (27.68, "logo", {"short": False}), (27.70, "brush", {"dur": 0.6, "dir": -1, "bright": 0.8}),
    (30.30, "brush", {"dur": 0.4, "dir": 1}),
    (31.30, "reverse_swell", {"until": bar(17), "note": 71}),
    (w(87), "shimmer", {}), (w(87) - 0.05, "brush", {"dur": 0.7, "dir": 1, "bright": 0.9}),
    (w(101) - 0.04, "brush", {"dur": 0.35, "dir": 1}), (36.10, "brush", {"dur": 0.35, "dir": 1, "bright": 1.2}),
    (bar(20), "impact_s", {}), (w(105) - 0.04, "brush", {"dur": 0.4, "dir": 1}),
    (w(112) - 0.05, "brush", {"dur": 0.4, "dir": 1}),
    (FINAL_STOP, "reverse_swell", {"until": TOP, "note": 76}),
    (TOP, "impact", {}), (TOP, "logo", {"short": False, "final": True}), (TOP - 0.04, "brush_big", {"dur": 0.3}),
    (43.55, "air", {"dur": 0.8}), (43.6, "bell", {"note": 76}),
]
