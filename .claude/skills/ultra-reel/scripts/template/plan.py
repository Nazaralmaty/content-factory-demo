"""Painterly cut ("холст") with ink-and-wash illustrations: one timeline shared by score and picture.

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


# ---------------------------------------------------------------- picture: one canvas, speaker node + painted worlds
# Speaker layouts (cx, cy, r, zoom_on_face). The node glides between them (Motion Canvas style); in a world it
# shrinks into a live card in the corner and the illustration takes the middle (abat: speaker as a live card).
L = {
    "hidden": (540, 900, 0, 1.0), "intro": (540, 860, 340, 1.0), "up": (540, 760, 320, 1.0),
    "mid": (540, 720, 300, 1.0), "center": (540, 820, 360, 1.0), "big": (540, 860, 440, 1.25),
    "card": (850, 330, 135, 1.3), "final": (540, 540, 220, 1.0), "logo": (540, 760, 250, 1.0),
}
# (start, layout, mood)
SCENES = [
    (0.00, "hidden", "canvas"), (0.85, "intro", "canvas"), (4.36, "up", "canvas"),
    (4.84, "card", "world"),     # двух маркетологов из компании ICON — Абубакир и Еркебулан
    (9.78, "intro", "canvas"),   # за то, что они мне очень сильно помогли
    (11.81, "card", "world"),    # в создании курса по трафику … академии ; они очень высокого уровня профессионалы
    (16.59, "big", "canvas"), (18.78, "mid", "canvas"),
    (20.16, "card", "world"),    # который умеет работать со всеми этими зарубежными инструментами … в СНГ
    (25.22, "center", "dusk"), (27.68, "center", "terra"), (29.17, "mid", "canvas"), (31.47, "center", "ochre"),
    (35.37, "card", "world"),    # высококачественная работа
    (36.62, "mid", "canvas"),
    (37.78, "card", "world"),    # если вам нужны классные подрядчики, идите к ним
    (40.60, "final", "canvas"), (42.40, "logo", "canvas"),
]
GLIDE = 0.62  # seconds for a layout change, centred on the boundary
# worlds: the illustration layer pushes in slowly 1.00 -> 1.04 (abat: the camera always pushes)
WORLDS = [(4.84, 9.78), (11.81, 14.62), (14.62, 16.59), (20.16, 25.36), (35.37, 36.62), (37.78, 40.60)]

# Big washes: (t_in, t_full, t_out_start, t_out_end, color, alpha, seed, w, h, centre)
WASHES = [
    (0.44, 0.70, 0.78, 1.05, "terra", 0.75, 20, 760, 460, (540, 1240)),
    (25.20, 25.95, 27.55, 28.10, "dusk", 0.93, 21, 1500, 2300, "speaker"),
    (27.64, 28.10, 29.00, 29.70, "terra", 0.86, 22, 1450, 2200, "speaker"),
    (31.45, 32.30, 35.10, 35.80, "ochre", 0.42, 23, 1300, 1900, "speaker"),
    (37.24, 37.70, 40.45, 40.75, "terra", 0.82, 24, 1500, 2300, (540, 960)),
    (41.10, 41.42, 42.20, 42.75, "terra", 0.88, 25, 1500, 2300, (540, 960)),
]

# Text items: (t_in, t_out, kind, text, font, size, x, y_baseline, color)
#   kind "rise" = serif up from below, blur->sharp;  "brush" = written on by a brush;
#   "label" = a world's script label, top-left, left-aligned (abat: a world opens with a label at the top)
#   color: ink / terra / slate / cream / grey / auto (ink on light, cream on dark washes)
TEXT = [
    (-0.30, 1.00, "brush", "Они", "brush", 330, 540, 770, "ink"),
    (0.20, 1.00, "brush", "точно", "brush", 330, 540, 1020, "ink"),
    (0.44, 1.00, "brush", "топ.", "brush", 420, 540, 1320, "terra"),
    (1.05, 3.15, "rise", "Антон Петраченков", "serif", 78, 540, 1330, "ink"),
    (w(6) - 0.05, 3.15, "rise", "ConvertMonster", "serif_i", 62, 540, 1410, "terra"),
    (w(11) - 0.04, 4.82, "brush", "лично", "brush", 217, 540, 440, "terra"),
    (4.33, 4.82, "rise", "поблагодарить", "serif", 104, 540, 1330, "ink"),
    # world 1 — the duo
    (w(13) - 0.04, 6.85, "rise", "двух маркетологов", "serif", 92, 540, 1350, "ink"),
    (w(15) - 0.03, 6.85, "rise", "из компании", "serif_i", 66, 540, 1440, "ink"),
    (w(19) - 0.04, 9.74, "rise", "Абубакир", "serif_i", 80, 330, 1335, "terra"),
    (w(20) - 0.03, 9.74, "rise", "и", "serif_i", 54, 540, 1335, "grey"),
    (w(21) - 0.04, 9.74, "rise", "Еркебулан", "serif_i", 80, 755, 1335, "slate"),
    # face
    (w(27) - 0.04, 11.76, "rise", "очень сильно", "serif_i", 73, 540, 1300, "ink"),
    (w(29) - 0.04, 11.76, "brush", "помогли", "brush", 275, 540, 1470, "terra"),
    # world 2 — the course
    (11.70, 14.58, "label", "в создании", "serif_i", 70, 80, 300, "ink"),
    (w(32) - 0.04, 14.58, "rise", "курса по трафику", "serif", 100, 540, 1215, "ink"),
    (w(35) - 0.04, 14.58, "rise", "в рамках нашей академии", "serif_i", 62, 540, 1305, "ink"),
    # world 3 — the stairs
    (w(40) - 0.04, 16.55, "label", "очень высокого уровня", "serif_i", 70, 80, 300, "ink"),
    (w(43) - 0.04, 16.55, "rise", "профессионалы", "serif", 118, 540, 1385, "ink"),
    # face — the search
    (w(45) - 0.04, 18.76, "rise", "если", "serif", 92, -1, 330, "ink"),
    (w(46) - 0.04, 18.76, "rise", "вам", "serif", 92, -2, 330, "ink"),
    (w(47) - 0.04, 18.76, "rise", "нужен", "serif", 92, -3, 330, "terra"),
    (w(51) - 0.04, 20.12, "rise", "классный подрядчик", "serif", 100, 540, 1210, "ink"),
    (w(53) - 0.04, 20.12, "brush", "по трафику", "brush", 217, 540, 1385, "terra"),
    # world 4 — the tools
    (20.12, 25.28, "label", "который умеет", "serif_i", 70, 80, 300, "ink"),
    (w(57) - 0.04, 22.72, "rise", "работать", "serif", 110, 540, 1425, "ink"),
    (w(58) - 0.04, 22.72, "rise", "со всеми этими", "serif_i", 64, 540, 1510, "ink"),
    (w(61) - 0.04, 25.28, "rise", "зарубежными инструментами", "serif", 84, 540, 1425, "ink"),
    (w(63) - 0.04, 25.28, "rise", "которые работают в СНГ", "serif_i", 64, 540, 1510, "ink"),
    # the stop and the drop
    (w(67) - 0.03, 27.62, "rise", "я точно", "serif_i", 70, 540, 1290, "cream"),
    (w(69) - 0.05, 27.62, "brush", "рекомендую", "brush", 275, 540, 1465, "cream"),
    (w(71) - 0.04, 27.62, "rise", "обращаться в агентство", "serif_i", 59, 540, 330, "cream"),
    (w(79) - 0.04, 31.45, "rise", "они у меня", "serif_i", 71, 540, 1170, "ink"),
    (w(83) - 0.04, 31.45, "rise", "подрядчики", "serif", 130, 540, 1320, "ink"),
    # thanks
    (w(86) - 0.04, 35.30, "rise", "ребят,", "serif_i", 71, 540, 330, "ink"),
    (w(87) - 0.05, 33.05, "brush", "спасибо", "brush", 333, 540, 1390, "terra"),
    (w(89) - 0.04, 33.05, "rise", "за ваш труд", "serif_i", 69, 540, 1490, "ink"),
    (w(92) - 0.04, 34.58, "rise", "я вам очень сильно благодарен", "serif", 66, 540, 1430, "ink"),
    (w(97) - 0.04, 35.33, "rise", "это реально", "serif_i", 71, 540, 1430, "ink"),
    # world 5 — the sealed work
    (w(100) - 0.04, 36.58, "rise", "высококачественная", "serif", 100, 540, 1235, "ink"),
    (w(101) - 0.04, 36.58, "brush", "работа", "brush", 290, 540, 1455, "terra"),
    # face
    (w(102) - 0.04, 37.74, "rise", "желаю вам всего", "serif_i", 73, 540, 1210, "ink"),
    (w(105) - 0.04, 37.74, "brush", "классного", "brush", 290, 540, 1390, "ink"),
    # world 6 — the way to them (on terracotta)
    (w(107) - 0.04, 40.58, "label", "если вам нужны", "serif_i", 70, 80, 300, "cream"),
    (w(110) - 0.04, 39.88, "rise", "классные подрядчики", "serif", 92, 540, 1430, "cream"),
    (w(112) - 0.05, 40.58, "brush", "идите к ним", "brush", 240, 540, 1440, "cream"),
    # final
    (w(115) - 0.04, 41.08, "rise", "они точно", "serif_i", 80, 540, 1330, "ink"),
    (TOP - 0.04, 42.38, "brush", "топ.", "brush", 380, 540, 1450, "cream"),
    (42.55, 99, "rise", "ICON", "serif_wide", 130, 540, 1180, "ink"),
    (42.75, 99, "rise", "подрядчик по трафику", "serif_i", 62, 540, 1265, "terra"),
    (42.95, 99, "rise", "отзыв · Антон Петраченков, ConvertMonster", "serif_small", 34, 540, 1370, "grey"),
]

# Brush strokes: (t_in, draw_dur, t_out, kind, params, color)
#   enso/enso2 = ring round the speaker; under = hand underline of a TEXT item; ring = hand ellipse round a TEXT item;
#   arrow = a brushed arrow along points (abat: a hand mark is red; on terracotta it turns cream)
STROKES = [
    (0.75, 0.85, 99, "enso", {"r_pad": 26, "width": 34, "seed": 5}, "ink"),
    (2.62, 0.32, 3.15, "under", {"text": "ConvertMonster", "seed": 30}, "terra"),
    (4.45, 0.30, 4.82, "under", {"text": "поблагодарить", "seed": 31}, "ochre"),
    (5.34, 0.32, 6.85, "under", {"text": "двух маркетологов", "seed": 32}, "terra"),
    (7.62, 0.30, 9.74, "under", {"text": "Абубакир", "seed": 43}, "terra"),
    (9.12, 0.30, 9.74, "under", {"text": "Еркебулан", "seed": 44}, "slate"),
    (12.90, 0.40, 14.58, "under", {"text": "курса по трафику", "seed": 33}, "ochre"),
    (19.40, 0.36, 20.12, "ring", {"text": "классный подрядчик", "seed": 37}, "terra"),
    (27.70, 0.60, 29.15, "enso2", {"r_pad": 46, "width": 26, "seed": 41}, "cream"),
    (30.30, 0.40, 31.45, "under", {"text": "подрядчики", "seed": 35}, "terra"),
    (36.10, 0.35, 36.58, "under", {"text": "работа", "seed": 36}, "ochre"),
    (39.95, 0.32, 40.58, "arrow", {"pts": [(95, 1215), (220, 1120), (370, 1092), (505, 1125)], "w": 40,
                                   "seed": 38}, "cream"),
    (41.16, 0.45, 42.38, "enso2", {"r_pad": 46, "width": 36, "seed": 42}, "cream"),
]

# Illustrations (abat: one through-object returns in new states — here the duo of marketers; every part lands on
# its own spoken word, sound by material).
#   key: object id (render.OBJECTS), w: px width, (x, y): centre, or feet when anchor="bottom"
#   anim: pop | drop (lands at t) | paint (wipe-in like a wash) | rise ; states: [(t, key)] swaps on words
#   hops: [(t, x, y)] jumps landing at t ; rim: paper cut-out edge ; snd: (kind, params) at the landing
def prop(t, out, key, w, x, y, anim="pop", anchor="center", states=(), hops=(), rim=False, shadow=False, snd=None,
         rot=0.0):
    return dict(t=t, out=out, key=key, w=w, x=x, y=y, anim=anim, anchor=anchor, states=list(states), hops=list(hops),
                rim=rim, shadow=shadow, snd=snd, rot=rot)


# stairs geometry for the hops (stairs drawn at w=760 centred on (540, 1010))
_SK = 760 / 820
_SX, _SY = 540 - (820 * _SK + 24) / 2 + 12, 1010 - (520 * _SK + 24) / 2 + 12
STEP = [(_SX + x * _SK, _SY + y * _SK) for x, y in ((120, 400), (320, 300), (520, 200), (710, 100))]

PROPS = [
    # face: a heart on "поблагодарить"
    prop(4.40, 4.82, "heart", 110, 540, 1185, rim=True, snd=("wood", {"note": 79})),
    # world 1: the two marketers, the ICON sign drops on "ICON", each one waves on his name
    prop(w(13) + 0.02, 9.72, "fig_terra", 300, 360, 1245, anchor="bottom", shadow=True,
         states=[(w(19), "fig_terra_wave")], snd=("wood", {"note": 64})),
    prop(w(14) + 0.15, 9.72, "fig_slate", 300, 720, 1245, anchor="bottom", shadow=True,
         states=[(w(21), "fig_slate_wave")], snd=("wood", {"note": 71})),
    prop(w(17), 9.72, "sign_icon", 420, 540, 615, anim="drop", snd=("thunk", {"note": 50})),
    # face: the duo's miniature is the icon of the caption cluster
    prop(w(25), 11.74, "duo", 170, 905, 1165, rim=True, snd=("wood", {"note": 76})),
    # world 2: the course — a blank book, the duo at its sides, pages fill on "трафику", the academy cap on top
    prop(11.86, 14.56, "book_blank", 600, 540, 900, anim="paint", states=[(w(34), "book_written")],
         snd=("paper", {})),
    prop(w(32), 14.56, "fig_terra_point", 210, 128, 1105, anchor="bottom", shadow=True, snd=("wood", {"note": 67})),
    prop(w(32) + 0.2, 14.56, "fig_slate_point", 210, 952, 1105, anchor="bottom", shadow=True,
         snd=("wood", {"note": 71})),
    prop(w(38), 14.56, "cap", 210, 778, 700, anim="drop", rot=8, snd=("thunk", {"note": 57})),
    # world 3: stairs; the duo climbs a step per word and reaches the top on "профессионалы"
    prop(14.64, 16.53, "stairs", 760, 540, 1010, anim="paint", snd=("brush", {"dur": 0.45})),
    prop(w(40), 16.53, "duo", 196, STEP[0][0], STEP[0][1], anchor="bottom",
         hops=[(w(41), *STEP[1]), (w(42), *STEP[2]), (w(43), *STEP[3])], snd=("wood", {"note": 76})),
    prop(w(43) + 0.08, 16.53, "sparkle", 110, 930, 720, snd=("bell", {"note": 88})),
    prop(w(43) + 0.24, 16.53, "sparkle", 70, 700, 770),
    # face: a magnifier looks for a contractor and finds the duo in its lens on "подрядчик"
    prop(18.95, 20.10, "magnifier", 190, 900, 960, rim=True, states=[(w(52), "magnifier_duo")],
         snd=("glass", {"note": 91})),
    # world 4: the toolbox — tools come out on the words, the globe on "зарубежными", a pin on "СНГ"
    prop(w(56), 25.36, "toolbox", 460, 540, 1150, anim="drop", shadow=True, snd=("clank", {"note": 45})),
    prop(w(57), 25.36, "wrench", 190, 245, 820, snd=("metal", {"note": 83})),
    prop(w(59), 25.36, "gear", 180, 835, 820, snd=("metal", {"note": 79})),
    prop(w(60), 25.36, "magnet", 165, 205, 1070, snd=("metal", {"note": 88})),
    prop(w(61), 25.36, "globe", 380, 540, 780, anim="paint", snd=("air", {"dur": 0.6})),
    prop(w(62), 25.36, "chart", 180, 880, 1075, snd=("wood", {"note": 84})),
    prop(w(66) - 0.02, 25.36, "pin", 104, 618, 712, anim="drop", anchor="bottom", snd=("thunk", {"note": 62})),
    # the drop: the ICON sign falls under the speaker on "ICON."
    prop(27.68, 29.13, "sign_icon", 470, 540, 1430, anim="drop", rim=True),
    # they stand beside him: "они у меня являются подрядчиками"
    prop(w(79) + 0.02, 31.42, "fig_terra", 200, 122, 1050, anchor="bottom", rim=True,
         states=[(w(83), "fig_terra_wave_out")], snd=("wood", {"note": 76})),
    prop(w(80), 31.42, "fig_slate", 200, 955, 1050, anchor="bottom", rim=True,
         states=[(w(83) + 0.12, "fig_slate_wave_out")], snd=("wood", {"note": 79})),
    # thanks: the duo icon, turning into a heart on "благодарен"
    prop(w(86), 35.30, "duo", 180, 905, 1140, rim=True, states=[(w(96), "heart")], snd=("wood", {"note": 83})),
    # world 5: the finished course, sealed on "работа"
    prop(w(100), 36.56, "book_plain", 340, 540, 880, states=[(w(101), "book_sealed")], snd=("paper", {})),
    # face: sparkles on "классного"
    prop(w(105) + 0.03, 37.72, "sparkle", 100, 880, 1060, snd=("bell", {"note": 88})),
    prop(w(105) + 0.18, 37.72, "sparkle", 70, 205, 1000, snd=("bell", {"note": 91})),
    # world 6: the sign on "нужны", the duo under it on "подрядчики", the arrow on "идите к ним"
    prop(w(109), 40.56, "sign_icon", 380, 700, 690, anim="drop", rim=True, snd=("thunk", {"note": 52})),
    prop(w(111), 40.56, "duo_wave", 330, 700, 1215, anchor="bottom", rim=True, snd=("wood", {"note": 76})),
    # final: the duo on a pedestal rises in the silence before "топ."
    prop(w(115) - 0.02, 42.36, "duo_pedestal", 360, 540, 1010, anim="rise", rim=True, snd=("air", {"dur": 0.45})),
    prop(TOP + 0.02, 42.36, "sparkle", 110, 250, 880),
    prop(TOP + 0.12, 42.36, "sparkle", 80, 845, 840),
    prop(TOP + 0.22, 42.36, "sparkle", 64, 300, 1120),
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
    (2.62, "brush", {"dur": 0.3, "dir": 1, "bright": 1.2}),
    (4.45, "brush", {"dur": 0.3, "dir": 1, "bright": 1.2}),
    (5.34, "brush", {"dur": 0.3, "dir": 1, "bright": 1.1}),
    (w(17), "logo", {"short": True}), (w(17), "impact_s", {}),
    (w(19), "bell", {"note": 76}), (w(21), "bell", {"note": 83}),
    (7.62, "brush", {"dur": 0.3, "dir": 1, "bright": 1.1}), (9.12, "brush", {"dur": 0.3, "dir": -1, "bright": 1.1}),
    (w(29) - 0.04, "brush", {"dur": 0.35, "dir": 1}),
    (w(34), "marker", {}), (12.90, "brush", {"dur": 0.4, "dir": 1, "bright": 1.1}),
    (19.40, "brush", {"dur": 0.36, "dir": -1, "bright": 1.1}),
    (w(53) - 0.04, "brush", {"dur": 0.32, "dir": 1}),
    (TAPE_STOP, "tape_stop", {}), (w(69) - 0.05, "brush_big", {"dur": 0.5}),
    (27.68, "impact", {}), (27.68, "logo", {"short": False}), (27.70, "brush", {"dur": 0.6, "dir": -1, "bright": 0.8}),
    (27.70, "thunk", {"note": 45}),
    (30.30, "brush", {"dur": 0.4, "dir": 1}),
    (31.30, "reverse_swell", {"until": bar(17), "note": 71}),
    (w(87), "shimmer", {}), (w(87) - 0.05, "brush", {"dur": 0.7, "dir": 1, "bright": 0.9}),
    (w(96), "bell", {"note": 88}),
    (w(101) - 0.04, "brush", {"dur": 0.35, "dir": 1}), (w(101), "stamp", {}),
    (36.10, "brush", {"dur": 0.35, "dir": 1, "bright": 1.2}),
    (bar(20), "impact_s", {}), (w(105) - 0.04, "brush", {"dur": 0.4, "dir": 1}),
    (w(112) - 0.05, "brush", {"dur": 0.45, "dir": 1}),
    (FINAL_STOP, "reverse_swell", {"until": TOP, "note": 76}),
    (TOP, "impact", {}), (TOP, "logo", {"short": False, "final": True}), (TOP - 0.04, "brush_big", {"dur": 0.3}),
    (43.55, "air", {"dur": 0.8}), (43.6, "bell", {"note": 76}),
]
# the speaker shrinking into a card / growing back: a soft breath of air under the glide
for _i, (_s, _lay, _m) in enumerate(SCENES[1:], 1):
    _prev = SCENES[_i - 1][1]
    if (_lay == "card") != (_prev == "card") and _s not in (25.22,):
        SFX.append((_s - GLIDE / 2, "air", {"dur": GLIDE + 0.1}))
# every illustration sounds by its material at its landing
for _p in PROPS:
    if _p["snd"]:
        SFX.append((_p["t"], _p["snd"][0], dict(_p["snd"][1])))
