"""Edit decision list for the ICON testimonial reel: one timeline shared by score and picture.

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
SCENES = [
    (0.00, 0.57, "hook", "cut"),
    (0.57, 4.36, "intro", "iris"),
    (4.36, 4.84, "thank", "slam"),
    (4.84, 6.04, "two", "whip_left"),
    (6.04, 6.46, "icon1", "flash"),
    (6.46, 9.78, "names", "slide_up"),
    (9.78, 11.45, "helped", "zoom"),
    (11.45, 14.62, "course", "whip_right"),
    (14.62, 16.59, "level", "glitch"),
    (16.59, 18.78, "ifyou", "zoom"),
    (18.78, 20.16, "search", "whip_left"),
    (20.16, 22.78, "tools", "slide_up"),
    (22.78, 25.22, "globe", "zoom"),
    (25.22, 27.68, "stop", "cut_black"),
    (27.68, 29.17, "drop1", "flash"),
    (29.17, 31.47, "contractors", "whip_right"),
    (31.47, 35.37, "thanks", "dissolve"),
    (35.37, 37.29, "quality", "zoom"),
    (37.29, 40.65, "final", "flash"),
    (40.65, 41.14, "pause", "cut_black"),
    (41.14, 42.40, "top", "flash"),
    (42.40, DURATION, "logo", "iris"),
]

# words that get on-screen headline treatment do not need the running caption
NO_CAPTION_SCENES = {"hook", "icon1", "top", "logo", "stop", "drop1"}

# ---------------------------------------------------------------- sound design cues
# (time, kind, params) — rendered by score.py; picture uses the same times
SFX = [
    (0.00, "hook_hit", {"note": 40}), (0.26, "hook_hit", {"note": 43}), (0.50, "hook_hit", {"note": 47}),
    (0.30, "reverse_swell", {"until": bar(1), "note": 64}),
    (w(3), "swish", {"dir": 1}), (w(6), "swish", {"dir": -1}),
    (w(11), "pop", {"note": 83}),
    (w(12), "whoosh", {"dir": 1, "dur": 0.32}), (w(12), "thud", {}),
    (4.84, "whoosh", {"dir": -1, "dur": 0.3}), (w(14), "stamp", {}),
    (w(17), "logo", {"short": True}), (w(17), "impact_s", {}),
    (6.46, "whoosh", {"dir": 1, "dur": 0.35}),
    (w(19), "card", {"note": 76}), (w(21), "card", {"note": 83}),
    (9.78, "zoom_whoosh", {}), (w(29), "thud", {}), (w(29) + 0.05, "marker", {}),
    (11.45, "whoosh", {"dir": 1, "dur": 0.3}),
    (bar(7, 0), "tick", {"note": 88}), (bar(7, 1), "tick", {"note": 91}), (bar(7, 2), "tick", {"note": 95}),
    (w(38), "ding", {"note": 88}),
    (14.62, "glitch", {"seed": 3}),
    (w(43), "thud", {}),
    (16.59, "zoom_whoosh", {}),
    (18.78, "whoosh", {"dir": -1, "dur": 0.28}), ("typing", 18.80, 20.10),
    (20.10, "click", {}), (20.16, "whoosh", {"dir": 1, "dur": 0.3}),
    (22.78, "zoom_whoosh", {}), (24.80, "pin", {}),
    (TAPE_STOP, "tape_stop", {}), (w(69), "stamp_big", {}),
    (27.68, "impact", {}), (27.68, "logo", {"short": False}),
    (29.17, "whoosh", {"dir": 1, "dur": 0.3}), (w(83), "stamp", {}),
    (31.30, "reverse_swell", {"until": bar(17), "note": 71}),
    (w(87), "shimmer", {}),
    (35.37, "zoom_whoosh", {}), (w(100), "stamp", {}), (w(101), "thud", {}),
    (bar(20), "impact_s", {}),
    (w(112), "whoosh", {"dir": 1, "dur": 0.35}),
    (FINAL_STOP, "reverse_swell", {"until": TOP, "note": 76}),
    (TOP, "impact", {}), (TOP, "logo", {"short": False, "final": True}),
    (42.40, "whoosh", {"dir": -1, "dur": 0.4}), (43.70, "pop", {"note": 76}),
]


def caption_groups(max_chars=22, max_words=4):
    """Split speech into short caption phrases (break on punctuation, pauses, length)."""
    groups, cur = [], []
    for i, wd in enumerate(WORDS):
        cur.append(i)
        text = " ".join(WORDS[j]["w"] for j in cur)
        nxt = WORDS[i + 1] if i + 1 < len(WORDS) else None
        brk = (wd["w"][-1] in ",.!?" or len(cur) >= max_words or len(text) >= max_chars
               or (nxt and nxt["s"] - wd["e"] > 0.25))
        if nxt is not None and not brk:
            projected = len(text) + 1 + len(nxt["w"])
            brk = projected > max_chars + 4
        if brk or nxt is None:
            groups.append(cur)
            cur = []
    return groups


if __name__ == "__main__":
    for n in range(1, 23):
        print(n, round(bar(n), 3), PROG.get(n))
    for g in caption_groups():
        print(round(WORDS[g[0]]["s"], 2), " ".join(WORDS[j]["w"] for j in g))

# ---------------------------------------------------------------- shared micro-timings
SEARCH_TEXT = "классный подрядчик по трафику"
TYPE_START, TYPE_END = 18.84, 20.04


def typing_times():
    import numpy as np
    rng = np.random.default_rng(11)
    n = len(SEARCH_TEXT)
    base = np.linspace(0, 1, n) ** 0.92
    jit = rng.uniform(-0.25, 0.25, n) / n
    jit[0] = 0
    t = TYPE_START + (TYPE_END - TYPE_START) * np.clip(base + jit, 0, 1)
    t = np.maximum.accumulate(t)
    return [float(x) for x in t]


# breakdown: stressed words that "play" the pluck (speech drives the melody)
PLUCK_WORDS = [85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98]
