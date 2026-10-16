"""Final cue sheet: aligned words + manual fixes + AV offset -> words.json on the VIDEO timeline.

Usage: make_words.py words_aligned.json media.json words.json ['{"17": [6.02, 6.44, "ICON,"], "22": null}']
Overrides are keyed by word index (as printed by align.py): [start, end] or [start, end, text];
null drops the word (e.g. a name the ASR split in two after you merged it into the previous one).
"""
import json
import sys

aligned, media, out = sys.argv[1], sys.argv[2], sys.argv[3]
over = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
words = json.load(open(aligned))["words"]
av = json.load(open(media)).get("av_offset", 0.0)
for k, v in over.items():
    i = int(k)
    if v is None:
        words[i]["w"] = ""
        continue
    words[i]["s"], words[i]["e"] = float(v[0]), float(v[1])
    if len(v) > 2:
        words[i]["w"] = v[2]
words = [w for w in words if w["w"]]
for i in range(1, len(words)):  # monotonic, non-overlapping
    if words[i]["s"] < words[i - 1]["s"] + 0.03:
        words[i]["s"] = round(words[i - 1]["s"] + 0.03, 3)
    if words[i - 1]["e"] > words[i]["s"]:
        words[i - 1]["e"] = words[i]["s"]
res = {"av_offset": av, "words": [{"w": w["w"], "s": round(w["s"] + av, 3), "e": round(w["e"] + av, 3)} for w in words]}
json.dump(res, open(out, "w"), ensure_ascii=False, indent=0)
print(" | ".join(f"{i}:{w['w']}@{w['s']:.2f}" for i, w in enumerate(res["words"])))
