"""Rhythm gate for a plan.py (abat / motion-canvas-pro laws, measured on the plan, not by eye).

python3 rhythm_qa.py <work_dir>

Prints: picture events per minute, the longest stretches without a new event while someone speaks,
how much of the speech the face is big (not a corner card), and how often the through-object returns.
Exit code 1 if a hard law fails.
"""
import importlib.util
import os
import sys

work = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.environ.get("ULTRA_ENGINE", os.path.join(os.path.dirname(__file__))))
spec = importlib.util.spec_from_file_location("plan", os.path.join(work, "plan.py"))
PL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PL)

ev = []
for it in PL.TEXT:
    if it[0] < PL.DURATION:
        ev.append((max(0.0, it[0]), "text  " + it[3]))
for st in PL.STROKES:
    ev.append((st[0], "stroke " + st[3]))
for p in getattr(PL, "PROPS", []):
    ev.append((p["t"], "prop  " + p["key"]))
    for ts, k in p["states"]:
        ev.append((ts, "state " + k))
    for h in p["hops"]:
        ev.append((h[0], "hop   " + p["key"]))
for s in PL.SCENES:
    ev.append((s[0], "scene " + s[1]))
for w_ in PL.WASHES:
    ev.append((w_[0], "wash  " + w_[4]))
ev.sort()
times = sorted({round(t, 2) for t, _ in ev})

s0, s1 = PL.WORDS[0]["s"], PL.WORDS[-1]["e"]
inside = [t for t in times if s0 - 0.5 <= t <= s1]
per_min = len(inside) / ((s1 - s0) / 60)
gaps = []
prev = s0
for t in inside + [s1]:
    if t - prev > 1.5:
        gaps.append((prev, t))
    prev = max(prev, t)

# face share: speech time while the speaker node is not a corner card
def layout_at(t):
    lay = PL.SCENES[0][1]
    for s, l_, _ in PL.SCENES:
        if t >= s:
            lay = l_
    return lay


speech = sum(w["e"] - w["s"] for w in PL.WORDS)
face = sum(w["e"] - w["s"] for w in PL.WORDS if layout_at((w["s"] + w["e"]) / 2) not in ("card", "hidden"))
through = getattr(PL, "THROUGH", ("duo", "fig_"))
returns = sorted({round(p["t"], 1) for p in getattr(PL, "PROPS", []) if p["key"].startswith(tuple(through))})
clusters = []
for t in returns:
    if not clusters or t - clusters[-1][-1] > 2.5:
        clusters.append([t])
    else:
        clusters[-1].append(t)

print(f"picture events during speech: {len(inside)}  ->  {per_min:.0f}/min   (abat 120-150, kinetic >= 120)")
print("stretches > 1.5 s without a new event:", ", ".join(f"{a:.2f}-{b:.2f}" for a, b in gaps) or "none")
print(f"face share of speech: {100 * face / speech:.0f}%   (law: >= 50%)")
print(f"through-object {through}: {len(clusters)} appearances at " + ", ".join(f"{c[0]:.1f}" for c in clusters)
      + "   (law: returns 2-4 times in new states)")
bad = per_min < 100 or face / speech < 0.5 or any(b - a > 3.0 for a, b in gaps) or (returns and len(clusters) < 3)
print("FAIL" if bad else "OK")
sys.exit(1 if bad else 0)
