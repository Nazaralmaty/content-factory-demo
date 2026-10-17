"""Contact sheet of every illustration in engine/illo.py on the canvas: python3 illo_sheet.py out.png"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ULTRA_ENGINE", HERE))
import numpy as np, skia
from engine import gfx as G, illo as I, paint as P
F = G.Fonts(os.environ.get("REEL_FONTS", os.path.expanduser("~/.cache/ultra-reel/fonts")), {"serif": "CormorantGaramond-600.ttf", "serif_i": "CormorantGaramond-600i.ttf"})
font = lambda s: F("serif", s)
objs = [
    ("duo", I.duo(420)), ("duo wave", I.duo(420, (("wave", None), (None, "point")))), ("sign", I.sign(420, "ICON", font)),
    ("book open", I.book_open(420)), ("book written", I.book_open(420, True, "Трафик", lambda s: F("serif_i", s))),
    ("book closed", I.book_closed(300, "Курс\nпо трафику", font, seal=True)), ("cap", I.cap(300)),
    ("stairs", I.stairs(420)), ("toolbox", I.toolbox(360)), ("wrench", I.wrench(200)), ("gear", I.gear(200)),
    ("magnet", I.magnet(200)), ("chart", I.chart(220)), ("globe", I.globe(300)), ("pin", I.pin(130)),
    ("heart", I.heart(180)), ("magnifier", I.magnifier(220)), ("magnifier duo", I.magnifier_duo(220)), ("duo pedestal", I.duo_on_pedestal(380)),
    ("foot", I.footstep(100)), ("sparkle", I.sparkle(140)), ("icon duo rim", I.with_rim(I.duo(150), 6)), ("icon heart rim", I.with_rim(I.heart(120), 6)), ("one", I.figure_one(200, "slate", hair="side", arm_r="point")),
]
cols, cw, ch = 4, 480, 520
rows = (len(objs) + cols - 1) // cols
surf = G.surface(cols * cw, rows * ch)
c = surf.getCanvas()
c.drawImage(P.canvas_texture(rows * ch, cols * cw, (241, 234, 223), seed=7), 0, 0)
for k, (name, img) in enumerate(objs):
    x, y = (k % cols) * cw, (k // cols) * ch
    c.drawImage(img, x + (cw - img.width()) / 2, y + 30 + (ch - 80 - img.height()) / 2)
    G.draw_text(c, name, x + cw / 2, y + ch - 20, F("serif_i", 34), skia.Color4f(0.3, 0.3, 0.3, 1), align="center")
from PIL import Image
Image.fromarray(np.asarray(G.snapshot(surf))[..., :3]).save(sys.argv[1])
print("ok")
