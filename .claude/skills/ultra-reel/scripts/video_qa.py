"""Final-file QA: stream sanity, loudness/true peak of the delivered audio, frame grid every 0.5 s.

Usage: video_qa.py reel.mp4 OUT_DIR [step_seconds]
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

src, out = sys.argv[1], sys.argv[2]
step = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
os.makedirs(out, exist_ok=True)
info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                  "stream=codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels,nb_frames"
                                  ":format=duration,bit_rate", "-of", "json", src], capture_output=True, text=True).stdout)
print(json.dumps(info, indent=1))
ln = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af", "ebur128=peak=true", "-f", "null", "-"],
                    capture_output=True, text=True).stderr
summary = ln[ln.rfind("Summary:"):]
print(summary.strip())

dur = float(info["format"]["duration"])
times = np.arange(0, dur, step)
tw, th = 180, 320
cols = 12
rows = int(np.ceil(len(times) / cols))
sheet = Image.new("RGB", (cols * tw, rows * (th + 16)), (30, 30, 30))
d = ImageDraw.Draw(sheet)
for i, t in enumerate(times):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1", "-vf",
                          f"scale={tw}:{th}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    if len(raw) != tw * th * 3:
        continue
    im = Image.fromarray(np.frombuffer(raw, np.uint8).reshape(th, tw, 3))
    x, y = (i % cols) * tw, (i // cols) * (th + 16)
    sheet.paste(im, (x, y + 16))
    d.text((x + 4, y + 2), f"{t:.1f}s", fill=(255, 255, 255))
sheet.save(os.path.join(out, "frames_grid.jpg"), quality=88)
print("grid:", os.path.join(out, "frames_grid.jpg"))
