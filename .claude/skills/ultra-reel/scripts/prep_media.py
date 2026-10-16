"""Prepare a speaker source: audio for ASR/mix, graded frames as .npy, AV offset, circle + face hints.

Usage: prep_media.py SOURCE.mp4 WORK_DIR [--fps 30]
Writes: voice16k.wav, voice48k.wav, speaker.npy (N x H x W x 3 uint8), media.json
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

src, work = sys.argv[1], sys.argv[2]
fps = int(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 30
os.makedirs(work, exist_ok=True)


def probe(sel, entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", entries, "-of", "json", src],
                         capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]


v = probe("v:0", "stream=width,height,r_frame_rate,start_time,duration")
a = probe("a:0", "stream=start_time,sample_rate")
w, h = int(v["width"]), int(v["height"])
av_offset = float(a.get("start_time", 0) or 0) - float(v.get("start_time", 0) or 0)

for sr, name in ((16000, "voice16k.wav"), (48000, "voice48k.wav")):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le",
                    os.path.join(work, name)], check=True)

raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                     capture_output=True, check=True).stdout
frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)

# gentle S-curve + warm grade + light unsharp (phone footage is soft once upscaled 2x)
x = np.linspace(0, 1, 256)
s = x + 0.06 * np.sin(2 * np.pi * x) * (1 - x) * x * 4
luts = [np.clip((s * k + o) * 255, 0, 255).astype(np.uint8) for k, o in ((1.03, 0.005), (1.0, 0.0), (0.95, 0.01))]
out = np.empty_like(frames)
for i, f in enumerate(frames):
    g = np.stack([luts[c][f[..., c]] for c in range(3)], axis=-1)
    out[i] = np.asarray(Image.fromarray(g).filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2)))
np.save(os.path.join(work, "speaker.npy"), out)

# round "video note" detection: near-white corners that never change
mean = frames[:: max(1, len(frames) // 50)].astype(np.float32).mean(axis=0).mean(axis=2)
corner = np.mean([mean[:8, :8].mean(), mean[:8, -8:].mean(), mean[-8:, :8].mean(), mean[-8:, -8:].mean()])
is_circle = bool(corner > 240 and abs(w - h) < 4)
info = {"width": w, "height": h, "fps": fps, "frames": int(len(frames)), "av_offset": round(av_offset, 4),
        "round_video_note": is_circle, "face_hint": [w // 2 - 4, int(h * 0.42)]}
json.dump(info, open(os.path.join(work, "media.json"), "w"), indent=1)
print(json.dumps(info))
