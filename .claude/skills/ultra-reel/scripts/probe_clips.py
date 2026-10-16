"""Re-transcribe short clips to verify doubtful word timings (names, brand, drop word, last word).

Usage: probe_clips.py voice16k.wav MODEL_DIR '[[t0, t1], ...]'   -> word[start-end] per clip on the WAV timeline
"""
import sys, json, soundfile as sf, numpy as np
from faster_whisper import WhisperModel
pcm, sr = sf.read(sys.argv[1], dtype='float32')
model = WhisperModel(sys.argv[2], device="cpu", compute_type="int8", cpu_threads=4)
clips = json.loads(sys.argv[3])
for a, b in clips:
    seg = pcm[int(a*sr):int(b*sr)]
    seg = np.concatenate([np.zeros(int(0.3*sr), np.float32), seg, np.zeros(int(0.3*sr), np.float32)])
    segments, info = model.transcribe(seg, language="ru", beam_size=5, word_timestamps=True, vad_filter=False, temperature=0.0)
    out = []
    for s in segments:
        for w in s.words:
            out.append(f"{w.word.strip()}[{a + w.start - 0.3:.2f}-{a + w.end - 0.3:.2f}]")
    print(f"{a:5.2f}-{b:5.2f}: {' '.join(out)}", flush=True)
