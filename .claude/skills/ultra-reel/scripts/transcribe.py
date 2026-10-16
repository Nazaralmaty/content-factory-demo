"""Word-level transcription with faster-whisper. Usage: transcribe.py audio.wav model_dir out.json [lang]"""
import json
import sys
import time

import soundfile as sf
from faster_whisper import WhisperModel

audio, model_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
lang = sys.argv[4] if len(sys.argv) > 4 else None

t0 = time.time()
model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=4)
pcm, sr = sf.read(audio, dtype='float32')
assert sr == 16000, 'expects 16 kHz mono wav'
segments, info = model.transcribe(
    pcm, language=lang, beam_size=5, word_timestamps=True, vad_filter=False,
    condition_on_previous_text=True, temperature=0.0,
)
words, segs = [], []
for s in segments:
    segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
    for w in s.words or []:
        words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.probability, 3)})
json.dump({"language": info.language, "lang_prob": info.language_probability, "segments": segs, "words": words},
          open(out, "w"), ensure_ascii=False, indent=1)
print(f"lang={info.language} p={info.language_probability:.2f} words={len(words)} took {time.time()-t0:.0f}s")
for s in segs:
    print(f"[{s['start']:6.2f}-{s['end']:6.2f}] {s['text']}")
