#!/usr/bin/env bash
# One-time environment for ULTRA REEL: python deps, libEGL for skia, Cyrillic display fonts.
# Usage: bash setup.sh [fonts_dir]   (default: $HOME/.cache/ultra-reel/fonts)
set -euo pipefail
FONTS="${1:-${REEL_FONTS:-$HOME/.cache/ultra-reel/fonts}}"
HERE="$(cd "$(dirname "$0")" && pwd)"

python3 -m pip install -q faster-whisper numba skia-python scipy soundfile pyloudnorm matplotlib pillow numpy 2>&1 | grep -v WARNING || true

# skia-python needs libEGL even for CPU raster surfaces
if ! python3 -c "import skia" 2>/dev/null; then
  (apt-get update -q >/dev/null 2>&1 && apt-get install -y -q libegl1 >/dev/null 2>&1) || true
fi
python3 -c "import skia, faster_whisper, numba, pyloudnorm; print('python deps ok, skia', skia.__version__)"
command -v ffmpeg >/dev/null || { echo "ffmpeg is required"; exit 1; }

mkdir -p "$FONTS"
python3 "$HERE/get_fonts.py" "$FONTS" \
  "Oswald:wght@600;700" "Dela Gothic One" "Unbounded:wght@900" \
  "Playfair Display:ital,wght@1,700" "JetBrains Mono:wght@500;800" "Montserrat:wght@500;700;900" \
  "Cormorant Garamond:ital,wght@0,500;0,600;1,600" "Comforter Brush"
echo "fonts in $FONTS  (export REEL_FONTS=$FONTS)"
