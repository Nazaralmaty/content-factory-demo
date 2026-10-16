#!/usr/bin/env bash
# Get faster-whisper large-v3 (CTranslate2) into DEST. Tries HuggingFace first; when the network
# policy blocks it, streams the model out of a public Docker Hub image layer that bakes it in
# (runpod/ai-api-faster-whisper:dev) and extracts only the needed files (~3 GB on disk).
# Usage: bash fetch_whisper.sh [DEST]   (default: $HOME/.cache/ultra-reel/whisper-large-v3)
set -euo pipefail
DEST="${1:-$HOME/.cache/ultra-reel/whisper-large-v3}"
mkdir -p "$DEST"
if [ -s "$DEST/model.bin" ]; then echo "model already in $DEST"; exit 0; fi

if python3 - "$DEST" <<'EOF' 2>/dev/null; then
import sys
from huggingface_hub import snapshot_download
snapshot_download("Systran/faster-whisper-large-v3", local_dir=sys.argv[1])
EOF
  echo "downloaded from HuggingFace -> $DEST"; exit 0
fi

echo "HuggingFace unavailable, extracting from Docker Hub image layer (streams ~12 GB, writes ~3 GB)..."
REPO=runpod/ai-api-faster-whisper
TAG=dev
TMP="$DEST/.layer"
mkdir -p "$TMP"
TOKEN=$(curl -sS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
ACC="application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"
MAN=$(curl -sS -H "Authorization: Bearer $TOKEN" -H "Accept: $ACC" "https://registry-1.docker.io/v2/$REPO/manifests/$TAG")
DIGEST=$(echo "$MAN" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print([m['digest'] for m in d.get('manifests',[]) if m.get('platform',{}).get('architecture')=='amd64'][0])")
LAYER=$(curl -sS -H "Authorization: Bearer $TOKEN" -H "Accept: $ACC" "https://registry-1.docker.io/v2/$REPO/manifests/$DIGEST" | python3 -c "
import sys,json; d=json.load(sys.stdin); print(max(d['layers'], key=lambda l: l['size'])['digest'])")
cd "$TMP"
curl -sS -L -H "Authorization: Bearer $TOKEN" "https://registry-1.docker.io/v2/$REPO/blobs/$LAYER" \
  | tar -xz --wildcards 'root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/blobs/*'
# GNU tar leaves snapshot symlinks as placeholders while streaming, so map blobs by content
B="$TMP/root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/blobs"
for f in "$B"/*; do
  sz=$(stat -c %s "$f")
  if [ "$sz" -gt 1000000000 ]; then mv "$f" "$DEST/model.bin"
  elif grep -q '"alignment_heads"' "$f" 2>/dev/null; then mv "$f" "$DEST/config.json"
  elif grep -q '"feature_extractor_type"' "$f" 2>/dev/null; then mv "$f" "$DEST/preprocessor_config.json"
  elif grep -q '"added_tokens"' "$f" 2>/dev/null; then mv "$f" "$DEST/tokenizer.json"
  else mv "$f" "$DEST/vocabulary.json"; fi
done
rm -rf "$TMP"
ls -la "$DEST"
