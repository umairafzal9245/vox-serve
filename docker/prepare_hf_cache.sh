#!/usr/bin/env bash
# Prepare a slim Hugging Face cache (~model+tokenizer only) for Docker builds.
# Usage:
#   ./docker/prepare_hf_cache.sh
#   export HF_CACHE_HOST="$(pwd)/docker/hf-cache-build"
#   docker compose build
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${HF_SOURCE_CACHE:-$HOME/.cache/huggingface}"
DST="${1:-$ROOT/docker/hf-cache-build}"
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
TOKENIZER="Qwen/Qwen3-TTS-Tokenizer-12Hz"

hub_name() { echo "models--${1//\//--}"; }

mkdir -p "$DST/hub"
for repo in "$MODEL" "$TOKENIZER"; do
  name="$(hub_name "$repo")"
  if [[ -d "$SRC/hub/$name" ]]; then
    echo "[prepare] rsync $name from $SRC"
    mkdir -p "$DST/hub"
    rsync -a --delete "$SRC/hub/$name" "$DST/hub/"
  else
    echo "[prepare] $name not in $SRC — will download during docker build"
  fi
done

du -sh "$DST" 2>/dev/null || true
echo "[prepare] HF_CACHE_HOST=$DST"
