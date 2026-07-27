#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${VOX_METRICS_DIR:-/var/log/vox-serve}" "${HF_HOME:-/data/hf-cache}" /data/voices

MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
PORT="${PORT:-2200}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
MAX_NUM_PAGES="${MAX_NUM_PAGES:-2048}"
DETOKENIZE_INTERVAL="${DETOKENIZE_INTERVAL:-5}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-online}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
OUTPUT_DIR="${VOX_OUTPUT_DIR:-/data/voices}"

ARGS=(
  -m vox_serve.launch
  --model "${MODEL}"
  --port "${PORT}"
  --max-batch-size "${MAX_BATCH_SIZE}"
  --max-num-pages "${MAX_NUM_PAGES}"
  --detokenize-interval "${DETOKENIZE_INTERVAL}"
  --scheduler-type "${SCHEDULER_TYPE}"
  --log-level "${LOG_LEVEL}"
  --output-dir "${OUTPUT_DIR}"
)

# Booleans: set env to 0/false/False to disable
if [[ "${ASYNC_SCHEDULING:-1}" != "0" && "${ASYNC_SCHEDULING:-1}" != "false" && "${ASYNC_SCHEDULING:-1}" != "False" ]]; then
  ARGS+=(--async-scheduling)
fi
if [[ "${UNROLL_DEPTH_CUDA_GRAPH:-1}" != "0" && "${UNROLL_DEPTH_CUDA_GRAPH:-1}" != "false" && "${UNROLL_DEPTH_CUDA_GRAPH:-1}" != "False" ]]; then
  ARGS+=(--unroll-depth-cuda-graph)
fi
if [[ "${ENABLE_CUDA_GRAPH:-1}" == "0" || "${ENABLE_CUDA_GRAPH:-1}" == "false" || "${ENABLE_CUDA_GRAPH:-1}" == "False" ]]; then
  ARGS+=(--disable-cuda-graph)
fi

echo "[entrypoint] starting: python ${ARGS[*]}"
exec python "${ARGS[@]}"
