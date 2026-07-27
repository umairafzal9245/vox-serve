# syntax=docker/dockerfile:1
# Qwen3-TTS on VoxServe — NVIDIA GPU image
#
# Build:  docker compose build
# Run:    docker compose up -d
#
# Requires: NVIDIA Container Toolkit on the host

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache \
    VOX_METRICS_DIR=/var/log/vox-serve \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System deps: Python 3.12, Opus (WS audio), espeak (phonemizer), build tools (flashinfer/opuslib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        git \
        curl \
        ca-certificates \
        build-essential \
        libopus0 \
        libopus-dev \
        espeak-ng \
        espeak-ng-data \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /app

# Install Python deps first (better layer cache). Torch cu128 matches host/runtime CUDA 12.8.
COPY pyproject.toml README.md LICENSE ./
COPY vox_serve ./vox_serve

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.8.0" "torchaudio==2.8.0" \
    && python -m pip install -e . \
    && python -m pip install "opuslib==3.0.1"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /var/log/vox-serve /data/hf-cache /data/voices

EXPOSE 2200

# Default production-ish Qwen3-TTS settings (override via compose/env)
ENV MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    PORT=2200 \
    MAX_BATCH_SIZE=64 \
    MAX_NUM_PAGES=2048 \
    DETOKENIZE_INTERVAL=5 \
    SCHEDULER_TYPE=online \
    ASYNC_SCHEDULING=1 \
    UNROLL_DEPTH_CUDA_GRAPH=1 \
    LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
