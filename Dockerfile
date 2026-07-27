# syntax=docker/dockerfile:1.7
# Qwen3-TTS on VoxServe — NVIDIA GPU image (model baked in at build time)
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
    TRANSFORMERS_CACHE=/data/hf-cache \
    HUGGINGFACE_HUB_CACHE=/data/hf-cache \
    VOX_METRICS_DIR=/var/log/vox-serve \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# System deps: Python 3.12, Opus (WS audio), espeak (phonemizer), build tools
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
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && python3.12 -m venv /opt/venv \
    && ln -sf /opt/venv/bin/python /usr/local/bin/python

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY vox_serve ./vox_serve

RUN pip install --upgrade pip setuptools wheel \
    && pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.8.0" "torchaudio==2.8.0" \
    && pip install -e . \
    && pip install "opuslib==3.0.1"

# ---- Bake model weights into the image ----
# Build context `hfcache` (see compose) is mounted at /host-hf.
# Prefer copying from that cache; otherwise download from Hugging Face Hub.
ARG MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ARG HF_TOKEN=""
ENV MODEL=${MODEL} \
    HF_TOKEN=${HF_TOKEN} \
    HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

COPY docker/prefetch_models.py /tmp/prefetch_models.py
RUN --mount=type=bind,from=hfcache,source=.,target=/host-hf,ro \
    python /tmp/prefetch_models.py \
    && rm -f /tmp/prefetch_models.py

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /var/log/vox-serve /data/voices

EXPOSE 2200

ENV PORT=2200 \
    MAX_BATCH_SIZE=64 \
    MAX_NUM_PAGES=2048 \
    DETOKENIZE_INTERVAL=5 \
    SCHEDULER_TYPE=online \
    ASYNC_SCHEDULING=1 \
    UNROLL_DEPTH_CUDA_GRAPH=1 \
    LOG_LEVEL=INFO \
    VOX_OUTPUT_DIR=/data/voices

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
