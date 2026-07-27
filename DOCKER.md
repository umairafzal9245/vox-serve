# One-command Docker deploy (Qwen3-TTS)

## Prerequisites

- Linux host with NVIDIA GPU
- Docker Engine + Compose v2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Start (build + run)

```bash
git clone https://github.com/reactivespace/Qwen3-TTS.git
cd Qwen3-TTS

# Optional: reuse your existing Hugging Face cache during build (much faster)
export HF_CACHE_HOST="$HOME/.cache/huggingface"

docker compose up -d --build
```

Model weights are **baked into the image at build time** (copied from `HF_CACHE_HOST` if present, otherwise downloaded from Hugging Face).

```bash
curl http://localhost:2200/health
curl http://localhost:2200/stats
```

## What is `docker-compose.override.yml`?

Compose **auto-merges** a file named `docker-compose.override.yml` next to `docker-compose.yml` when you run `docker compose up`.

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Shared / committed config for everyone |
| `docker-compose.override.yml` | **Local-only** tweaks (extra volume mounts, debug ports). Not committed. |

Example local override:

```yaml
services:
  qwen3-tts:
    environment:
      LOG_LEVEL: DEBUG
```

You do **not** need an override file for normal deploy. If it exists, Compose applies it automatically — that can surprise you (e.g. mounting an empty host folder over `/data/hf-cache` would hide the baked-in model).

## Configure

Copy `.env.example` → `.env`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `HF_CACHE_HOST` | `./docker/empty-hf` | Host HF cache used **during build** |
| `HF_TOKEN` | — | Token for gated / rate-limited Hub access |
| `MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Model id |
| `MAX_BATCH_SIZE` | `64` | Concurrent decode batch |
| `PORT` | `2200` | Host port |

## Logs

Host path `/var/log/vox-serve` is mounted into the container:

```bash
docker compose logs -f qwen3-tts
curl http://localhost:2200/stats
ls /var/log/vox-serve/
```

## Stop

```bash
docker compose down
```
