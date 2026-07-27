# One-command Docker deploy (Qwen3-TTS)

## Prerequisites

- Linux host with NVIDIA GPU
- [Docker](https://docs.docker.com/get-docker/) + [Compose v2](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Verify GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Start

```bash
git clone https://github.com/reactivespace/Qwen3-TTS.git
cd Qwen3-TTS
docker compose up -d --build
```

First boot downloads the Hugging Face model into a Docker volume (can take several minutes).

## Check

```bash
curl http://localhost:2200/health
curl http://localhost:2200/stats
curl http://localhost:2200/voices
```

WebSocket TTS: `ws://HOST:2200/ws`  
Browser tester: open `index.html` and point at `http://HOST:2200`

## Logs (persistent)

Inside the container / volume:

- `/var/log/vox-serve/requests.jsonl`
- `/var/log/vox-serve/stats.json`
- `/var/log/vox-serve/stats.log`
- `/var/log/vox-serve/server.log` (compose stdout also via `docker compose logs -f`)

```bash
docker compose logs -f qwen3-tts
docker compose exec qwen3-tts cat /var/log/vox-serve/stats.json
```

## Configure

Copy `.env.example` → `.env` and set e.g.:

```bash
HF_TOKEN=hf_...          # if the model gated / rate-limited
MAX_BATCH_SIZE=32        # lower if GPU OOM
NVIDIA_VISIBLE_DEVICES=0
```

## Stop

```bash
docker compose down
```

Volumes (`hf_cache`, `vox_logs`, `vox_voices`) keep models and metrics unless you run `docker compose down -v`.
