# Qwen3-TTS

High-performance streaming TTS server for **[Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts)**, built on [VoxServe](https://github.com/vox-serve/vox-serve).

Optimized for call-center / real-time voice agents: WebSocket streaming, voice cloning, 8/16/24 kHz output, Opus or PCM, continuous batching on a single GPU.

**Default model:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

---

## Requirements

- Linux + NVIDIA GPU (CUDA 12.8 recommended)
- For Docker: Docker Engine, Compose v2, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- For native run: Python 3.12+

---

## Quick start (Docker)

```bash
git clone https://github.com/reactivespace/Qwen3-TTS.git
cd Qwen3-TTS
docker compose up -d --build
```

First boot downloads model weights into a Docker volume (several minutes). Then:

```bash
curl http://localhost:2200/health
# {"status":"healthy"}
```

More detail: [DOCKER.md](DOCKER.md)

Optional config — copy `.env.example` to `.env`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `PORT` | `2200` | Host port |
| `MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Hugging Face model id |
| `MAX_BATCH_SIZE` | `64` | Max concurrent decode batch |
| `MAX_NUM_PAGES` | `2048` | KV cache pages |
| `DETOKENIZE_INTERVAL` | `5` | Codec frames before each audio chunk (~400 ms @ 24 kHz) |
| `HF_TOKEN` | — | Hugging Face token if needed |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU selection |

---

## Native run (without Docker)

```bash
git clone https://github.com/reactivespace/Qwen3-TTS.git
cd Qwen3-TTS
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .

export VOX_METRICS_DIR=/var/log/vox-serve
mkdir -p /var/log/vox-serve

python -m vox_serve.launch \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --host 0.0.0.0 --port 2200 \
  --scheduler-type online \
  --async-scheduling \
  --max-batch-size 64 \
  --max-num-pages 2048 \
  --detokenize-interval 5 \
  --unroll-depth-cuda-graph \
  --log-level INFO
```

---

## API

Base URL: `http://HOST:2200`  
Interactive docs: `http://HOST:2200/docs`

> **`POST /generate` is disabled** (returns `410`). Use **WebSocket `/ws`** for synthesis.

### Health & metrics

```bash
curl http://HOST:2200/health
curl http://HOST:2200/stats
curl "http://HOST:2200/stats/requests?limit=20"
curl "http://HOST:2200/stats/log?lines=40"
```

Metrics are also written under `/var/log/vox-serve/` (or `$VOX_METRICS_DIR`):

- `requests.jsonl` — one JSON line per finished request (TTFA, total ms, KV memory)
- `stats.json` — latest live snapshot
- `stats.log` — rolling text log

### Voices (register once, reuse by id)

```bash
# Add
curl -X POST http://HOST:2200/voices \
  -F "audio=@reference.wav" -F "voice_id=laura"

# List
curl http://HOST:2200/voices

# Delete
curl -X DELETE http://HOST:2200/voices/laura
```

### WebSocket `/ws` (primary)

Open one connection and send many utterances.

**Client → server** (JSON per utterance):

```json
{
  "text": "Thank you for calling. How can I help you today?",
  "voice_id": "laura",
  "x_vector_only_mode": true,
  "sample_rate": 8000,
  "format": "pcm"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `text` | yes | Text to speak |
| `voice_id` | recommended | Registered voice |
| `x_vector_only_mode` | recommended `true` | Clone from reference audio without `ref_text` |
| `sample_rate` | no | `8000`, `16000`, or `24000` (default) |
| `format` | no | `pcm` (default) or `opus` |
| `language` | no | e.g. `english` |
| `audio_base64` | no | One-off reference WAV instead of `voice_id` |

**Server → client**

1. `{"type":"start","request_id":"...","sample_rate":8000,"channels":1,"format":"pcm_s16le"}`
2. Binary frames — PCM int16 mono chunks (~400 ms each), or Opus 20 ms packets
3. `{"type":"end","request_id":"..."}`

Errors: `{"type":"error","detail":"..."}` (socket stays open).

**PCM chunk sizes (approx.)**

| `sample_rate` | Bytes / frame | Duration |
|---------------|---------------|----------|
| 24000 | 19200 | ~400 ms |
| 16000 | 12800 | ~400 ms |
| 8000 | 6400 | ~400 ms |

Browser tester: open [`index.html`](index.html) and set the server URL.

---

## Useful launch flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | HF model id (use Qwen3-TTS Base for cloning) |
| `--port` | `8000` | HTTP / WS port |
| `--scheduler-type` | `base` | Use `online` for streaming agents |
| `--async-scheduling` | off | Overlap scheduling with GPU work |
| `--max-batch-size` | `8` | Concurrent decode capacity |
| `--max-num-pages` | `2048` | KV pages (memory for concurrency) |
| `--detokenize-interval` | model default | Frames per audio chunk (lower → faster TTFA) |
| `--unroll-depth-cuda-graph` | off | Faster depth transformer for Qwen3 |
| `--output-dir` | `/tmp/vox_serve_audio` | Uploads + registered voices (`VOX_OUTPUT_DIR`) |

---

## Project layout

```
Dockerfile / docker-compose.yml   # one-command GPU deploy
docker/entrypoint.sh
vox_serve/                        # server + scheduler + Qwen3 model
index.html                        # simple WebSocket TTS tester
DOCKER.md                         # Docker notes
```

---

## License

See [LICENSE](LICENSE).
