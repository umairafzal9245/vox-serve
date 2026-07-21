#!/usr/bin/env python3
"""WebSocket streaming client + concurrency tester for vox-serve.

The whole point of the WebSocket endpoint is that each client opens ONE
connection and reuses it for many utterances, paying the TCP+TLS handshake
only once instead of on every request (which is what dominates latency over
long network distances).

Requires: pip install websockets   (also opuslib to save/hear Opus output)

Saved audio: demo writes ws_demo_1.wav.. ; concurrency writes ws_sample.wav.

Usage:
    # 1) register a voice once (over HTTP) so we can clone without upload:
    curl -X POST http://HOST:2200/voices -F "audio=@reference_hf.wav" -F "voice_id=umair"

    # 2) demo: one socket, several utterances, prints per-utterance TTFA
    python ws_client.py demo          # PCM output
    python ws_client.py demo opus     # Opus output (reports payload savings)

    # 3) concurrency: N sockets opened once, then one utterance each at once
    #    (uses WebSocket + voice_id; add "opus" for Opus output)
    python ws_client.py 64                 # 64 users, voice_id clone, PCM
    python ws_client.py 64 opus            # 64 users, voice_id clone, Opus (~8x smaller)
    python ws_client.py 64 noclone         # 64 users, default voice, PCM
    python ws_client.py 64 noclone opus    # 64 users, default voice, Opus
"""

import asyncio
import json
import statistics
import sys
import time
import wave

import websockets

SAMPLE_RATE = 24000  # server streams 24 kHz mono int16


def _save_wav(path, pcm_bytes):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_bytes)
    print(f"  saved {path} ({len(pcm_bytes) / 2 / SAMPLE_RATE:.2f}s)")

# ---------------- CONFIG: edit these ----------------
HOST = "127.0.0.1"      # <-- your server IP, e.g. "178.63.124.87"
PORT = 2200             # <-- server port
VOICE_ID = "umair"      # <-- must be pre-registered via POST /voices
TEXT = "Hello, this is a concurrency test of the streaming text to speech system."
# ----------------------------------------------------

WS_URL = f"ws://{HOST}:{PORT}/ws"


def pct(arr, p):
    if not arr:
        return 0.0
    s = sorted(arr)
    k = (p / 100) * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


async def synth(ws, text, voice_id, fmt="pcm", stats=None, collect=False):
    """Send one utterance on an open socket, return TTFA (ms) to first audio.

    If ``stats`` (a dict) is passed, records total wire ``bytes`` received and,
    when ``collect`` is True, the decoded PCM under ``stats["pcm"]`` (Opus is
    decoded back to PCM via opuslib so the result is directly playable).
    """
    req = {"text": text, "format": fmt}
    if voice_id:
        req["voice_id"] = voice_id
    t0 = time.perf_counter()
    await ws.send(json.dumps(req))

    ttfa = None
    nbytes = 0
    pcm = bytearray() if collect else None
    decoder = None
    frame_samples = None

    while True:
        m = await ws.recv()
        if isinstance(m, (bytes, bytearray)):
            if ttfa is None:
                ttfa = (time.perf_counter() - t0) * 1000.0
            nbytes += len(m)
            if collect:
                if fmt == "opus":
                    if decoder is None:
                        import opuslib
                        decoder = opuslib.Decoder(SAMPLE_RATE, 1)
                        frame_samples = SAMPLE_RATE * 20 // 1000  # 20 ms frames
                    pcm.extend(decoder.decode(bytes(m), frame_samples))
                else:
                    pcm.extend(m)
        else:
            j = json.loads(m)
            if j.get("type") == "start" and collect and fmt == "opus":
                frame_samples = SAMPLE_RATE * (j.get("frame_ms") or 20) // 1000
            if j.get("type") == "end":
                if stats is not None:
                    stats["bytes"] = nbytes
                    if collect:
                        stats["pcm"] = bytes(pcm)
                return ttfa
            if j.get("type") == "error":
                raise RuntimeError(j.get("detail"))


async def demo(fmt="pcm"):
    async with websockets.connect(WS_URL, max_size=None) as ws:
        print(f"connected to {WS_URL}; format={fmt}; reusing ONE socket:")
        for i in range(4):
            stats = {}
            ttfa = await synth(ws, TEXT, VOICE_ID, fmt=fmt, stats=stats, collect=True)
            print(f"  utterance {i + 1}: ttfa={ttfa:.0f}ms  wire_bytes={stats.get('bytes')}")
            _save_wav(f"ws_demo_{i + 1}.wav", stats.get("pcm", b""))
        await ws.send(json.dumps({"type": "close"}))
    if fmt == "opus":
        print("(compare wire_bytes vs a PCM run to see the ~8x reduction)")


async def user(voice_id, fmt, connect_times, results, collect=False):
    """One user: open a socket (timed), then send a single utterance (timed)."""
    try:
        c0 = time.perf_counter()
        ws = await websockets.connect(WS_URL, max_size=None)
        connect_times.append((time.perf_counter() - c0) * 1000.0)
    except Exception as e:  # noqa: BLE001
        results.append({"ok": False, "err": f"connect: {e}"})
        return
    try:
        stats = {}
        ttfa = await synth(ws, TEXT, voice_id, fmt=fmt, stats=stats, collect=collect)
        results.append({
            "ok": ttfa is not None, "ttfa": ttfa,
            "bytes": stats.get("bytes", 0), "pcm": stats.get("pcm"),
        })
    except Exception as e:  # noqa: BLE001
        results.append({"ok": False, "err": str(e)})
    finally:
        await ws.close()


async def concurrency(n, voice_id, fmt):
    mode = "clone(voice_id)" if voice_id else "no-clone"
    print(f"Target {WS_URL} | concurrency={n} | mode={mode} | format={fmt}")
    connect_times, results = [], []

    t0 = time.perf_counter()
    # first user collects its audio so we can save one playable sample
    await asyncio.gather(
        user(voice_id, fmt, connect_times, results, collect=True),
        *[user(voice_id, fmt, connect_times, results) for _ in range(n - 1)],
    )
    wall = (time.perf_counter() - t0) * 1000.0

    ttfas = [r["ttfa"] for r in results if r["ok"]]
    byts = [r["bytes"] for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    sample = next((r["pcm"] for r in results if r["ok"] and r.get("pcm")), None)
    print(f"\nok={len(ttfas)}/{len(results)}  failed={len(fails)}  wall={wall:.0f}ms")
    if sample:
        _save_wav("ws_sample.wav", sample)
    if connect_times:
        print(f"connect (ms): mean={statistics.mean(connect_times):.0f} p50={pct(connect_times, 50):.0f}")
    if ttfas:
        print(
            "TTFA after connect (ms): "
            f"mean={statistics.mean(ttfas):.0f} p50={pct(ttfas, 50):.0f} "
            f"p90={pct(ttfas, 90):.0f} p99={pct(ttfas, 99):.0f} "
            f"min={min(ttfas):.0f} max={max(ttfas):.0f}"
        )
    if byts:
        print(f"wire bytes/utterance: mean={int(statistics.mean(byts))} ({fmt})")
    if fails:
        print("errors:", sorted({f['err'] for f in fails})[:5])


def main():
    args = sys.argv[1:]
    if not args or args[0] == "demo":
        fmt = "opus" if "opus" in args else "pcm"
        asyncio.run(demo(fmt))
    else:
        n = int(args[0])
        rest = args[1:]
        voice_id = None if "noclone" in rest else VOICE_ID
        fmt = "opus" if "opus" in rest else "pcm"
        asyncio.run(concurrency(n, voice_id, fmt))


if __name__ == "__main__":
    main()
