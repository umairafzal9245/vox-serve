#!/usr/bin/env python3
"""Concurrency / TTFA tester for vox-serve streaming.

Requires: pip install aiohttp

Usage:
    python concurrency_test.py               # uses config below
    python concurrency_test.py 32 clone      # 32 users, upload reference each request
    python concurrency_test.py 64 noclone    # 64 users, default voice
    python concurrency_test.py 64 byid       # 64 users, register-once voice_id (no upload)

Register-once (byid) first uploads the reference a single time to /voices,
then every request sends only the voice_id -> no per-request audio upload.
"""

import asyncio
import statistics
import sys
import time

import aiohttp

# ---------------- CONFIG: edit these ----------------
HOST = "127.0.0.1"          # <-- your server IP, e.g. "203.0.113.5"
PORT = 2200                 # <-- server port
REF_WAV = "./reference_hf.wav"  # <-- clean 3-10s speech wav for cloning
TEXT = "Hello, this is a concurrency test of the streaming text to speech system."
CONCURRENCY = 64            # number of simultaneous users
MODE = "clone"             # "clone" (upload each req) | "noclone" | "byid" (register once)
# ----------------------------------------------------

# optional CLI overrides: python concurrency_test.py <N> <clone|noclone|byid>
if len(sys.argv) > 1:
    CONCURRENCY = int(sys.argv[1])
if len(sys.argv) > 2:
    MODE = sys.argv[2]

URL = f"http://{HOST}:{PORT}/generate"
VOICES_URL = f"http://{HOST}:{PORT}/voices"


def pct(arr, p):
    if not arr:
        return 0.0
    s = sorted(arr)
    k = (p / 100) * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


async def one_request(session, ref_bytes, voice_id):
    form = aiohttp.FormData()
    form.add_field("text", TEXT)
    form.add_field("streaming", "true")
    if voice_id is not None:
        form.add_field("voice_id", voice_id)          # register-once: no upload
    elif ref_bytes is not None:
        form.add_field("audio", ref_bytes, filename="ref.wav", content_type="audio/wav")

    start = time.perf_counter()
    try:
        async with session.post(URL, data=form) as resp:
            if resp.status != 200:
                return {"ok": False, "err": f"HTTP {resp.status}"}
            chunk_idx = 0
            ttfa = None
            async for _chunk in resp.content.iter_any():
                chunk_idx += 1
                if chunk_idx == 1:          # first chunk is the WAV header
                    continue
                if ttfa is None:
                    ttfa = (time.perf_counter() - start) * 1000.0
            return {"ok": ttfa is not None, "ttfa": ttfa}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": str(e)}


async def main():
    ref_bytes = None
    voice_id = None
    print(f"Target {URL} | concurrency={CONCURRENCY} | mode={MODE}")

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if MODE == "byid":
            # Upload the reference ONCE, then reference it by id every request
            with open(REF_WAV, "rb") as f:
                data = f.read()
            form = aiohttp.FormData()
            form.add_field("audio", data, filename="ref.wav", content_type="audio/wav")
            async with session.post(VOICES_URL, data=form) as r:
                voice_id = (await r.json())["voice_id"]
            print(f"registered voice_id={voice_id} from {REF_WAV} ({len(data)} bytes, uploaded once)")
        elif MODE == "clone":
            with open(REF_WAV, "rb") as f:
                ref_bytes = f.read()
            print(f"reference: {REF_WAV} ({len(ref_bytes)} bytes, uploaded per request)")

        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[one_request(session, ref_bytes, voice_id) for _ in range(CONCURRENCY)]
        )
        wall = (time.perf_counter() - t0) * 1000.0

    ttfas = [r["ttfa"] for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]

    print(f"\nok={len(ttfas)}/{len(results)}  failed={len(fails)}  wall={wall:.0f}ms")
    if ttfas:
        print(
            "TTFA (ms): "
            f"mean={statistics.mean(ttfas):.0f} p50={pct(ttfas, 50):.0f} "
            f"p90={pct(ttfas, 90):.0f} p99={pct(ttfas, 99):.0f} "
            f"min={min(ttfas):.0f} max={max(ttfas):.0f}"
        )
    if fails:
        print("errors:", sorted({f['err'] for f in fails}))


if __name__ == "__main__":
    asyncio.run(main())
