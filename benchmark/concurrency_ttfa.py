#!/usr/bin/env python3
"""
Fixed-concurrency TTFA benchmark for vox-serve streaming.

Unlike goodput.py (Poisson arrivals), this fires N streaming requests at (nearly)
the same instant and reports Time-To-First-Audio (TTFA) percentiles. This matches
the scenario of "N concurrent streaming requests" and is the right metric for a
latency target like "300ms for 32 concurrent streams".

Usage:
    python concurrency_ttfa.py --host localhost --port 8000 --concurrency 32 --rounds 3
"""

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp

SAMPLE_TEXTS = [
    "Hello world, this is a test of streaming text to speech latency.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "We are measuring how quickly the first chunk of audio is returned.",
    "Speech synthesis systems should feel instantaneous to the listener.",
    "Concurrent requests stress the scheduler and the batching pipeline.",
]


@dataclass
class Req:
    idx: int
    start: float
    ttfa: Optional[float] = None
    end: Optional[float] = None
    audio_bytes: int = 0
    ok: bool = False
    err: Optional[str] = None
    chunk_arrivals: List[float] = field(default_factory=list)
    chunk_bytes: List[int] = field(default_factory=list)


def pcm_dur(nbytes: int, sr: int = 24000) -> float:
    return nbytes / (sr * 2 * 1)


async def one(session, base_url, idx, text, audio_bytes=None, audio_name=None) -> Req:
    r = Req(idx=idx, start=time.time())
    try:
        form = aiohttp.FormData()
        form.add_field("text", text)
        form.add_field("streaming", "true")
        if audio_bytes is not None:
            form.add_field("audio", audio_bytes, filename=audio_name or "ref.wav",
                           content_type="audio/wav")
        async with session.post(
            f"{base_url}/generate", data=form,
            timeout=aiohttp.ClientTimeout(total=None, sock_read=120),
        ) as resp:
            if resp.status != 200:
                r.err = f"HTTP {resp.status}"
                return r
            n = 0
            async for chunk in resp.content.iter_any():
                if not chunk:
                    break
                n += 1
                now = time.time()
                if n == 1:
                    # WAV header, skip for TTFA/audio accounting
                    continue
                if r.ttfa is None:
                    r.ttfa = now - r.start
                r.chunk_arrivals.append(now)
                r.chunk_bytes.append(len(chunk))
                r.audio_bytes += len(chunk)
            r.end = time.time()
            r.ok = r.ttfa is not None
    except asyncio.TimeoutError:
        r.err = "timeout"
    except Exception as e:  # noqa: BLE001
        r.err = str(e)
    if not r.end:
        r.end = time.time()
    return r


def streaming_viability(r: Req) -> Optional[float]:
    """Fraction of chunks (after the first) that arrived before their audio must play."""
    if len(r.chunk_arrivals) < 2:
        return None
    satisfied = 0
    total = 0
    for i in range(1, len(r.chunk_arrivals)):
        cum_audio = sum(pcm_dur(b) for b in r.chunk_bytes[:i])
        latency = r.chunk_arrivals[i] - r.chunk_arrivals[0]
        if cum_audio > latency:
            satisfied += 1
        total += 1
    return 100.0 * satisfied / total if total else None


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (p / 100.0) * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    w = k - lo
    return s[lo] * (1 - w) + s[hi] * w


async def run_round(base_url, concurrency, audio_bytes=None, audio_name=None) -> List[Req]:
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(one(session, base_url, i, SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)],
                                    audio_bytes=audio_bytes, audio_name=audio_name))
            for i in range(concurrency)
        ]
        return await asyncio.gather(*tasks)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="warmup rounds excluded from stats")
    ap.add_argument("--audio", default=None, help="reference wav path for voice cloning (sent with every request)")
    args = ap.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    audio_bytes = None
    audio_name = None
    if args.audio:
        import os
        with open(args.audio, "rb") as f:
            audio_bytes = f.read()
        audio_name = os.path.basename(args.audio)

    mode = f"cloning (ref={audio_name})" if audio_bytes is not None else "no-clone (default voice)"
    print(f"Target {base_url} | concurrency={args.concurrency} | rounds={args.rounds} "
          f"(warmup={args.warmup}) | mode={mode}")
    all_ttfa: List[float] = []
    all_viab: List[float] = []
    fails = 0
    total = 0

    for rd in range(args.rounds + args.warmup):
        results = await run_round(base_url, args.concurrency, audio_bytes=audio_bytes, audio_name=audio_name)
        is_warm = rd < args.warmup
        ttfas = [r.ttfa for r in results if r.ok and r.ttfa is not None]
        rfails = sum(1 for r in results if not r.ok)
        tag = "WARMUP" if is_warm else f"round {rd - args.warmup + 1}"
        if ttfas:
            print(f"[{tag}] ok={len(ttfas)}/{len(results)} fails={rfails} "
                  f"TTFA ms: mean={1000*statistics.mean(ttfas):.0f} "
                  f"p50={1000*pct(ttfas,50):.0f} p90={1000*pct(ttfas,90):.0f} "
                  f"p99={1000*pct(ttfas,99):.0f} max={1000*max(ttfas):.0f}")
        else:
            errs = {r.err for r in results if r.err}
            print(f"[{tag}] ok=0/{len(results)} fails={rfails} errs={errs}")
        if not is_warm:
            all_ttfa.extend(ttfas)
            total += len(results)
            fails += rfails
            for r in results:
                v = streaming_viability(r)
                if v is not None:
                    all_viab.append(v)
        await asyncio.sleep(1.0)

    print("\n==================== AGGREGATE (excluding warmup) ====================")
    print(f"requests: {total}  ok: {len(all_ttfa)}  failed: {fails}")
    if all_ttfa:
        print("TTFA (ms):")
        print(f"  mean={1000*statistics.mean(all_ttfa):.1f}  p50={1000*pct(all_ttfa,50):.1f}  "
              f"p90={1000*pct(all_ttfa,90):.1f}  p95={1000*pct(all_ttfa,95):.1f}  "
              f"p99={1000*pct(all_ttfa,99):.1f}  min={1000*min(all_ttfa):.1f}  max={1000*max(all_ttfa):.1f}")
    if all_viab:
        print(f"streaming viability (per-chunk %): mean={statistics.mean(all_viab):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
