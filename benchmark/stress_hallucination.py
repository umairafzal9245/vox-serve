#!/usr/bin/env python3
"""Fire N /stream requests and flag unusually long audio (hallucination / runaway)."""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

TEXT = "I'm here! Could you please confirm if you own your home?"
# ~4s spoken is typical for this line; >12s is suspicious; >20s is runaway.
WARN_SEC = 12.0
FAIL_SEC = 20.0


def synth_one(url: str, voice_id: str, idx: int, timeout: float) -> dict:
    t0 = time.perf_counter()
    pcm = bytearray()
    request_id = None
    err = None
    try:
        with requests.post(
            f"{url.rstrip('/')}/stream",
            json={
                "text": TEXT,
                "voice_id": voice_id,
                "x_vector_only_mode": True,
                "sample_rate": 16000,
                "format": "pcm",
            },
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            event = None
            data_lines: list[str] = []
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                if raw.startswith("event:"):
                    event = raw[6:].strip()
                    data_lines = []
                elif raw.startswith("data:"):
                    data_lines.append(raw[5:].lstrip())
                elif raw == "":
                    if event and data_lines:
                        payload = json.loads("\n".join(data_lines))
                        if event == "start":
                            request_id = payload.get("request_id")
                        elif event == "audio":
                            pcm.extend(base64.b64decode(payload["data"]))
                        elif event == "error":
                            err = payload.get("detail") or str(payload)
                    event = None
                    data_lines = []
    except Exception as e:  # noqa: BLE001
        err = str(e)

    elapsed = time.perf_counter() - t0
    # 16-bit mono @ 16 kHz
    audio_sec = len(pcm) / (2 * 16000) if pcm else 0.0
    return {
        "idx": idx,
        "request_id": request_id,
        "elapsed_sec": round(elapsed, 3),
        "audio_sec": round(audio_sec, 3),
        "pcm_bytes": len(pcm),
        "error": err,
        "status": (
            "error"
            if err
            else "FAIL"
            if audio_sec >= FAIL_SEC
            else "WARN"
            if audio_sec >= WARN_SEC
            else "ok"
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:2200")
    p.add_argument("--voice", default="laura")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--concurrency", type=int, default=1, help="1 = sequential (better for repro)")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out-dir", default="benchmark/hallucination_out")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"POST {args.url}/stream  n={args.n} concurrency={args.concurrency} voice={args.voice}")
    print(f"text: {TEXT!r}")
    print(f"thresholds: WARN>={WARN_SEC}s  FAIL>={FAIL_SEC}s\n")

    results = []
    t_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {
            ex.submit(synth_one, args.url, args.voice, i + 1, args.timeout): i + 1
            for i in range(args.n)
        }
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            flag = r["status"]
            print(
                f"[{r['idx']:02d}] {flag:5}  audio={r['audio_sec']:6.2f}s  "
                f"wall={r['elapsed_sec']:6.2f}s  id={r['request_id']}  "
                f"{('ERR=' + r['error']) if r['error'] else ''}"
            )

    results.sort(key=lambda x: x["idx"])
    wall = time.perf_counter() - t_all
    ok = [r for r in results if r["status"] == "ok"]
    warn = [r for r in results if r["status"] == "WARN"]
    fail = [r for r in results if r["status"] == "FAIL"]
    errs = [r for r in results if r["status"] == "error"]
    durations = [r["audio_sec"] for r in results if r["audio_sec"] > 0]

    summary = {
        "text": TEXT,
        "n": args.n,
        "concurrency": args.concurrency,
        "voice": args.voice,
        "wall_sec": round(wall, 2),
        "ok": len(ok),
        "warn": len(warn),
        "fail": len(fail),
        "error": len(errs),
        "audio_sec_min": min(durations) if durations else None,
        "audio_sec_max": max(durations) if durations else None,
        "audio_sec_mean": round(statistics.mean(durations), 3) if durations else None,
        "audio_sec_stdev": round(statistics.stdev(durations), 3) if len(durations) > 1 else 0.0,
        "results": results,
    }
    out_path = out_dir / f"run_{int(time.time())}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(
        f"ok={len(ok)} warn={len(warn)} fail={len(fail)} error={len(errs)}  "
        f"audio_sec min/mean/max="
        f"{summary['audio_sec_min']}/{summary['audio_sec_mean']}/{summary['audio_sec_max']}  "
        f"wall={wall:.1f}s"
    )
    print(f"wrote {out_path}")
    return 1 if fail or errs else 0


if __name__ == "__main__":
    sys.exit(main())
