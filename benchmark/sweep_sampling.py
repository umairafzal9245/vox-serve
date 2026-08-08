#!/usr/bin/env python3
"""Parameter sweep for Qwen3-TTS runaway / EOS reliability.

Restarts the docker compose service with each sampling config (fixed max_tokens),
runs N sequential /stream requests, and scores EOS vs max_tokens failures.

Literature sources baked into CONFIGS:
- official: checkpoint generation_config.json / qwen_tts hard defaults
- prod_arkodeep: arkodeepsen runaway-fix defaults
- docs_perf: mintlify performance guide suggestion
- mild / tight / vox_original: our prior experiments
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
TEXT = "I'm here! Could you please confirm if you own your home?"
VOICE = "laura"
URL = "http://127.0.0.1:2200"
N = 20
WARN_SEC = 12.0
FAIL_SEC = 20.0
# Fixed duration budget so configs are comparable on EOS rate, not cap height.
MAX_TOKENS = "512"
EXPECTED_CAP_SEC = 35.0  # ~512 frames @ 12.5Hz minus prompt ≈ high 30s

# name -> (temp, top_p, top_k, rep_penalty, note)
CONFIGS = [
    ("official", 0.9, 1.0, 50, 1.05, "Qwen generation_config.json / package defaults"),
    ("official_temp07", 0.7, 1.0, 50, 1.05, "Official nucleus; lower temperature"),
    ("official_rep12", 0.9, 1.0, 50, 1.2, "Official nucleus; stronger rep penalty"),
    ("prod_arkodeep", 0.8, 0.9, 50, 1.1, "arkodeepsen stability defaults"),
    ("docs_perf", 0.7, 0.9, 50, 1.05, "Mintlify performance guide-ish"),
    ("mild", 0.7, 0.9, 40, 1.2, "Our mild preset"),
    ("vox_original", 0.7, 1.0, 40, 1.2, "Prior vox-serve qwen3 defaults"),
    ("high_rep_nucleus", 0.7, 1.0, 50, 1.3, "Keep full nucleus; strong anti-loop"),
    ("tight", 0.6, 0.8, 25, 1.3, "Our tight preset (may exclude EOS)"),
]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=True)


def recreate(temp: float, top_p: float, top_k: int, rep: float) -> None:
    env_prefix = {
        "TEMPERATURE": str(temp),
        "TOP_P": str(top_p),
        "TOP_K": str(top_k),
        "REPETITION_PENALTY": str(rep),
        "MAX_TOKENS": MAX_TOKENS,
    }
    # Force env into compose via shell export for this process tree
    cmd = ["docker", "compose", "up", "-d", "--force-recreate", "--no-build"]
    subprocess.run(cmd, cwd=ROOT, check=True, env={**dict(**{k: v for k, v in __import__("os").environ.items()}), **env_prefix})


def wait_healthy(timeout: float = 180.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{URL}/health", timeout=2)
            if r.ok:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("server did not become healthy")


def confirm_args(temp: float, top_p: float, top_k: int, rep: float) -> str:
    out = subprocess.check_output(["docker", "logs", "qwen3-tts"], text=True, stderr=subprocess.STDOUT)
    last = ""
    for line in out.splitlines():
        if "[entrypoint] starting:" in line:
            last = line
    need = [
        f"--temperature {temp}",
        f"--top-p {top_p}",
        f"--top-k {top_k}",
        f"--repetition-penalty {rep}",
        f"--max-tokens {MAX_TOKENS}",
    ]
    missing = [x for x in need if x not in last]
    if missing:
        raise RuntimeError(f"entrypoint args mismatch; missing {missing}; got: {last}")
    return last


def synth_one(timeout: float = 180.0) -> dict:
    import base64

    t0 = time.perf_counter()
    pcm = bytearray()
    request_id = None
    err = None
    try:
        with requests.post(
            f"{URL}/stream",
            json={
                "text": TEXT,
                "voice_id": VOICE,
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

    audio_sec = len(pcm) / (2 * 16000) if pcm else 0.0
    if err:
        status = "error"
    elif audio_sec >= FAIL_SEC:
        status = "FAIL"
    elif audio_sec >= WARN_SEC:
        status = "WARN"
    else:
        status = "ok"
    return {
        "request_id": request_id,
        "audio_sec": round(audio_sec, 3),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "status": status,
        "error": err,
        "hit_cap": audio_sec >= EXPECTED_CAP_SEC,
    }


def finish_reasons_since(since_iso: str) -> dict[str, str]:
    """Parse scheduler completion logs: Request <id> completed: {'status':..., 'reason':...}"""
    out = subprocess.check_output(
        ["docker", "logs", "--since", since_iso, "qwen3-tts"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    found = {}
    pat = re.compile(
        r"Request ([0-9a-f-]+) completed: \{[^}]*'reason': '([^']+)'"
    )
    # also allow double quotes from json.dumps
    pat2 = re.compile(
        r'Request ([0-9a-f-]+) completed: \{[^}]*"reason": "([^"]+)"'
    )
    for line in out.splitlines():
        m = pat.search(line) or pat2.search(line)
        if m:
            found[m.group(1)] = m.group(2)
    return found


def score(results: list[dict], reasons: dict[str, str]) -> dict:
    for r in results:
        rid = r.get("request_id")
        r["finish_reason"] = reasons.get(rid) if rid else None

    ok = [r for r in results if r["status"] == "ok"]
    warn = [r for r in results if r["status"] == "WARN"]
    fail = [r for r in results if r["status"] == "FAIL"]
    errs = [r for r in results if r["status"] == "error"]
    durations = [r["audio_sec"] for r in results if r["audio_sec"] > 0]
    ok_durs = [r["audio_sec"] for r in ok]
    stop = sum(1 for r in results if r.get("finish_reason") == "stop_id_encountered")
    maxed = sum(1 for r in results if r.get("finish_reason") == "max_tokens_reached")
    unknown = sum(1 for r in results if not r.get("finish_reason"))

    return {
        "n": len(results),
        "ok": len(ok),
        "warn": len(warn),
        "fail": len(fail),
        "error": len(errs),
        "eos_rate": round(stop / len(results), 3) if results else 0,
        "max_tokens_rate": round(maxed / len(results), 3) if results else 0,
        "stop_id_count": stop,
        "max_tokens_count": maxed,
        "unknown_reason_count": unknown,
        "fail_or_warn_rate": round((len(warn) + len(fail)) / len(results), 3) if results else 0,
        "audio_sec_min": min(durations) if durations else None,
        "audio_sec_max": max(durations) if durations else None,
        "audio_sec_mean": round(statistics.mean(durations), 3) if durations else None,
        "ok_audio_mean": round(statistics.mean(ok_durs), 3) if ok_durs else None,
        "ok_audio_stdev": round(statistics.stdev(ok_durs), 3) if len(ok_durs) > 1 else 0.0,
    }


def main() -> int:
    out_dir = ROOT / "benchmark" / "hallucination_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = {
        "text": TEXT,
        "voice": VOICE,
        "n_per_config": N,
        "max_tokens": int(MAX_TOKENS),
        "configs": [],
    }

    print(f"Sweep {len(CONFIGS)} configs × {N} requests; max_tokens={MAX_TOKENS} fixed\n")

    for name, temp, top_p, top_k, rep, note in CONFIGS:
        print("=" * 72)
        print(f"CONFIG {name}: temp={temp} top_p={top_p} top_k={top_k} rep={rep}")
        print(f"  {note}")
        recreate(temp, top_p, top_k, rep)
        wait_healthy()
        args_line = confirm_args(temp, top_p, top_k, rep)
        print(f"  {args_line.strip()}")

        # Warmup (CUDA graphs / first-request tax)
        print("  warmup...")
        warm = synth_one()
        print(f"  warmup -> {warm['status']} audio={warm['audio_sec']}s wall={warm['elapsed_sec']}s")

        since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        results = []
        for i in range(N):
            r = synth_one()
            results.append(r)
            print(
                f"  [{i+1:02d}] {r['status']:5} audio={r['audio_sec']:6.2f}s "
                f"wall={r['elapsed_sec']:6.2f}s id={r['request_id']}"
            )

        # give logs a moment to flush
        time.sleep(1)
        reasons = finish_reasons_since(since)
        summary = score(results, reasons)
        summary.update(
            {
                "name": name,
                "temperature": temp,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": rep,
                "note": note,
                "warmup": warm,
                "results": results,
            }
        )
        sweep["configs"].append(summary)
        print(
            f"  >> eos={summary['eos_rate']:.0%} maxed={summary['max_tokens_rate']:.0%} "
            f"ok={summary['ok']} warn={summary['warn']} fail={summary['fail']} "
            f"ok_mean={summary['ok_audio_mean']}s"
        )

    # Rank: primary = lowest max_tokens_rate (EOS failures), then highest eos_rate, then lowest fail_or_warn
    ranked = sorted(
        sweep["configs"],
        key=lambda c: (c["max_tokens_rate"], -c["eos_rate"], c["fail_or_warn_rate"], -(c["ok"] or 0)),
    )
    sweep["ranking"] = [
        {
            "rank": i + 1,
            "name": c["name"],
            "eos_rate": c["eos_rate"],
            "max_tokens_rate": c["max_tokens_rate"],
            "ok": c["ok"],
            "fail": c["fail"],
            "warn": c["warn"],
            "params": {
                "temperature": c["temperature"],
                "top_p": c["top_p"],
                "top_k": c["top_k"],
                "repetition_penalty": c["repetition_penalty"],
            },
        }
        for i, c in enumerate(ranked)
    ]

    out_path = out_dir / f"sweep_{int(time.time())}.json"
    out_path.write_text(json.dumps(sweep, indent=2))

    print("\n" + "=" * 72)
    print("RANKING (best = fewest max_tokens / highest EOS)")
    for row in sweep["ranking"]:
        p = row["params"]
        print(
            f"  #{row['rank']} {row['name']:18} eos={row['eos_rate']:.0%} "
            f"maxed={row['max_tokens_rate']:.0%} ok={row['ok']:2} fail={row['fail']:2}  "
            f"t={p['temperature']} p={p['top_p']} k={p['top_k']} rep={p['repetition_penalty']}"
        )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
