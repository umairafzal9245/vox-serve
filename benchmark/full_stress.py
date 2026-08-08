#!/usr/bin/env python3
"""Full stress suite: varied sentence types × sequential + concurrent load.

Measures EOS reliability (vs max_tokens runaway), audio length vs expected
duration, errors, and latency under concurrency.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

# Expected speech ~2.8 words/sec for IVR-ish delivery; floor/ceiling for scoring.
WORDS_PER_SEC = 2.8
MIN_EXPECTED = 1.2
# Soft/hard multipliers over expected length
WARN_MULT = 3.0
FAIL_MULT = 5.0
# Absolute runaway near max_tokens=512 ≈ ~39s generated
CAP_SEC = 35.0


@dataclass(frozen=True)
class Case:
    category: str
    text: str


CASES: list[Case] = [
    # --- short / greetings ---
    Case("short", "Hello."),
    Case("short", "Hi there!"),
    Case("short", "Thanks."),
    Case("short", "Goodbye."),
    Case("short", "Yes."),
    Case("short", "No problem."),
    # --- original repro ---
    Case("repro", "I'm here! Could you please confirm if you own your home?"),
    # --- questions / IVR ---
    Case("ivr", "Could you please confirm if you own your home?"),
    Case("ivr", "What is the best phone number to reach you at?"),
    Case("ivr", "Are you calling about an existing claim or a new one?"),
    Case("ivr", "Would you like me to send that confirmation by email or text message?"),
    Case("ivr", "Can you verify the last four digits of your social security number?"),
    # --- statements ---
    Case("statement", "Your appointment is confirmed for tomorrow afternoon."),
    Case("statement", "I have successfully updated your account information."),
    Case("statement", "Please hold while I transfer you to a specialist."),
    Case("statement", "We received your payment of one hundred twenty dollars."),
    # --- numbers / dates / amounts ---
    Case("numbers", "Your confirmation number is 4 8 2 9 1 7."),
    Case("numbers", "Please call us back at 5 5 5 0 1 2 3."),
    Case("numbers", "The total amount due is $1,247.56."),
    Case("numbers", "Your flight departs on March 15th at 6:45 PM."),
    Case("numbers", "The reference ID is ABC-2094-XZ."),
    # --- punctuation / emphasis ---
    Case("punct", "Wait — are you sure that's correct?!"),
    Case("punct", "Okay... let me check that for you."),
    Case("punct", "Great! You're all set."),
    Case("punct", "Hmm, I don't see that on my end."),
    # --- multi-sentence ---
    Case(
        "multi",
        "Thanks for calling. I can help with that. First, I'll need your account number.",
    ),
    Case(
        "multi",
        "I'm sorry for the inconvenience. Let me look into this right away. It should only take a moment.",
    ),
    Case(
        "multi",
        "Your order has shipped. You should receive it by Friday. Is there anything else I can help you with today?",
    ),
    # --- longer (still IVR-length) ---
    Case(
        "long",
        "Before we continue, please note that this call may be recorded for quality assurance and training purposes. "
        "If you do not wish to be recorded, you may hang up now.",
    ),
    Case(
        "long",
        "I understand this has been frustrating. I've noted the issue on your account and a supervisor will "
        "review it within one business day. You'll receive an update by email.",
    ),
    # --- edge / tricky ---
    Case("edge", "UM... uh, hold on a second."),
    Case("edge", "Dr. Smith's office will call you back."),
    Case("edge", "The Wi-Fi password is case-sensitive."),
    Case("edge", "Say 'cancel' at any time to return to the main menu."),
    Case("edge", "Email us at support@example.com for help."),
]


def expected_sec(text: str) -> float:
    words = max(1, len(text.split()))
    return max(MIN_EXPECTED, words / WORDS_PER_SEC)


def classify(audio_sec: float, text: str) -> str:
    if audio_sec <= 0:
        return "error"
    if audio_sec >= CAP_SEC:
        return "FAIL"
    exp = expected_sec(text)
    if audio_sec >= exp * FAIL_MULT or audio_sec >= 25.0:
        return "FAIL"
    if audio_sec >= exp * WARN_MULT or audio_sec >= 15.0:
        return "WARN"
    return "ok"


def synth(url: str, voice: str, text: str, timeout: float = 300.0) -> dict:
    t0 = time.perf_counter()
    pcm = bytearray()
    rid = None
    err = None
    try:
        with requests.post(
            f"{url.rstrip('/')}/stream",
            json={
                "text": text,
                "voice_id": voice,
                "x_vector_only_mode": True,
                "sample_rate": 16000,
                "format": "pcm",
            },
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            event = None
            lines: list[str] = []
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                if raw.startswith("event:"):
                    event = raw[6:].strip()
                    lines = []
                elif raw.startswith("data:"):
                    lines.append(raw[5:].lstrip())
                elif raw == "" and event and lines:
                    payload = json.loads("\n".join(lines))
                    if event == "start":
                        rid = payload.get("request_id")
                    elif event == "audio":
                        pcm.extend(base64.b64decode(payload["data"]))
                    elif event == "error":
                        err = payload.get("detail") or str(payload)
                    event = None
                    lines = []
    except Exception as e:  # noqa: BLE001
        err = str(e)

    audio = len(pcm) / (2 * 16000) if pcm else 0.0
    status = "error" if err else classify(audio, text)
    return {
        "request_id": rid,
        "audio_sec": round(audio, 3),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "expected_sec": round(expected_sec(text), 3),
        "status": status,
        "error": err,
        "hit_cap": audio >= CAP_SEC,
    }


def parse_reasons(since_iso: str) -> dict[str, str]:
    try:
        out = subprocess.check_output(
            ["docker", "logs", "--since", since_iso, "qwen3-tts"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return {}
    found: dict[str, str] = {}
    pat = re.compile(r"Request ([0-9a-f-]+) completed: \{[^}]*'reason': '([^']+)'")
    for line in out.splitlines():
        m = pat.search(line)
        if m:
            found[m.group(1)] = m.group(2)
    return found


def attach_reasons(rows: list[dict], since_iso: str) -> None:
    time.sleep(1.0)
    reasons = parse_reasons(since_iso)
    for r in rows:
        r["finish_reason"] = reasons.get(r.get("request_id") or "")


def summarize(rows: list[dict], label: str) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    warn = [r for r in rows if r["status"] == "WARN"]
    fail = [r for r in rows if r["status"] == "FAIL"]
    errs = [r for r in rows if r["status"] == "error"]
    durs = [r["audio_sec"] for r in rows if r["audio_sec"] > 0]
    walls = [r["elapsed_sec"] for r in rows]
    stop = sum(1 for r in rows if r.get("finish_reason") == "stop_id_encountered")
    maxed = sum(1 for r in rows if r.get("finish_reason") == "max_tokens_reached")
    n = len(rows) or 1
    by_cat: dict[str, dict] = {}
    for r in rows:
        cat = r.get("category", "?")
        bucket = by_cat.setdefault(cat, {"n": 0, "ok": 0, "warn": 0, "fail": 0, "error": 0, "maxed": 0})
        bucket["n"] += 1
        bucket[r["status"]] = bucket.get(r["status"], 0) + 1
        if r.get("finish_reason") == "max_tokens_reached" or r.get("hit_cap"):
            bucket["maxed"] += 1
    return {
        "label": label,
        "n": len(rows),
        "ok": len(ok),
        "warn": len(warn),
        "fail": len(fail),
        "error": len(errs),
        "eos_rate": round(stop / n, 3),
        "max_tokens_rate": round(maxed / n, 3),
        "stop_id_count": stop,
        "max_tokens_count": maxed,
        "audio_sec_min": min(durs) if durs else None,
        "audio_sec_max": max(durs) if durs else None,
        "audio_sec_mean": round(statistics.mean(durs), 3) if durs else None,
        "wall_sec_mean": round(statistics.mean(walls), 3) if walls else None,
        "wall_sec_p95": round(sorted(walls)[int(0.95 * (len(walls) - 1))], 3) if walls else None,
        "by_category": by_cat,
    }


def run_sequential(url: str, voice: str, cases: list[Case], repeats: int) -> list[dict]:
    rows = []
    total = len(cases) * repeats
    i = 0
    for rep in range(repeats):
        for case in cases:
            i += 1
            r = synth(url, voice, case.text)
            r.update({"idx": i, "category": case.category, "text": case.text, "voice": voice, "rep": rep + 1})
            rows.append(r)
            flag = r["status"]
            print(
                f"  [seq {i:03d}/{total}] {flag:5} cat={case.category:10} "
                f"audio={r['audio_sec']:6.2f}s exp~{r['expected_sec']:4.1f}s "
                f"wall={r['elapsed_sec']:6.2f}s  {case.text[:56]!r}"
            )
    return rows


def run_concurrent(url: str, voice: str, jobs: list[tuple[str, str, str]], conc: int) -> list[dict]:
    """jobs: list of (category, text, voice)"""
    rows: list[dict] = []

    def one(item):
        idx, category, text, v = item
        r = synth(url, v, text)
        r.update({"idx": idx, "category": category, "text": text, "voice": v})
        return r

    items = [(i + 1, c, t, v) for i, (c, t, v) in enumerate(jobs)]
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(one, it) for it in items]
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            print(
                f"  [par {r['idx']:03d}] {r['status']:5} cat={r['category']:10} "
                f"voice={r['voice']:10} audio={r['audio_sec']:6.2f}s wall={r['elapsed_sec']:6.2f}s"
            )
    rows.sort(key=lambda x: x["idx"])
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:2200")
    p.add_argument("--voice", default="laura")
    p.add_argument("--seq-repeats", type=int, default=2, help="repeats of full case list sequentially")
    p.add_argument("--conc", type=int, default=40, help="concurrency for parallel phases")
    p.add_argument("--burst", type=int, default=50, help="size of mixed concurrent burst")
    p.add_argument("--repro-burst", type=int, default=40, help="concurrent copies of original repro line")
    p.add_argument("--multi-voice", action="store_true", default=True)
    p.add_argument("--out-dir", default=str(ROOT / "benchmark" / "hallucination_out"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # health + warmup
    requests.get(f"{args.url}/health", timeout=5).raise_for_status()
    voices = requests.get(f"{args.url}/voices", timeout=5).json().get("voices") or [args.voice]
    print(f"voices={voices}")
    print("warmup...")
    w = synth(args.url, args.voice, "Warmup check.")
    print(f"warmup -> {w['status']} audio={w['audio_sec']}s wall={w['elapsed_sec']}s\n")

    report: dict = {
        "params_note": "server sampling from live entrypoint (expect official)",
        "text_cases": len(CASES),
        "phases": [],
    }

    # Phase 1: sequential coverage
    print("=" * 72)
    print(f"PHASE 1: sequential  cases={len(CASES)} × repeats={args.seq_repeats}  voice={args.voice}")
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    t0 = time.perf_counter()
    seq_rows = run_sequential(args.url, args.voice, CASES, args.seq_repeats)
    attach_reasons(seq_rows, since)
    seq_sum = summarize(seq_rows, "sequential_coverage")
    seq_sum["wall_total_sec"] = round(time.perf_counter() - t0, 2)
    report["phases"].append({"summary": seq_sum, "results": seq_rows})
    print(
        f"  >> ok={seq_sum['ok']} warn={seq_sum['warn']} fail={seq_sum['fail']} err={seq_sum['error']} "
        f"eos={seq_sum['eos_rate']:.0%} maxed={seq_sum['max_tokens_rate']:.0%}\n"
    )

    # Phase 2: mixed concurrent burst (cycle cases)
    print("=" * 72)
    print(f"PHASE 2: mixed concurrent burst  n={args.burst} conc={args.conc}")
    jobs = []
    for i in range(args.burst):
        case = CASES[i % len(CASES)]
        jobs.append((case.category, case.text, args.voice))
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    t0 = time.perf_counter()
    mix_rows = run_concurrent(args.url, args.voice, jobs, args.conc)
    attach_reasons(mix_rows, since)
    mix_sum = summarize(mix_rows, "mixed_concurrent")
    mix_sum["wall_total_sec"] = round(time.perf_counter() - t0, 2)
    report["phases"].append({"summary": mix_sum, "results": mix_rows})
    print(
        f"  >> ok={mix_sum['ok']} warn={mix_sum['warn']} fail={mix_sum['fail']} err={mix_sum['error']} "
        f"eos={mix_sum['eos_rate']:.0%} maxed={mix_sum['max_tokens_rate']:.0%} "
        f"batch_wall={mix_sum['wall_total_sec']}s\n"
    )

    # Phase 3: repro-line concurrent hammer
    repro = "I'm here! Could you please confirm if you own your home?"
    print("=" * 72)
    print(f"PHASE 3: repro concurrent  n={args.repro_burst} conc={args.conc}")
    jobs = [("repro", repro, args.voice)] * args.repro_burst
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    t0 = time.perf_counter()
    repro_rows = run_concurrent(args.url, args.voice, jobs, args.conc)
    attach_reasons(repro_rows, since)
    repro_sum = summarize(repro_rows, "repro_concurrent")
    repro_sum["wall_total_sec"] = round(time.perf_counter() - t0, 2)
    report["phases"].append({"summary": repro_sum, "results": repro_rows})
    print(
        f"  >> ok={repro_sum['ok']} warn={repro_sum['warn']} fail={repro_sum['fail']} err={repro_sum['error']} "
        f"eos={repro_sum['eos_rate']:.0%} maxed={repro_sum['max_tokens_rate']:.0%}\n"
    )

    # Phase 4: multi-voice concurrent (same texts rotating voices)
    if args.multi_voice and len(voices) > 1:
        print("=" * 72)
        n_mv = min(48, len(CASES) * 2)
        print(f"PHASE 4: multi-voice concurrent  n={n_mv} voices={voices}")
        jobs = []
        for i in range(n_mv):
            case = CASES[i % len(CASES)]
            v = voices[i % len(voices)]
            jobs.append((case.category, case.text, v))
        since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        t0 = time.perf_counter()
        mv_rows = run_concurrent(args.url, args.voice, jobs, args.conc)
        attach_reasons(mv_rows, since)
        mv_sum = summarize(mv_rows, "multi_voice_concurrent")
        mv_sum["wall_total_sec"] = round(time.perf_counter() - t0, 2)
        # per-voice
        per_v: dict[str, dict] = {}
        for r in mv_rows:
            b = per_v.setdefault(r["voice"], {"n": 0, "ok": 0, "fail": 0, "warn": 0, "error": 0})
            b["n"] += 1
            b[r["status"]] = b.get(r["status"], 0) + 1
        mv_sum["by_voice"] = per_v
        report["phases"].append({"summary": mv_sum, "results": mv_rows})
        print(
            f"  >> ok={mv_sum['ok']} warn={mv_sum['warn']} fail={mv_sum['fail']} err={mv_sum['error']} "
            f"eos={mv_sum['eos_rate']:.0%} maxed={mv_sum['max_tokens_rate']:.0%}"
        )
        print(f"  >> by_voice={per_v}\n")

    # Overall
    all_rows = []
    for ph in report["phases"]:
        all_rows.extend(ph["results"])
    overall = summarize(all_rows, "OVERALL")
    report["overall"] = overall

    # Worst offenders
    bad = [r for r in all_rows if r["status"] in ("FAIL", "WARN", "error")]
    bad.sort(key=lambda r: r["audio_sec"], reverse=True)
    report["worst"] = bad[:20]

    out = out_dir / f"full_stress_{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=2))

    print("=" * 72)
    print("OVERALL")
    print(
        f"  n={overall['n']}  ok={overall['ok']} warn={overall['warn']} "
        f"fail={overall['fail']} error={overall['error']}"
    )
    print(
        f"  eos={overall['eos_rate']:.0%}  maxed={overall['max_tokens_rate']:.0%}  "
        f"audio mean/max={overall['audio_sec_mean']}/{overall['audio_sec_max']}s"
    )
    print("  by_category:")
    for cat, b in sorted(overall["by_category"].items()):
        print(
            f"    {cat:10} n={b['n']:3} ok={b.get('ok',0):3} warn={b.get('warn',0):2} "
            f"fail={b.get('fail',0):2} err={b.get('error',0):2} maxed~{b.get('maxed',0)}"
        )
    if bad:
        print("\n  worst cases:")
        for r in bad[:10]:
            print(
                f"    {r['status']:5} audio={r['audio_sec']:6.2f}s cat={r.get('category')} "
                f"voice={r.get('voice')} reason={r.get('finish_reason')} text={r.get('text','')[:70]!r}"
            )
    print(f"\nwrote {out}")
    return 1 if overall["fail"] or overall["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
