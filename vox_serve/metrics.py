"""Persistent server metrics: active requests, latency, per-request GPU memory.

Writes under ``VOX_METRICS_DIR`` (default ``/var/log/vox-serve``):
  - ``requests.jsonl``  — one JSON line per completed request
  - ``stats.json``      — latest periodic snapshot (overwritten)
  - ``stats.log``       — human-readable rolling log of snapshots

All timestamps are ISO-8601 UTC (``...Z``) for production log shipping.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(dt: Optional[datetime] = None) -> str:
    """Return ISO-8601 UTC timestamp with millisecond precision, e.g. 2026-07-27T15:38:14.123Z"""
    d = dt or utc_now()
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


class MetricsTracker:
    def __init__(
        self,
        metrics_dir: Optional[str] = None,
        bytes_per_kv_page: int = 0,
        max_num_pages: int = 0,
        page_size: int = 128,
        interval_sec: float = 5.0,
        model_name: Optional[str] = None,
    ):
        self.metrics_dir = Path(
            metrics_dir or os.environ.get("VOX_METRICS_DIR", "/var/log/vox-serve")
        )
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.requests_path = self.metrics_dir / "requests.jsonl"
        self.stats_json_path = self.metrics_dir / "stats.json"
        self.stats_log_path = self.metrics_dir / "stats.log"

        self.bytes_per_kv_page = bytes_per_kv_page
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.interval_sec = interval_sec
        self.model_name = model_name or os.environ.get("VOX_MODEL_NAME", "unknown")
        self.hostname = socket.gethostname()
        self._started_at = utc_iso()
        self._last_dump = 0.0
        self._lock = threading.Lock()
        self._completed_count = 0
        self._ttfa_ms_recent: List[float] = []
        self._total_ms_recent: List[float] = []
        self._max_recent = 200

        # Bootstrap empty snapshot so /stats works before first dump
        if not self.stats_json_path.exists():
            self.maybe_dump_snapshot(
                active_requests=[],
                free_pages=max_num_pages,
                gpu_alloc_gib=None,
                gpu_reserved_gib=None,
                force=True,
            )

    def kv_bytes_for_pages(self, n_pages: int) -> int:
        return int(n_pages) * int(self.bytes_per_kv_page)

    def kv_mib_for_pages(self, n_pages: int) -> float:
        return self.kv_bytes_for_pages(n_pages) / (1024 * 1024)

    def record_complete(self, record: Dict[str, Any]) -> None:
        now = utc_now()
        record.setdefault("ts", utc_iso(now))
        record.setdefault("ts_unix", round(now.timestamp(), 3))
        record.setdefault("hostname", self.hostname)
        record.setdefault("model", self.model_name)
        with self._lock:
            self._completed_count += 1
            if record.get("ttfa_ms") is not None:
                self._ttfa_ms_recent.append(float(record["ttfa_ms"]))
                self._ttfa_ms_recent = self._ttfa_ms_recent[-self._max_recent :]
            if record.get("total_ms") is not None:
                self._total_ms_recent.append(float(record["total_ms"]))
                self._total_ms_recent = self._total_ms_recent[-self._max_recent :]
            with open(self.requests_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _pct(values: List[float], q: float) -> Optional[float]:
        if not values:
            return None
        s = sorted(values)
        idx = min(len(s) - 1, max(0, int(q * len(s))))
        return round(s[idx], 1)

    def maybe_dump_snapshot(
        self,
        *,
        active_requests: list,
        free_pages: int,
        gpu_alloc_gib: Optional[float],
        gpu_reserved_gib: Optional[float],
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        now_mono = time.time()
        if not force and (now_mono - self._last_dump) < self.interval_sec:
            return None
        self._last_dump = now_mono

        now = utc_now()
        active = []
        used_pages = 0
        for req in active_requests:
            n_pages = len(req.kv_pages) if getattr(req, "kv_pages", None) else 0
            used_pages += n_pages
            admit = getattr(req, "admit_time", None) or now_mono
            age_ms = (now_mono - admit) * 1000.0
            ttfa = None
            if getattr(req, "first_audio_time", None) is not None:
                ttfa = (req.first_audio_time - admit) * 1000.0
            n_tok = len(req.lm_output_audio_tokens) if req.lm_output_audio_tokens else 0
            active.append(
                {
                    "request_id": req.request_id,
                    "age_ms": round(age_ms, 1),
                    "ttfa_ms": round(ttfa, 1) if ttfa is not None else None,
                    "prefill_done": bool(req.done_lm_prefill),
                    "lm_done": bool(req.done_lm_generation),
                    "pressing": bool(req.is_pressing),
                    "tokens": n_tok,
                    "kv_pages": n_pages,
                    "kv_mib": round(self.kv_mib_for_pages(n_pages), 2),
                    "chunks_sent": len(getattr(req, "chunk_send_timestamps", []) or []),
                    "voice": (getattr(req, "model_kwargs", None) or {}).get("voice_id"),
                }
            )

        with self._lock:
            snap = {
                "ts": utc_iso(now),
                "ts_unix": round(now.timestamp(), 3),
                "hostname": self.hostname,
                "model": self.model_name,
                "started_at": self._started_at,
                "active_count": len(active_requests),
                "completed_count": self._completed_count,
                "pages": {
                    "used": used_pages,
                    "free": free_pages,
                    "max": self.max_num_pages,
                    "page_size": self.page_size,
                    "bytes_per_page": self.bytes_per_kv_page,
                    "used_mib": round(self.kv_mib_for_pages(used_pages), 2),
                    "free_mib": round(self.kv_mib_for_pages(free_pages), 2),
                },
                "gpu": {
                    "allocated_gib": gpu_alloc_gib,
                    "reserved_gib": gpu_reserved_gib,
                },
                "latency_recent": {
                    "n": len(self._ttfa_ms_recent),
                    "ttfa_p50_ms": self._pct(self._ttfa_ms_recent, 0.50),
                    "ttfa_p90_ms": self._pct(self._ttfa_ms_recent, 0.90),
                    "ttfa_p99_ms": self._pct(self._ttfa_ms_recent, 0.99),
                    "total_p50_ms": self._pct(self._total_ms_recent, 0.50),
                    "total_p90_ms": self._pct(self._total_ms_recent, 0.90),
                    "total_p99_ms": self._pct(self._total_ms_recent, 0.99),
                },
                "active": active,
            }

        self.stats_json_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        line = (
            f"{snap['ts']} host={self.hostname} model={self.model_name} "
            f"active={snap['active_count']} completed={snap['completed_count']} "
            f"pages={used_pages}/{self.max_num_pages} "
            f"kv={snap['pages']['used_mib']:.1f}MiB "
            f"gpu_alloc={gpu_alloc_gib}GiB "
            f"ttfa_p50={snap['latency_recent']['ttfa_p50_ms']} "
            f"ttfa_p90={snap['latency_recent']['ttfa_p90_ms']} "
            f"total_p50={snap['latency_recent']['total_p50_ms']}"
        )
        with open(self.stats_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            for a in active:
                f.write(
                    f"  {a['request_id'][:8]} age={a['age_ms']:.0f}ms "
                    f"ttfa={a['ttfa_ms']} tok={a['tokens']} "
                    f"pages={a['kv_pages']} ({a['kv_mib']:.2f}MiB) "
                    f"voice={a.get('voice')} pressing={a['pressing']} lm_done={a['lm_done']}\n"
                )
        return snap
