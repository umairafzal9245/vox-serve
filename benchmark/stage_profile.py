#!/usr/bin/env python3
"""
Per-stage latency profiler for the Chatterbox pipeline in vox-serve.

Drives CudaGraphWorker directly (no server) and times each stage:
  1. model.preprocess()            - per request, runs serialized on scheduler thread
  2. prepare_lm_inputs + prefill   - batch of B_prefill requests
  3. decode step                   - CUDA graph decode at batch 32
  4. sampling task (CPU coroutine) - part of decode step in sync mode
  5. run_detokenize                - S3Gen postprocess for N chunks

Usage:
    python benchmark/stage_profile.py
"""

import argparse
import asyncio
import time

import torch

from vox_serve.requests import Request
from vox_serve.worker import CudaGraphWorker

TEXT = "Hello world, this is a test of streaming text to speech latency."


def sync():
    torch.cuda.synchronize()


def make_requests(n, offset=0):
    return [
        Request(request_id=f"prof_{offset + i}", prompt=TEXT, is_streaming=True)
        for i in range(n)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chatterbox")
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--max-num-pages", type=int, default=2048)
    args = ap.parse_args()

    max_batch = args.max_batch_size
    print(f"Initializing CudaGraphWorker for model={args.model} (captures CUDA graphs, ~1 min)...")
    t0 = time.perf_counter()
    worker = CudaGraphWorker(
        model_name=args.model,
        max_batch_size=max_batch,
        max_num_pages=args.max_num_pages,
        page_size=128,
    )
    print(f"worker init: {time.perf_counter() - t0:.1f}s")
    print(f"detokenize_interval={worker.detokenize_interval} overlap={worker.detokenize_overlap}")
    print(f"prefill_graph_batch_size={worker.prefill_graph_batch_size}")

    # ---- Stage 1: preprocess ----
    # warmup
    for _ in range(3):
        worker.model.preprocess(prompt=TEXT)
    sync()
    t0 = time.perf_counter()
    n_pre = 32
    for _ in range(n_pre):
        worker.model.preprocess(prompt=TEXT)
    sync()
    pre_ms = (time.perf_counter() - t0) / n_pre * 1000
    print(f"\n[1] preprocess: {pre_ms:.2f} ms/request  ({n_pre} serialized = {pre_ms * n_pre:.0f} ms)")

    # ---- Stage 2: prefill (batch of 8) ----
    def do_prefill(reqs):
        lm_inputs = worker.prepare_lm_inputs(reqs, [])
        assert lm_inputs["is_prefill"]
        task = worker.run_lm_prefill(reqs, lm_inputs)
        if task is not None:
            asyncio.run(task)
        sync()

    # warmup prefill wave and free it
    warm = make_requests(8, offset=1000)
    t0 = time.perf_counter()
    do_prefill(warm)
    print(f"[2] prefill bs=8 (warmup incl. tokenizer JIT): {(time.perf_counter() - t0) * 1000:.1f} ms")
    for r in warm:
        worker.free_kv_cache(r)

    all_reqs = make_requests(max_batch)
    prefill_times = []
    for wave in range(4):
        batch = all_reqs[wave * 8 : (wave + 1) * 8]
        sync()
        t0 = time.perf_counter()
        do_prefill(batch)
        prefill_times.append((time.perf_counter() - t0) * 1000)
    print(f"[2] prefill bs=8 waves: {[f'{t:.1f}' for t in prefill_times]} ms "
          f"(input_length={all_reqs[0].input_length})")

    # ---- Stage 3+4: decode steps at batch 32 ----
    def decode_step(reqs):
        lm_inputs = worker.prepare_lm_inputs(reqs, [])
        assert not lm_inputs["is_prefill"]
        task = worker.run_lm_decode(reqs, lm_inputs)
        if task is not None:
            asyncio.run(task)

    # warmup decode
    for _ in range(3):
        decode_step(all_reqs)
    sync()

    n_steps = 25
    t0 = time.perf_counter()
    for _ in range(n_steps):
        decode_step(all_reqs)
    sync()
    dec_ms = (time.perf_counter() - t0) / n_steps * 1000
    interval = worker.detokenize_interval
    print(f"[3] decode step bs=32 (incl. sampling, sync): {dec_ms:.2f} ms/step "
          f"-> {interval} steps (=1st chunk) = {dec_ms * interval:.0f} ms")

    # breakdown: forward only vs sampling
    lm_inputs = worker.prepare_lm_inputs(all_reqs, [])
    sync()
    t0 = time.perf_counter()
    for _ in range(10):
        lm_inputs2 = worker.prepare_lm_inputs(all_reqs, [])
    prep_ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"    prepare_lm_inputs bs=32: {prep_ms:.2f} ms")

    # ---- Stage 5: detokenize ----
    # Ensure requests have >= interval audio tokens
    while min(len(r.lm_output_audio_tokens) for r in all_reqs) < worker.detokenize_interval + 5:
        decode_step(all_reqs)
    sync()

    for bs in [1, 8, 16, 32]:
        reqs = all_reqs[:bs]
        for r in reqs:
            r.audio_decode_idx = [0]
        # warmup
        worker.run_detokenize(reqs)
        sync()
        t0 = time.perf_counter()
        for _ in range(5):
            worker.run_detokenize(reqs)
            sync()
        det_ms = (time.perf_counter() - t0) / 5 * 1000
        print(f"[5] detokenize chunks={bs}: {det_ms:.2f} ms")

    # ---- Summary model ----
    print("\n--- TTFA model (32 concurrent, worst request) ---")
    total_pre = pre_ms * 32
    total_prefill = sum(prefill_times)
    dec_total = dec_ms * interval
    print(f"preprocess 32x serialized:      {total_pre:.0f} ms")
    print(f"prefill waves of 8:             {total_prefill:.0f} ms")
    print(f"{interval} decode steps bs32 (1st chunk): {dec_total:.0f} ms")
    print("+ detokenize of first chunk + IPC/HTTP overhead")


if __name__ == "__main__":
    main()
