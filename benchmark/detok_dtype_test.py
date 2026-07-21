#!/usr/bin/env python3
"""
Measure Chatterbox detokenizer (S3Gen) latency under fp32 / TF32 / bf16-autocast,
and validate output quality (SNR vs fp32 reference) before enabling in the server.
"""

import time

import torch

from vox_serve.model.chatterbox import ChatterboxModel


def bench(fn, iters=5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def snr_db(ref, test):
    noise = (ref - test).pow(2).mean()
    sig = ref.pow(2).mean()
    return 10 * torch.log10(sig / noise).item()


def main():
    print("Loading ChatterboxModel...")
    model = ChatterboxModel("chatterbox")
    interval = model.detokenize_interval

    torch.manual_seed(0)
    for bs in [8, 32]:
        tokens = torch.randint(0, 6561, (bs, interval, 1), device="cuda")

        # fp32 strict baseline
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for _ in range(2):
            ref = model.postprocess(tokens)
        t_fp32 = bench(lambda: model.postprocess(tokens))
        ref = model.postprocess(tokens).float()

        # TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        for _ in range(2):
            model.postprocess(tokens)
        t_tf32 = bench(lambda: model.postprocess(tokens))
        out_tf32 = model.postprocess(tokens).float()

        # bf16 autocast (+TF32)
        def run_bf16():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.postprocess(tokens)

        for _ in range(2):
            run_bf16()
        t_bf16 = bench(run_bf16)
        out_bf16 = run_bf16().float()

        print(f"\nbatch={bs}:")
        print(f"  fp32 strict : {t_fp32:8.1f} ms")
        print(f"  tf32        : {t_tf32:8.1f} ms  snr_vs_fp32={snr_db(ref, out_tf32):.1f} dB")
        print(f"  bf16 autocast: {t_bf16:8.1f} ms  snr_vs_fp32={snr_db(ref, out_bf16):.1f} dB")
        print(f"  audio rms fp32={ref.pow(2).mean().sqrt():.4f} "
              f"bf16={out_bf16.pow(2).mean().sqrt():.4f} "
              f"nan_bf16={torch.isnan(out_bf16).any().item()}")


if __name__ == "__main__":
    main()
