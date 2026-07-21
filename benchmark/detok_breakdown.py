#!/usr/bin/env python3
"""Break down Chatterbox detokenizer cost: flow (CFM estimator) vs HiFT vocoder,
and CFM timestep scaling, under bf16 autocast."""

import time

import torch

from vox_serve.model.chatterbox import ChatterboxModel
from vox_serve.tokenizer import chatterbox as ct


def bench(fn, iters=5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = ChatterboxModel("chatterbox")
    dec = model.audio_decoder
    interval = model.detokenize_interval
    ref = model.default_conds.gen
    print(f"prompt_token_len={ref['prompt_token_len']}, prompt_feat={ref['prompt_feat'].shape}")

    for bs in [8, 32]:
        tokens = torch.randint(0, 6561, (bs, interval), device="cuda")

        def flow_only():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sp = torch.concat([ref["prompt_token"].repeat(tokens.size(0), 1), tokens], dim=1)
                lens = torch.ones(sp.size(0), device=sp.device, dtype=torch.long) * (
                    ref["prompt_token_len"] + interval
                )
                mels, _ = dec.flow(
                    token=sp, token_len=lens,
                    prompt_feat=ref["prompt_feat"], prompt_feat_len=ref["prompt_feat_len"],
                    embedding=ref["embedding"], streaming=True, finalize=False,
                )
            return mels

        mels = flow_only()
        mels_c = mels[:, :, ref["prompt_feat_len"]:] if ref["prompt_feat_len"] else mels

        def vocoder_only():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cache_source = torch.zeros(1, 1, 0, device="cuda")
                return dec.mel2wav.forward(mels_c, cache_source)

        flow_only()
        vocoder_only()
        t_flow = bench(flow_only)
        t_voc = bench(vocoder_only)
        print(f"bs={bs}: flow={t_flow:.1f} ms  vocoder={t_voc:.1f} ms  (mel shape {mels_c.shape})")

    # CFM timestep scaling on full postprocess
    cfm = dec.flow.decoder  # CausalConditionalCFM

    orig_forward = cfm.forward

    for steps in [10, 6, 4]:
        def patched(mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming=False, _s=steps):
            return orig_forward(mu, mask, _s, temperature, spks, cond, streaming)

        cfm.forward = patched
        tokens = torch.randint(0, 6561, (32, interval, 1), device="cuda")

        def run():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.postprocess(tokens)

        run()
        t = bench(run)
        out = run()
        print(f"cfm_steps={steps}: postprocess bs=32 = {t:.1f} ms  rms={out.float().pow(2).mean().sqrt():.4f} "
              f"nan={torch.isnan(out).any().item()}")

    cfm.forward = orig_forward


if __name__ == "__main__":
    main()
