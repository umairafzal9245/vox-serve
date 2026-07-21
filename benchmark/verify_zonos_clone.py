#!/usr/bin/env python3
"""Verify Zonos voice cloning by comparing speaker embeddings.

Loads the same speaker encoder Zonos uses, embeds each reference and each
generated output, and reports cosine similarity. A working clone should score
much higher against its own reference than the default-voice output does.
"""
import struct
import sys

import torch
import torch.nn.functional as F
import torchaudio

from vox_serve.encoder.zonos import ZonosSpeakerEmbeddingLDA


def load_wav(path):
    """Load a wav; fall back to raw 24kHz mono int16 PCM if the header is bad."""
    try:
        wav, sr = torchaudio.load(path)
        if wav.numel() > 0:
            return wav, sr
    except Exception:
        pass
    with open(path, "rb") as f:
        data = f.read()
    pcm = data[44:]
    n = len(pcm) // 2
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    wav = torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0
    return wav, 24000


def main():
    device = "cuda:0"
    enc = ZonosSpeakerEmbeddingLDA(device=device).eval()

    files = {
        "REF ref_voice": "/tmp/ref_voice.wav",
        "REF default_test": "/tmp/default_test.wav",
        "OUT default(no-clone)": "/tmp/zonos_default.wav",
        "OUT clone<-ref_voice": "/tmp/zonos_clone_ref_voice.wav",
        "OUT clone<-default_test": "/tmp/zonos_clone_default_test.wav",
    }

    embs = {}
    for name, path in files.items():
        wav, sr = load_wav(path)
        with torch.no_grad():
            _, e = enc(wav.to(device), sr)
        embs[name] = F.normalize(e.reshape(1, -1).float(), dim=-1)
        print(f"{name:28s} frames={wav.shape[-1]:7d} sr={sr}")

    def cos(a, b):
        return float((embs[a] * embs[b]).sum())

    print("\nCosine similarity (higher = more similar voice):")
    pairs = [
        ("clone<-ref_voice vs its REF", "OUT clone<-ref_voice", "REF ref_voice"),
        ("default    vs REF ref_voice", "OUT default(no-clone)", "REF ref_voice"),
        ("clone<-default_test vs its REF", "OUT clone<-default_test", "REF default_test"),
        ("default    vs REF default_test", "OUT default(no-clone)", "REF default_test"),
    ]
    for label, a, b in pairs:
        print(f"  {label:34s} = {cos(a, b):+.3f}")


if __name__ == "__main__":
    main()
