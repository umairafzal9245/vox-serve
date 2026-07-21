#!/usr/bin/env python3
"""Objective sanity/quality metrics for generated WAV samples."""

import glob
import wave

import numpy as np


def load(path):
    # Streaming responses write a WAV header with a zero data-size field, so
    # wave.getnframes() returns 0. Read the raw bytes and skip the 44-byte header.
    with open(path, "rb") as f:
        data = f.read()
    sr = 24000
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
    except Exception:
        pass
    pcm = data[44:]
    pcm = pcm[: len(pcm) - (len(pcm) % 2)]
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def spectral_centroid(x, sr):
    # crude average spectral centroid over the whole clip
    if len(x) < 1024:
        return 0.0
    f = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return float((freqs * f).sum() / (f.sum() + 1e-9))


for path in sorted(glob.glob("samples/*.wav")):
    x, sr = load(path)
    dur = len(x) / sr
    rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    peak = float(np.abs(x).max()) if len(x) else 0.0
    clip = float((np.abs(x) > 0.999).mean() * 100)
    sc = spectral_centroid(x, sr)
    print(f"{path.split('/')[-1]:32s} dur={dur:5.2f}s sr={sr} rms={rms:.4f} "
          f"peak={peak:.3f} clip%={clip:.3f} centroid={sc:6.0f}Hz")
