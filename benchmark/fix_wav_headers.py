#!/usr/bin/env python3
"""Rewrite streamed WAV files with a correct RIFF/data size so players work.

Streaming responses emit a WAV header whose data-size field is 0, followed by
raw PCM. This reads the PCM (skipping the 44-byte header) and writes a proper
24 kHz mono 16-bit WAV in place next to the original as <name>_fixed.wav.
"""

import glob
import os
import wave


def main():
    for path in sorted(glob.glob("samples/*.wav")):
        if path.endswith("_fixed.wav"):
            continue
        with open(path, "rb") as f:
            data = f.read()
        pcm = data[44:]
        pcm = pcm[: len(pcm) - (len(pcm) % 2)]
        out = path.replace(".wav", "_fixed.wav")
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm)
        dur = len(pcm) / 2 / 24000
        print(f"wrote {os.path.basename(out)}  ({dur:.2f}s, {len(pcm)} bytes PCM)")


if __name__ == "__main__":
    main()
