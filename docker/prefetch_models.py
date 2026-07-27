#!/usr/bin/env python3
"""Prefetch Qwen3-TTS weights into HF_HOME during Docker image build.

If a host Hugging Face cache is provided as build context ``hfcache``
(mounted at /host-hf), copy matching repos from there. Otherwise download
from Hugging Face Hub (needs network; optional HF_TOKEN).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

HF_HOME = Path(os.environ.get("HF_HOME", "/data/hf-cache"))
HOST_HUB = Path("/host-hf/hub")
MODELS = [
    os.environ.get("MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
]


def hub_dirname(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def main() -> int:
    HF_HOME.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

    for repo_id in MODELS:
        name = hub_dirname(repo_id)
        src = HOST_HUB / name
        if src.is_dir() and any(src.iterdir()):
            dst = HF_HOME / "hub" / name
            print(f"[prefetch] copying {repo_id} from build-context cache → {dst}", flush=True)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, symlinks=True)
        else:
            print(f"[prefetch] downloading {repo_id} into {HF_HOME}", flush=True)
            snapshot_download(
                repo_id=repo_id,
                cache_dir=str(HF_HOME),
                token=token,
            )
        print(f"[prefetch] ready: {repo_id}", flush=True)

    print("[prefetch] all models ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
