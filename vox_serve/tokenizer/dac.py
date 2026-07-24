import math

import torch
import torchaudio
from transformers.models.dac import DacModel


class DAC:
    def __init__(self, enable_torch_compile: bool = False):
        super().__init__()
        self.dac = DacModel.from_pretrained("descript/dac_44khz")
        self.dac.eval().requires_grad_(False)
        if enable_torch_compile:
            self.dac.decode = torch.compile(self.dac.decode, mode="max-autotune-no-cudagraphs")
        self.codebook_size = self.dac.config.codebook_size
        self.num_codebooks = self.dac.quantizer.n_codebooks
        self.sampling_rate = self.dac.config.sampling_rate

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        wav = torchaudio.functional.resample(wav, sr, 44_100)
        right_pad = math.ceil(wav.shape[-1] / 512) * 512 - wav.shape[-1]
        return torch.nn.functional.pad(wav, (0, right_pad))

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.dac.encode(wav).audio_codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # When the DAC runs in fp32 (for quality), decode in full fp32 — an fp16
        # autocast would defeat the point by downcasting the conv/matmul ops. When
        # the DAC is in a reduced dtype, keep the fp16 autocast as before.
        param_dtype = next(self.dac.parameters()).dtype
        use_autocast = self.dac.device.type != "cpu" and param_dtype != torch.float32
        with torch.autocast(self.dac.device.type, torch.float16, enabled=use_autocast):
            return self.dac.decode(audio_codes=codes).audio_values.unsqueeze(1).float()
