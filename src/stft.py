"""Torch STFT front end.

The constants and the sample->frame mapping live in `framing.py`, which has no
PyTorch dependency, and are re-exported here so existing imports keep working.
Anything that only needs the framing numbers - the ONNX inference path, the
classical baselines - should import from `src.framing` instead, so it stays
runnable on a machine without PyTorch.

These constants must match `src/models/gtcrn.py` exactly or the pretrained
weights are meaningless.
"""
from __future__ import annotations

import torch

from .framing import (HOP, N_FFT, SR, WIN, n_frames, np_window,
                      samples_to_frame_mask)

__all__ = ["SR", "N_FFT", "HOP", "WIN", "n_frames", "samples_to_frame_mask",
           "np_window", "window", "stft", "istft"]

_WINDOW_CACHE: dict = {}


def window(device=None, dtype=torch.float32) -> torch.Tensor:
    key = (str(device), dtype)
    if key not in _WINDOW_CACHE:
        _WINDOW_CACHE[key] = torch.hann_window(WIN, device=device, dtype=dtype).pow(0.5)
    return _WINDOW_CACHE[key]


def stft(x: torch.Tensor) -> torch.Tensor:
    """(..., n) waveform -> (..., 257, T, 2) real/imag, GTCRN's expected input."""
    spec = torch.stft(x, N_FFT, HOP, WIN, window(x.device), return_complex=True)
    return torch.view_as_real(spec)


def istft(spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
    """(..., 257, T, 2) -> (..., n) waveform."""
    return torch.istft(
        torch.view_as_complex(spec.contiguous()),
        N_FFT, HOP, WIN, window(spec.device), length=length,
    )
