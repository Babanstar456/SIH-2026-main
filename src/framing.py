"""Framing constants and sample/frame mapping - NumPy only, no PyTorch.

Split out of `stft.py` deliberately. The deployed inference path (ONNX Runtime)
and the classical baselines need these constants but have no business pulling in
PyTorch: a teammate who only wants to run the shipped model should not need a
2 GB CUDA download, and on a machine where PyTorch cannot load at all - Windows
Smart App Control blocks its unsigned DLLs, for instance - the model must still
run.

`stft.py` re-exports everything here, so existing imports keep working.

These values are NOT free parameters: they must match `src/models/gtcrn.py`
exactly or the pretrained weights are meaningless.

    n_fft = 512  ->  32 ms analysis window
    hop   = 256  ->  16 ms, the chunk size agreed with the hardware team
    window = hann(512) ** 0.5   (sqrt-Hann, analysis AND synthesis)

The 32 ms window is also where the latency budget goes: algorithmic latency
equals the window length, so it is 32 ms before any compute. See src/bench.py.
"""
from __future__ import annotations

import numpy as np

SR = 16000
N_FFT = 512
HOP = 256
WIN = 512


# Streaming cache shapes, fixed by the GTCRN architecture (see
# third_party/gtcrn/stream/gtcrn_stream.py). Plain constants, kept here rather
# than in export_onnx.py so the inference path never imports PyTorch.
CONV_CACHE = (2, 1, 16, 16, 33)
TRA_CACHE = (2, 3, 1, 1, 16)
INTER_CACHE = (2, 1, 33, 16)


def zero_caches_np():
    """Fresh zeroed caches for the start of a stream."""
    return (np.zeros(CONV_CACHE, dtype="float32"),
            np.zeros(TRA_CACHE, dtype="float32"),
            np.zeros(INTER_CACHE, dtype="float32"))


def np_window() -> np.ndarray:
    """sqrt-Hann analysis/synthesis window, matching torch.hann_window(512)**0.5."""
    return (np.hanning(WIN + 1)[:WIN] ** 0.5).astype(np.float32)


def n_frames(n_samples: int) -> int:
    """Frame count torch.stft produces for n_samples with center=True."""
    return n_samples // HOP + 1


def samples_to_frame_mask(sample_mask, n_frame: int) -> np.ndarray:
    """Collapse a sample-level boolean mask to STFT frames.

    center=True means frame t is centred on sample t*hop and spans
    [t*hop - win/2, t*hop + win/2). A frame counts as active if ANY of its
    samples are. Getting this alignment wrong silently poisons the
    transient-weighted loss and the burst-local metrics, so it is unit-tested
    in tests/test_core.py.
    """
    sample_mask = np.asarray(sample_mask, dtype=bool)
    out = np.zeros(n_frame, dtype=bool)
    half = WIN // 2
    for t in range(n_frame):
        lo = max(0, t * HOP - half)
        hi = min(len(sample_mask), t * HOP + half)
        if lo < hi and sample_mask[lo:hi].any():
            out[t] = True
    return out
