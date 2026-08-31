"""Classical single-channel noise suppressors.

These exist to answer the question the report has to answer: how much does the
AI actually add over what already exists? They are real implementations run on
the identical audio - not numbers copied out of papers.

Both assume the noise is roughly STATIONARY, which is precisely why they are
expected to fail on gunfire. That expected failure is a load-bearing result, not
an embarrassment: if these do NOT collapse on the gunshot category, our test set
is too easy and needs rebuilding.
"""
from __future__ import annotations

import numpy as np

from ..framing import HOP, N_FFT, WIN

_EPS = 1e-10


def _analyse(x: np.ndarray):
    """Periodic Hann, 50% overlap.

    Padded by a full window at BOTH ends so every real sample - including the
    very first - is covered by the same number of overlapping frames. torch.stft
    centre-pads for the same reason; without this the baselines would take a
    32 ms taper at clip start that GTCRN does not, quietly handicapping them in
    the comparison table.
    """
    w = np.hanning(WIN + 1)[:WIN].astype(np.float64)
    xp = np.pad(np.asarray(x, dtype=np.float64), (WIN, WIN))
    n_frames = int(np.ceil(max(len(xp) - WIN, 0) / HOP)) + 1
    total = (n_frames - 1) * HOP + WIN
    xp = np.pad(xp, (0, max(0, total - len(xp))))
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = xp[idx] * w
    return np.fft.rfft(frames, N_FFT, axis=1), w, n_frames


def _synthesise(spec, w, n_frames, length):
    """Weighted overlap-add.

    The window-sum floor matters: at the first and last half-window only one
    frame contributes, so `norm` decays towards zero there. Dividing by it
    unclamped amplifies the edges by orders of magnitude - which is exactly the
    100x peak this produced before the floor was added.
    """
    frames = np.fft.irfft(spec, N_FFT, axis=1)[:, :WIN] * w
    total = (n_frames - 1) * HOP + WIN
    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    for i in range(n_frames):
        s = i * HOP
        out[s:s + WIN] += frames[i]
        norm[s:s + WIN] += w ** 2
    floor = max(norm.max() * 1e-3, _EPS)
    out = out / np.maximum(norm, floor)
    out = out[WIN:WIN + length]          # undo the analysis pad
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _noise_floor(mag: np.ndarray, init_frames: int = 6) -> np.ndarray:
    """Estimate the noise magnitude from the quietest frames.

    Using the leading frames alone (the textbook approach) is fragile when
    speech starts immediately, so we take a low percentile over time instead -
    still a stationary assumption, just a less brittle one.
    """
    k = min(init_frames, mag.shape[0])
    lead = mag[:k].mean(axis=0)
    quiet = np.percentile(mag, 10, axis=0)
    return np.maximum(np.minimum(lead, quiet), _EPS)


def spectral_subtraction(x: np.ndarray, sr: int = 16000,
                         alpha: float = 2.0, beta: float = 0.01) -> np.ndarray:
    """Boll (1979), with over-subtraction factor and a spectral floor."""
    spec, w, nf = _analyse(np.asarray(x, dtype=np.float64))
    mag, phase = np.abs(spec), np.angle(spec)
    noise = _noise_floor(mag)
    clean = mag ** 2 - alpha * noise[None, :] ** 2
    floor = (beta * noise[None, :]) ** 2
    clean = np.sqrt(np.maximum(clean, floor))
    return _synthesise(clean * np.exp(1j * phase), w, nf, len(x))


def wiener(x: np.ndarray, sr: int = 16000,
           alpha: float = 0.98, snr_floor_db: float = -25.0) -> np.ndarray:
    """Decision-directed a priori SNR (Ephraim & Malah 1984) + Wiener gain."""
    spec, w, nf = _analyse(np.asarray(x, dtype=np.float64))
    mag, phase = np.abs(spec), np.angle(spec)
    noise_pow = _noise_floor(mag) ** 2
    snr_min = 10.0 ** (snr_floor_db / 10.0)

    gain_prev = np.ones(mag.shape[1])
    post_prev = np.ones(mag.shape[1])
    out = np.empty_like(mag)
    for t in range(mag.shape[0]):
        post = mag[t] ** 2 / np.maximum(noise_pow, _EPS)          # a posteriori
        prio = alpha * (gain_prev ** 2) * post_prev \
             + (1 - alpha) * np.maximum(post - 1.0, 0.0)          # decision-directed
        prio = np.maximum(prio, snr_min)
        gain = prio / (1.0 + prio)
        out[t] = gain * mag[t]
        gain_prev, post_prev = gain, post
    return _synthesise(out * np.exp(1j * phase), w, nf, len(x))
