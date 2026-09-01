"""Synthetic reference-microphone channel model.

Deliberately NOT wired into `mixer.py::Mixer.build()`. That module drives the
trained model's single-channel input and every existing measured result in
this project rests on its output being unchanged; adding a second channel to
it is a real change that needs its own mixture-QA pass (`qa_mixtures.py`)
before anything trains on it, and this machine does not have the corpus
(LibriSpeech/MUSAN/manifests) to run that QA. This module exists only for the
standalone two-mic experiment in `scripts/eval_multimic.py`.

Models a second mic (e.g. an ear-cup "shell" mic feeding a reference channel,
per the DRDO block diagram's primary+reference layout) that:

  - sees the SAME noise sources as the primary ("boom") mic, through a
    slightly different acoustic path — a handful of samples of propagation
    delay for a plausible helmet mic spacing, and a different coupling gain,
    and
  - sees much LESS of the talker's voice, because it sits away from the mouth.

This is a SIMPLIFICATION of an acoustic transfer function, not a measured
one — no real dual-mic recording exists in this project. Every number
produced downstream of this function is `HYPOTHESIS`, not a validated
result, until checked against real two-microphone hardware.
"""
from __future__ import annotations

import numpy as np


def synthesize_reference(
    speech: np.ndarray,
    noise: np.ndarray,
    mic_delay_samples: int = 5,      # ~10 cm mic spacing / speed of sound @ 16 kHz
    speech_leak_db: float = -20.0,   # mouth -> ear-cup mic attenuation
    noise_gain_db: float = 3.0,      # ear-cup mic sits closer to ambient noise
) -> np.ndarray:
    """Return a synthetic reference-mic signal aligned to `speech`/`noise`.

    `noise` must be the SAME noise track already scaled into the primary
    mixture (not re-scaled here) so the reference channel is coherent with
    what the canceller is actually trying to remove from the primary.
    """
    speech = np.asarray(speech, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    assert len(speech) == len(noise), "speech and noise tracks must be aligned"

    delayed_noise = np.zeros_like(noise)
    if mic_delay_samples > 0:
        delayed_noise[mic_delay_samples:] = noise[:-mic_delay_samples]
    else:
        delayed_noise = noise.copy()

    ref = (speech * (10.0 ** (speech_leak_db / 20.0)) +
          delayed_noise * (10.0 ** (noise_gain_db / 20.0)))
    return ref.astype(np.float32)
