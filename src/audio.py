"""Audio I/O and level maths.

Deliberately uses soundfile + soxr rather than librosa: this code runs inside
the training dataloader for every sample, and librosa.load is far too slow to
sit in that path.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import soxr

SR = 16000


# --------------------------------------------------------------------------- io

def load_audio(path, sr: int = SR, mono: bool = True) -> np.ndarray:
    """Read any soundfile-supported format as float32, resampled to `sr`."""
    x, in_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if mono and x.shape[1] > 1:
        x = x.mean(axis=1, keepdims=True)
    x = x[:, 0] if mono else x
    if in_sr != sr:
        x = soxr.resample(x, in_sr, sr, quality="HQ").astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def save_audio(path, x: np.ndarray, sr: int = SR) -> None:
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr, subtype="PCM_16")


def audio_info(path):
    """Duration/rate without decoding the file - used when building manifests."""
    info = sf.info(str(path))
    return info.frames / info.samplerate, info.samplerate, info.channels


# ------------------------------------------------------------------- levels

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + 1e-20))


def active_speech_mask(
    x: np.ndarray, sr: int = SR, frame_ms: float = 20.0,
    hop_ms: float = 10.0, thresh_db: float = -40.0,
) -> np.ndarray:
    """Energy VAD: frames within `thresh_db` of the loudest frame are active.

    A crude stand-in for ITU-T P.56 active speech level, but honest and
    deterministic. It matters because SNR computed over a whole utterance is
    skewed by leading/trailing silence - two clips at "0 dB SNR" can differ by
    6+ dB in how loud the speech actually is against the noise.
    """
    n = max(1, int(sr * frame_ms / 1000))
    h = max(1, int(sr * hop_ms / 1000))
    mask = np.zeros(len(x), dtype=bool)
    if len(x) < n:
        return np.ones(len(x), dtype=bool)
    starts = np.arange(0, len(x) - n + 1, h)
    energy = np.array([np.mean(x[s:s + n] ** 2) for s in starts])
    peak = energy.max()
    if peak <= 0:
        return mask
    active = energy > peak * (10.0 ** (thresh_db / 10.0))
    for s in starts[active]:
        mask[s:s + n] = True
    return mask


def active_rms(x: np.ndarray, sr: int = SR) -> float:
    """RMS over active-speech frames only."""
    m = active_speech_mask(x, sr)
    return rms(x[m]) if m.any() else rms(x)


def scale_noise_for_snr(speech_ref_rms: float, noise: np.ndarray, snr_db: float) -> float:
    """Gain to apply to `noise` so speech sits `snr_db` above it."""
    n_rms = rms(noise)
    if n_rms <= 1e-12:
        return 0.0
    return float(speech_ref_rms / (n_rms * (10.0 ** (snr_db / 20.0))))


def scale_burst_for_peak_snr(speech_ref_rms: float, burst: np.ndarray, snr_db: float) -> float:
    """Gain so the burst's PEAK sits `snr_db` relative to speech RMS.

    Negative snr_db => the burst peak is louder than the speech, which is the
    realistic case for gunfire beside the microphone.
    """
    peak = float(np.max(np.abs(burst))) if burst.size else 0.0
    if peak <= 1e-12:
        return 0.0
    return float(speech_ref_rms / (peak * (10.0 ** (snr_db / 20.0))))


# --------------------------------------------------------------- shaping

def soft_limit(x: np.ndarray, threshold: float) -> np.ndarray:
    """Soft-knee limiter modelling the analog limiter before the ADC.

    Linear below `threshold`, tanh-compressed above, asymptotic to 1.0. The doc
    specifies a fast analog limiter between mic preamp and ADC, so the model
    must be trained on limited audio - that is what it receives in the field.
    A model trained on unlimited audio meets a different signal at deployment.
    """
    t = float(np.clip(threshold, 1e-3, 0.999))
    a = np.abs(x)
    over = a > t
    if not over.any():
        return x
    y = x.copy()
    y[over] = np.sign(x[over]) * (t + (1.0 - t) * np.tanh((a[over] - t) / (1.0 - t)))
    return y.astype(np.float32)


def peak_normalise(x: np.ndarray, target_peak: float) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-12:
        return x, 1.0
    g = target_peak / peak
    return (x * g).astype(np.float32), g


def fit_length(x: np.ndarray, n: int, rng: np.random.Generator, loop: bool = True) -> np.ndarray:
    """Crop or extend `x` to exactly n samples."""
    if len(x) == n:
        return x
    if len(x) > n:
        s = int(rng.integers(0, len(x) - n + 1))
        return x[s:s + n]
    if not loop:
        out = np.zeros(n, dtype=np.float32)
        out[:len(x)] = x
        return out
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n].astype(np.float32)


def load_random_window(path, n: int, rng, sr: int = SR) -> np.ndarray:
    """Read a random n-sample window without decoding the whole file.

    MUSAN background files run to several minutes; decoding one in full for
    every training sample would dominate dataloader time. All files reaching
    this function are already 16 kHz (see scripts/prepare_data.py), so
    frame-accurate seeking is valid with no resampling.
    """
    try:
        info = sf.info(str(path))
    except Exception:  # noqa: BLE001
        return np.zeros(n, dtype=np.float32)
    total = info.frames
    if total <= 0:
        return np.zeros(n, dtype=np.float32)
    if total <= n:
        x, in_sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        if in_sr != sr:
            x = soxr.resample(x, in_sr, sr, quality="HQ").astype(np.float32)
        return fit_length(x, n, rng, loop=True)
    start = int(rng.integers(0, total - n))
    x, in_sr = sf.read(str(path), dtype="float32", start=start,
                       frames=n, always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if in_sr != sr:
        x = soxr.resample(x, in_sr, sr, quality="HQ").astype(np.float32)
    return fit_length(np.ascontiguousarray(x, dtype=np.float32), n, rng, loop=True)


def load_layered(paths, n: int, rng, layers: int = 1, sr: int = SR) -> np.ndarray:
    """Sum `layers` independent random windows into one noise track.

    Needed for babble. MUSAN's `speech/` directory holds SINGLE-SPEAKER
    recordings, so using one clip gives an interfering talker, not babble -
    a quite different and much easier problem. Real multi-talker babble is
    built by summing several unrelated voices, which is what this does.

    Each layer is level-matched before summing so one loud recording cannot
    dominate and collapse the result back to a single talker. The sum is then
    restored to the mean level of its constituents, so the returned track sits
    at a natural level rather than an arbitrary one - for k=1 this reduces
    exactly to `load_random_window`, keeping every other category unchanged.
    """
    if not len(paths):
        return np.zeros(n, dtype=np.float32)
    k = max(1, int(layers))
    idx = np.atleast_1d(rng.choice(len(paths), size=min(k, len(paths)),
                                   replace=len(paths) < k))
    out = np.zeros(n, dtype=np.float32)
    levels = []
    for i in idx:
        c = load_random_window(paths[int(i)], n, rng, sr)
        r = rms(c)
        if r > 1e-9:
            out += (c / r).astype(np.float32)
            levels.append(r)
    if not levels:
        return out
    target = float(np.mean(levels))
    cur = rms(out)
    if cur > 1e-9:
        out = out * (target / cur)
    return out.astype(np.float32)


def load_event(path, t_center: float, pre_s: float = 0.02,
               post_s: float = 0.48, sr: int = SR) -> np.ndarray:
    """Read a window around an ANNOTATED event time.

    Used for the gunshot corpus, which ships ground-truth shot timestamps. A
    labelled cut is strictly better than an energy-detected one: the recordings
    contain several shots plus range chatter, and an energy detector will
    sometimes lock onto the wrong transient.
    """
    try:
        info = sf.info(str(path))
    except Exception:  # noqa: BLE001
        return np.zeros(int(sr * (pre_s + post_s)), dtype=np.float32)
    in_sr, total = info.samplerate, info.frames
    start = max(0, int((t_center - pre_s) * in_sr))
    frames = int((pre_s + post_s) * in_sr)
    if start >= total:
        start = max(0, total - frames)
    x, _ = sf.read(str(path), dtype="float32", start=start,
                   frames=min(frames, total - start), always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if in_sr != sr:
        x = soxr.resample(x, in_sr, sr, quality="HQ").astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def load_burst(rec, rng, sr: int = SR) -> np.ndarray:
    """Load one impulsive event from a manifest record.

    Uses a ground-truth timestamp when the corpus provides one, otherwise falls
    back to energy-based onset detection inside the mixer.
    """
    path = rec["path"] if isinstance(rec, dict) else rec
    shots = rec.get("shots") if isinstance(rec, dict) else None
    if shots:
        t = float(shots[int(rng.integers(0, len(shots)))])
        return load_event(path, t, sr=sr)
    return load_audio(path, sr)


def find_onset(x: np.ndarray, rel_thresh: float = 0.1) -> int:
    """First sample reaching `rel_thresh` of the clip peak.

    Transient clips routinely have a second or two of silence before the event;
    placing a burst by file offset rather than by onset would scatter the actual
    impulse anywhere in the mixture, which is exactly the failure mode the doc
    warns about.
    """
    if x.size == 0:
        return 0
    a = np.abs(x)
    peak = a.max()
    if peak <= 1e-12:
        return 0
    idx = np.nonzero(a >= rel_thresh * peak)[0]
    return int(idx[0]) if idx.size else 0
