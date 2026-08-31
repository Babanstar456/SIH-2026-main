"""Objective quality metrics.

Reference implementations only - `pesq` is the ITU-T P.862 reference code and
`pystoi` the reference STOI/ESTOI. We do not hand-roll perceptual metrics: a
home-grown PESQ would produce numbers nobody could check against the literature,
which defeats the point of reporting them.

If PESQ is unavailable the value is recorded as NaN and a `pesq_available=False`
column is written, so a missing metric can never be mistaken for a computed one.
"""
from __future__ import annotations

import warnings

import numpy as np

SR = 16000
_EPS = 1e-12

try:
    from pesq import pesq as _pesq_fn
    PESQ_AVAILABLE = True
except Exception:  # noqa: BLE001
    _pesq_fn = None
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi as _stoi_fn
    STOI_AVAILABLE = True
except Exception:  # noqa: BLE001
    _stoi_fn = None
    STOI_AVAILABLE = False


def _align(ref: np.ndarray, est: np.ndarray):
    n = min(len(ref), len(est))
    return (np.asarray(ref[:n], dtype=np.float64),
            np.asarray(est[:n], dtype=np.float64))


# ------------------------------------------------------------------ SNR family

def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    """Plain SNR of `est` against `ref`, in dB."""
    ref, est = _align(ref, est)
    noise = est - ref
    return float(10.0 * np.log10((np.sum(ref ** 2) + _EPS) / (np.sum(noise ** 2) + _EPS)))


def si_sdr_db(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR. Immune to any overall gain the method applies,
    which plain SNR is not."""
    ref, est = _align(ref, est)
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + _EPS)
    proj = alpha * ref
    noise = est - proj
    return float(10.0 * np.log10((np.sum(proj ** 2) + _EPS) / (np.sum(noise ** 2) + _EPS)))


def seg_snr_db(ref: np.ndarray, est: np.ndarray, sr: int = SR,
               frame_ms: float = 32.0, lo: float = -10.0, hi: float = 35.0) -> float:
    """Segmental SNR over active frames, clamped to [lo, hi] as is conventional
    (unclamped, silent frames dominate the average and the number stops meaning
    anything)."""
    ref, est = _align(ref, est)
    n = int(sr * frame_ms / 1000)
    if len(ref) < n:
        return float(np.clip(snr_db(ref, est), lo, hi))
    k = len(ref) // n
    r = ref[:k * n].reshape(k, n)
    e = est[:k * n].reshape(k, n)
    rp = np.sum(r ** 2, axis=1)
    active = rp > (rp.max() * 1e-4)
    if not active.any():
        return float("nan")
    npow = np.sum((e - r) ** 2, axis=1)
    seg = 10.0 * np.log10((rp[active] + _EPS) / (npow[active] + _EPS))
    return float(np.mean(np.clip(seg, lo, hi)))


# --------------------------------------------------------------- perceptual

def pesq_wb(ref: np.ndarray, est: np.ndarray, sr: int = SR) -> float:
    """Wideband PESQ (ITU-T P.862.2), range roughly 1.0 - 4.5."""
    if not PESQ_AVAILABLE:
        return float("nan")
    ref, est = _align(ref, est)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(_pesq_fn(sr, ref.astype(np.float32),
                                  est.astype(np.float32), "wb"))
    except Exception:  # noqa: BLE001
        # PESQ raises on degenerate input (all-silence, no detectable speech).
        # At very low SNR this happens legitimately - record NaN rather than
        # crashing a sweep of thousands of clips.
        return float("nan")


def stoi_score(ref: np.ndarray, est: np.ndarray, sr: int = SR,
               extended: bool = False) -> float:
    if not STOI_AVAILABLE:
        return float("nan")
    ref, est = _align(ref, est)
    try:
        return float(_stoi_fn(ref, est, sr, extended=extended))
    except Exception:  # noqa: BLE001
        return float("nan")


# ------------------------------------------------------------------ bundle

def masked_metrics(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray,
                   mask: np.ndarray) -> dict:
    """Metrics computed INSIDE the burst regions only.

    This exists because a whole-clip score hides the exact failure the doc warns
    about. Bursts occupy roughly 12% of a clip; the other 88% is ordinary speech
    the model already handles. A model that removes none of the gunfire but
    cleans the rest still posts a respectable whole-clip SI-SDR, and the gunshot
    category looks fine while being completely unsolved.

    Restricting to the masked samples asks the direct question: in the moment
    the rifle fired, did the voice survive?
    """
    n = min(len(clean), len(noisy), len(enhanced), len(mask))
    m = np.asarray(mask[:n], dtype=bool)
    if not m.any():
        return {}
    c, no, e = clean[:n][m], noisy[:n][m], enhanced[:n][m]
    out = {
        "burst_si_sdr": si_sdr_db(c, e),
        "burst_si_sdr_noisy": si_sdr_db(c, no),
        "burst_snr": snr_db(c, e),
        "burst_snr_noisy": snr_db(c, no),
        "burst_frac": float(m.mean()),
    }
    out["burst_si_sdr_gain"] = out["burst_si_sdr"] - out["burst_si_sdr_noisy"]
    out["burst_snr_gain"] = out["burst_snr"] - out["burst_snr_noisy"]
    return out


def reference_metrics(clean: np.ndarray, noisy: np.ndarray, sr: int = SR) -> dict:
    """Metrics of the UNPROCESSED clip against clean.

    Split out because these depend only on the test clip, not on the method, so
    a sweep over N methods can compute them once instead of N times - roughly
    halving evaluation time, which is dominated by PESQ.
    """
    return {
        "pesq_noisy":   pesq_wb(clean, noisy, sr),
        "stoi_noisy":   stoi_score(clean, noisy, sr),
        "estoi_noisy":  stoi_score(clean, noisy, sr, extended=True),
        "si_sdr_noisy": si_sdr_db(clean, noisy),
        "snr_noisy":    snr_db(clean, noisy),
        "segsnr_noisy": seg_snr_db(clean, noisy, sr),
    }


def evaluate_pair(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray,
                  sr: int = SR, ref: dict | None = None) -> dict:
    """All metrics for one clip, including the before/after deltas.

    `clean` is the reference the mixer produced, so every delta here is against
    ground truth rather than an estimate. Pass `ref` from `reference_metrics`
    to skip recomputing the unprocessed-clip metrics.
    """
    out = {
        "pesq":        pesq_wb(clean, enhanced, sr),
        "stoi":        stoi_score(clean, enhanced, sr),
        "estoi":       stoi_score(clean, enhanced, sr, extended=True),
        "si_sdr":      si_sdr_db(clean, enhanced),
        "snr":         snr_db(clean, enhanced),
        "segsnr":      seg_snr_db(clean, enhanced, sr),
        "pesq_available": PESQ_AVAILABLE,
    }
    out.update(ref if ref is not None else reference_metrics(clean, noisy, sr))
    out["snr_gain"] = out["snr"] - out["snr_noisy"]
    out["si_sdr_gain"] = out["si_sdr"] - out["si_sdr_noisy"]
    out["pesq_gain"] = out["pesq"] - out["pesq_noisy"]
    out["stoi_gain"] = out["stoi"] - out["stoi_noisy"]
    return out
