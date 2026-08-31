"""Invariant tests for the pieces that fail SILENTLY when wrong.

The bugs that matter in this project do not raise exceptions. A misaligned
transient mask, an SNR that is not the SNR you asked for, a baseline with a
broken overlap-add - all of these train, run, and produce plausible-looking
numbers that are wrong. These tests pin the invariants down.

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None
    HAS_TORCH = False
needs_torch = pytest.mark.skipif(
    not HAS_TORCH,
    reason="PyTorch unavailable (not installed, or blocked by Windows Smart App "
           "Control). The NumPy invariants below still run.")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import audio as A          # noqa: E402
from src import metrics as M        # noqa: E402
from src import framing as S        # noqa: E402

# `framing` is the NumPy-only half and deliberately has no stft/istft - importing
# it here is what lets the rest of this file run on a machine where torch cannot
# load. The two tests that genuinely exercise the torch STFT need the other half,
# so it is imported separately and only when torch is actually available.
# Binding both names to `framing` made those two ERROR with an AttributeError
# instead of skipping, which is how this went unnoticed.
if HAS_TORCH:
    from src import stft as ST      # noqa: E402
else:
    ST = None
from src.baselines.classical import _analyse, _synthesise, spectral_subtraction, wiener  # noqa: E402
from src.mixer import Mixer         # noqa: E402

SR = 16000


@pytest.fixture
def cfg():
    import yaml
    with open(ROOT / "configs" / "data.yaml") as f:
        return yaml.safe_load(f)


def _speech(n=SR * 6, seed=0):
    """Speech-ish: harmonic stack with a syllabic envelope and real pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    f0 = 120.0
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)) ** 2
    env[env < 0.15] = 0.0                       # silences, so active-RMS matters
    return (0.3 * x * env + 1e-4 * rng.standard_normal(n)).astype(np.float32)


# --------------------------------------------------------------------- stft

@needs_torch
def test_stft_roundtrip_is_identity():
    x = torch.from_numpy(_speech(SR * 2))
    y = ST.istft(ST.stft(x), length=len(x))
    assert torch.allclose(x, y, atol=1e-5), f"max err {(x-y).abs().max():.2e}"


@needs_torch
def test_n_frames_matches_torch():
    for n in (SR, SR * 2, 64000, 12345):
        got = ST.stft(torch.zeros(n)).shape[-2]
        assert S.n_frames(n) == got, f"n={n}: predicted {S.n_frames(n)}, actual {got}"


def test_frame_mask_alignment():
    """A burst in the middle must mark frames covering that time, and no others."""
    n = 64000
    mask = np.zeros(n, dtype=bool)
    start, end = 32000, 32000 + 4000          # 2.00s - 2.25s
    mask[start:end] = True
    nf = S.n_frames(n)
    fm = S.samples_to_frame_mask(mask, nf)

    assert fm.any(), "no frames marked for a clearly present burst"
    marked = np.nonzero(fm)[0]
    # Every marked frame's window must actually intersect the burst.
    for t in marked:
        lo, hi = max(0, t * S.HOP - S.WIN // 2), t * S.HOP + S.WIN // 2
        assert lo < end and hi > start, f"frame {t} marked but does not overlap burst"
    # And the burst's centre must be marked.
    centre = (start + end) // 2 // S.HOP
    assert fm[centre], "burst centre frame not marked"


def test_frame_mask_empty_stays_empty():
    nf = S.n_frames(64000)
    assert not S.samples_to_frame_mask(np.zeros(64000, bool), nf).any()


# -------------------------------------------------------------------- audio

def test_active_rms_ignores_silence():
    """Active-speech RMS must not be dragged down by padded silence.

    This is why the mixer references active RMS: whole-clip RMS would make two
    clips at the same nominal SNR sound very different.
    """
    x = _speech(SR * 2)
    padded = np.concatenate([x, np.zeros(SR * 6, np.float32)])
    assert A.rms(padded) < 0.6 * A.rms(x)                 # whole-clip RMS collapses
    assert A.active_rms(padded) == pytest.approx(A.active_rms(x), rel=0.15)


def test_scale_noise_for_snr_hits_the_target():
    rng = np.random.default_rng(0)
    speech = _speech(SR * 3)
    noise = rng.standard_normal(len(speech)).astype(np.float32)
    ref = A.active_rms(speech)
    for want in (-5.0, 0.0, 10.0, 15.0):
        g = A.scale_noise_for_snr(ref, noise, want)
        got = 20 * np.log10(ref / (A.rms(noise * g) + 1e-12))
        assert got == pytest.approx(want, abs=0.1), f"asked {want}, got {got:.2f}"


def test_burst_peak_snr_is_peak_referenced():
    """Negative peak-SNR must make the burst PEAK louder than the speech RMS."""
    speech = _speech(SR * 2)
    ref = A.active_rms(speech)
    burst = np.zeros(1000, np.float32)
    burst[500] = 1.0
    g = A.scale_burst_for_peak_snr(ref, burst, -15.0)
    peak = np.abs(burst * g).max()
    assert peak > ref, "a -15 dB peak-SNR burst should exceed the speech RMS"
    assert 20 * np.log10(ref / peak) == pytest.approx(-15.0, abs=0.1)


def test_soft_limit_bounds_and_preserves_small_signals():
    x = np.linspace(-2, 2, 4001).astype(np.float32)
    y = A.soft_limit(x, 0.8)
    assert np.abs(y).max() < 1.0, "limiter must stay below full scale"
    small = np.abs(x) <= 0.8
    assert np.allclose(y[small], x[small]), "below threshold must be untouched"
    assert np.all(np.diff(y) >= -1e-6), "limiter must stay monotonic"


def test_find_onset_skips_leading_silence():
    x = np.zeros(SR, np.float32)
    x[8000] = 1.0
    assert abs(A.find_onset(x) - 8000) < 10


# ----------------------------------------------------------------- baselines

def test_wola_roundtrip_is_exact():
    """Unmodified analysis->synthesis must be identity to machine precision.

    This caught a real bug: an unclamped window-sum divide amplified clip edges
    by 100x while the middle of the signal looked perfectly fine.
    """
    x = _speech(SR * 2).astype(np.float64)
    spec, w, nf = _analyse(x)
    y = _synthesise(spec, w, nf, len(x))
    assert np.abs(y - x).max() < 1e-9, f"max err {np.abs(y-x).max():.2e}"


@pytest.mark.parametrize("fn", [wiener, spectral_subtraction])
def test_baselines_do_not_amplify(fn):
    rng = np.random.default_rng(1)
    x = _speech(SR * 2) + 0.05 * rng.standard_normal(SR * 2).astype(np.float32)
    y = fn(x, SR)
    assert len(y) == len(x)
    assert np.isfinite(y).all()
    assert np.abs(y).max() <= np.abs(x).max() * 1.5, "suppressor should not amplify"


# ------------------------------------------------------------------ metrics

def test_metrics_are_maximal_for_identity():
    x = _speech(SR * 3)
    assert M.si_sdr_db(x, x) > 100
    if M.STOI_AVAILABLE:
        assert M.stoi_score(x, x) > 0.99
    if M.PESQ_AVAILABLE:
        assert M.pesq_wb(x, x) > 4.0


def test_degradation_lowers_scores():
    rng = np.random.default_rng(2)
    x = _speech(SR * 3)
    noisy = x + 0.1 * rng.standard_normal(len(x)).astype(np.float32)
    assert M.si_sdr_db(x, noisy) < M.si_sdr_db(x, x)
    if M.STOI_AVAILABLE:
        assert M.stoi_score(x, noisy) < M.stoi_score(x, x)


def test_si_sdr_is_scale_invariant():
    x = _speech(SR * 2)
    rng = np.random.default_rng(3)
    est = x + 0.05 * rng.standard_normal(len(x)).astype(np.float32)
    assert M.si_sdr_db(x, est) == pytest.approx(M.si_sdr_db(x, est * 7.3), abs=1e-6)


# -------------------------------------------------------------------- mixer

def test_mixer_marks_bursts_where_they_are(cfg):
    mix = Mixer(cfg)
    rng = np.random.default_rng(0)
    burst = np.zeros(2400, np.float32)
    burst[100:400] = np.linspace(1, 0, 300)        # sharp attack, short decay
    res = mix.build(rng, _speech(SR * 8), bg_clips=[], burst_clips=[burst],
                    category="gunshot", force_burst=True)
    assert res is not None
    assert res.meta["n_bursts"] >= 1, "force_burst must guarantee an event"

    resid = res.noisy - res.target
    m = res.transient_mask
    assert m.any() and (~m).any()
    e_in = np.mean(resid[m] ** 2)
    e_out = np.mean(resid[~m] ** 2)
    assert e_in > e_out * 100, "masked region must hold the burst energy"


def test_mixer_scales_target_and_noisy_together(cfg):
    """The pair must share one gain, or the model learns a level offset."""
    mix = Mixer(cfg)
    rng = np.random.default_rng(5)
    res = mix.build(rng, _speech(SR * 8), bg_clips=[], burst_clips=[],
                    category="engine")
    assert res is not None
    quiet = res.noisy - res.target                 # with no noise sources added,
    assert np.abs(quiet).max() < 1e-6, "no noise given, so noisy must equal target"


def test_mixer_respects_background_snr(cfg):
    import copy
    c = copy.deepcopy(cfg)
    c["background"]["snr_db"] = [5.0, 5.0]         # pin it
    c["reverb"]["prob"] = 0.0
    c["limiter"]["prob"] = 0.0
    mix = Mixer(c)
    rng = np.random.default_rng(7)
    bg = rng.standard_normal(mix.n).astype(np.float32)
    res = mix.build(rng, _speech(SR * 8), bg_clips=[bg], burst_clips=[],
                    category="engine")
    assert res is not None
    noise = res.noisy - res.target
    got = 20 * np.log10(A.active_rms(res.target) / (A.rms(noise) + 1e-12))
    assert got == pytest.approx(5.0, abs=1.0), f"asked 5 dB, measured {got:.2f} dB"


def test_mixer_output_is_in_range(cfg):
    mix = Mixer(cfg)
    rng = np.random.default_rng(11)
    burst = np.zeros(3200, np.float32)
    burst[0:200] = 1.0
    for i in range(8):
        res = mix.build(np.random.default_rng(i), _speech(SR * 8, seed=i),
                        bg_clips=[rng.standard_normal(mix.n).astype(np.float32)],
                        burst_clips=[burst], category="gunshot", force_burst=True)
        if res is None:
            continue
        assert np.abs(res.noisy).max() <= 1.0
        assert np.abs(res.target).max() <= 1.0
        assert np.isfinite(res.noisy).all() and np.isfinite(res.target).all()


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__)), "-q"]))
