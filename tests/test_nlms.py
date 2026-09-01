"""Unit tests for the NLMS adaptive canceller (src/baselines/nlms.py).

These are sanity checks on synthetic signals with a KNOWN coupling path
(delay + gain), not a claim about real acoustic performance - there is no
real dual-mic recording in this project yet.
"""
import numpy as np

from src.baselines.nlms import NLMSCanceller

SR = 16000


def _coupled_noise(reference: np.ndarray, delay: int, gain: float) -> np.ndarray:
    coupled = np.zeros_like(reference)
    coupled[delay:] = reference[:-delay] * gain
    return coupled


def test_nlms_removes_correlated_noise():
    """With no speech at all, the filter should converge to near-silence."""
    rng = np.random.default_rng(0)
    n = SR  # 1 s
    reference = (rng.standard_normal(n) * 0.1).astype(np.float32)
    coupled = _coupled_noise(reference, delay=3, gain=0.7)
    primary = coupled  # no speech - isolates the cancellation question

    out = NLMSCanceller(order=16, mu=0.1).process_block(primary, reference)

    first_quarter = float(np.mean(out[:n // 4] ** 2))
    last_quarter = float(np.mean(out[3 * n // 4:] ** 2))
    noise_power = float(np.mean(coupled[3 * n // 4:] ** 2))

    assert last_quarter < first_quarter, "filter should be converging, not static"
    assert last_quarter < 0.05 * noise_power, ">=13 dB cancellation once converged"


def test_nlms_preserves_uncorrelated_speech():
    """Speech in `primary` that is absent from `reference` must survive -
    an NLMS canceller has no way to predict it and shouldn't touch it."""
    rng = np.random.default_rng(1)
    n = SR
    reference = (rng.standard_normal(n) * 0.1).astype(np.float32)
    coupled = _coupled_noise(reference, delay=3, gain=0.7)
    t = np.arange(n) / SR
    speech = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    primary = speech + coupled

    # mu=0.1, not the more aggressive 0.5: NLMS misadjustment (excess
    # steady-state error from using a noisy instantaneous gradient) scales
    # with step size, and at mu=0.5 enough of it leaks into the output to
    # fail this bar even though cancellation itself still converges.
    out = NLMSCanceller(order=16, mu=0.1).process_block(primary, reference)

    tail = slice(3 * n // 4, n)
    err_power = float(np.mean((out[tail] - speech[tail]) ** 2))
    speech_power = float(np.mean(speech[tail] ** 2))

    assert err_power < 0.1 * speech_power, "recovered signal should track speech"


def test_vad_gated_adaptation_protects_speech_better():
    """If the reference channel leaks some speech (the realistic case - no
    physical mic is perfectly isolated from the talker), plain NLMS partially
    learns to cancel that speech too, because it cannot distinguish
    'correlated because it's noise' from 'correlated because it's leaked
    speech'. Freezing adaptation during detected speech activity (VAD-gated,
    src/baselines/nlms.py's `update_mask`) is the standard real-headset fix -
    check it actually helps, not just that it runs."""
    rng = np.random.default_rng(3)
    n = SR * 2  # 2 s, alternating quarter-second speech-on/off blocks
    reference = (rng.standard_normal(n) * 0.1).astype(np.float32)
    coupled = _coupled_noise(reference, delay=3, gain=0.7)

    block = SR // 4
    speech_active = (np.arange(n) // block) % 2 == 0
    t = np.arange(n) / SR
    tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    speech = np.where(speech_active, tone, 0.0).astype(np.float32)

    leak_gain = 0.3  # reference is NOT perfectly isolated from the talker
    reference_with_leak = reference + np.where(speech_active, speech * leak_gain, 0.0)
    reference_with_leak = reference_with_leak.astype(np.float32)
    primary = speech + coupled

    out_unmasked = NLMSCanceller(order=16, mu=0.1).process_block(
        primary, reference_with_leak)
    out_gated = NLMSCanceller(order=16, mu=0.1).process_block(
        primary, reference_with_leak, update_mask=~speech_active)

    err_unmasked = float(np.mean((out_unmasked[speech_active] - speech[speech_active]) ** 2))
    err_gated = float(np.mean((out_gated[speech_active] - speech[speech_active]) ** 2))

    assert err_gated < err_unmasked, "VAD-gated adaptation should preserve speech better"


def test_nlms_zero_reference_is_a_no_op():
    """A silent/absent reference channel must not corrupt the primary signal -
    the required single-mic fallback behaviour."""
    rng = np.random.default_rng(2)
    n = 4000
    primary = (rng.standard_normal(n) * 0.1).astype(np.float32)
    reference = np.zeros(n, dtype=np.float32)

    out = NLMSCanceller(order=16, mu=0.5).process_block(primary, reference)

    assert np.allclose(out, primary, atol=1e-6)
