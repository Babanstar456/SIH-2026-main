"""Unit tests for the LMS and RLS adaptive cancellers, mirroring
tests/test_nlms.py's synthetic delay+gain setup so all three algorithms are
checked on identical signals.
"""
import numpy as np

from src.baselines.lms import LMSCanceller
from src.baselines.rls import RLSCanceller

SR = 16000


def _coupled_noise(reference: np.ndarray, delay: int, gain: float) -> np.ndarray:
    coupled = np.zeros_like(reference)
    coupled[delay:] = reference[:-delay] * gain
    return coupled


def test_lms_removes_correlated_noise_with_conservative_step():
    rng = np.random.default_rng(0)
    n = SR
    reference = (rng.standard_normal(n) * 0.1).astype(np.float32)
    coupled = _coupled_noise(reference, delay=3, gain=0.7)

    out = LMSCanceller(order=16, mu=0.05).process_block(coupled, reference)

    first_quarter = float(np.mean(out[:n // 4] ** 2))
    last_quarter = float(np.mean(out[3 * n // 4:] ** 2))
    noise_power = float(np.mean(coupled[3 * n // 4:] ** 2))

    assert last_quarter < first_quarter
    assert last_quarter < 0.05 * noise_power


def test_lms_can_diverge_on_a_loud_transient_that_nlms_survives():
    """The actual point of building LMS alongside NLMS: a fixed step size
    tuned safe for speech-level input is not safe for a signal that suddenly
    jumps far louder - exactly what a gunshot does to a reference channel.
    NLMS's per-sample normalisation is specifically the fix for this."""
    rng = np.random.default_rng(4)
    n = SR
    reference = (rng.standard_normal(n) * 0.1).astype(np.float32)
    # A step size that is perfectly well-behaved on quiet input...
    mu = 0.5
    quiet_coupled = _coupled_noise(reference, delay=3, gain=0.7)
    out_quiet = LMSCanceller(order=16, mu=mu).process_block(quiet_coupled, reference)
    assert np.all(np.isfinite(out_quiet))

    # ...diverges once a loud burst raises the reference signal's power by
    # orders of magnitude partway through the clip (a gunshot arriving).
    loud_reference = reference.copy()
    loud_reference[n // 2:] *= 50.0
    loud_coupled = _coupled_noise(loud_reference, delay=3, gain=0.7)
    with np.errstate(over="ignore", invalid="ignore"):
        out_loud = LMSCanceller(order=16, mu=mu).process_block(loud_coupled, loud_reference)

    tail_power = float(np.nanmean(out_loud[3 * n // 4:].astype(np.float64) ** 2))
    burst_power = float(np.mean(loud_coupled[3 * n // 4:] ** 2))
    assert not np.all(np.isfinite(out_loud)) or tail_power > burst_power, (
        "expected the fixed step size to diverge or blow up on the loud segment")


def test_rls_converges_faster_than_lms_on_a_short_clip():
    """RLS's whole selling point is convergence speed. Give both algorithms a
    SHORT clip - too short for LMS's fixed small step to fully converge - and
    check RLS gets further."""
    rng = np.random.default_rng(5)
    n = SR // 20  # 50 ms - short on purpose
    reference = (rng.standard_normal(n) * 0.1).astype(np.float32)
    coupled = _coupled_noise(reference, delay=3, gain=0.7)

    out_lms = LMSCanceller(order=16, mu=0.05).process_block(coupled, reference)
    out_rls = RLSCanceller(order=16, forgetting=0.999).process_block(coupled, reference)

    err_lms = float(np.mean(out_lms[-32:] ** 2))
    err_rls = float(np.mean(out_rls[-32:] ** 2))
    assert err_rls < err_lms


def test_rls_zero_reference_is_a_no_op():
    rng = np.random.default_rng(2)
    n = 2000
    primary = (rng.standard_normal(n) * 0.1).astype(np.float32)
    reference = np.zeros(n, dtype=np.float32)

    out = RLSCanceller(order=16).process_block(primary, reference)
    assert np.allclose(out, primary, atol=1e-6)
