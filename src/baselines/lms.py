"""Plain LMS adaptive noise cancellation from a reference microphone.

Sibling of `nlms.py` - same two-input reference-mic setup, same reasons for
being kept out of `methods.py`'s single-channel registry. This exists to
answer a specific comparison question (see section 14 of the project's own
scoping discussion): is the extra per-sample normalisation in NLMS actually
buying anything over plain LMS on this problem, or is it dead weight?

Plain LMS (Widrow & Hoff 1960) uses a FIXED step size, unlike NLMS's
step-size-normalised-by-signal-power. That makes it simpler and slightly
cheaper per sample, but the fixed step has to be chosen conservatively enough
to stay stable across the loudest signal the filter will ever see - which for
this project is a gunshot peaking many dB above the speech. A step size tuned
for speech-level input can genuinely diverge (weights blow up) the moment a
burst arrives; NLMS's normalisation exists specifically to avoid this.
"""
from __future__ import annotations

import numpy as np


class LMSCanceller:
    """Same interface as `NLMSCanceller`, fixed (not normalised) step size.

    `mu` here is an ABSOLUTE step size, not the (0, 1]-normalised one NLMS
    uses - it must be scaled to the expected input power, and there is no
    single value that is simultaneously safe for quiet speech and a gunshot
    peak. That fragility is the point of building this alongside NLMS: it is
    the concrete answer to "why not the simpler algorithm".
    """

    def __init__(self, order: int = 64, mu: float = 0.01, leakage: float = 1.0):
        assert mu > 0.0
        assert 0.0 < leakage <= 1.0
        self.order = order
        self.mu = mu
        self.leakage = leakage
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(self.order, dtype=np.float64)
        self.xbuf = np.zeros(self.order, dtype=np.float64)

    def process_sample(self, d: float, x: float, adapt: bool = True) -> float:
        self.xbuf[1:] = self.xbuf[:-1]
        self.xbuf[0] = x
        y = float(self.w @ self.xbuf)
        e = d - y
        if adapt:
            self.w = self.leakage * self.w + self.mu * e * self.xbuf
        return e

    def process_block(self, primary: np.ndarray, reference: np.ndarray,
                      update_mask: np.ndarray | None = None) -> np.ndarray:
        primary = np.asarray(primary, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        assert len(primary) == len(reference), "primary and reference must be aligned"
        if update_mask is not None:
            assert len(update_mask) == len(primary)
        out = np.empty(len(primary), dtype=np.float64)
        for n in range(len(primary)):
            adapt = True if update_mask is None else bool(update_mask[n])
            out[n] = self.process_sample(primary[n], reference[n], adapt=adapt)
        return out.astype(np.float32)


def lms_cancel(primary: np.ndarray, reference: np.ndarray,
               order: int = 64, mu: float = 0.01,
               update_mask: np.ndarray | None = None) -> np.ndarray:
    """One-shot wrapper: fresh filter state, whole clip at once."""
    return LMSCanceller(order=order, mu=mu).process_block(primary, reference, update_mask)
