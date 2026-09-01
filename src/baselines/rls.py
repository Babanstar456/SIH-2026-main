"""RLS adaptive noise cancellation from a reference microphone.

Third sibling of `nlms.py` / `lms.py`. Recursive Least Squares (Haykin,
_Adaptive Filter Theory_) converges in far fewer samples than LMS/NLMS -
useful when the correlated noise itself is non-stationary (a gunshot arrives
and is gone before an LMS-family filter has adapted to it at all) - at the
cost of O(order^2) work per sample instead of O(order). For `order=64` that
is 64x the arithmetic of NLMS per sample, which matters directly for the
16 ms/frame embedded budget this project measures everything against
(see CLAUDE.md's latency-budget invariants). Whether RLS's faster
convergence is worth that cost on THIS problem is an empirical question,
not assumed - that is exactly why it needs to be run through the same
evaluation as LMS/NLMS rather than argued from first principles.
"""
from __future__ import annotations

import numpy as np


class RLSCanceller:
    """Same primary/reference interface as `NLMSCanceller` and `LMSCanceller`.

    `delta` initialises the inverse correlation matrix `P = delta^-1 * I`;
    large `delta` (small P) means "start unsure", which is the standard
    choice absent prior knowledge of the coupling path. `forgetting` (lambda)
    close to but below 1.0 lets the filter track a slowly time-varying
    coupling path instead of weighting all of history equally.
    """

    def __init__(self, order: int = 64, forgetting: float = 0.999, delta: float = 1.0):
        assert 0.0 < forgetting <= 1.0
        assert delta > 0.0
        self.order = order
        self.lam = forgetting
        self.delta = delta
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(self.order, dtype=np.float64)
        self.xbuf = np.zeros(self.order, dtype=np.float64)
        self.P = np.eye(self.order, dtype=np.float64) / self.delta

    def process_sample(self, d: float, x: float, adapt: bool = True) -> float:
        self.xbuf[1:] = self.xbuf[:-1]
        self.xbuf[0] = x
        y = float(self.w @ self.xbuf)
        e = d - y
        if adapt:
            Px = self.P @ self.xbuf
            denom = self.lam + float(self.xbuf @ Px)
            k = Px / denom                      # Kalman gain vector
            self.w = self.w + k * e
            self.P = (self.P - np.outer(k, Px)) / self.lam
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


def rls_cancel(primary: np.ndarray, reference: np.ndarray,
              order: int = 64, forgetting: float = 0.999,
              update_mask: np.ndarray | None = None) -> np.ndarray:
    """One-shot wrapper: fresh filter state, whole clip at once."""
    return RLSCanceller(order=order, forgetting=forgetting).process_block(
        primary, reference, update_mask)
