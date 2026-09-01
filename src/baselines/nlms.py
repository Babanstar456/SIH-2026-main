"""NLMS adaptive noise cancellation from a reference microphone.

Every other method in this package is single-channel, matching the deployed
model's contract (16 kHz mono in — see CLAUDE.md). This one genuinely needs a
SECOND signal, correlated with the noise component of the primary microphone
and (ideally) largely free of the talker's voice. That is a different
interface, which is exactly why it is kept OUT of `methods.py`'s registry
rather than shoehorned into `f(noisy, sr) -> enhanced`: folding a two-input
method into that signature would either silently drop the reference channel
or break the uniformity that makes the single-mic comparison table
trustworthy. See `scripts/eval_multimic.py` for how this is actually
evaluated, and `src/baselines/reference_mic.py` for where the second channel
comes from (synthetic — no real dual-mic recording exists in this project
yet).

Standard normalised LMS (Widrow & Hoff), run causally sample-by-sample so it
is a real streaming filter and not an offline approximation: an embedded
implementation runs this same update inside the 16 ms mic callback.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


class NLMSCanceller:
    """Predicts the reference-correlated component of `primary` and removes it.

    `order` is the adaptive filter length in samples. It has to cover the real
    acoustic path between the two microphones (propagation delay plus any
    reflection difference) - too short and the filter cannot represent the
    coupling at all; too long and it converges more slowly and starts also
    adapting to whatever small amount of speech leaks into the reference
    channel, which is the opposite of what it is for.
    """

    def __init__(self, order: int = 64, mu: float = 0.1, leakage: float = 1.0):
        assert 0.0 < mu <= 1.0, "mu is a normalised step size, keep it in (0, 1]"
        assert 0.0 < leakage <= 1.0
        self.order = order
        self.mu = mu
        self.leakage = leakage      # <1.0 bleeds weights toward zero; guards against drift
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(self.order, dtype=np.float64)
        self.xbuf = np.zeros(self.order, dtype=np.float64)

    def process_sample(self, d: float, x: float, adapt: bool = True) -> float:
        """One sample in from each mic (primary `d`, reference `x`), one
        enhanced sample out.

        `adapt=False` runs the filter (still subtracts its current noise
        estimate) but freezes the weight update - see `process_block` for why
        this exists. Measured to matter: with no reference mic acoustically
        isolated from the talker's mouth, ordinary NLMS has no way to tell
        "correlated because it's noise" from "correlated because it's leaked
        speech", and will happily reduce its output error by cancelling part
        of the speech too.
        """
        self.xbuf[1:] = self.xbuf[:-1]
        self.xbuf[0] = x
        y = float(self.w @ self.xbuf)
        e = d - y
        if adapt:
            norm = float(self.xbuf @ self.xbuf) + _EPS
            self.w = self.leakage * self.w + (self.mu / norm) * e * self.xbuf
        return e

    def process_block(self, primary: np.ndarray, reference: np.ndarray,
                      update_mask: np.ndarray | None = None) -> np.ndarray:
        """Whole-array convenience wrapper. NLMS is inherently sequential -
        each weight update depends on the previous one - so this is a plain
        Python loop, not vectorised; fine for offline evaluation on clips a
        few seconds long, not how an embedded target would be written (that
        would keep `xbuf` as a ring buffer to avoid the per-sample shift).

        `update_mask`, if given, is a per-sample bool array: True where the
        filter is allowed to adapt. Pass False during detected speech activity
        (VAD-gated adaptation) to stop the filter learning to cancel the
        talker along with the noise - the standard technique real headset ANC
        uses for exactly this reason. None adapts on every sample.
        """
        primary = np.asarray(primary, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        assert len(primary) == len(reference), "primary and reference must be aligned"
        if update_mask is not None:
            assert len(update_mask) == len(primary), "update_mask must be aligned too"
        out = np.empty(len(primary), dtype=np.float64)
        for n in range(len(primary)):
            adapt = True if update_mask is None else bool(update_mask[n])
            out[n] = self.process_sample(primary[n], reference[n], adapt=adapt)
        return out.astype(np.float32)


def nlms_cancel(primary: np.ndarray, reference: np.ndarray,
                order: int = 64, mu: float = 0.1,
                update_mask: np.ndarray | None = None) -> np.ndarray:
    """One-shot wrapper: fresh filter state, whole clip at once."""
    return NLMSCanceller(order=order, mu=mu).process_block(primary, reference, update_mask)
