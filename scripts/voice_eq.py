"""Correct a recording's spectral balance toward normal speech.

WHY. Measured on this project's own reference, the test speaker's DRY recording
- no noise, no model - sits 10.4 dB low at 1-2 kHz and 6.2 dB low at 2-3 kHz.
That is the formant region that carries word identity, so the recording reads as
muffled before anything is done to it. The model then removes a further 2.9 to
4.0 dB across the same range. Neither deficit is noise; both are a level
imbalance in bands that DO contain speech, which is why correcting them is sound
where re-opening empty bands was not.

Read that distinction carefully before reusing this. A sibling script (since
deleted) failed on an earlier recording precisely because it lifted bands that
held only noise. The test is whether the speech energy is present-but-quiet (fix
it here) or absent (nothing to fix). Check the SNR in the target band first.

NOTE ON STOI. Correcting toward normal speech will LOWER STOI measured against
the untouched recording, because STOI scores fidelity to that reference - and
the reference is the thing being corrected. A high STOI against a muffled
reference means "faithfully muffled", not "intelligible". Judge this by ear.

    python scripts/voice_eq.py --input in.wav --out out.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import audio as A                                        # noqa: E402
from src.framing import N_FFT, SR                                 # noqa: E402
from src.baselines.classical import _analyse, _synthesise         # noqa: E402

_EPS = 1e-10
DEFAULT_TARGET = ROOT / "results" / "demo60" / "reference_clean.wav"


def ltas(x: np.ndarray) -> np.ndarray:
    """Long-term average spectrum over ACTIVE frames, normalised to its peak.

    Active-frames-only for the same reason SNR is measured that way: silence
    drags the average down and the correction would chase it.
    """
    m = A.active_speech_mask(x, SR)
    x = x[m] if m.any() else x
    p = 20 * np.log10(np.abs(_analyse(x)[0]).mean(axis=0) + 1e-9)
    return p - p.max()


def correction(src: np.ndarray, tgt: np.ndarray, smooth_bins: int,
               lo_db: float, hi_db: float, hp_hz: float) -> np.ndarray:
    """Per-bin gain taking `src` toward `tgt`, smoothed and clamped."""
    g = tgt - src
    k = max(1, int(smooth_bins))
    g = np.convolve(g, np.ones(k) / k, mode="same")
    g = np.clip(g, lo_db, hi_db)
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    g[freqs < hp_hz] = lo_db          # rumble below the voice buys nothing
    return g


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                   help="recording whose spectral balance to aim at")
    p.add_argument("--max-boost-db", type=float, default=14.0)
    p.add_argument("--max-cut-db", type=float, default=-6.0)
    p.add_argument("--smooth-bins", type=int, default=9)
    p.add_argument("--hp-hz", type=float, default=120.0)
    p.add_argument("--target-dbfs", type=float, default=-20.0)
    p.add_argument("--limit", type=float, default=0.89)
    p.add_argument("--no-level", action="store_true")
    args = p.parse_args()

    x = A.load_audio(args.input)
    t = A.load_audio(args.target)
    g = correction(ltas(x), ltas(t), args.smooth_bins,
                   args.max_cut_db, args.max_boost_db, args.hp_hz)

    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    print("correction applied (dB):")
    for a, b in [(100, 300), (300, 500), (500, 1000), (1000, 2000),
                 (2000, 3000), (3000, 4000), (4000, 6000)]:
        s = (freqs >= a) & (freqs < b)
        print("   %5d-%5d Hz  %+6.1f" % (a, b, g[s].mean()))

    S, w, nf = _analyse(x)
    y = _synthesise(S * 10 ** (g[None, :] / 20.0), w, nf, len(x))
    if not args.no_level:
        y = A.soft_limit(y * (10 ** (args.target_dbfs / 20.0)
                              / max(A.active_rms(y, SR), _EPS)), args.limit)
    y = y.astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    A.save_audio(args.out, y)
    print("\n%-14s active %6.1f dBFS -> %-14s active %6.1f dBFS"
          % (args.input.name, 20 * np.log10(A.active_rms(x, SR) + _EPS),
             args.out.name, 20 * np.log10(A.active_rms(y, SR) + _EPS)))


if __name__ == "__main__":
    main()
