"""Push word intelligibility, not just noise removal.

WHY. Suppression gets the voice out of the noise; a presence EQ puts the formant
region back at the right AVERAGE level. Neither touches the reason degraded
speech is hard to follow: consonants are 20-30 dB quieter and far shorter than
the vowels around them, so they are the first thing a damaged channel loses, and
they are what distinguishes one word from another. A flat EQ lifts the vowels by
exactly as much, so it cannot help here.

The number that matters is the consonant-to-vowel ratio. Measured on this
project's own clean reference it is -10.68 dB. On the test speaker:

    clean speech reference        -10.68 dB
    speaker's dry recording       -16.07 dB   <- 5.4 dB lost at the microphone
    after the model               -18.07 dB   <- model costs another 2.0 dB
    after presence EQ             -19.65 dB   <- EQ costs 1.6 dB more

So three stages each take a bite out of exactly the cue that carries word
identity, and the EQ that makes the audio sound better makes this WORSE.

WHAT WORKS. Boosting frames by how fricative-like they are - ranked by spectral
tilt, so bursts and fricatives are raised and vowels are not. At 12 dB this
restores CVR to -10.96 dB, matching clean speech.

WHAT DOES NOT WORK, MEASURED. Multiband upward compression - raising quiet
moments toward loud ones per band - is the textbook move here and it is wrong
for this problem. It lifts every quiet frame, and most quiet frames are pauses
and vowel tails rather than consonants, so it drives CVR the wrong way: -19.65
to -22.92 dB at 15 dB of boost, worse still with spectral contrast on top. It
was implemented, measured, and removed. Do not reintroduce it without measuring
CVR.

NOTHING HERE INVENTS SIGNAL. Every stage redistributes energy already present.
Bandwidth extension - hallucinating the missing 4-8 kHz - would sound clearer
still and is deliberately NOT done: on a communications path a misheard grid
reference is worse than an obviously muffled one.

    python scripts/intelligibility.py --input in.wav --out out.wav
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

_FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)
_VOWEL_BAND = (_FREQS >= 200) & (_FREQS < 800)     # first formant region
_FRIC_BAND = (_FREQS >= 2000) & (_FREQS < 6000)    # fricatives and stop bursts


def _ltas(x: np.ndarray) -> np.ndarray:
    m = A.active_speech_mask(x, SR)
    x = x[m] if m.any() else x
    p = 20 * np.log10(np.abs(_analyse(x)[0]).mean(axis=0) + 1e-9)
    return p - p.max()


def _band_db(mag: np.ndarray, sel: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.sqrt((mag[:, sel] ** 2).mean(axis=1)) + _EPS)


def _active(mag: np.ndarray) -> np.ndarray:
    fdb = 20 * np.log10(np.sqrt((mag ** 2).mean(axis=1)) + _EPS)
    return fdb > np.percentile(fdb, 95) - 35.0


def spectral_tilt(mag: np.ndarray) -> np.ndarray:
    """High-band minus low-band level per frame. High = fricative-like."""
    return _band_db(mag, _FRIC_BAND) - _band_db(mag, _VOWEL_BAND)


def consonant_boost(mag: np.ndarray, boost_db: float) -> np.ndarray:
    """Raise frames in proportion to how fricative-like they are.

    Ranked against this recording's own tilt distribution rather than an
    absolute threshold, so it adapts to a speaker and a microphone instead of
    assuming one. Vowel frames sit at the bottom of that ranking and are left
    untouched, which is the whole difference between this and a compressor.
    """
    if boost_db <= 0:
        return mag
    act = _active(mag)
    if act.sum() < 8:
        return mag
    t = spectral_tilt(mag)
    lo, hi = np.percentile(t[act], 20), np.percentile(t[act], 90)
    w = np.clip((t - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * act
    return mag * 10 ** ((boost_db * w)[:, None] / 20.0)


def spectral_contrast(mag: np.ndarray, beta: float) -> np.ndarray:
    """Sharpen each frame's peaks against its own valleys.

    Formants carry vowel identity and place of articulation; noise fills the
    valleys between them. Worth about 1.5 dB of CVR on its own.
    """
    if beta <= 0:
        return mag
    ker = np.ones(9) / 9
    smooth = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, mag)
    ratio = (mag + _EPS) / (smooth + _EPS)
    return mag * np.clip(ratio, 0.25, 4.0) ** beta


def presence_eq(y: np.ndarray, target: np.ndarray, max_boost_db: float,
                max_cut_db: float, hp_hz: float) -> np.ndarray:
    """Correct the long-term balance toward `target`'s spectrum."""
    g = np.convolve(target - _ltas(y), np.ones(9) / 9, mode="same")
    g = np.clip(g, max_cut_db, max_boost_db)
    g[_FREQS < hp_hz] = max_cut_db
    S, w, nf = _analyse(y)
    return _synthesise(S * 10 ** (g[None, :] / 20.0), w, nf, len(y))


def cvr_db(x: np.ndarray) -> float:
    """Consonant-to-vowel ratio: fricative-like frames vs vowel-like frames."""
    mag = np.abs(_analyse(x)[0])
    fdb = 20 * np.log10(np.sqrt((mag ** 2).mean(axis=1)) + _EPS)
    act = _active(mag)
    if act.sum() < 16:
        return float("nan")
    t = spectral_tilt(mag)
    con = act & (t >= np.percentile(t[act], 67))
    vow = act & (t <= np.percentile(t[act], 33))
    if con.sum() < 4 or vow.sum() < 4:
        return float("nan")
    return float(fdb[con].mean() - fdb[vow].mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    p.add_argument("--consonant-db", type=float, default=12.0,
                   help="consonant-frame boost; 12 matches clean-speech CVR here")
    p.add_argument("--contrast", type=float, default=0.4)
    p.add_argument("--eq-boost-db", type=float, default=14.0)
    p.add_argument("--target-dbfs", type=float, default=-20.0)
    p.add_argument("--limit", type=float, default=0.89)
    args = p.parse_args()

    x = A.load_audio(args.input)
    tgt_audio = A.load_audio(args.target)

    y = presence_eq(x, _ltas(tgt_audio), args.eq_boost_db, -6.0, 120.0)
    S, w, nf = _analyse(y)
    mag, phase = np.abs(S), np.angle(S)
    mag = spectral_contrast(mag, args.contrast)
    mag = consonant_boost(mag, args.consonant_db)
    y = _synthesise(mag * np.exp(1j * phase), w, nf, len(y))
    y = A.soft_limit(y * (10 ** (args.target_dbfs / 20.0)
                          / max(A.active_rms(y, SR), _EPS)), args.limit).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    A.save_audio(args.out, y)
    print("%s -> %s" % (args.input.name, args.out.name))
    print("  CVR  %+.2f dB -> %+.2f dB      (target, clean speech: %+.2f dB)"
          % (cvr_db(x), cvr_db(y), cvr_db(tgt_audio)))
    print("  level %.1f -> %.1f dBFS"
          % (20 * np.log10(A.active_rms(x, SR) + _EPS),
             20 * np.log10(A.active_rms(y, SR) + _EPS)))


if __name__ == "__main__":
    main()
