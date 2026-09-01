"""What does the model actually remove when it over-suppresses?

RESUME.md's own "Where to pick up" list flagged this as unanswered: the floor
sweep shows suppression-vs-word-survival as a curve, never the mechanism. This
compares two outputs of the SAME streaming run on the SAME input at two
`--floor-db` settings (so they are sample-aligned to each other with no
timing correction needed - unlike comparing either one against a separate
clean recording, see CLAUDE.md invariant #10) and asks, band by band and
frame by frame, where the EXTRA suppression between the floor-capped and the
full model actually lands.

Uses this project's own established consonant/vowel band definitions
(`scripts/intelligibility.py`: vowel 200-800 Hz first formant, fricative/stop
2000-6000 Hz) rather than inventing a new measure.

    python scripts/spectrogram_diff.py \\
        --a test-result/floors/floor_18dB.wav \\
        --b test-result/floors/floor_full_model.wav \\
        --unprocessed test-result/floors/floor_none_unprocessed.wav
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
from src.framing import HOP, N_FFT, SR                            # noqa: E402
from src.baselines.classical import _analyse                      # noqa: E402

_EPS = 1e-10
_FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)
_VOWEL_BAND = (_FREQS >= 200) & (_FREQS < 800)
_FRIC_BAND = (_FREQS >= 2000) & (_FREQS < 6000)


def _mag_db(x: np.ndarray) -> np.ndarray:
    spec, _, _ = _analyse(np.asarray(x, dtype=np.float64))
    return 20 * np.log10(np.abs(spec) + _EPS)  # (frames, freq_bins)


def _band_energy_db(mag_db: np.ndarray, sel: np.ndarray) -> np.ndarray:
    lin = 10 ** (mag_db[:, sel] / 20.0)
    return 20 * np.log10(np.sqrt((lin ** 2).mean(axis=1)) + _EPS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="less-suppressed file, e.g. floor_18dB.wav")
    ap.add_argument("--b", required=True, help="more-suppressed file, e.g. floor_full_model.wav")
    ap.add_argument("--unprocessed", default=None,
                    help="optional, for active-speech masking context")
    ap.add_argument("--out-png", default="results/spectrogram_diff.png")
    args = ap.parse_args()

    xa = A.load_audio(args.a)
    xb = A.load_audio(args.b)
    n = min(len(xa), len(xb))
    xa, xb = xa[:n], xb[:n]

    db_a = _mag_db(xa)
    db_b = _mag_db(xb)
    m = min(db_a.shape[0], db_b.shape[0])
    db_a, db_b = db_a[:m], db_b[:m]
    diff = db_b - db_a  # negative = full model suppressed MORE here than floor-capped

    vowel_a, vowel_b = _band_energy_db(db_a, _VOWEL_BAND), _band_energy_db(db_b, _VOWEL_BAND)
    fric_a, fric_b = _band_energy_db(db_a, _FRIC_BAND), _band_energy_db(db_b, _FRIC_BAND)
    cvr_a = float(np.median(fric_a - vowel_a))
    cvr_b = float(np.median(fric_b - vowel_b))

    extra_suppression_vowel = float(np.median(diff[:, _VOWEL_BAND]))
    extra_suppression_fric = float(np.median(diff[:, _FRIC_BAND]))

    # Is the extra suppression concentrated in short bursts, or spread evenly?
    # High kurtosis => concentrated in a few frames (consistent with "gating
    # brief high-frequency events"); near-Gaussian => spread through the clip.
    frame_diff = diff[:, _FRIC_BAND].mean(axis=1)
    frame_diff_z = (frame_diff - frame_diff.mean()) / (frame_diff.std() + _EPS)
    kurtosis = float(np.mean(frame_diff_z ** 4) - 3.0)  # excess kurtosis

    hop_ms = 1000.0 * HOP / SR
    worst_frames = np.argsort(frame_diff)[:10]
    worst_times_s = sorted((worst_frames * hop_ms / 1000.0).tolist())

    print(f"CVR, floor-capped (--a): {cvr_a:.2f} dB")
    print(f"CVR, full model   (--b): {cvr_b:.2f} dB")
    print(f"(clean-speech reference CVR, per scripts/intelligibility.py: -10.68 dB)")
    print()
    print(f"Extra suppression going floor-capped -> full model:")
    print(f"  vowel band   (200-800 Hz):  {extra_suppression_vowel:+.2f} dB (median)")
    print(f"  fricative band (2-6 kHz):   {extra_suppression_fric:+.2f} dB (median)")
    print()
    print(f"Excess kurtosis of fricative-band extra-suppression across time: {kurtosis:.2f}")
    print("  (near 0 = spread evenly through the clip; high = concentrated in a few frames)")
    print(f"  10 most-suppressed frames (seconds into clip): {[round(t,2) for t in worst_times_s]}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        extent = [0, m * hop_ms / 1000.0, 0, SR / 2 / 1000.0]
        axes[0].imshow(db_a.T, aspect="auto", origin="lower", extent=extent,
                      cmap="magma", vmin=-80, vmax=0)
        axes[0].set_title(f"{Path(args.a).name} (floor-capped)")
        axes[0].set_ylabel("kHz")

        axes[1].imshow(db_b.T, aspect="auto", origin="lower", extent=extent,
                      cmap="magma", vmin=-80, vmax=0)
        axes[1].set_title(f"{Path(args.b).name} (full model)")
        axes[1].set_ylabel("kHz")

        im = axes[2].imshow(diff.T, aspect="auto", origin="lower", extent=extent,
                            cmap="RdBu", vmin=-30, vmax=30)
        axes[2].set_title("difference (full model - floor-capped); blue = extra suppression")
        axes[2].set_ylabel("kHz")
        axes[2].set_xlabel("seconds")
        fig.colorbar(im, ax=axes[2], label="dB")

        fig.tight_layout()
        out_png = ROOT / args.out_png
        fig.savefig(out_png, dpi=150)
        print(f"\nwrote {out_png}")
    except ImportError:
        print("\nmatplotlib not available, skipped the plot")


if __name__ == "__main__":
    main()
