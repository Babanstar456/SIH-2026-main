"""Run the shipped model over one noise recording at a range of input SNRs.

WHY. When a field recording comes back unintelligible there are two candidate
explanations - the model failed, or the recording never carried the speech in
the first place - and they are indistinguishable by ear. This settles it. Take
clean speech, mix in the SAME noise at a sweep of known SNRs, enhance each, and
listen down the ladder until it breaks. Where it breaks is the model's operating
limit; if the field recording sits below that, the recording is the problem.

Speech is scaled by ACTIVE-speech RMS, not whole-clip RMS, for the reason in
audio.active_speech_mask: silence in the clip otherwise skews the ratio and two
clips at nominally the same SNR differ audibly.

    python scripts/snr_sweep.py --speech results/demo60/reference_clean.wav \
        --noise test-result/before.wav --noise-start 2.5 \
        --out-dir test-result/snr_sweep
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
from src.stream_demo import StreamingEnhancer                     # noqa: E402
from src.baselines.classical import _analyse                      # noqa: E402

_EPS = 1e-10
BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def band_snr(speech: np.ndarray, noise: np.ndarray) -> list[float]:
    """Per-band SNR - the number that actually predicts intelligibility.

    A single broadband SNR hides the failure this script exists to find: a clip
    can read -13 dB overall while the 1-4 kHz band that carries the consonants
    sits 30 dB under. Whole-clip SNR then looks merely difficult instead of
    impossible.
    """
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    ms = np.abs(_analyse(speech)[0]) ** 2
    mn = np.abs(_analyse(noise)[0]) ** 2
    n = min(len(ms), len(mn))
    out = []
    for f1, f2 in BANDS:
        sel = (freqs >= f1) & (freqs < f2)
        out.append(10 * np.log10((ms[:n, sel].mean() + _EPS) / (mn[:n, sel].mean() + _EPS)))
    return out


def enhance(x: np.ndarray, onnx: Path) -> np.ndarray:
    """Stream the clip through the shipped ONNX, one 16 ms frame at a time."""
    enh = StreamingEnhancer(onnx)
    n = len(x) // HOP
    y = np.zeros(n * HOP, dtype=np.float32)
    for i in range(n):
        y[i * HOP:(i + 1) * HOP] = enh.process_chunk(x[i * HOP:(i + 1) * HOP])
    return y


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speech", type=Path, required=True, help="CLEAN speech wav")
    p.add_argument("--noise", type=Path, required=True)
    p.add_argument("--noise-start", type=float, default=0.0,
                   help="seconds to skip into the noise file")
    p.add_argument("--onnx", type=Path, default=ROOT / "artifacts" / "model_simple.onnx")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--snr", type=float, nargs="+",
                   default=[15.0, 10.0, 5.0, 0.0, -5.0, -13.0])
    p.add_argument("--dur", type=float, default=20.0)
    p.add_argument("--limit", type=float, default=0.89,
                   help="analog limiter modelled ahead of the ADC")
    args = p.parse_args()

    speech = A.load_audio(args.speech)
    noise = A.load_audio(args.noise)[int(args.noise_start * SR):]
    n = int(args.dur * SR)
    if len(speech) < n or len(noise) < n:
        n = min(len(speech), len(noise))
        print("note: trimmed to %.1f s by the shorter input" % (n / SR))
    speech, noise = speech[:n], noise[:n]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    A.save_audio(args.out_dir / "clean_reference.wav", speech)
    sp_rms = A.active_rms(speech, SR)

    print("speech %.1f s, active %.1f dBFS | noise %.1f dBFS"
          % (n / SR, 20 * np.log10(sp_rms + _EPS),
             20 * np.log10(A.rms(noise) + _EPS)))
    print("\n%-8s %-26s %s" % ("", "per-band SNR of the MIX (dB)", "enhanced"))
    print("%-8s %6s %6s %6s %6s %6s   %s"
          % ("SNR", "0-.5k", ".5-1k", "1-2k", "2-4k", "4-8k", "active dBFS"))

    for snr in args.snr:
        g = A.scale_noise_for_snr(sp_rms, noise, snr)
        nz = noise * g
        mix = A.soft_limit(speech + nz, args.limit)
        out = enhance(mix, args.onnx)

        bs = band_snr(speech, nz)
        tag = ("m%d" % abs(snr)) if snr < 0 else ("p%d" % snr)
        A.save_audio(args.out_dir / ("snr_%s_before.wav" % tag), mix)
        A.save_audio(args.out_dir / ("snr_%s_after.wav" % tag), out)
        print("%-+8.0f %6.1f %6.1f %6.1f %6.1f %6.1f   %6.1f"
              % (snr, *bs, 20 * np.log10(A.active_rms(out, SR) + _EPS)))

    print("\nwrote %d pairs to %s" % (len(args.snr), args.out_dir))
    print("Listen DOWN the ladder. The SNR where speech stops surviving is the "
          "model's limit on this noise.")


if __name__ == "__main__":
    main()
