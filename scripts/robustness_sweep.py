"""Operating envelope: how far can the talker degrade before the model fails?

WHY. A test recording made in a quiet room with a cooperative speaker says
almost nothing about the field, where the voice is quieter, strained, and
muffled by a mask or a helmet. This sweeps both axes that actually move -
how far the voice sits under the noise, and how much high frequency the voice
has lost - and reports where the output stops being usable.

This is the first measurement in the project that can use REFERENCE metrics on
this speaker: the mixture is built from a dry take, so that take IS the clean
reference and `pesq`/`pystoi` apply directly. Everything measured on the raw
field recordings was a proxy, because no clean reference for them exists.

Alignment matters here and is easy to get wrong - the streaming model lags its
input by exactly win-hop (256 samples). PESQ hides that by time-aligning
internally; SI-SDR and STOI do not, and an uncompensated comparison scores a
good model at absurdly negative SI-SDR. See CLAUDE.md invariant 10.

    python scripts/robustness_sweep.py --speech test-result/voice/voice_dry.wav \
        --noise test-result/before.wav --noise-start 2.5 \
        --out-dir test-result/envelope
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
from src import metrics as M                                      # noqa: E402
from src.framing import HOP, N_FFT, SR, WIN                       # noqa: E402
from src.stream_demo import StreamingEnhancer                     # noqa: E402
from src.baselines.classical import _analyse, _synthesise         # noqa: E402

_EPS = 1e-10
LAG = WIN - HOP          # 256 samples: the streaming overlap-add delay


def muffle(x: np.ndarray, db_at_4k: float, knee_hz: float = 800.0) -> np.ndarray:
    """Roll the highs off the way a mask, a helmet or a strained voice does.

    A first-order tilt above `knee_hz` reaching -db_at_4k at 4 kHz, held flat
    above. Modelled on the measured difference between the Bluetooth capture and
    a normal one, which is a smooth tilt rather than a codec brick wall.
    """
    if db_at_4k <= 0:
        return x
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    oct_above = np.log2(np.maximum(freqs, knee_hz) / knee_hz)
    span = np.log2(4000.0 / knee_hz)
    g = -db_at_4k * np.clip(oct_above / span, 0.0, 1.0)
    S, w, nf = _analyse(x)
    return _synthesise(S * 10 ** (g[None, :] / 20.0), w, nf, len(x))


def enhance(x: np.ndarray, onnx: Path) -> np.ndarray:
    enh = StreamingEnhancer(onnx)
    n = len(x) // HOP
    y = np.zeros(n * HOP, dtype=np.float32)
    for i in range(n):
        y[i * HOP:(i + 1) * HOP] = enh.process_chunk(x[i * HOP:(i + 1) * HOP])
    return y


def scored(clean: np.ndarray, est: np.ndarray, compensate_lag: bool) -> dict:
    """Metrics against the clean reference, lag-compensated where needed."""
    if compensate_lag:
        est = est[LAG:]
    n = min(len(clean), len(est))
    c, e = clean[:n], est[:n]
    return {"pesq": M.pesq_wb(c, e, SR), "stoi": M.stoi_score(c, e, SR),
            "si_sdr": M.si_sdr_db(c, e)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speech", type=Path, required=True, help="DRY take, no noise")
    p.add_argument("--noise", type=Path, required=True)
    p.add_argument("--noise-start", type=float, default=0.0)
    p.add_argument("--onnx", type=Path, default=ROOT / "artifacts" / "model_simple.onnx")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--snr", type=float, nargs="+", default=[10., 5., 0., -5.])
    p.add_argument("--muffle", type=float, nargs="+", default=[0., 8.],
                   help="dB of high-frequency loss at 4 kHz")
    p.add_argument("--dur", type=float, default=20.0)
    p.add_argument("--speech-start", type=float, default=0.0)
    p.add_argument("--limit", type=float, default=0.89)
    args = p.parse_args()

    speech = A.load_audio(args.speech)[int(args.speech_start * SR):]
    noise = A.load_audio(args.noise)[int(args.noise_start * SR):]
    n = int(args.dur * SR)
    if len(speech) < n or len(noise) < n:
        n = min(len(speech), len(noise))
        print("note: trimmed to %.1f s by the shorter input" % (n / SR))
    speech, noise = speech[:n], noise[:n]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not M.PESQ_AVAILABLE:
        print("WARNING: pesq unavailable - PESQ columns will be NaN, never read as 0")

    print("\n%-7s %-7s | %-19s | %-19s" % ("muffle", "SNR", "NOISY", "ENHANCED"))
    print("%-7s %-7s | %6s %6s %6s | %6s %6s %6s   %s"
          % ("dB@4k", "dB", "PESQ", "STOI", "SI-SDR", "PESQ", "STOI", "SI-SDR", "verdict"))

    for mf in args.muffle:
        clean = muffle(speech, mf).astype(np.float32)
        sp_rms = A.active_rms(clean, SR)
        A.save_audio(args.out_dir / ("clean_muffle%d.wav" % mf), clean)
        for snr in args.snr:
            g = A.scale_noise_for_snr(sp_rms, noise, snr)
            mix = A.soft_limit(clean + noise * g, args.limit)
            out = enhance(mix, args.onnx)

            nz = scored(clean, mix, compensate_lag=False)
            en = scored(clean, out, compensate_lag=True)
            tag = "m%d_snr%s" % (mf, ("m%d" % abs(snr)) if snr < 0 else str(int(snr)))
            A.save_audio(args.out_dir / (tag + "_before.wav"), mix)
            A.save_audio(args.out_dir / (tag + "_after.wav"), out)

            good = en["stoi"] >= 0.75 and en["pesq"] >= 1.8
            fair = en["stoi"] >= 0.65
            print("%-7.0f %-+7.0f | %6.3f %6.3f %+6.1f | %6.3f %6.3f %+6.1f   %s"
                  % (mf, snr, nz["pesq"], nz["stoi"], nz["si_sdr"],
                     en["pesq"], en["stoi"], en["si_sdr"],
                     "usable" if good else ("marginal" if fair else "FAILS")))

    print("\nwrote %s" % args.out_dir)
    print("STOI is the intelligibility number: >0.75 usable, 0.65-0.75 marginal, "
          "<0.65 words start disappearing.")


if __name__ == "__main__":
    main()
