"""Trade suppression depth against word survival, and measure both.

WHY. The model suppresses gunfire well and costs words: on a real -5.9 dB
recording it took ASR word score from 62% (unprocessed) to 35%. Those two facts
are the same fact - at low SNR the mask cuts so hard that speech goes with the
noise. A mask FLOOR buys words back by refusing to cut any bin deeper than a set
depth:

    G_final = max(G_model, floor)

At floor = 0 dB nothing is suppressed and you have the microphone back. At
floor = -inf you have the model untouched. Somewhere between is the best
available operating point, and it cannot be reasoned about - it has to be
measured on both axes at once, which is what this does.

Pair it with `asr_score.py` on the files this writes. Suppression is reported
here; word score is reported there. Optimise the pair, never one alone - that is
how the earlier chain ended up destroying intelligibility while every proxy
metric improved.

    python scripts/floor_sweep.py --input noisy.wav --ckpt checkpoints/lowsnr_best.pt \
        --out-dir test-result/floors
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
from src.framing import SR                                        # noqa: E402
from src.baselines.classical import _analyse, _synthesise         # noqa: E402

_EPS = 1e-10


def suppression_db(noisy: np.ndarray, out: np.ndarray) -> tuple[float, float]:
    """(noise-floor reduction, speech-to-noise contrast gain) in dB.

    Noise frames are taken as the quietest 15% of the NOISY signal and speech
    frames the loudest 15%, so both are defined on the input and stay fixed
    across every candidate - otherwise each candidate would be scored against
    its own moved goalposts.
    """
    mb = np.abs(_analyse(noisy)[0])
    ma = np.abs(_analyse(out)[0])
    n = min(len(mb), len(ma))
    fb = 20 * np.log10(np.sqrt((mb[:n] ** 2).mean(1)) + _EPS)
    fa = 20 * np.log10(np.sqrt((ma[:n] ** 2).mean(1)) + _EPS)
    quiet = fb <= np.percentile(fb, 15)
    loud = fb >= np.percentile(fb, 85)
    drop = fb[quiet].mean() - fa[quiet].mean()
    contrast = (fa[loud].mean() - fa[quiet].mean()) - (fb[loud].mean() - fb[quiet].mean())
    return drop, contrast


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--ckpt", default="checkpoints/lowsnr_best.pt")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--floors", type=float, nargs="+",
                   default=[-6.0, -9.0, -12.0, -18.0, -24.0])
    p.add_argument("--target-dbfs", type=float, default=-20.0)
    p.add_argument("--limit", type=float, default=0.89)
    args = p.parse_args()

    from src import methods

    x = A.load_audio(args.input)
    enh = np.asarray(methods.get("gtcrn:" + str(args.ckpt))(x, SR), dtype=np.float32)
    n = min(len(x), len(enh))
    x, enh = x[:n], enh[:n]

    def lvl(y):
        return A.soft_limit(y * (10 ** (args.target_dbfs / 20.0)
                                 / max(A.active_rms(y, SR), _EPS)),
                            args.limit).astype(np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    Sx, w, nf = _analyse(x)
    Se, _, _ = _analyse(enh)
    nfr = min(len(Sx), len(Se))
    Sx, Se = Sx[:nfr], Se[:nfr]
    g_model = np.abs(Se) / (np.abs(Sx) + _EPS)

    A.save_audio(args.out_dir / "floor_none_unprocessed.wav", lvl(x))
    A.save_audio(args.out_dir / "floor_full_model.wav", lvl(enh))

    print("%-32s %14s %14s" % ("candidate", "noise drop dB", "contrast dB"))
    d, c = suppression_db(x, x)
    print("%-32s %14.1f %14.1f" % ("unprocessed", d, c))
    d, c = suppression_db(x, enh)
    print("%-32s %14.1f %14.1f" % ("full model (no floor)", d, c))

    for f in args.floors:
        g = np.maximum(g_model, 10 ** (f / 20.0))
        y = _synthesise(g * Sx, w, nfr, len(x))
        d, c = suppression_db(x, y)
        name = "floor_%ddB.wav" % abs(f)
        A.save_audio(args.out_dir / name, lvl(y))
        print("%-32s %14.1f %14.1f" % ("floor %+.0f dB" % f, d, c))

    print("\nwrote %s" % args.out_dir)
    print("Now score the SAME files for words:")
    print("  python scripts/asr_score.py --inputs %s/*.wav" % args.out_dir)


if __name__ == "__main__":
    main()
