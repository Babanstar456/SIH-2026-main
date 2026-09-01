"""Ad-hoc verification for the rnnoise/deepfilternet baselines - real measured
PESQ/STOI/SI-SDR on real audio, not a smoke test. Not part of the pipeline."""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import methods
from src.metrics import evaluate_pair

PAIRS = [
    ("voice3", ROOT / "test-result/voice/voice_dry3.wav", ROOT / "test-result/voice/voice_noisy3.wav"),
    ("multimic", ROOT / "results/multimic_demo/target.wav", ROOT / "results/multimic_demo/primary.wav"),
]

METHOD_NAMES = ["unprocessed", "rnnoise", "deepfilternet", "gtcrn:checkpoints/shipped_best.pt"]


def load(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def main():
    rows = []
    for tag, clean_path, noisy_path in PAIRS:
        if not clean_path.exists() or not noisy_path.exists():
            print(f"[skip] {tag}: missing {clean_path if not clean_path.exists() else noisy_path}")
            continue
        clean, sr_c = load(clean_path)
        noisy, sr_n = load(noisy_path)
        assert sr_c == sr_n, (sr_c, sr_n)
        sr = sr_c
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n], noisy[:n]

        for name in METHOD_NAMES:
            try:
                f = methods.get(name)
            except Exception as exc:
                rows.append((tag, name, f"UNAVAILABLE: {exc}"))
                continue
            try:
                t0 = time.time()
                enh = f(noisy, sr)
                dt = time.time() - t0
                enh = np.asarray(enh, dtype=np.float32)
                m = evaluate_pair(clean, noisy, enh, sr)
                rows.append((tag, name, m, dt))
                print(f"{tag:10s} {name:35s} pesq={m['pesq']:.3f} stoi={m['stoi']:.3f} "
                      f"si_sdr={m['si_sdr']:.2f} si_sdr_gain={m['si_sdr_gain']:+.2f} "
                      f"(pesq_noisy={m['pesq_noisy']:.3f} stoi_noisy={m['stoi_noisy']:.3f} "
                      f"si_sdr_noisy={m['si_sdr_noisy']:.2f})  [{dt:.1f}s]")
            except Exception as exc:
                import traceback
                traceback.print_exc()
                rows.append((tag, name, f"ERROR: {exc}"))
                print(f"{tag:10s} {name:35s} ERROR: {exc}")

    print()
    print("=== raw rows ===")
    for r in rows:
        print(r)

    import json
    out = ROOT / "results" / "baseline_comparison.json"
    payload = []
    for r in rows:
        tag, name = r[0], r[1]
        if isinstance(r[2], dict):
            payload.append({"pair": tag, "method": name, "metrics": r[2],
                            "seconds": r[3]})
        else:
            payload.append({"pair": tag, "method": name, "error": r[2]})
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
