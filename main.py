#!/usr/bin/env python3
"""Live microphone -> speech enhancement -> speaker output.

Entry point for running the shipped model live on this machine's own audio
devices. Thin wrapper around `src.stream_demo` - the same `StreamingEnhancer`
used for every latency/fidelity measurement in this project, so what comes
out of your speakers is the real streaming contract (one 16 ms frame per
audio callback, caches carried forward), not an offline approximation.

    python main.py --list                          # find your device indices
    python main.py                                  # run with system defaults
    python main.py --in-device 1 --out-device 4
    python main.py --floor-db none                  # raw model, no suppression cap
    python main.py --onnx artifacts/model_simple.onnx --floor-db -18

WEAR HEADPHONES for the output. Through a speaker the microphone hears the
processed audio played back and you get a feedback loop. Do not select a
Bluetooth headset as the INPUT - in call mode it collapses to narrowband and
destroys consonants before the model ever sees them (see CLAUDE.md).

Suppression depth is a dial, not something to maximise: at negative input
SNR the unconstrained model removes speech along with the noise (measured -
see README's status table). `--floor-db -18` is this project's current best
operating point, and is the default here; the raw model is one flag away for
comparison.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Env vars let dhwanik.service configure this without editing the unit file -
# set them in an EnvironmentFile (see dhwanik.env.example). CLI flags, when
# given, always win over these.
DEFAULT_ONNX = os.environ.get("DHWANIK_ONNX", "artifacts/model_lowsnr_simple.onnx")
DEFAULT_FLOOR_DB = os.environ.get("DHWANIK_FLOOR_DB", "-18")
DEFAULT_IN_DEVICE = os.environ.get("DHWANIK_IN_DEVICE")
DEFAULT_OUT_DEVICE = os.environ.get("DHWANIK_OUT_DEVICE")


def _parse_floor(value: str | None) -> float | None:
    if value is None or value.lower() == "none":
        return None
    return float(value)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=DEFAULT_ONNX,
                    help=f"path to the streaming ONNX model (default: {DEFAULT_ONNX}, "
                         f"override with $DHWANIK_ONNX)")
    ap.add_argument("--floor-db", type=str, default=str(DEFAULT_FLOOR_DB),
                    help="suppression depth cap in dB, e.g. -18. Pass 'none' for "
                         "the raw, uncapped model (default: -18, override with "
                         "$DHWANIK_FLOOR_DB)")
    ap.add_argument("--in-device", default=DEFAULT_IN_DEVICE,
                    help="input device index or name substring ($DHWANIK_IN_DEVICE)")
    ap.add_argument("--out-device", default=DEFAULT_OUT_DEVICE,
                    help="output device index or name substring ($DHWANIK_OUT_DEVICE)")
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    a = ap.parse_args()

    if a.list:
        try:
            import sounddevice as sd
            print(sd.query_devices())
        except OSError as e:
            raise SystemExit(
                f"could not load the audio backend ({e}).\n"
                f"On Linux this needs the system PortAudio library, e.g. "
                f"`sudo apt install libportaudio2`.")
        return

    onnx_path = Path(a.onnx)
    if not onnx_path.is_absolute():
        onnx_path = ROOT / onnx_path
    if not onnx_path.exists():
        raise SystemExit(
            f"{onnx_path} not found.\n"
            f"Available models: {sorted(p.name for p in (ROOT / 'artifacts').glob('*.onnx'))}\n"
            f"Pass --onnx <path> to pick one, or run `python -m src.export_onnx` "
            f"to build one from a checkpoint (needs PyTorch).")

    try:
        floor_db = _parse_floor(a.floor_db)
    except ValueError:
        raise SystemExit(f"--floor-db must be a number or 'none', got {a.floor_db!r}")

    def dev(v):
        return int(v) if v is not None and str(v).isdigit() else v

    from src.stream_demo import run_live

    print(f"model      : {onnx_path.name}")
    print(f"floor-db   : {floor_db if floor_db is not None else 'none (raw model)'}")
    print("WEAR HEADPHONES for the output - see this file's module docstring.\n")

    try:
        run_live(onnx_path, dev(a.in_device), dev(a.out_device), floor_db)
    except OSError as e:
        raise SystemExit(
            f"could not open an audio device ({e}).\n"
            f"Run `python main.py --list` to see valid device indices, then "
            f"pass --in-device / --out-device.")


if __name__ == "__main__":
    main()
