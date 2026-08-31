"""Prepare the VoiceBank-DEMAND test set as a second, external benchmark.

Our own frozen test set measures what we care about (gunfire, artillery, rotor).
VoiceBank-DEMAND measures something different and equally necessary: whether our
numbers are in the same universe as everyone else's. It is the set essentially
every speech-enhancement paper reports on, so a PESQ here can be compared
directly against published results - which a score on a test set we invented
cannot be.

The corpus ships at 48 kHz; we resample to 16 kHz, which is what every paper
reporting on it does. Skipping that step would make our numbers quietly
incomparable to the literature we are comparing against.

    python scripts/make_vbd_testset.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SR = 16000
ZIPS = {"vbd_clean_testset.zip": "clean_testset_wav",
        "vbd_noisy_testset.zip": "noisy_testset_wav"}


def _resample_one(args):
    src, dst = Path(args[0]), Path(args[1])
    if dst.exists():
        return True
    try:
        x, in_sr = sf.read(str(src), dtype="float32", always_2d=True)
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        if in_sr != SR:
            x = soxr.resample(x, in_sr, SR, quality="HQ")
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), x.astype(np.float32), SR, subtype="PCM_16")
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    with open(ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    raw = Path(cfg["paths"]["raw"])
    out = Path(cfg["paths"]["testset"]).parent / "voicebank_demand"
    ext = raw / "extracted"

    for zname, sub in ZIPS.items():
        zpath = raw / zname
        if not (raw / f"{zname}.done").exists():
            raise SystemExit(f"{zname} not downloaded yet")
        marker = ext / f".{zname}.extracted"
        if not marker.exists():
            print(f"[unpack] {zname}")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(ext)
            marker.touch()

    clean_src = ext / ZIPS["vbd_clean_testset.zip"]
    noisy_src = ext / ZIPS["vbd_noisy_testset.zip"]
    if not clean_src.is_dir() or not noisy_src.is_dir():
        raise SystemExit(f"expected {clean_src} and {noisy_src}")

    names = sorted(p.name for p in clean_src.glob("*.wav"))
    paired = [n for n in names if (noisy_src / n).exists()]
    if len(paired) != len(names):
        print(f"[warn ] {len(names)-len(paired)} clean files have no noisy pair; "
              f"using the {len(paired)} matched pairs only")

    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for n in paired:
        jobs.append((str(clean_src / n), str(out / "clean" / n)))
        jobs.append((str(noisy_src / n), str(out / "noisy" / n)))

    ok = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_resample_one, j) for j in jobs]
        for fu in tqdm(as_completed(futs), total=len(futs), ncols=80,
                       unit="f", desc="48k->16k"):
            ok += bool(fu.result())

    items = [{"id": Path(n).stem, "category": "voicebank_demand",
              "noisy": f"noisy/{n}", "clean": f"clean/{n}",
              "mask": None, "speaker": Path(n).stem.split("_")[0]}
             for n in paired]

    index = {
        "seed": None, "sr": SR, "source": "VoiceBank-DEMAND test set (10283/2791)",
        "note": ("External benchmark, resampled 48 kHz -> 16 kHz as is standard "
                 "practice. Unseen speakers and unseen noise types by the "
                 "corpus's own design. Used for comparability with published "
                 "results, NOT as our primary defence-noise evaluation."),
        "items": items,
    }
    with open(out / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)

    print(f"\n{ok}/{len(jobs)} files converted")
    print(f"wrote {len(items)} pairs -> {out}")
    print(f"\nevaluate with:\n  python -m src.evaluate --testset "
          f"\"{out}\" --methods unprocessed gtcrn_vctk --tag vbd")


if __name__ == "__main__":
    main()
