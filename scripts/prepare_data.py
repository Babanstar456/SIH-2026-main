"""Extract downloaded archives, then normalise to 16 kHz mono where needed.

Deliberately does NOT convert everything. LibriSpeech, MUSAN and the OpenSLR
RIRs are already 16 kHz, so we index them in place - converting would burn an
hour and ~25 GB for no gain. Only the transient corpora (44.1/48 kHz, mixed
rates) actually need resampling.

Resumable: each step drops a .done marker and is skipped on re-run.

    python scripts/prepare_data.py --step all
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
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

# archive -> directory it unpacks into (relative to raw/extracted)
ARCHIVES = {
    "dev-clean.tar.gz":       "LibriSpeech",
    "test-clean.tar.gz":      "LibriSpeech",
    "train-clean-100.tar.gz": "LibriSpeech",
    "musan.tar.gz":           "musan",
    "rirs_noises.zip":        "RIRS_NOISES",
    "esc50.zip":              "ESC-50-master",
    "UrbanSound8K.tar.gz":    "UrbanSound8K",
    "gunshot_edge.zip":       "edge-collected-gunshot-audio",
}

# Corpora that need resampling to 16 kHz mono. Everything else is already there.
NEEDS_CONVERT = {
    "esc50":        ("ESC-50-master/audio", "*.wav"),
    "urbansound8k": ("UrbanSound8K/audio", "**/*.wav"),
    "gunshot_edge": ("edge-collected-gunshot-audio", "**/*.wav"),
}


def load_cfg() -> dict:
    with open(ROOT / "configs" / "data.yaml") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------- extract

def extract_all(raw: Path) -> None:
    out = raw / "extracted"
    out.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        src = raw / name
        marker = out / f".{name}.extracted"
        if marker.exists():
            print(f"[skip ] {name}")
            continue
        if not src.exists() or not (raw / f"{name}.done").exists():
            print(f"[wait ] {name} not downloaded yet")
            continue
        print(f"[unpack] {name} ...", flush=True)
        try:
            if name.endswith((".tar.gz", ".tgz")):
                with tarfile.open(src, "r:gz") as t:
                    t.extractall(out, filter="data")
            else:
                with zipfile.ZipFile(src) as z:
                    z.extractall(out)
            marker.touch()
            print(f"[ok    ] {name}")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL  ] {name}: {e}")


# ------------------------------------------------------------------- convert

def _convert_one(args) -> tuple[str, bool]:
    src, dst = Path(args[0]), Path(args[1])
    if dst.exists():
        return str(dst), True
    try:
        x, in_sr = sf.read(str(src), dtype="float32", always_2d=True)
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        if in_sr != SR:
            x = soxr.resample(x, in_sr, SR, quality="HQ")
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak < 1e-6:
            return str(dst), False          # silent file, drop it
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), x.astype(np.float32), SR, subtype="PCM_16")
        return str(dst), True
    except Exception:  # noqa: BLE001
        return str(dst), False


def convert_all(raw: Path, prepared: Path, workers: int) -> None:
    ext = raw / "extracted"
    for tag, (subdir, pattern) in NEEDS_CONVERT.items():
        marker = prepared / f".{tag}.converted"
        if marker.exists():
            print(f"[skip ] convert {tag}")
            continue
        src_root = ext / subdir
        if not src_root.exists():
            print(f"[wait ] {tag}: {src_root} missing")
            continue
        files = sorted(src_root.glob(pattern))
        if not files:
            print(f"[wait ] {tag}: no wavs under {src_root}")
            continue
        dst_root = prepared / tag
        jobs = [(str(f), str(dst_root / f.relative_to(src_root))) for f in files]
        ok = 0
        print(f"[conv ] {tag}: {len(jobs)} files -> {dst_root}", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_convert_one, j) for j in jobs]
            for fu in tqdm(as_completed(futs), total=len(futs), unit="f", ncols=80):
                ok += bool(fu.result()[1])
        print(f"[ok   ] {tag}: {ok}/{len(jobs)} converted")
        # Copy the metadata CSVs alongside - we need the label columns and,
        # for the gunshot corpus, the ground-truth shot timestamps.
        for csv in src_root.rglob("*.csv"):
            shutil.copy2(csv, dst_root / csv.name)
        for csv in (src_root.parent).glob("*.csv"):
            shutil.copy2(csv, dst_root / csv.name)
        for meta_dir in ("meta", "metadata"):
            d = src_root.parent / meta_dir
            if d.is_dir():
                for csv in d.glob("*.csv"):
                    shutil.copy2(csv, dst_root / csv.name)
        marker.touch()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["extract", "convert", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    cfg = load_cfg()
    raw = Path(cfg["paths"]["raw"])
    prepared = Path(cfg["paths"]["prepared"])
    prepared.mkdir(parents=True, exist_ok=True)

    if a.step in ("extract", "all"):
        extract_all(raw)
    if a.step in ("convert", "all"):
        convert_all(raw, prepared, a.workers)


if __name__ == "__main__":
    main()
