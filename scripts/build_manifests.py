"""Build train/val/test manifests, and PROVE the splits are disjoint.

The splitting rule everywhere is: split by RECORDING, never by file. Several
corpora here ship many files derived from one physical recording, and splitting
naively would put the same acoustic event in train and test:

  * ESC-50      - many 5 s clips cut from one Freesound upload (`src_file`).
  * UrbanSound8K- many slices from one Freesound upload (`fsID`).
  * gunshot_edge- one gunshot appears as several per-channel files PLUS a
                  channel-mean file, all sharing a uuid. This one is the
                  nastiest: 8 microphone channels of the identical shot.
  * LibriSpeech - split by speaker (the official splits already are).

A leak here does not crash anything. It silently inflates every number we
report, which for a system intended for military comms is the worst possible
failure mode. So `assert_disjoint` raises rather than warns.

    python scripts/build_manifests.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SR = 16000


def load_cfg() -> dict:
    with open(ROOT / "configs" / "data.yaml") as f:
        return yaml.safe_load(f)


def probe(path: Path):
    """(duration_s, samplerate) without decoding."""
    try:
        i = sf.info(str(path))
        return i.frames / i.samplerate, i.samplerate
    except Exception:  # noqa: BLE001
        return 0.0, 0


def probe_many(paths, workers: int = 16, desc: str = ""):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(probe, paths), total=len(paths), desc=desc,
                         unit="f", ncols=80, leave=False))


# --------------------------------------------------------------------- speech

def collect_speech(ext: Path, min_dur: float):
    """LibriSpeech, split by official subset (already speaker-disjoint)."""
    out = {"train": [], "val": [], "test": []}
    subsets = {"train-clean-100": "train", "dev-clean": "val", "test-clean": "test"}
    for subset, split in subsets.items():
        d = ext / "LibriSpeech" / subset
        if not d.is_dir():
            print(f"[wait ] speech: {subset} not extracted yet")
            continue
        files = sorted(d.rglob("*.flac"))
        durs = probe_many(files, desc=f"probe {subset}")
        kept = 0
        for f, (dur, sr) in zip(files, durs):
            # Require the full crop length so the mixer never has to pad speech
            # with silence, which would skew the active-speech SNR reference.
            if dur < min_dur or sr != SR:
                continue
            out[split].append({
                "path": str(f), "group": f.parts[-3],   # speaker id
                "source": "librispeech", "subset": subset, "dur": round(dur, 3),
            })
            kept += 1
        print(f"[ok   ] speech {subset:16s} {kept:6d} / {len(files)} utts >= {min_dur}s")
    return out


# ---------------------------------------------------------------------- noise

def _read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def collect_esc50(prepared: Path):
    """label -> [records]. group = src_file (the source Freesound upload)."""
    root = prepared / "esc50"
    rows = _read_csv(root / "esc50.csv")
    by_label = defaultdict(list)
    for r in rows:
        p = root / r["filename"]
        if p.exists():
            by_label[r["category"]].append(
                {"path": str(p), "group": f"esc50:{r['src_file']}", "source": "esc50"})
    return by_label


def collect_urbansound8k(prepared: Path):
    """label -> [records]. group = fsID (the source Freesound upload)."""
    root = prepared / "urbansound8k"
    rows = _read_csv(root / "UrbanSound8K.csv")
    by_label = defaultdict(list)
    for r in rows:
        p = root / f"fold{r['fold']}" / r["slice_file_name"]
        if p.exists():
            by_label[r["class"]].append(
                {"path": str(p), "group": f"us8k:{r['fsID']}", "source": "urbansound8k"})
    return by_label


def _gunshot_edge_shot_times(root: Path) -> dict[str, list[float]]:
    """Parse the ground-truth shot timestamps.

    The CSV is `filename, num_gunshots, t1, t2, ...` with a ragged tail of empty
    columns, so it is parsed positionally rather than by header name.

    The key is the FULL stem including the channel suffix (`_chan0`, `_mean`) -
    the annotations are per-file, not per-recording, and each channel carries its
    own slightly different arrival times. Stripping the channel to key on the
    recording silently drops every multichannel file (~61% of the corpus).
    """
    times: dict[str, list[float]] = {}
    for name in ("gunshot-audio-labels-only.csv",
                 "gunshot-audio-gunshot-locations-only.csv"):
        path = root / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) < 3 or not row[0] or row[0].strip() == "filename":
                    continue
                try:
                    n = int(float(row[1]))
                except (ValueError, IndexError):
                    continue
                ts = []
                for cell in row[2:2 + max(n, 0)]:
                    cell = (cell or "").strip()
                    if cell:
                        try:
                            ts.append(float(cell))
                        except ValueError:
                            pass
                if ts:
                    times[row[0].strip()] = ts
    return times


def collect_gunshot_edge(prepared: Path):
    """group = uuid of the ORIGINAL recording.

    Critical: the same shot exists as chan0..chanN plus a channel-mean file. All
    share the uuid prefix, so grouping on it keeps every copy on one side of the
    split. Grouping on filename would leak the same physical gunshot ~8x.

    Where ground-truth shot timestamps exist they are attached, so bursts can be
    cut at the annotated event rather than wherever an energy detector guesses.
    """
    root = prepared / "gunshot_edge"
    if not root.is_dir():
        return {}
    shot_times = _gunshot_edge_shot_times(root)
    recs, with_labels = [], 0
    for p in sorted(root.rglob("*.wav")):
        uuid = p.stem.split("_")[0]                # grouping stays per-recording
        shots = shot_times.get(p.stem)             # annotations are per-file
        if shots:
            with_labels += 1
        recs.append({"path": str(p), "group": f"gse:{uuid}",
                     "source": "gunshot_edge", "firearm": p.parent.name,
                     **({"shots": shots} if shots else {})})
    if recs:
        print(f"[ok   ] gunshot_edge: {len(recs)} clips, "
              f"{with_labels} with ground-truth shot timestamps")
    return {"gunshot": recs} if recs else {}


def collect_musan(ext: Path, sub: str):
    """MUSAN is already 16 kHz - indexed in place, not copied.

    Group per FILE, not per directory. MUSAN's subdirectories are *provenance*
    labels (free-sound, sound-bible, librivox, us-gov, fma, ...) - only 2 to 5
    of them per type - and each file under them is an independent recording.

    Grouping on the directory gives 2 groups for babble, and an 85/5/10 split of
    2 groups puts everything in train: babble then vanishes from val and test
    entirely, and the background pool loses its whole validation set. That is
    silent - the disjointness assertion still passes, because degenerate splits
    are trivially disjoint - so it has to be got right here.
    """
    d = ext / "musan" / sub
    if not d.is_dir():
        return []
    return [{"path": str(p), "group": f"musan:{sub}:{p.stem}", "source": f"musan/{sub}"}
            for p in sorted(d.rglob("*.wav"))]


def collect_rirs(ext: Path):
    """OpenSLR 28. group = ROOM, so a test room is never seen in training.

    The archive mixes three things and only one of them is impulse responses:
      simulated_rirs/{small,medium,large}room/RoomNNN/  - real RIRs
      real_rirs_isotropic_noises/RVB2014_type1_rir_*    - real RIRs
      real_rirs_isotropic_noises/RVB2014_type1_noise_*  - isotropic NOISE
      pointsource_noises/noise-free-sound-*             - NOISE, no RIRs at all

    Convolving speech with a noise recording instead of an impulse response
    would produce nonsense that still trains, so the non-RIR content is excluded
    by directory rather than trusting a filename substring.

    Room numbering restarts in each size bucket (Room001 exists under smallroom,
    mediumroom AND largeroom), so the size is part of the group key - otherwise
    three acoustically different rooms collapse into one and can straddle splits.
    """
    root = ext / "RIRS_NOISES"
    if not root.is_dir():
        return []
    recs = []

    for p in sorted((root / "simulated_rirs").rglob("*.wav")):
        size, room = p.parent.parent.name, p.parent.name
        recs.append({"path": str(p), "group": f"rir:sim:{size}:{room}",
                     "source": "openslr28/simulated"})

    real = root / "real_rirs_isotropic_noises"
    if real.is_dir():
        for p in sorted(real.glob("*.wav")):
            if "_rir_" not in p.name.lower():
                continue                        # isotropic noise, not an RIR
            parts = p.stem.split("_")
            # RVB2014_type1_rir_<room>_<dist>_<angle> -> room token
            room = parts[3] if len(parts) > 3 else p.stem
            recs.append({"path": str(p), "group": f"rir:real:{room}",
                         "source": "openslr28/real"})
    return recs


# --------------------------------------------------------------------- splits

def split_by_group(records, rng, fracs=(0.85, 0.05, 0.10)):
    """Partition on the group key so no recording spans two splits."""
    groups = sorted({r["group"] for r in records})
    rng.shuffle(groups)
    n = len(groups)
    n_tr = max(1, int(round(n * fracs[0])))
    n_va = max(1, int(round(n * fracs[1]))) if n - n_tr > 1 else 0
    assign = {}
    for i, g in enumerate(groups):
        assign[g] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    out = {"train": [], "val": [], "test": []}
    for r in records:
        out[assign[r["group"]]].append(r)
    return out


def assert_disjoint(name: str, splits: dict) -> None:
    keys = {s: {r["group"] for r in rs} for s, rs in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = keys[a] & keys[b]
        if overlap:
            raise AssertionError(
                f"{name}: {len(overlap)} recording group(s) appear in BOTH "
                f"'{a}' and '{b}' - this would leak test data into training. "
                f"Examples: {sorted(overlap)[:5]}")
    print(f"[ok   ] {name:22s} disjoint  "
          + "  ".join(f"{s}={len(rs)}" for s, rs in splits.items()))


# ----------------------------------------------------------------------- main

def main() -> None:
    cfg = load_cfg()
    raw = Path(cfg["paths"]["raw"])
    ext = raw / "extracted"
    prepared = Path(cfg["paths"]["prepared"])
    man = ROOT / "manifests"
    man.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg["splits"]["seed"])
    min_dur = float(cfg["mixture"]["dur_s"])

    bundle: dict = {"speech": {}, "noise": {}, "rir": {}}

    # -- speech -------------------------------------------------------------
    print("\n--- speech ---")
    speech = collect_speech(ext, min_dur)
    if any(speech.values()):
        assert_disjoint("librispeech", speech)
        bundle["speech"] = speech

    # -- noise, resolved through the category map in data.yaml --------------
    print("\n--- noise ---")
    pools = {
        "esc50": collect_esc50(prepared),
        "urbansound8k": collect_urbansound8k(prepared),
        "gunshot_edge": collect_gunshot_edge(prepared),
    }
    for cat, spec in cfg["categories"].items():
        recs = []
        for src in spec["sources"]:
            ds, _, label = src.partition("/")
            if ds == "musan":
                recs += collect_musan(ext, label)
            elif ds == "gunshot_edge":
                recs += pools["gunshot_edge"].get("gunshot", [])
            elif ds in pools:
                recs += pools[ds].get(label, [])
        if not recs:
            print(f"[wait ] {cat:10s} no files yet")
            continue
        for r in recs:
            r["category"] = cat
        splits = split_by_group(recs, rng)
        assert_disjoint(f"noise/{cat}", splits)
        bundle["noise"][cat] = splits

    # -- background ---------------------------------------------------------
    print("\n--- background ---")
    bg = []
    for src in cfg["background"]["sources"]:
        _, _, sub = src.partition("/")
        bg += collect_musan(ext, sub)
    if bg:
        for r in bg:
            r["category"] = "background"
        splits = split_by_group(bg, rng)
        assert_disjoint("background", splits)
        bundle["noise"]["_background"] = splits

    # -- rirs ---------------------------------------------------------------
    print("\n--- rirs ---")
    rirs = collect_rirs(ext)
    if rirs:
        splits = split_by_group(rirs, rng)
        assert_disjoint("rir", splits)
        bundle["rir"] = splits

    out = man / "manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=1)

    print(f"\nwrote {out}")
    for kind in ("speech", "rir"):
        if bundle[kind]:
            print(f"  {kind:12s} " + "  ".join(
                f"{s}={len(v)}" for s, v in bundle[kind].items()))
    for cat, splits in bundle["noise"].items():
        print(f"  noise/{cat:10s} " + "  ".join(
            f"{s}={len(v)}" for s, v in splits.items()))


if __name__ == "__main__":
    main()
