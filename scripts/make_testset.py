"""Render the FROZEN evaluation set, once.

Everything here is deliberately rigid:

  * fixed seed, so the set is byte-reproducible;
  * drawn only from the `test` manifest split - held-out speakers, held-out
    noise recordings, held-out rooms;
  * written to disk and never regenerated.

The reason is simple. If the test set is regenerated between experiments, then
"model B beats model A" might only mean "B got an easier draw". Every number in
the report has to be traceable to these exact files.

Refusing to overwrite is intentional - pass --force only when you intend to
invalidate every previously reported result.

    python scripts/make_testset.py --per-category 120
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import audio as A                      # noqa: E402
from src.mixer import Mixer                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--force", action="store_true",
                    help="delete and re-render, invalidating prior results")
    ap.add_argument("--out", default=None,
                    help="write elsewhere (e.g. a scratch set for harness "
                         "validation) instead of the canonical frozen set")
    a = ap.parse_args()

    with open(ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    out = Path(a.out) if a.out else Path(cfg["paths"]["testset"])

    if out.exists() and (out / "index.json").exists():
        if not a.force:
            print(f"{out} already exists. Refusing to overwrite - every number "
                  f"already reported was measured on it.\nUse --force only if "
                  f"you intend to invalidate those results.")
            return
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "manifests" / "manifest.json", encoding="utf-8") as f:
        man = json.load(f)

    speech = man["speech"].get("test", [])
    if not speech:
        raise SystemExit("no test speech in manifest - run build_manifests.py")
    noise = man.get("noise", {})
    background = [r["path"] for r in noise.get("_background", {}).get("test", [])]
    rirs = [r["path"] for r in man.get("rir", {}).get("test", [])]
    mixer = Mixer(cfg)

    items, rng = [], np.random.default_rng(a.seed)
    for cat, spec in cfg["categories"].items():
        recs = noise.get(cat, {}).get("test", [])
        if not recs:
            print(f"[skip] {cat}: no held-out noise available")
            continue
        pool = recs                       # full records: may carry shot timestamps
        impulsive = bool(spec.get("impulsive", False))
        (out / cat).mkdir(exist_ok=True)

        made = 0
        pbar = tqdm(total=a.per_category, desc=f"{cat:10s}", ncols=80, unit="clip")
        attempts = 0
        while made < a.per_category and attempts < a.per_category * 20:
            attempts += 1
            sp = speech[int(rng.integers(0, len(speech)))]
            wav = A.load_audio(sp["path"])

            bg = [A.load_random_window(p, mixer.n, rng) for p in
                  (rng.choice(background, size=min(2, len(background)), replace=False)
                   if background else [])]
            bursts = []
            if impulsive:
                k = int(rng.integers(1, 4))
                for i in rng.choice(len(pool), size=min(k, len(pool)), replace=False):
                    bursts.append(A.load_burst(pool[i], rng))
            else:
                lo, hi = spec.get("layers", [1, 1])
                bg.append(A.load_layered([r["path"] for r in pool], mixer.n, rng,
                                         layers=int(rng.integers(lo, hi + 1))))

            rs = rn = None
            if rirs:
                rs = A.load_audio(rirs[int(rng.integers(0, len(rirs)))])
                rn = A.load_audio(rirs[int(rng.integers(0, len(rirs)))])

            res = mixer.build(rng, wav, bg, bursts, rs, rn, category=cat,
                              force_burst=impulsive)
            if res is None:
                continue

            stem = f"{cat}/{made:04d}"
            A.save_audio(out / f"{stem}_noisy.wav", res.noisy)
            A.save_audio(out / f"{stem}_clean.wav", res.target)
            mask_rel = None
            if res.transient_mask.any():
                np.savez_compressed(out / f"{stem}_mask.npz",
                                    mask=res.transient_mask)
                mask_rel = f"{stem}_mask.npz"
            items.append({
                "id": stem, "category": cat,
                "noisy": f"{stem}_noisy.wav", "clean": f"{stem}_clean.wav",
                "mask": mask_rel, "speaker": sp["group"],
                **{k: v for k, v in res.meta.items() if k != "events"},
                "n_events": len(res.meta["events"]),
            })
            made += 1
            pbar.update(1)
        pbar.close()

    index = {
        "seed": a.seed,
        "per_category": a.per_category,
        "sr": cfg["audio"]["sr"],
        "dur_s": cfg["mixture"]["dur_s"],
        "note": ("Frozen evaluation set. Held-out speakers, noise recordings "
                 "and rooms. Do not regenerate - reported results reference "
                 "these exact files."),
        "items": items,
    }
    with open(out / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)

    print(f"\nwrote {len(items)} clips to {out}")
    from collections import Counter
    for c, n in sorted(Counter(i["category"] for i in items).items()):
        print(f"  {c:12s} {n}")


if __name__ == "__main__":
    main()
