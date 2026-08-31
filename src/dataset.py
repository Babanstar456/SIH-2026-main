"""Training / validation datasets.

Training mixtures are generated ON THE FLY: with a few thousand speech files and
a few thousand noise files, the number of distinct mixtures is effectively
unbounded, and regenerating every epoch is far better regularisation than a
fixed pre-rendered set.

Validation and test are the opposite - FROZEN. A val set that changes every
epoch cannot tell you whether the model improved or the data got easier. `FrozenSet`
reads pre-rendered pairs written once by scripts/make_testset.py.

Category semantics: every mixture gets a steady MUSAN background. The category
then decides what is layered on top -

    impulsive categories (gunshot, artillery) -> discrete BURSTS
    steady categories (rotor, engine, siren, babble) -> extra BACKGROUND

which is what makes "per-category results" mean something specific.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import audio as A
from . import stft as S
from .mixer import Mixer


def load_manifest(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class MixtureDataset(Dataset):
    """On-the-fly mixture generation for training."""

    def __init__(self, manifest: dict, cfg: dict, split: str = "train",
                 epoch_size: int = 20000, seed: int = 0,
                 category_weights: dict | None = None):
        self.cfg = cfg
        self.split = split
        self.epoch_size = int(epoch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.mixer = Mixer(cfg)
        self.n = self.mixer.n

        self.speech = manifest["speech"].get(split, [])
        if not self.speech:
            raise RuntimeError(f"no speech in manifest split {split!r} - "
                               "run scripts/build_manifests.py first")

        noise = manifest.get("noise", {})
        self.background = [r["path"] for r in
                           noise.get("_background", {}).get(split, [])]
        self.rirs = [r["path"] for r in manifest.get("rir", {}).get(split, [])]

        # Full records, not bare paths: impulsive corpora carry ground-truth
        # shot timestamps that A.load_burst uses to cut at the real event.
        self.cats: dict[str, list[dict]] = {}
        self.impulsive: dict[str, bool] = {}
        self.layers: dict[str, list] = {}
        for cat, spec in cfg["categories"].items():
            recs = noise.get(cat, {}).get(split, [])
            if recs:
                self.cats[cat] = recs
                self.impulsive[cat] = bool(spec.get("impulsive", False))
                self.layers[cat] = spec.get("layers", [1, 1])
        if not self.cats:
            raise RuntimeError(f"no noise categories present for split {split!r}")

        # Weight the hard, impulsive categories up: they are the ones the doc
        # says the model must not fail on, and the ones a classical filter
        # cannot touch. Weights come from the config because they are not purely
        # a function of difficulty - a category with a tiny source pool gets
        # held back regardless, or the model memorises those few recordings.
        default = cfg.get("category_weights") or {
            c: (3.0 if self.impulsive.get(c) else 1.0) for c in self.cats}
        w = category_weights or default
        self.cat_names = list(self.cats)
        p = np.array([float(w.get(c, 1.0)) for c in self.cat_names], dtype=np.float64)
        self.cat_p = p / p.sum()

    def set_epoch(self, epoch: int) -> None:
        """Reseed so each epoch draws different mixtures, reproducibly."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, self.epoch, idx))

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        for _ in range(8):                       # retry on a bad draw
            item = self._try_build(rng)
            if item is not None:
                return item
        # Degenerate corner: return silence rather than crash a long run.
        z = torch.zeros(self.n)
        return z, z, torch.zeros(S.n_frames(self.n), dtype=torch.bool), 0

    def _try_build(self, rng):
        sp = self.speech[int(rng.integers(0, len(self.speech)))]
        speech = A.load_audio(sp["path"])

        cat = self.cat_names[int(rng.choice(len(self.cat_names), p=self.cat_p))]
        pool = self.cats[cat]

        bg_paths = list(rng.choice(self.background, size=min(2, len(self.background)),
                                   replace=False)) if self.background else []
        bg_clips = [A.load_random_window(p, self.n, rng) for p in bg_paths]

        burst_clips = []
        if self.impulsive[cat]:
            k = int(rng.integers(1, 4))
            idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
            for i in idx:
                burst_clips.append(A.load_burst(pool[i], rng))
        else:
            # steady category: it becomes another background layer. `layers`
            # is >1 only for babble, where several voices must be summed.
            lo, hi = self.layers.get(cat, [1, 1])
            k = int(rng.integers(lo, hi + 1))
            bg_clips.append(A.load_layered([r["path"] for r in pool],
                                           self.n, rng, layers=k))

        rir_s = rir_n = None
        if self.rirs:
            rir_s = A.load_audio(self.rirs[int(rng.integers(0, len(self.rirs)))])
            rir_n = A.load_audio(self.rirs[int(rng.integers(0, len(self.rirs)))])

        res = self.mixer.build(rng, speech, bg_clips, burst_clips,
                               rir_s, rir_n, category=cat,
                               force_burst=self.impulsive[cat])
        if res is None:
            return None

        nf = S.n_frames(len(res.noisy))
        fmask = S.samples_to_frame_mask(res.transient_mask, nf)
        return (
            torch.from_numpy(res.noisy),
            torch.from_numpy(res.target),
            torch.from_numpy(fmask),
            self.cat_names.index(cat),
        )


class FrozenSet(Dataset):
    """Pre-rendered pairs from scripts/make_testset.py - never regenerated."""

    def __init__(self, root):
        self.root = Path(root)
        index = self.root / "index.json"
        if not index.exists():
            raise FileNotFoundError(
                f"{index} missing - run scripts/make_testset.py first")
        with open(index, encoding="utf-8") as f:
            self.items = json.load(f)["items"]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        noisy = A.load_audio(self.root / it["noisy"])
        clean = A.load_audio(self.root / it["clean"])
        mask = np.load(self.root / it["mask"])["mask"] if it.get("mask") else None
        nf = S.n_frames(len(noisy))
        fmask = (S.samples_to_frame_mask(mask, nf) if mask is not None
                 else np.zeros(nf, dtype=bool))
        return (torch.from_numpy(noisy), torch.from_numpy(clean),
                torch.from_numpy(fmask), it["category"])

    def raw(self, i: int):
        """Numpy access for evaluate.py, which needs metadata too."""
        it = self.items[i]
        return (A.load_audio(self.root / it["noisy"]),
                A.load_audio(self.root / it["clean"]), it)
