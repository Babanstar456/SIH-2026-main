"""Smoke-test the full training step on whatever data is present.

Runs the real dataset -> STFT -> GTCRN -> loss -> backward path for a handful of
batches. Catches shape errors, device errors, dataloader-worker problems and
NaN losses in ~30 seconds instead of 20 minutes into a real run.

    python scripts/smoke_train.py --steps 6 --workers 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import stft as S                              # noqa: E402
from src.dataset import MixtureDataset, load_manifest  # noqa: E402
from src.losses import TransientWeightedLoss           # noqa: E402
from src.models.gtcrn import GTCRN                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--split", default=None,
                    help="defaults to 'train', falling back to any split with speech")
    a = ap.parse_args()

    with open(ROOT / "configs" / "data.yaml") as f:
        dcfg = yaml.safe_load(f)
    man = load_manifest(ROOT / "manifests" / "manifest.json")

    split = a.split
    if split is None:
        for s in ("train", "val", "test"):
            if man["speech"].get(s):
                split = s
                break
    print(f"using split: {split!r}  (speech files: {len(man['speech'].get(split, []))})")

    ds = MixtureDataset(man, dcfg, split, epoch_size=a.steps * a.batch, seed=0)
    ld = DataLoader(ds, batch_size=a.batch, num_workers=a.workers,
                    persistent_workers=a.workers > 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GTCRN().to(device).train()
    obj = torch.load(ROOT / "checkpoints/pretrained/model_trained_on_dns3.tar",
                     map_location=device, weights_only=False)
    model.load_state_dict(obj["model"])
    loss_fn = TransientWeightedLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"device: {device}\n")
    t0 = time.time()
    n_seen = 0
    # Windows spawns dataloader workers, and each one re-imports torch. On a
    # short run that startup swamps the measurement, so steady-state throughput
    # is timed from a later step with the workers already warm.
    warm_after = 3
    t_warm, n_warm = None, 0
    for i, (noisy, clean, fmask, cat) in enumerate(ld):
        if i == warm_after:
            t_warm, n_warm = time.time(), 0
        noisy, clean, fmask = noisy.to(device), clean.to(device), fmask.to(device)
        spec_n, spec_c = S.stft(noisy), S.stft(clean)
        out = model(spec_n)

        assert out.shape == spec_c.shape, f"{out.shape} != {spec_c.shape}"
        assert fmask.shape[1] == spec_n.shape[2], \
            f"frame mask {tuple(fmask.shape)} misaligned with spec {tuple(spec_n.shape)}"

        loss, parts = loss_fn(out, spec_c, fmask)
        assert torch.isfinite(loss), f"non-finite loss at step {i}: {loss}"

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        n_seen += noisy.shape[0]
        if t_warm is not None:
            n_warm += noisy.shape[0]

        print(f"step {i}  wav={tuple(noisy.shape)}  spec={tuple(spec_n.shape)}  "
              f"mask_frames={float(fmask.float().mean())*100:5.1f}%  "
              f"loss={float(loss.detach()):8.4f}  transient={float(parts['transient']):8.4f}  "
              f"|g|={float(gnorm):.2f}")

    dt = time.time() - t0
    print(f"\nOK - {n_seen} mixtures in {dt:.1f}s "
          f"({n_seen/dt:.1f}/s including startup, {a.workers} workers)")
    if t_warm is not None and n_warm > 0:
        dtw = time.time() - t_warm
        rate = n_warm / dtw
        print(f"steady state: {rate:.1f} mixtures/s "
              f"(from step {warm_after}, workers warm)")
        print(f"-> one 10k-mixture epoch ~= {10000/rate/60:.1f} min")
    else:
        print(f"-> one 10k-mixture epoch ~= {10000/(n_seen/dt)/60:.1f} min")


if __name__ == "__main__":
    main()
