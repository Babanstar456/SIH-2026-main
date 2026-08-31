"""Fine-tune GTCRN on defence-noise mixtures.

    python -m src.train                       # start / resume
    python -m src.train --w-transient 0.0     # ablation: upstream objective

Resumable by design: every epoch writes `last.pt` with optimiser and scheduler
state, so an interrupted run costs one epoch rather than the whole session.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import metrics as M                    # noqa: E402
from src import stft as S                       # noqa: E402
from src.dataset import MixtureDataset, load_manifest   # noqa: E402
from src.losses import TransientWeightedLoss    # noqa: E402
from src.models.gtcrn import GTCRN              # noqa: E402


def build_scheduler(opt, cfg, steps_per_epoch: int):
    total = cfg["optim"]["epochs"] * steps_per_epoch
    warm = int(cfg["optim"]["warmup_steps"])
    base, mn = float(cfg["optim"]["lr"]), float(cfg["optim"]["min_lr"])

    def fn(step: int) -> float:
        if step < warm:
            return (step + 1) / max(warm, 1)
        p = (step - warm) / max(total - warm, 1)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
        return (mn + (base - mn) * cos) / base

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


@torch.no_grad()
def validate(model, loader, loss_fn, device, pesq_clips: int = 120):
    """Val loss over the whole set + PESQ over a capped subset.

    PESQ costs ~30 ms a clip, so it is measured on a fixed prefix rather than
    everything - enough to steer model selection without doubling epoch time.
    """
    model.eval()
    tot, n = 0.0, 0
    pesqs, stois = [], []
    for noisy, clean, fmask, _ in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        fmask = fmask.to(device)
        spec_n, spec_c = S.stft(noisy), S.stft(clean)
        out = model(spec_n)
        loss, _ = loss_fn(out, spec_c, fmask)
        tot += float(loss) * noisy.shape[0]
        n += noisy.shape[0]

        if len(pesqs) < pesq_clips:
            enh = S.istft(out, length=noisy.shape[-1]).cpu().numpy()
            ref = clean.cpu().numpy()
            for i in range(min(len(enh), pesq_clips - len(pesqs))):
                pesqs.append(M.pesq_wb(ref[i], enh[i]))
                stois.append(M.stoi_score(ref[i], enh[i]))
    model.train()
    valid = [p for p in pesqs if not np.isnan(p)]
    return {
        "val_loss": tot / max(n, 1),
        "val_pesq": float(np.mean(valid)) if valid else float("nan"),
        "val_stoi": float(np.nanmean(stois)) if stois else float("nan"),
        "val_pesq_n": len(valid),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--w-transient", type=float, default=None,
                    help="override loss.w_transient (0.0 = upstream objective)")
    ap.add_argument("--w-consonant", type=float, default=None,
                    help="override loss.w_consonant (0.0 = off). Weights the "
                         "1-4 kHz bins that carry consonant identity.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--epoch-size", type=int, default=None,
                    help="override mixtures per epoch (small values smoke-test "
                         "the trainer end to end in a minute)")
    ap.add_argument("--val-size", type=int, default=None)
    ap.add_argument("--tag", default="ft")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    with open(ROOT / a.config) as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / a.data_config) as f:
        dcfg = yaml.safe_load(f)
    if a.w_transient is not None:
        cfg["loss"]["w_transient"] = a.w_transient
    if a.w_consonant is not None:
        cfg["loss"]["w_consonant"] = a.w_consonant
    if a.epochs is not None:
        cfg["optim"]["epochs"] = a.epochs
    if a.epoch_size is not None:
        cfg["data"]["epoch_size"] = a.epoch_size
    if a.val_size is not None:
        cfg["data"]["val_size"] = a.val_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("!! CUDA not available - this will be extremely slow !!")
    torch.manual_seed(cfg["data"]["seed"])

    man = load_manifest(ROOT / "manifests" / "manifest.json")
    d = cfg["data"]
    train_ds = MixtureDataset(man, dcfg, "train", d["epoch_size"], d["seed"])
    # Val uses a fixed seed and never advances its epoch, so it draws the
    # identical mixtures every time - frozen in effect, without pre-rendering.
    val_ds = MixtureDataset(man, dcfg, "val", d["val_size"], seed=999_777)

    common = dict(batch_size=d["batch_size"], num_workers=d["num_workers"],
                  pin_memory=(device == "cuda"), drop_last=False,
                  persistent_workers=d["num_workers"] > 0)
    train_ld = DataLoader(train_ds, shuffle=False, **common)
    val_ld = DataLoader(val_ds, shuffle=False, **common)

    model = GTCRN().to(device)
    init = cfg["init"]["checkpoint"]
    if init:
        obj = torch.load(ROOT / init, map_location=device, weights_only=False)
        state = obj.get("model", obj.get("state_dict", obj))
        model.load_state_dict(state)
        print(f"initialised from {init}")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"GTCRN parameters: {n_par:,}   device: {device}")

    loss_fn = TransientWeightedLoss(**cfg["loss"])
    print(f"loss: transient weight = {cfg['loss']['w_transient']}, "
          f"consonant weight = {cfg['loss'].get('w_consonant', 0.0)}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]),
                            weight_decay=float(cfg["optim"]["weight_decay"]),
                            betas=tuple(cfg["optim"]["betas"]))
    steps_per_epoch = math.ceil(len(train_ds) / d["batch_size"])
    sched = build_scheduler(opt, cfg, steps_per_epoch)

    ck_dir = ROOT / cfg["checkpoint"]["dir"]
    ck_dir.mkdir(exist_ok=True)
    last_path = ck_dir / f"{a.tag}_last.pt"
    best_path = ck_dir / f"{a.tag}_best.pt"
    log_path = ROOT / "results" / f"train_log_{a.tag}.csv"
    log_path.parent.mkdir(exist_ok=True)

    start_epoch, best, bad = 0, -float("inf"), 0
    if a.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best, bad = ck["epoch"] + 1, ck["best"], ck.get("bad", 0)
        print(f"resumed from epoch {start_epoch}")

    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "mag", "ri", "sisnr", "transient",
                 "val_loss", "val_pesq", "val_stoi", "lr", "secs"])

    clip = float(cfg["optim"]["grad_clip"])
    for epoch in range(start_epoch, cfg["optim"]["epochs"]):
        train_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        run = {"loss": 0.0, "mag": 0.0, "ri": 0.0, "sisnr": 0.0, "transient": 0.0}
        seen = 0

        for step, (noisy, clean, fmask, _) in enumerate(train_ld):
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            fmask = fmask.to(device, non_blocking=True)

            out = model(S.stft(noisy))
            loss, parts = loss_fn(out, S.stft(clean), fmask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()

            b = noisy.shape[0]
            seen += b
            run["loss"] += float(loss.detach()) * b
            for k in ("mag", "ri", "sisnr", "transient"):
                run[k] += float(parts[k]) * b

            if step % 50 == 0:
                print(f"  e{epoch:03d} s{step:04d}/{steps_per_epoch}  "
                      f"loss={float(loss.detach()):8.4f}  lr={sched.get_last_lr()[0]:.2e}",
                      flush=True)

        for k in run:
            run[k] /= max(seen, 1)
        v = validate(model, val_ld, loss_fn, device)
        secs = time.time() - t0

        print(f"epoch {epoch:03d}  train={run['loss']:.4f}  "
              f"val={v['val_loss']:.4f}  val_pesq={v['val_pesq']:.3f}  "
              f"val_stoi={v['val_stoi']:.3f}  ({secs/60:.1f} min)")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, run["loss"], run["mag"], run["ri"], run["sisnr"],
                run["transient"], v["val_loss"], v["val_pesq"], v["val_stoi"],
                sched.get_last_lr()[0], round(secs, 1)])

        # Update `best` BEFORE writing last.pt. Writing it first stores a value
        # one epoch stale, so a resumed run starts with an understated best and
        # can overwrite a genuinely better best.pt with a worse checkpoint.
        score = v[cfg["checkpoint"]["monitor"]]
        improved = not np.isnan(score) and score > best
        if improved:
            best, bad = score, 0
        else:
            bad += 1

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch, "best": best,
                 "bad": bad, "cfg": cfg, "val": v}
        torch.save(state, last_path)
        if improved:
            torch.save(state, best_path)
            print(f"  new best {cfg['checkpoint']['monitor']}={best:.4f} -> {best_path.name}")
        elif bad >= int(cfg["checkpoint"]["patience"]):
            print(f"early stop: no improvement for {bad} epochs")
            break

    print(f"\ndone. best {cfg['checkpoint']['monitor']} = {best:.4f}")
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
