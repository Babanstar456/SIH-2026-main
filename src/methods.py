"""Enhancement method registry.

Every method has the same signature - `f(noisy: np.ndarray, sr: int) -> np.ndarray`
returning audio of identical length - so evaluate.py can sweep all of them over
identical audio without special-casing. That uniformity is what makes the
comparison table trustworthy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .baselines.classical import spectral_subtraction, wiener

# PyTorch is imported lazily, inside the neural methods only. `unprocessed` and
# the classical baselines are pure NumPy, so evaluating them must not require a
# 2 GB CUDA install - and must keep working on a machine where PyTorch cannot
# load at all (Windows Smart App Control blocks its unsigned DLLs, for one).

ROOT = Path(__file__).resolve().parents[1]
_MODEL_CACHE: dict = {}


# ------------------------------------------------------------------- neural

def load_gtcrn(ckpt: str | Path, device: str = "cpu"):
    """Load GTCRN from a checkpoint. Accepts upstream .tar and our own .pt."""
    import torch

    key = (str(ckpt), device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from .models.gtcrn import GTCRN

    model = GTCRN().to(device).eval()
    obj = torch.load(str(ckpt), map_location=device, weights_only=False)
    state = obj.get("model", obj.get("state_dict", obj)) if isinstance(obj, dict) else obj
    model.load_state_dict(state)
    _MODEL_CACHE[key] = model
    return model


def enhance_gtcrn(x: np.ndarray, sr: int, ckpt, device: str = "cpu") -> np.ndarray:
    import torch

    from . import stft as S

    model = load_gtcrn(ckpt, device)
    with torch.no_grad():
        wav = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        spec = S.stft(wav)[None]
        out = model(spec)[0]
        enh = S.istft(out, length=len(x))
    return enh.cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------ registry

def _identity(x: np.ndarray, sr: int) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _noisereduce(x: np.ndarray, sr: int) -> np.ndarray:
    """Third classical baseline. Stands in for RNNoise where RNNoise will not
    build on Windows; whichever is used is stated plainly in the results."""
    import noisereduce as nr
    return nr.reduce_noise(y=np.asarray(x, dtype=np.float32), sr=sr,
                           stationary=False).astype(np.float32)


BUILTIN = {
    "unprocessed": _identity,
    "specsub": spectral_subtraction,
    "wiener": wiener,
    "noisereduce": _noisereduce,
}


def get(name: str, device: str = "cpu"):
    """Resolve a method name to a callable.

    Names:
      unprocessed | wiener | specsub | noisereduce
      gtcrn_dns3  | gtcrn_vctk          (upstream pretrained checkpoints)
      gtcrn:<path/to/checkpoint>        (anything we train)
    """
    if name in BUILTIN:
        return BUILTIN[name]

    presets = {
        "gtcrn_dns3": ROOT / "checkpoints" / "pretrained" / "model_trained_on_dns3.tar",
        "gtcrn_vctk": ROOT / "checkpoints" / "pretrained" / "model_trained_on_vctk.tar",
    }
    if name in presets:
        ckpt = presets[name]
    elif name.startswith("gtcrn:"):
        ckpt = Path(name.split(":", 1)[1])
    else:
        raise KeyError(f"unknown method {name!r}. known: "
                       f"{sorted(BUILTIN) + sorted(presets)} or 'gtcrn:<ckpt>'")
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    return lambda x, sr, _c=ckpt: enhance_gtcrn(x, sr, _c, device)


def available() -> list[str]:
    out = list(BUILTIN)
    for n, p in (("gtcrn_dns3", "model_trained_on_dns3.tar"),
                 ("gtcrn_vctk", "model_trained_on_vctk.tar")):
        if (ROOT / "checkpoints" / "pretrained" / p).exists():
            out.append(n)
    return out
