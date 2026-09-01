"""Enhancement method registry.

Every method has the same signature - `f(noisy: np.ndarray, sr: int) -> np.ndarray`
returning audio of identical length - so evaluate.py can sweep all of them over
identical audio without special-casing. That uniformity is what makes the
comparison table trustworthy.
"""
from __future__ import annotations

import subprocess
import tempfile
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


def _rnnoise(x: np.ndarray, sr: int) -> np.ndarray:
    """Reference RNNoise (github.com/xiph/rnnoise), via the `pyrnnoise` package's
    bundled, compiled `librnnoise`/`rnnoise.dll` and its built-in pretrained
    model - there is no separate weights file to load.

    RNNoise operates at 48 kHz in 480-sample (10 ms) frames; this project runs
    at 16 kHz, so we resample up with `soxr`, run the reference C library frame
    by frame through its low-level ctypes binding, then resample back down.
    Measured round trip (resample 16k->48k->16k on clean speech, no RNNoise in
    between): ~50.5 dB SNR - negligible next to what RNNoise itself does to the
    signal, so it does not confound the comparison.

    We call `pyrnnoise.rnnoise.process_mono_frame` directly rather than the
    package's higher-level `RNNoise.denoise_wav`/`denoise_chunk` wrappers: those
    are broken against the `audiolab` version pulled in by pip (`Reader` object
    has no attribute `rate`; `Graph.__init__() got an unexpected keyword
    argument 'rate'`) - a real upstream version-skew bug, not something to route
    around by faking a result. The low-level binding calls the identical
    compiled reference library and needs none of that wrapper.
    """
    try:
        from pyrnnoise.rnnoise import FRAME_SIZE, SAMPLE_RATE, create, destroy, process_mono_frame
    except ImportError as exc:
        raise ImportError(
            "rnnoise baseline requires `pyrnnoise` (pip install pyrnnoise). "
            f"Import failed: {exc}"
        ) from exc
    import soxr

    xf = np.asarray(x, dtype=np.float32)
    x48 = xf if sr == SAMPLE_RATE else soxr.resample(xf, sr, SAMPLE_RATE).astype(np.float32)

    n = len(x48)
    pad = (-n) % FRAME_SIZE
    x48p = np.pad(x48, (0, pad))
    state = create()
    try:
        out_chunks = []
        for i in range(0, len(x48p), FRAME_SIZE):
            denoised, _prob = process_mono_frame(state, x48p[i:i + FRAME_SIZE])
            out_chunks.append(denoised)
    finally:
        destroy(state)
    # process_mono_frame returns int16-range float32; RNNoise's own convention.
    out48 = np.concatenate(out_chunks).astype(np.float32)[:n] / 32768.0

    out = out48 if sr == SAMPLE_RATE else soxr.resample(out48, SAMPLE_RATE, sr).astype(np.float32)
    if len(out) < len(xf):
        out = np.pad(out, (0, len(xf) - len(out)))
    else:
        out = out[:len(xf)]
    return out.astype(np.float32)


def _dfn_venv_python() -> Path | None:
    """Locate the isolated `.venv-dfn` interpreter DeepFilterNet lives in.

    DeepFilterNet pins `numpy<2.0`, which is incompatible with this project's
    numpy 2.x (installing it into the main venv silently broke pystoi/scipy
    here - numpy 1.26 is not ABI-compatible with the scipy build in use). It is
    kept in a sibling venv and invoked as a subprocess instead of imported.
    """
    for name in ("python", "python3"):
        p = ROOT / ".venv-dfn" / "bin" / name
        if p.exists():
            return p
    return None


def _deepfilternet(x: np.ndarray, sr: int) -> np.ndarray:
    """DeepFilterNet3 (Rikorose/DeepFilterNet), official pretrained checkpoint,
    downloaded automatically on first use to `~/.cache/DeepFilterNet`.

    Runs out-of-process in `.venv-dfn` (see `_dfn_venv_python`) via
    `scripts/dfn_worker.py`, which resamples 16 kHz <-> 48 kHz around the model
    exactly as `_rnnoise` does, for the same reason (DeepFilterNet's native
    rate is also 48 kHz).
    """
    py = _dfn_venv_python()
    if py is None:
        raise ImportError(
            "deepfilternet baseline requires the isolated '.venv-dfn' "
            "environment (numpy<2.0 conflicts with this project's numpy 2.x). "
            "Create it and `pip install deepfilternet torch torchaudio` there; "
            "see scripts/dfn_worker.py."
        )
    worker = ROOT / "scripts" / "dfn_worker.py"
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.wav"
        out_path = Path(td) / "out.wav"
        import soundfile as sf
        sf.write(in_path, np.asarray(x, dtype=np.float32), sr)
        result = subprocess.run(
            [str(py), str(worker), str(in_path), str(out_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"deepfilternet worker failed (exit {result.returncode}):\n"
                f"{result.stderr[-4000:]}"
            )
        out, out_sr = sf.read(out_path, dtype="float32")
    assert out_sr == sr
    return np.asarray(out, dtype=np.float32)


BUILTIN = {
    "unprocessed": _identity,
    "specsub": spectral_subtraction,
    "wiener": wiener,
    "noisereduce": _noisereduce,
    "rnnoise": _rnnoise,
    "deepfilternet": _deepfilternet,
}


def get(name: str, device: str = "cpu"):
    """Resolve a method name to a callable.

    Names:
      unprocessed | wiener | specsub | noisereduce
      rnnoise                          (reference RNNoise, pretrained)
      deepfilternet                    (DeepFilterNet3, pretrained; needs .venv-dfn)
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
    # rnnoise/deepfilternet are dependency-gated, unlike the rest of BUILTIN
    # (whose only optional dep, noisereduce, has always been listed
    # unconditionally here and fails loudly at call time instead) - they pull
    # in a compiled binding and, for deepfilternet, an entire separate venv, so
    # actually checking avoids advertising a method that will raise.
    out = [n for n in BUILTIN if n not in ("rnnoise", "deepfilternet")]

    import importlib.util
    if importlib.util.find_spec("pyrnnoise") is not None:
        out.append("rnnoise")
    if _dfn_venv_python() is not None:
        out.append("deepfilternet")

    for n, p in (("gtcrn_dns3", "model_trained_on_dns3.tar"),
                 ("gtcrn_vctk", "model_trained_on_vctk.tar")):
        if (ROOT / "checkpoints" / "pretrained" / p).exists():
            out.append(n)
    return out
