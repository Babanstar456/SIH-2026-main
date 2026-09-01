"""DeepFilterNet worker, run inside the isolated `.venv-dfn` interpreter.

Why this exists as a subprocess rather than an in-process import: DeepFilterNet
pins `numpy<2.0`, which conflicts with the rest of this project's numpy 2.x
(installing it in the main venv silently broke pystoi/scipy - numpy 1.26 is not
ABI-compatible with the scipy build here). Keeping it in a separate venv
(`.venv-dfn`, created alongside `.venv`) avoids poisoning the shared
environment. `src/methods.py::_deepfilternet` shells out to this script.

Usage: <venv-dfn python> dfn_worker.py <in_wav> <out_wav>
Reads a mono wav at any sample rate, resamples to DeepFilterNet's native
48 kHz, denoises, resamples back to the input rate, writes `out_wav`.
"""
from __future__ import annotations

import sys

import numpy as np
import soundfile as sf
import soxr
import torch

from df.enhance import enhance, init_df


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    x, sr = sf.read(in_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)

    model, df_state, _ = init_df()
    native_sr = df_state.sr()  # 48000

    x_native = x if sr == native_sr else soxr.resample(x, sr, native_sr).astype(np.float32)
    t = torch.from_numpy(x_native).unsqueeze(0)
    with torch.no_grad():
        out = enhance(model, df_state, t)
    out_native = out.squeeze(0).numpy().astype(np.float32)

    out_sr = out_native if sr == native_sr else soxr.resample(out_native, native_sr, sr).astype(np.float32)
    # Trim/pad to the exact input length - resampling twice can shift length by
    # a sample or two, and every method in the registry must return audio of
    # identical length to its input.
    n = len(x)
    if len(out_sr) < n:
        out_sr = np.pad(out_sr, (0, n - len(out_sr)))
    else:
        out_sr = out_sr[:n]
    sf.write(out_path, out_sr, sr)


if __name__ == "__main__":
    main()
