"""Build the hardware-team handoff bundle.

Two modes:

  --stub    Day-1 deliverable: an identity ("pass-through") ONNX with the exact
            input/output/cache signature the real model will have, so the
            hardware team can build and test their integration while training is
            still running. It is clearly named, produces no metrics, and is
            replaced by the trained model.

  --model   Final deliverable: the trained model plus a spec sheet, example
            inference code, and the measured numbers.

    python scripts/make_handoff.py --stub
    python scripts/make_handoff.py --model artifacts/model_simple.onnx
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.framing import CONV_CACHE, INTER_CACHE, TRA_CACHE  # noqa: E402
from src.framing import HOP, N_FFT, SR, WIN                     # noqa: E402

OUT = ROOT / "artifacts"


class PassThrough(nn.Module):
    """Identity with the real model's exact signature.

    Deliberately touches the caches so the graph keeps them as genuine
    inputs/outputs - if they were dropped, the hardware team would build against
    a signature that changes the day the real model lands.
    """

    def forward(self, mix, conv_cache, tra_cache, inter_cache):
        return mix, conv_cache * 1.0, tra_cache * 1.0, inter_cache * 1.0


def make_stub() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "passthrough_stub.onnx"
    m = PassThrough().eval()
    dummy = (torch.randn(1, 257, 1, 2), torch.zeros(*CONV_CACHE),
             torch.zeros(*TRA_CACHE), torch.zeros(*INTER_CACHE))
    torch.onnx.export(
        m, dummy, str(path),
        input_names=["mix", "conv_cache", "tra_cache", "inter_cache"],
        output_names=["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"],
        opset_version=11)

    import onnx
    m = onnx.load(str(path), load_external_data=True)
    onnx.save(m, str(path), save_as_external_data=False)
    side = path.with_suffix(path.suffix + ".data")
    if side.exists():
        side.unlink()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 257, 1, 2).astype("float32")
    out = sess.run([], {"mix": x,
                        "conv_cache": np.zeros(CONV_CACHE, "float32"),
                        "tra_cache": np.zeros(TRA_CACHE, "float32"),
                        "inter_cache": np.zeros(INTER_CACHE, "float32")})
    assert np.allclose(out[0], x), "stub is not identity"
    print(f"wrote {path}  ({path.stat().st_size/1024:.1f} KB) - verified identity")
    return path


def spec_sheet(onnx_name: str) -> str:
    bench = ROOT / "results" / "bench.json"
    b = json.loads(bench.read_text(encoding="utf-8")) if bench.exists() else None
    ver = ROOT / "results" / "onnx_verify.json"
    v = json.loads(ver.read_text(encoding="utf-8")) if ver.exists() else None

    lat = "_not yet measured - run `python -m src.bench`_"
    if b and "latency_budget_ms" in b:
        L = b["latency_budget_ms"]
        single = next((r for r in b["onnx_runs"] if r["threads"] == 1), None)
        lat = (
            f"| stage | ms |\n|---|---|\n"
            f"| chunk buffering (hop) | {L['chunk_buffering']:.2f} |\n"
            f"| overlap-add delay (measured) | {L['overlap_add_delay_measured']:.2f} |\n"
            f"| model compute (p95, 1 thread) | {L['model_compute_p95_1thread']:.2f} |\n"
            f"| **total** | **{L['total_p95']:.2f}** |\n\n"
            f"RTF (mean, 1 thread): **{single['rtf_mean']:.4f}**\n" if single else lat)

    match = ""
    if v:
        match = (f"\nStreaming ONNX verified against offline PyTorch: "
                 f"max abs diff `{v['max_abs_diff']:.2e}`, "
                 f"relative to RMS `{v['rel_to_rms']:.2e}` "
                 f"({'MATCH' if v['match'] else 'MISMATCH'}).\n")

    opset = "unknown"
    try:
        import onnx
        mm = onnx.load(str(OUT / onnx_name))
        opset = ", ".join(f"{i.domain or 'ai.onnx'} v{i.version}"
                          for i in mm.opset_import)
    except Exception:  # noqa: BLE001
        pass

    return f"""# Model spec sheet — SIH 26052 Path 1

Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}

## Interface

- **File**: `{onnx_name}` (weights inline - no sidecar file required)
- **ONNX opset**: {opset}

  > PyTorch's exporter upgrades the graph to opset 18 regardless of the
  > requested version. Confirm your runtime supports it before porting to an
  > embedded target or converting to TensorRT; older ONNX Runtime builds and
  > some converters cap out lower.
- **Sample rate**: {SR} Hz, mono, float32 in [-1, 1]
- **Framing**: `n_fft={N_FFT}`, `hop={HOP}` ({1000*HOP/SR:.0f} ms), `win={WIN}` ({1000*WIN/SR:.0f} ms)
- **Window**: `hann(512) ** 0.5` — sqrt-Hann, used for BOTH analysis and synthesis
- **Runtime**: ONNX Runtime, `CPUExecutionProvider`

The model is **stateful and streaming**: it consumes ONE frame per call and you
must feed its output caches back in on the next call.

### Inputs
| name | shape | dtype |
|---|---|---|
| `mix` | `(1, 257, 1, 2)` | float32 |
| `conv_cache` | `{CONV_CACHE}` | float32 |
| `tra_cache` | `{TRA_CACHE}` | float32 |
| `inter_cache` | `{INTER_CACHE}` | float32 |

`mix` is the STFT of one frame as `(batch, freq_bins, time=1, [real, imag])`.
Initialise all three caches to **zeros** at stream start.

### Outputs
`enh` (same shape as `mix`), then `conv_cache_out`, `tra_cache_out`,
`inter_cache_out` — pass these back as the next call's caches.

## Latency

{lat}
**Note.** Algorithmic latency equals the {1000*WIN/SR:.0f} ms analysis window, not the
{1000*HOP/SR:.0f} ms hop — a frame cannot be transformed until all {WIN} of its samples
exist. This is inherent to `n_fft={N_FFT}` and cannot be reduced without changing the
framing. Reported in full rather than quoting the hop alone.
{match}
## Signal chain assumptions

The model is trained on audio that has passed through a **fast analog limiter**
between mic preamp and ADC (soft-knee, threshold 0.6–0.95 full scale). Feeding it
unlimited audio with saturating gunshot peaks is a domain mismatch.

The model performs **suppression only** — it does not apply gain. Voice
amplification belongs downstream in the analog path.
"""


EXAMPLE = '''"""Minimal streaming inference example — SIH 26052 Path 1.

Feeds a wav file through the model one 16 ms frame at a time, exactly as a live
audio callback would.
"""
import numpy as np
import onnxruntime as ort
import soundfile as sf

SR, N_FFT, HOP, WIN = 16000, 512, 256, 512
MODEL = "model.onnx"

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
conv_cache = np.zeros((2, 1, 16, 16, 33), dtype="float32")
tra_cache = np.zeros((2, 3, 1, 1, 16), dtype="float32")
inter_cache = np.zeros((2, 1, 33, 16), dtype="float32")

x, sr = sf.read("input.wav", dtype="float32")
assert sr == SR and x.ndim == 1

window = np.hanning(WIN) ** 0.5
n_frames = 1 + len(x) // HOP
padded = np.pad(x, (N_FFT // 2, N_FFT // 2 + HOP))

out_spec = []
for t in range(n_frames):
    frame = padded[t * HOP: t * HOP + WIN] * window
    spec = np.fft.rfft(frame, N_FFT).astype("complex64")
    mix = np.stack([spec.real, spec.imag], -1)[None, :, None, :].astype("float32")

    enh, conv_cache, tra_cache, inter_cache = sess.run(
        [], {"mix": mix, "conv_cache": conv_cache,
             "tra_cache": tra_cache, "inter_cache": inter_cache})
    out_spec.append(enh)

spec = np.concatenate(out_spec, axis=2)[0]
comp = spec[..., 0] + 1j * spec[..., 1]

# overlap-add
y = np.zeros(len(x) + N_FFT, dtype="float32")
norm = np.zeros_like(y)
for t in range(comp.shape[1]):
    frame = np.fft.irfft(comp[:, t], N_FFT)[:WIN] * window
    y[t * HOP: t * HOP + WIN] += frame
    norm[t * HOP: t * HOP + WIN] += window ** 2
y = y[N_FFT // 2: N_FFT // 2 + len(x)] / np.maximum(norm[N_FFT // 2: N_FFT // 2 + len(x)], 1e-8)

sf.write("output.wav", y, SR)
print("wrote output.wav")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if a.stub:
        p = make_stub()
        (OUT / "STUB_README.md").write_text(
            "# passthrough_stub.onnx\n\n"
            "**This is not the real model.** It is an identity pass-through with the\n"
            "exact input/output/cache signature the trained model will have, so\n"
            "integration can be built and tested before training finishes.\n\n"
            "It removes no noise and produces no metrics. Replace it with\n"
            "`model.onnx` when that is delivered - the interface will not change.\n\n"
            + spec_sheet(p.name), encoding="utf-8")
        print(f"wrote {OUT/'STUB_README.md'}")

    if a.model:
        src = Path(a.model) if Path(a.model).is_absolute() else ROOT / a.model
        if not src.exists():
            raise SystemExit(f"{src} not found")
        dst = OUT / "model.onnx"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        (OUT / "SPEC.md").write_text(spec_sheet(dst.name), encoding="utf-8")
        (OUT / "example_inference.py").write_text(EXAMPLE, encoding="utf-8")
        for name in ("results.md", "results.csv"):
            p = ROOT / "results" / name
            if p.exists():
                shutil.copy2(p, OUT / name)
        print(f"handoff bundle -> {OUT}")
        for f in sorted(OUT.iterdir()):
            print(f"  {f.name}  ({f.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
