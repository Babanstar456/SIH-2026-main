# Model spec sheet — SIH 26052 Path 1

Generated 2026-08-28T16:35:00+00:00

## Interface

- **File**: `model.onnx` (weights inline - no sidecar file required)
- **ONNX opset**: ai.onnx v18

  > PyTorch's exporter upgrades the graph to opset 18 regardless of the
  > requested version. Confirm your runtime supports it before porting to an
  > embedded target or converting to TensorRT; older ONNX Runtime builds and
  > some converters cap out lower.
- **Sample rate**: 16000 Hz, mono, float32 in [-1, 1]
- **Framing**: `n_fft=512`, `hop=256` (16 ms), `win=512` (32 ms)
- **Window**: `hann(512) ** 0.5` — sqrt-Hann, used for BOTH analysis and synthesis
- **Runtime**: ONNX Runtime, `CPUExecutionProvider`

The model is **stateful and streaming**: it consumes ONE frame per call and you
must feed its output caches back in on the next call.

### Inputs
| name | shape | dtype |
|---|---|---|
| `mix` | `(1, 257, 1, 2)` | float32 |
| `conv_cache` | `(2, 1, 16, 16, 33)` | float32 |
| `tra_cache` | `(2, 3, 1, 1, 16)` | float32 |
| `inter_cache` | `(2, 1, 33, 16)` | float32 |

`mix` is the STFT of one frame as `(batch, freq_bins, time=1, [real, imag])`.
Initialise all three caches to **zeros** at stream start.

### Outputs
`enh` (same shape as `mix`), then `conv_cache_out`, `tra_cache_out`,
`inter_cache_out` — pass these back as the next call's caches.

## Latency

| stage | ms |
|---|---|
| chunk buffering (hop) | 16.00 |
| overlap-add delay (measured) | 16.00 |
| model compute (p95, 1 thread) | 6.01 |
| **total** | **38.01** |

RTF (mean, 1 thread): **0.2955**

**Note.** Algorithmic latency equals the 32 ms analysis window, not the
16 ms hop — a frame cannot be transformed until all 512 of its samples
exist. This is inherent to `n_fft=512` and cannot be reduced without changing the
framing. Reported in full rather than quoting the hop alone.

Streaming ONNX verified against offline PyTorch: max abs diff `3.99e-07`, relative to RMS `6.27e-06` (MATCH).

## Signal chain assumptions

The model is trained on audio that has passed through a **fast analog limiter**
between mic preamp and ADC (soft-knee, threshold 0.6–0.95 full scale). Feeding it
unlimited audio with saturating gunshot peaks is a domain mismatch.

The model performs **suppression only** — it does not apply gain. Voice
amplification belongs downstream in the analog path.
