"""Minimal streaming inference example — SIH 26052 Path 1.

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
