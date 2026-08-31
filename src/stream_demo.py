"""Live microphone demo and before/after renderer.

Two modes:

    python -m src.stream_demo --list                       # show audio devices
    python -m src.stream_demo --live --onnx artifacts/model.onnx
    python -m src.stream_demo --file input.wav --onnx artifacts/model.onnx

The live path runs the real streaming contract - one 16 ms frame per callback,
caches carried forward - so it demonstrates the same code path the hardware team
will run, not an offline approximation that happens to sound good.
"""
from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import audio as A                                          # noqa: E402
from src.framing import (CONV_CACHE, HOP, INTER_CACHE, N_FFT, SR, TRA_CACHE,
                          WIN)            # noqa: E402


class StreamingEnhancer:
    """Frame-by-frame enhancement with explicit overlap-add.

    Deliberately does NOT use torch.stft: this mirrors what an embedded
    implementation does, so the numbers and the artefacts are the real ones.
    """

    def __init__(self, onnx_path: Path, threads: int = 1,
                 floor_db: float | None = None):
        """`floor_db` caps how deep any bin may be suppressed, e.g. -18.

        None reproduces the model untouched. The cap exists because at negative
        SNR - which is the deployment case, gunfire louder than the talker - the
        unconstrained mask cuts speech away along with the noise. Measured word
        recognition falls as suppression depth rises, so depth is a dial to be
        set, not maximised.
        """
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), so,
                                         providers=["CPUExecutionProvider"])
        self.window = (np.hanning(WIN) ** 0.5).astype(np.float32)
        self.floor_lin = None if floor_db is None else 10.0 ** (floor_db / 20.0)
        self.reset()

    def reset(self) -> None:
        self.conv = np.zeros(CONV_CACHE, dtype="float32")
        self.tra = np.zeros(TRA_CACHE, dtype="float32")
        self.inter = np.zeros(INTER_CACHE, dtype="float32")
        self.in_buf = np.zeros(WIN, dtype=np.float32)
        self.ola = np.zeros(WIN, dtype=np.float32)
        self.norm = np.zeros(WIN, dtype=np.float32)
        self.wsq = self.window ** 2

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """One HOP-sized chunk in, one HOP-sized enhanced chunk out."""
        assert len(chunk) == HOP, f"expected {HOP} samples, got {len(chunk)}"
        self.in_buf = np.concatenate([self.in_buf[HOP:], chunk.astype(np.float32)])

        spec = np.fft.rfft(self.in_buf * self.window, N_FFT).astype("complex64")
        mix = np.stack([spec.real, spec.imag], -1)[None, :, None, :].astype("float32")

        enh, self.conv, self.tra, self.inter = self.sess.run(
            [], {"mix": mix, "conv_cache": self.conv,
                 "tra_cache": self.tra, "inter_cache": self.inter})

        comp = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]

        if self.floor_lin is not None:
            # Recover the gain the model chose, clamp its depth, reapply to the
            # NOISY spectrum so the noisy phase is kept - the same construction
            # every mask-based suppressor uses, and identical to the offline
            # floor sweep so live and file paths cannot diverge.
            g = np.abs(comp) / (np.abs(spec) + 1e-10)
            comp = np.maximum(g, self.floor_lin) * spec

        frame = np.fft.irfft(comp, N_FFT)[:WIN].astype(np.float32) * self.window

        self.ola = np.concatenate([self.ola[HOP:], np.zeros(HOP, np.float32)])
        self.norm = np.concatenate([self.norm[HOP:], np.zeros(HOP, np.float32)])
        self.ola += frame
        self.norm += self.wsq
        out = self.ola[:HOP] / np.maximum(self.norm[:HOP], 1e-6)
        return np.clip(out, -1.0, 1.0)


def run_file(onnx: Path, wav_in: Path, out_dir: Path,
             floor_db: float | None = None) -> None:
    x = A.load_audio(wav_in)
    enh = StreamingEnhancer(onnx, floor_db=floor_db)
    n_chunks = len(x) // HOP
    y = np.zeros(n_chunks * HOP, dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n_chunks):
        y[i * HOP:(i + 1) * HOP] = enh.process_chunk(x[i * HOP:(i + 1) * HOP])
    dt = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    A.save_audio(out_dir / "before.wav", x[:len(y)])
    A.save_audio(out_dir / "after.wav", y)
    dur = len(y) / SR
    print(f"processed {dur:.1f}s in {dt:.2f}s   RTF={dt/dur:.4f}")
    print(f"wrote {out_dir/'before.wav'} and {out_dir/'after.wav'}")


def run_live(onnx: Path, in_dev, out_dev, floor_db: float | None = None) -> None:
    import sounddevice as sd
    enh = StreamingEnhancer(onnx, floor_db=floor_db)
    q: queue.Queue = queue.Queue()

    def callback(indata, outdata, frames, t, status):
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, 0]
        outdata[:, 0] = enh.process_chunk(mono)
        q.put(float(np.abs(mono).max()))

    print(f"streaming: {SR} Hz, {HOP} samples ({1000*HOP/SR:.0f} ms) per callback")
    print("Ctrl+C to stop.\n")
    with sd.Stream(samplerate=SR, blocksize=HOP, dtype="float32",
                   channels=1, callback=callback,
                   device=(in_dev, out_dev)):
        try:
            while True:
                peak = q.get()
                bars = int(min(peak, 1.0) * 40)
                print(f"\rin {'#'*bars}{' '*(40-bars)} {peak:.3f}", end="")
        except KeyboardInterrupt:
            print("\nstopped.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="artifacts/gtcrn_dns3_simple.onnx")
    ap.add_argument("--file", default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--in-device", default=None)
    ap.add_argument("--out-device", default=None)
    ap.add_argument("--out-dir", default="results/demo")
    ap.add_argument("--floor-db", type=float, default=None,
                    help="cap suppression depth, e.g. -18. Omit for the raw model.")
    a = ap.parse_args()

    if a.list:
        import sounddevice as sd
        print(sd.query_devices())
        return

    onnx = Path(a.onnx) if Path(a.onnx).is_absolute() else ROOT / a.onnx
    if not onnx.exists():
        raise SystemExit(f"{onnx} not found - run src.export_onnx first")

    def dev(v):
        return int(v) if v is not None and str(v).isdigit() else v

    if a.live:
        run_live(onnx, dev(a.in_device), dev(a.out_device), a.floor_db)
    elif a.file:
        run_file(onnx, Path(a.file), ROOT / a.out_dir, a.floor_db)
    else:
        ap.error("pass --file, --live or --list")


if __name__ == "__main__":
    main()
