"""Torch-free latency/RTF benchmark for the target embedded device.

Unlike `src/bench.py` (which imports PyTorch to also benchmark the offline
reference model), this script only needs numpy + onnxruntime — the same
dependency set as `src/stream_demo.py` — so it can be copied onto hardware
that will never have PyTorch installed (a Raspberry Pi, for instance) and
still produce a real PASS/FAIL against the latency and RTF targets.

    python scripts/bench_edge.py --onnx artifacts/model_lowsnr_simple.onnx

Run this ON the target device. A number measured on a laptop does not
transfer to an ARM board — different microarchitecture, different ONNX
Runtime kernel coverage, different memory subsystem.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framing import CONV_CACHE, HOP, INTER_CACHE, SR, TRA_CACHE, WIN  # noqa: E402

HOP_MS = 1000.0 * HOP / SR
WIN_MS = 1000.0 * WIN / SR


def bench_onnx(onnx_path: Path, n_frames: int, threads: int, warmup: int = 50) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    frames = rng.standard_normal((n_frames + warmup, 1, 257, 1, 2)).astype("float32")
    conv_c = np.zeros(CONV_CACHE, dtype="float32")
    tra_c = np.zeros(TRA_CACHE, dtype="float32")
    inter_c = np.zeros(INTER_CACHE, dtype="float32")

    times = []
    for i in range(n_frames + warmup):
        t0 = time.perf_counter()
        _, conv_c, tra_c, inter_c = sess.run(
            [], {"mix": frames[i], "conv_cache": conv_c,
                 "tra_cache": tra_c, "inter_cache": inter_c})
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    t = np.array(times) * 1000.0
    return {
        "threads": threads, "frames": len(t),
        "ms_p50": float(np.percentile(t, 50)),
        "ms_p95": float(np.percentile(t, 95)),
        "ms_p99": float(np.percentile(t, 99)),
        "ms_mean": float(t.mean()), "ms_max": float(t.max()),
        "rtf_mean": float(t.mean() / HOP_MS),
        "rtf_p95": float(np.percentile(t, 95) / HOP_MS),
    }


def measure_streaming_delay(onnx_path: Path, seconds: float = 4.0) -> dict:
    """Cross-correlate output against input to MEASURE the OLA delay."""
    from src.stream_demo import StreamingEnhancer

    rng = np.random.default_rng(0)
    n = int(SR * seconds)
    x = rng.standard_normal(n).astype(np.float32) * 0.1
    x[:HOP * 4] = 0.0

    enh = StreamingEnhancer(onnx_path)
    chunks = n // HOP
    y = np.zeros(chunks * HOP, dtype=np.float32)
    for i in range(chunks):
        y[i * HOP:(i + 1) * HOP] = enh.process_chunk(x[i * HOP:(i + 1) * HOP])

    m = len(y)
    seg = slice(SR // 2, min(m, SR * 3))
    a, b = y[seg], x[:m][seg]
    c = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    lag = int(np.argmax(np.abs(c)) - (len(b) - 1))
    return {"delay_samples": lag, "delay_ms": 1000.0 * lag / SR,
            "equals_win_minus_hop": lag == (WIN - HOP)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="artifacts/model_lowsnr_simple.onnx")
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--rtf-target", type=float, default=0.5)
    ap.add_argument("--latency-target-ms", type=float, default=32.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    onnx_path = Path(a.onnx) if Path(a.onnx).is_absolute() else ROOT / a.onnx
    if not onnx_path.exists():
        raise SystemExit(f"{onnx_path} not found")

    report: dict = {
        "device": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "onnx": str(onnx_path),
        "model_size_kb": onnx_path.stat().st_size / 1024,
        "hop_ms": HOP_MS, "window_ms": WIN_MS,
        "onnx_runs": [],
    }

    print(f"device     : {platform.machine()}  ({platform.platform()})")
    print(f"model      : {onnx_path.name}  ({report['model_size_kb']:.0f} KB)")

    for th in a.threads:
        r = bench_onnx(onnx_path, a.frames, th)
        report["onnx_runs"].append(r)
        print(f"ONNX streaming, {th} thread(s): "
              f"p50={r['ms_p50']:.3f}ms  p95={r['ms_p95']:.3f}ms  "
              f"max={r['ms_max']:.3f}ms  RTF={r['rtf_mean']:.4f}")

    delay = measure_streaming_delay(onnx_path)
    report["measured_streaming_delay"] = delay
    print(f"\nmeasured streaming delay: {delay['delay_samples']} samples "
          f"({delay['delay_ms']:.2f} ms)"
          + ("  == win - hop, as expected" if delay["equals_win_minus_hop"] else
             "  !! does not equal win-hop, investigate"))

    single = next(r for r in report["onnx_runs"] if r["threads"] == a.threads[0])
    compute_p95 = single["ms_p95"]
    total = HOP_MS + delay["delay_ms"] + compute_p95
    report["latency_budget_ms"] = {
        "chunk_buffering": HOP_MS,
        "overlap_add_delay_measured": delay["delay_ms"],
        "model_compute_p95": compute_p95,
        "total_p95": total,
    }

    rtf_pass = single["rtf_mean"] < a.rtf_target
    lat_pass = total < a.latency_target_ms
    print("\n--- latency budget (this device, p95 compute) ---")
    print(f"  chunk buffering (hop)      {HOP_MS:6.2f} ms")
    print(f"  overlap-add delay          {delay['delay_ms']:6.2f} ms   <- MEASURED")
    print(f"  model compute (p95)        {compute_p95:6.2f} ms")
    print(f"  {'-'*40}")
    print(f"  total                      {total:6.2f} ms")
    print(f"\n  RTF (mean, {a.threads[0]} thread)      {single['rtf_mean']:.4f}   "
          f"target < {a.rtf_target}  -> {'PASS' if rtf_pass else 'FAIL'}")
    print(f"  latency total              {total:.2f} ms  target < {a.latency_target_ms} ms  -> "
          f"{'PASS' if lat_pass else 'OVER by %.1f ms' % (total - a.latency_target_ms)}")
    report["rtf_pass"] = rtf_pass
    report["latency_pass"] = lat_pass

    if a.out:
        out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {out}")

    print("\nNOTE: run this on an OTHERWISE IDLE device. Numbers measured on a "
          "different CPU (e.g. a laptop) do not transfer to this one.")


if __name__ == "__main__":
    main()
