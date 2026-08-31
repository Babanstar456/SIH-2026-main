"""Latency and real-time-factor benchmark.

The point of this file is to report the latency budget HONESTLY, because the
problem statement's numbers do not quite close and pretending otherwise would
fail at hardware integration.

GTCRN frames at n_fft=512 / hop=256 @ 16 kHz. That means:

    chunk buffering (hop)     16 ms   - the agreed chunk size
    analysis window           32 ms   - a frame cannot be transformed until all
                                        512 samples of it exist
    model compute            measured

Algorithmic latency equals the ANALYSIS WINDOW, not the hop: 32 ms, before a
single multiply. The PS asks for "16 ms chunks" and "under 32 ms delay", and
both of those trace back to this same config - so the target sits exactly on the
boundary and the true total lands slightly over it.

We report the breakdown rather than quote the hop and call it 16 ms. If the
32 ms ceiling turns out to be hard rather than nominal, the fix is a 320/160
retrain (20 ms window), which is a real change and is listed as an open item.

    python -m src.bench --onnx artifacts/model_simple.onnx
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import stft as S                       # noqa: E402
from src.models.gtcrn import GTCRN              # noqa: E402

SR_LOCAL = S.SR
HOP_MS = 1000.0 * S.HOP / S.SR      # 16.0
WIN_MS = 1000.0 * S.WIN / S.SR      # 32.0


def bench_onnx(onnx_path: Path, n_frames: int = 800, threads: int = 1,
               warmup: int = 50) -> dict:
    import onnxruntime as ort
    from src.export_onnx import zero_caches

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    frames = rng.standard_normal((n_frames + warmup, 1, 257, 1, 2)).astype("float32")
    conv_c, tra_c, inter_c = zero_caches(np_mode=True)

    times = []
    for i in range(n_frames + warmup):
        t0 = time.perf_counter()
        _, conv_c, tra_c, inter_c = sess.run(
            [], {"mix": frames[i], "conv_cache": conv_c,
                 "tra_cache": tra_c, "inter_cache": inter_c})
        if i >= warmup:                       # discard warmup: first calls pay
            times.append(time.perf_counter() - t0)   # allocation + cache-warm costs
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
    """MEASURE the input->output delay instead of asserting it.

    Runs a signal through the real streaming path and cross-correlates the
    output against the input. This catches the implementation delay - the
    overlap-add buffer holding a sample back until every frame covering it has
    been computed - which a purely theoretical budget tends to miss.
    """
    from src.stream_demo import StreamingEnhancer

    rng = np.random.default_rng(0)
    n = int(SR_LOCAL * seconds)
    # Band-limited noise: broadband enough for a sharp correlation peak, and it
    # keeps the model out of its "this is pure silence" regime.
    x = rng.standard_normal(n).astype(np.float32) * 0.1
    x[: S.HOP * 4] = 0.0

    enh = StreamingEnhancer(onnx_path)
    chunks = n // S.HOP
    y = np.zeros(chunks * S.HOP, dtype=np.float32)
    for i in range(chunks):
        y[i * S.HOP:(i + 1) * S.HOP] = enh.process_chunk(x[i * S.HOP:(i + 1) * S.HOP])

    m = len(y)
    seg = slice(S.SR // 2, min(m, S.SR * 3))
    a, b = y[seg], x[:m][seg]
    c = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    lag = int(np.argmax(np.abs(c)) - (len(b) - 1))
    return {"delay_samples": lag, "delay_ms": 1000.0 * lag / S.SR,
            "equals_win_minus_hop": lag == (S.WIN - S.HOP)}


def bench_torch_offline(seconds: float = 10.0, threads: int = 1) -> dict:
    """Whole-utterance PyTorch CPU path, for reference against the ONNX path."""
    torch.set_num_threads(threads)
    model = GTCRN().eval()
    x = torch.randn(int(S.SR * seconds))
    with torch.no_grad():
        for _ in range(2):
            S.istft(model(S.stft(x)[None])[0], length=len(x))
        runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            S.istft(model(S.stft(x)[None])[0], length=len(x))
            runs.append(time.perf_counter() - t0)
    return {"threads": threads, "seconds": seconds,
            "wall_s": float(np.median(runs)),
            "rtf": float(np.median(runs) / seconds)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="artifacts/gtcrn_dns3_simple.onnx")
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--out", default="results/bench.json")
    a = ap.parse_args()

    onnx_path = Path(a.onnx) if Path(a.onnx).is_absolute() else ROOT / a.onnx
    if not onnx_path.exists():
        raise SystemExit(f"{onnx_path} not found - run src.export_onnx first")

    n_par = sum(p.numel() for p in GTCRN().parameters())
    report: dict = {
        "platform": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "onnx": str(onnx_path),
        "parameters": n_par,
        "hop_ms": HOP_MS, "window_ms": WIN_MS,
        "onnx_runs": [], "torch_offline": [],
    }

    print(f"model      : {onnx_path.name}  ({onnx_path.stat().st_size/1024:.0f} KB)")
    print(f"parameters : {n_par:,}")
    print(f"cpu        : {report['platform']}\n")

    for th in a.threads:
        r = bench_onnx(onnx_path, a.frames, th)
        report["onnx_runs"].append(r)
        print(f"ONNX streaming, {th} thread(s): "
              f"p50={r['ms_p50']:.3f}ms  p95={r['ms_p95']:.3f}ms  "
              f"max={r['ms_max']:.3f}ms  RTF={r['rtf_mean']:.4f}")

    for th in a.threads:
        r = bench_torch_offline(threads=th)
        report["torch_offline"].append(r)
        print(f"PyTorch offline, {th} thread(s): RTF={r['rtf']:.4f}")

    delay = measure_streaming_delay(onnx_path)
    report["measured_streaming_delay"] = delay
    print(f"\nmeasured streaming delay: {delay['delay_samples']} samples "
          f"({delay['delay_ms']:.2f} ms)"
          + ("  == win - hop, as expected" if delay["equals_win_minus_hop"] else
             "  !! does not equal win-hop, investigate"))

    single = next(r for r in report["onnx_runs"] if r["threads"] == 1)
    compute_p95 = single["ms_p95"]
    # Measured OLA delay, not an assumed one, plus the chunk you must collect
    # before you can call the model at all, plus the time the call takes.
    total = HOP_MS + delay["delay_ms"] + compute_p95
    report["latency_budget_ms"] = {
        "chunk_buffering": HOP_MS,
        "overlap_add_delay_measured": delay["delay_ms"],
        "model_compute_p95_1thread": compute_p95,
        "total_p95": total,
        "analysis_window_for_reference": WIN_MS,
        "note": ("Chunk buffering (16 ms) + measured overlap-add delay (16 ms) "
                 "together equal the 32 ms analysis window: a sample is not "
                 "released until every frame covering it has been computed. This "
                 "is inherent to n_fft=512 and cannot be reduced without changing "
                 "the framing. Reported in full rather than quoting the hop."),
    }

    print("\n--- latency budget (single thread, p95 compute) ---")
    print(f"  chunk buffering (hop)      {HOP_MS:6.2f} ms")
    print(f"  overlap-add delay          {delay['delay_ms']:6.2f} ms   <- MEASURED")
    print(f"  model compute (p95)        {compute_p95:6.2f} ms")
    print(f"  {'-'*40}")
    print(f"  total                      {total:6.2f} ms")
    print(f"\n  RTF (mean, 1 thread)       {single['rtf_mean']:.4f}   "
          f"target < 0.5  -> {'PASS' if single['rtf_mean'] < 0.5 else 'FAIL'}")
    print(f"  latency total              {total:.2f} ms  target < 32 ms  -> "
          f"{'PASS' if total < 32 else 'OVER by %.1f ms' % (total - 32)}")

    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {out}")
    print("\nNOTE: benchmark on an OTHERWISE IDLE machine. Background downloads,\n"
          "      extraction or training will inflate these numbers substantially.")


if __name__ == "__main__":
    main()
