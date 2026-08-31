"""Export a trained checkpoint to streaming ONNX - and PROVE it still works.

The hardware team gets a single `model.onnx` that consumes one 16 ms frame at a
time and carries its own recurrent state. Converting an offline model to a
streaming one silently changes behaviour if the cache wiring is wrong: it still
loads, still runs, still emits plausible audio, and sounds subtly worse. So this
script does not just export - it runs the exported graph frame by frame and
compares against the offline PyTorch output, and refuses to declare success if
they diverge.

    python -m src.export_onnx --ckpt checkpoints/ft_best.pt --out artifacts/model.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The upstream streaming model imports `modules.convolution` relative to its own
# directory, so that directory has to be importable as a top-level package root.
STREAM_DIR = ROOT / "third_party" / "gtcrn" / "stream"
if str(STREAM_DIR) not in sys.path:
    sys.path.insert(0, str(STREAM_DIR))

from src import stft as S                      # noqa: E402
from src.models.gtcrn import GTCRN             # noqa: E402

from src.framing import CONV_CACHE, INTER_CACHE, TRA_CACHE  # noqa: E402,F401


def zero_caches(np_mode: bool = False):
    mk = (lambda s: np.zeros(s, dtype="float32")) if np_mode else \
         (lambda s: torch.zeros(*s))
    return mk(CONV_CACHE), mk(TRA_CACHE), mk(INTER_CACHE)


def build_stream_model(ckpt: Path, device: str = "cpu"):
    from gtcrn_stream import StreamGTCRN            # noqa: PLC0415
    from modules.convert import convert_to_stream   # noqa: PLC0415

    offline = GTCRN().to(device).eval()
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    state = obj.get("model", obj.get("state_dict", obj)) if isinstance(obj, dict) else obj
    offline.load_state_dict(state)

    stream = StreamGTCRN().to(device).eval()
    convert_to_stream(stream, offline)
    return offline, stream


def export(ckpt: Path, out: Path, simplify: bool = True) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    _, stream = build_stream_model(ckpt)
    conv_c, tra_c, inter_c = zero_caches()
    dummy = torch.randn(1, 257, 1, 2)

    torch.onnx.export(
        stream, (dummy, conv_c, tra_c, inter_c), str(out),
        input_names=["mix", "conv_cache", "tra_cache", "inter_cache"],
        output_names=["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"],
        opset_version=11, verbose=False,
    )
    import onnx
    # torch.onnx may split weights into an external `.onnx.data` sidecar. The
    # hardware team is handed ONE file, so fold everything back inline - a model
    # that silently depends on a sidecar loads fine here and fails there.
    m = onnx.load(str(out), load_external_data=True)
    onnx.save(m, str(out), save_as_external_data=False)
    sidecar = out.with_suffix(out.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()
    onnx.checker.check_model(onnx.load(str(out)))
    print(f"exported {out}  ({out.stat().st_size/1024:.0f} KB, weights inline)")

    if simplify:
        try:
            from onnxsim import simplify as onnxsim_simplify
            m, ok = onnxsim_simplify(onnx.load(str(out)))
            if ok:
                simp = out.with_name(out.stem + "_simple.onnx")
                onnx.save(m, str(simp))
                print(f"simplified -> {simp}  ({simp.stat().st_size/1024:.0f} KB)")
                return simp
            print("onnxsim could not validate the simplified graph; keeping original")
        except ImportError:
            print("onnxsim not installed; skipping simplification")
    return out


def verify(onnx_path: Path, ckpt: Path, wav: np.ndarray, tol: float = 1e-3) -> dict:
    """Frame-by-frame ONNX vs whole-utterance PyTorch."""
    import onnxruntime as ort

    offline, _ = build_stream_model(ckpt)
    with torch.no_grad():
        spec = S.stft(torch.from_numpy(wav))[None]
        ref = S.istft(offline(spec)[0], length=len(wav)).numpy()

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    x = spec.numpy()
    conv_c, tra_c, inter_c = zero_caches(np_mode=True)
    outs, times = [], []
    for i in range(x.shape[-2]):
        t0 = time.perf_counter()
        o, conv_c, tra_c, inter_c = sess.run(
            [], {"mix": x[..., i:i + 1, :], "conv_cache": conv_c,
                 "tra_cache": tra_c, "inter_cache": inter_c})
        times.append(time.perf_counter() - t0)
        outs.append(o)

    spec_out = torch.from_numpy(np.concatenate(outs, axis=2))
    got = S.istft(spec_out[0], length=len(wav)).numpy()

    n = min(len(ref), len(got))
    diff = np.abs(ref[:n] - got[:n])
    denom = np.sqrt(np.mean(ref[:n] ** 2)) + 1e-12
    rel = float(diff.max() / denom)
    t = np.array(times) * 1000.0
    return {
        "max_abs_diff": float(diff.max()),
        "rel_to_rms": rel,
        "match": bool(rel < tol),
        "frames": len(times),
        "ms_p50": float(np.percentile(t, 50)),
        "ms_p95": float(np.percentile(t, 95)),
        "ms_mean": float(t.mean()),
        "rtf": float(t.mean() / 16.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="artifacts/model.onnx")
    ap.add_argument("--wav", default="third_party/gtcrn/test_wavs/mix.wav")
    ap.add_argument("--no-simplify", action="store_true")
    a = ap.parse_args()

    from src import audio as A
    ckpt = Path(a.ckpt) if Path(a.ckpt).is_absolute() else ROOT / a.ckpt
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out

    final = export(ckpt, out, simplify=not a.no_simplify)

    # Prove the shipped file stands alone: load it from a scratch copy with no
    # neighbouring files at all.
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        solo = Path(td) / "solo.onnx"
        shutil.copy2(final, solo)
        try:
            import onnxruntime as ort
            ort.InferenceSession(str(solo), providers=["CPUExecutionProvider"])
            print("self-contained: loads with no sidecar files present")
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"{final.name} is NOT self-contained ({e}). It would fail on the "
                f"hardware team's machine. Do not ship it.")

    wav = A.load_audio(ROOT / a.wav)
    r = verify(final, ckpt, wav)

    print("\n--- streaming ONNX vs offline PyTorch ---")
    print(f"  frames            : {r['frames']}")
    print(f"  max abs diff      : {r['max_abs_diff']:.3e}")
    print(f"  relative to RMS   : {r['rel_to_rms']:.3e}")
    print(f"  per-frame p50/p95 : {r['ms_p50']:.3f} / {r['ms_p95']:.3f} ms")
    print(f"  RTF               : {r['rtf']:.4f}")
    if r["match"]:
        print("  MATCH - the streaming export is faithful to the offline model.")
    else:
        print("  !! MISMATCH - do NOT ship this. The cache wiring is wrong; the\n"
              "     model will sound subtly worse on hardware while appearing to work.")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "onnx_verify.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(ckpt), "onnx": str(final), **r}, f, indent=1)
    sys.exit(0 if r["match"] else 1)


if __name__ == "__main__":
    main()
