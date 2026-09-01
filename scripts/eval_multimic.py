"""Two-microphone evaluation: NLMS reference-mic cancellation vs the shipped
single-mic neural model vs a DSP+neural hybrid.

SCOPE AND HONESTY NOTE, read before trusting any number this prints:

  - The reference-mic channel is SYNTHETIC (`src/baselines/reference_mic.py`)
    - a plausible but unmeasured acoustic-coupling model, built because this
    project has no real dual-microphone recording. Treat every result here as
    HYPOTHESIS, not validated hardware performance.
  - This does NOT use the corpus-driven `Mixer` (`src/mixer.py`) - that
    machine's dataset (LibriSpeech/MUSAN/manifests) is not present here. It
    builds one controlled mixture from two REAL recordings already in the
    repo: `test-result/voice/voice_dry3.wav` (clean dry speech) and
    `test-result/voice/noise_bed2.wav` (real gunfire extracted from a live
    take), so the noise and (before mixing) the speech are both genuine
    audio, only their combination is constructed here.
  - PESQ/STOI/SI-SDR moving the right way is NOT sufficient evidence of an
    improvement in this project - see CLAUDE.md's central finding. This
    script does not run ASR word-scoring (`scripts/asr_score.py`); treat any
    PESQ/STOI win here as a lead to check with that script, not a result.

Usage:
    python scripts/eval_multimic.py
    python scripts/eval_multimic.py --snr-db -6 --duration 10 --out results/multimic.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import audio as A                                    # noqa: E402
from src import metrics as M                                  # noqa: E402
from src.baselines.lms import LMSCanceller                     # noqa: E402
from src.baselines.nlms import NLMSCanceller                   # noqa: E402
from src.baselines.reference_mic import synthesize_reference   # noqa: E402
from src.baselines.rls import RLSCanceller                     # noqa: E402
from src.framing import WIN, HOP                               # noqa: E402
from src.stream_demo import StreamingEnhancer                  # noqa: E402


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _timed_cancel(canceller, primary, reference, update_mask=None):
    """Run one adaptive filter and time it - the computational-cost half of
    the comparison, not just quality. RLS is O(order^2)/sample vs LMS/NLMS's
    O(order); this is what actually measures that difference on real signal
    lengths rather than quoting big-O."""
    t0 = time.perf_counter()
    out = canceller.process_block(primary, reference, update_mask)
    dt = time.perf_counter() - t0
    return out, {"wall_s": dt, "ms_per_sample": 1000.0 * dt / len(primary),
                "order": canceller.order}

SPEECH_WAV = ROOT / "test-result/voice/voice_dry3.wav"
NOISE_WAV = ROOT / "test-result/voice/noise_bed2.wav"

# The streaming enhancer lags its input by exactly WIN - HOP samples (16 ms) -
# the overlap-add buffer withholds a sample until every frame covering it has
# been computed (CLAUDE.md invariant #10). PESQ time-aligns internally and
# hides this; SI-SDR and STOI do not, and comparing unaligned once scored a
# perfectly good model at -32.55 dB SI-SDR. NLMS has no such buffering delay
# (a plain sample-domain filter), so only STFT-based outputs need this shift.
STREAMING_DELAY = WIN - HOP


def _align_to_streaming_output(clean: np.ndarray, out: np.ndarray):
    """`out` at sample n corresponds to `clean` at sample n - STREAMING_DELAY."""
    d = STREAMING_DELAY
    m = min(len(out) - d, len(clean))
    return clean[:m], out[d:d + m]


def build_mixture(duration_s: float, snr_db: float, seed: int = 0):
    """Construct primary + reference channels from real, separately-recorded
    speech and noise. Mirrors `Mixer.build()`'s peak-referenced burst scaling
    and shared final gain (src/mixer.py), so the numbers stay comparable with
    the rest of the project's methodology even though this bypasses `Mixer`
    itself."""
    rng = np.random.default_rng(seed)
    sr = A.SR
    n = int(duration_s * sr)

    speech_full = A.load_audio(SPEECH_WAV)
    offset = int(5.0 * sr)  # skip likely lead-in silence/handling noise
    speech = speech_full[offset:offset + n]
    if len(speech) < n:
        speech = A.fit_length(speech, n, rng, loop=False)

    noise_full = A.load_audio(NOISE_WAV)
    noise = A.fit_length(noise_full, n, rng, loop=True)

    speech_rms = A.active_rms(speech, sr)
    gain = A.scale_burst_for_peak_snr(speech_rms, noise, snr_db)
    noise_scaled = (noise * gain).astype(np.float32)

    primary_raw = speech + noise_scaled
    primary_raw = A.soft_limit(primary_raw, threshold=0.8)
    primary, g = A.peak_normalise(primary_raw, target_peak=0.7)
    primary = np.clip(primary, -1.0, 1.0).astype(np.float32)

    # Same shared gain `g` applied to the clean target, exactly as Mixer.build
    # does - scaling independently would teach/measure a spurious gain offset.
    target = np.clip(speech * g, -1.0, 1.0).astype(np.float32)

    reference_raw = synthesize_reference(speech, noise_scaled)
    reference = np.clip(reference_raw * g, -1.0, 1.0).astype(np.float32)

    return primary, reference, target, {"snr_db": snr_db, "gain": float(g),
                                        "duration_s": duration_s}


def run_neural(onnx_path: Path, x: np.ndarray) -> np.ndarray:
    from src.framing import HOP
    enh = StreamingEnhancer(onnx_path)
    n_chunks = len(x) // HOP
    y = np.zeros(n_chunks * HOP, dtype=np.float32)
    for i in range(n_chunks):
        y[i * HOP:(i + 1) * HOP] = enh.process_chunk(x[i * HOP:(i + 1) * HOP])
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--snr-db", type=float, default=-6.0,
                    help="burst PEAK relative to speech RMS, negative = burst louder (see src/mixer.py)")
    ap.add_argument("--onnx", default="artifacts/model_lowsnr_simple.onnx")
    ap.add_argument("--nlms-order", type=int, default=64)
    ap.add_argument("--nlms-mu", type=float, default=0.1)
    ap.add_argument("--lms-mu", type=float, default=0.01)
    ap.add_argument("--rls-forgetting", type=float, default=0.999)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    onnx_path = ROOT / a.onnx
    primary, reference, target, meta = build_mixture(a.duration, a.snr_db)
    print(f"built {meta['duration_s']:.1f}s mixture at peak SNR {meta['snr_db']} dB "
          f"(speech: {SPEECH_WAV.name}, noise: {NOISE_WAV.name})\n")

    n = min(len(primary), len(target))
    primary, reference, target = primary[:n], reference[:n], target[:n]
    speech_active = A.active_speech_mask(primary, A.SR)

    print("running NLMS (reference-mic)...")
    nlms_out, nlms_cost = _timed_cancel(
        NLMSCanceller(order=a.nlms_order, mu=a.nlms_mu), primary, reference)

    print("running NLMS, VAD-gated adaptation...")
    # Freeze weight updates during detected speech activity on the PRIMARY mic
    # (the only signal a real single-primary-mic system has to gate on). This
    # is what a real headset ANC does to stop the filter learning to cancel
    # the talker along with the noise - see src/baselines/nlms.py docstring.
    nlms_gated_out, _ = _timed_cancel(
        NLMSCanceller(order=a.nlms_order, mu=a.nlms_mu), primary, reference,
        update_mask=~speech_active)

    print("running LMS (fixed step size)...")
    lms_out, lms_cost = _timed_cancel(
        LMSCanceller(order=a.nlms_order, mu=a.lms_mu), primary, reference)

    print("running RLS (recursive least squares)...")
    rls_out, rls_cost = _timed_cancel(
        RLSCanceller(order=a.nlms_order, forgetting=a.rls_forgetting), primary, reference)

    print("running neural (single-mic, shipped model)...")
    neural_out = run_neural(onnx_path, primary)

    print("running NLMS + neural hybrid...")
    hybrid_out = run_neural(onnx_path, nlms_out)

    # (output, needs_delay_alignment) - only outputs that went through the
    # STFT-based streaming enhancer carry the WIN-HOP group delay.
    methods = {
        "unprocessed (primary mic only)": (primary, False),
        "NLMS (reference mic, DSP only)": (nlms_out, False),
        "NLMS, VAD-gated adaptation": (nlms_gated_out, False),
        "LMS (fixed step, DSP only)": (lms_out, False),
        "RLS (DSP only)": (rls_out, False),
        "neural (single mic, shipped)": (neural_out, True),
        "NLMS + neural (hybrid)": (hybrid_out, True),
    }
    compute_cost = {"NLMS (reference mic, DSP only)": nlms_cost,
                    "LMS (fixed step, DSP only)": lms_cost,
                    "RLS (DSP only)": rls_cost}

    ref = M.reference_metrics(target, primary)
    rows = []
    print(f"\n{'method':<32} {'PESQ':>6} {'STOI':>6} {'SI-SDR':>8} {'SI-SDR gain':>12}")
    for name, (out, needs_align) in methods.items():
        if needs_align:
            tgt, out_m = _align_to_streaming_output(target, out)
            noisy_m = primary[:len(tgt)]
        else:
            m = min(len(target), len(out))
            tgt, out_m, noisy_m = target[:m], out[:m], primary[:m]
        r = M.evaluate_pair(tgt, noisy_m, out_m, ref=ref if not needs_align else None)
        if name in compute_cost:
            r["compute"] = compute_cost[name]
        rows.append({"method": name, **r})
        print(f"{name:<32} {r['pesq']:6.3f} {r['stoi']:6.3f} "
              f"{r['si_sdr']:8.3f} {r['si_sdr_gain']:12.3f}")

    print(f"\n{'method':<32} {'ms/sample':>10} {'ms/16ms-frame':>14}")
    for name, cost in compute_cost.items():
        print(f"{name:<32} {cost['ms_per_sample']:10.5f} "
              f"{cost['ms_per_sample'] * HOP:14.4f}")

    print(f"\npesq_available={M.PESQ_AVAILABLE}  stoi_available={M.STOI_AVAILABLE}")
    print("\nNOTE: this is a SYNTHETIC reference-mic construction (see module "
          "docstring). PESQ/STOI/SI-SDR moving the right way is not sufficient "
          "evidence of an intelligibility improvement in this project - run "
          "scripts/asr_score.py on the written-out clips before trusting any "
          "ranking here.")

    if a.out:
        out_path = ROOT / a.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        experiment = {
            "experiment_id": f"multimic_{int(time.time())}",
            "git_commit": _git_commit(),
            "hardware": platform.platform(),
            "python": platform.python_version(),
            "meta": meta,
            "results": rows,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(experiment, f, indent=1)
        print(f"\nwrote {out_path}")

    out_dir = ROOT / "results" / "multimic_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    A.save_audio(out_dir / "primary.wav", primary)
    A.save_audio(out_dir / "reference.wav", reference)
    A.save_audio(out_dir / "target.wav", target)
    A.save_audio(out_dir / "nlms.wav", nlms_out)
    A.save_audio(out_dir / "nlms_gated.wav", nlms_gated_out)
    A.save_audio(out_dir / "lms.wav", lms_out)
    A.save_audio(out_dir / "rls.wav", rls_out)
    A.save_audio(out_dir / "neural.wav", neural_out)
    A.save_audio(out_dir / "hybrid.wav", hybrid_out)
    print(f"wrote audio to {out_dir}")


if __name__ == "__main__":
    main()
