"""Render the 60-second before/after demonstration.

Built from HELD-OUT test material only, and processed through the streaming
ONNX file the hardware team actually receives - one 16 ms frame at a time, caches
carried forward. A demo run through the offline PyTorch path would sound slightly
better than the shipped artefact and would therefore be a misrepresentation.

The clip is deliberately staged so a listener can hear the model work rather than
just take a number on trust: it opens on speech in light background, brings in
vehicle noise, then puts gunfire over the top of continuing speech.

    python scripts/make_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import audio as A                      # noqa: E402
from src.framing import SR                      # noqa: E402

DUR_S = 60.0


def build_speech(rng, speech_recs, n_target: int) -> np.ndarray:
    """Concatenate held-out utterances with natural pauses."""
    out = []
    total = 0
    while total < n_target:
        rec = speech_recs[int(rng.integers(0, len(speech_recs)))]
        x = A.load_audio(rec["path"])
        if len(x) < SR:
            continue
        x = x / (np.max(np.abs(x)) + 1e-9) * 0.6
        gap = np.zeros(int(rng.uniform(0.25, 0.7) * SR), dtype=np.float32)
        out.append(x.astype(np.float32))
        out.append(gap)
        total += len(x) + len(gap)
    return np.concatenate(out)[:n_target]


def main() -> None:
    with open(ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / "manifests" / "manifest.json", encoding="utf-8") as f:
        man = json.load(f)

    rng = np.random.default_rng(20260828)
    n = int(DUR_S * SR)

    speech = build_speech(rng, man["speech"]["test"], n)
    speech_rms = A.active_rms(speech, SR)

    # --- steady background: vehicle/engine, faded in over the first third ----
    eng = [r["path"] for r in man["noise"]["engine"]["test"]]
    bg = A.load_random_window(eng[int(rng.integers(0, len(eng)))], n, rng)
    bg = bg * A.scale_noise_for_snr(speech_rms, bg, 6.0)
    ramp = np.clip(np.linspace(-0.6, 1.6, n), 0.0, 1.0).astype(np.float32)
    bg = bg * ramp

    # --- gunfire: discrete bursts over the back two-thirds -------------------
    gun = man["noise"]["gunshot"]["test"]
    bursts = np.zeros(n, dtype=np.float32)
    marks = []
    t = DUR_S * 0.34
    while t < DUR_S - 1.2:
        rec = gun[int(rng.integers(0, len(gun)))]
        ev = A.load_burst(rec, rng)[: int(0.5 * SR)]
        if ev.size > 16:
            ev = ev * A.scale_burst_for_peak_snr(
                speech_rms, ev, float(rng.uniform(-18.0, -6.0)))
            s = int(t * SR)
            e = min(n, s + ev.size)
            bursts[s:e] += ev[: e - s]
            marks.append(round(t, 2))
        t += float(rng.uniform(2.0, 5.5))

    noisy = speech + bg + bursts
    noisy = A.soft_limit(noisy, 0.85)           # the analog limiter before the ADC
    noisy, g = A.peak_normalise(noisy, 0.89)
    clean_ref = np.clip(speech * g, -1, 1).astype(np.float32)

    # --- enhance through the SHIPPED streaming ONNX --------------------------
    from src.stream_demo import StreamingEnhancer
    from src.framing import HOP

    onnx = ROOT / "artifacts" / "model_simple.onnx"
    if not onnx.exists():
        raise SystemExit(f"{onnx} not found - run src.export_onnx first")
    enh = StreamingEnhancer(onnx)
    chunks = len(noisy) // HOP
    out = np.zeros(chunks * HOP, dtype=np.float32)
    for i in range(chunks):
        out[i * HOP:(i + 1) * HOP] = enh.process_chunk(noisy[i * HOP:(i + 1) * HOP])

    d = ROOT / "results" / "demo60"
    d.mkdir(parents=True, exist_ok=True)
    A.save_audio(d / "before.wav", noisy[:len(out)])
    A.save_audio(d / "after.wav", out)
    A.save_audio(d / "reference_clean.wav", clean_ref[:len(out)])

    # --- measure what the listener is about to hear -------------------------
    # The streaming path lags its input by exactly win - hop (256 samples,
    # 16 ms; measured in src/bench.py). PESQ time-aligns internally and hides
    # this, but SI-SDR and STOI do not: comparing unaligned makes a working
    # model score about -32 dB SI-SDR. Compensate before measuring.
    from src import metrics as M
    from src.framing import WIN
    lag = WIN - HOP
    out_al = out[lag:]
    ref = clean_ref[:len(out_al)]
    out_m = out_al[:len(ref)]
    noisy_m = noisy[:len(ref)]
    res = {
        "alignment_lag_samples": lag,
        "duration_s": round(len(out) / SR, 2),
        "gunshot_times_s": marks,
        "n_gunshots": len(marks),
        "pesq_before": round(M.pesq_wb(ref, noisy_m), 3),
        "pesq_after": round(M.pesq_wb(ref, out_m), 3),
        "stoi_before": round(M.stoi_score(ref, noisy_m), 3),
        "stoi_after": round(M.stoi_score(ref, out_m), 3),
        "si_sdr_before": round(M.si_sdr_db(ref, noisy_m), 2),
        "si_sdr_after": round(M.si_sdr_db(ref, out_m), 2),
    }
    with open(d / "demo_metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)

    print(f"wrote {d}")
    print(f"  duration      : {res['duration_s']} s, {res['n_gunshots']} gunshots")
    print(f"  PESQ          : {res['pesq_before']} -> {res['pesq_after']}")
    print(f"  STOI          : {res['stoi_before']} -> {res['stoi_after']}")
    print(f"  SI-SDR        : {res['si_sdr_before']} -> {res['si_sdr_after']} dB")
    print(f"  gunshots at   : {marks}")


if __name__ == "__main__":
    main()
