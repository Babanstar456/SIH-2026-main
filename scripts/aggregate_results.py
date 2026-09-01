"""Consolidate every method measured on the SAME clip into one table + plot.

HONESTY NOTE: this deliberately does NOT mix these numbers with the frozen
720-clip defence testset (`results/results.csv` / `results/results.md`).
Those are the statistically robust, per-category numbers for the methods
that existed before this session (GTCRN variants, Wiener, spectral
subtraction) and remain the primary evidence for this project - see them for
that comparison. This script instead answers a narrower, newer question:
"on one identical piece of audio, how do ALL the methods this session added
(NLMS/LMS/RLS, RNNoise, DeepFilterNet) compare against the pre-existing
ones?" - which only makes sense measured on the same clip, since NLMS/LMS/RLS
have never been run through the 720-clip testset (they need a second
reference-mic channel the testset doesn't have).

The clip: `results/multimic_demo/{target,primary}.wav` - a SYNTHETIC 45 s
mixture (real dry speech + real gunfire, see scripts/eval_multimic.py), one
clip, not a statistically powered set. Read every number here as a single
data point, not a category average.

Reads results/multimic.json, results/baseline_comparison.json (the
"multimic" pair only - the "voice3" pair is flagged unreliable, misaligned
audio, and excluded here), and results/classical_multimic.json. Writes
results/method_comparison.md and results/pareto_latency_quality.png.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    p = ROOT / "results" / name
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    rows = {}  # method -> {pesq, stoi, si_sdr_gain, ms_per_16ms_frame}

    multimic = _load("multimic.json")
    for r in multimic.get("results", []):
        rows[r["method"]] = {
            "pesq": r["pesq"], "stoi": r["stoi"], "si_sdr_gain": r["si_sdr_gain"],
            "ms_per_16ms_frame": r.get("compute", {}).get("ms_per_sample", None),
        }
        if rows[r["method"]]["ms_per_16ms_frame"] is not None:
            rows[r["method"]]["ms_per_16ms_frame"] *= 256  # HOP samples

    baseline_comp = _load("baseline_comparison.json")
    for r in baseline_comp:
        if r.get("pair") != "multimic" or "metrics" not in r:
            continue
        m = r["metrics"]
        name = {"unprocessed": "unprocessed (primary mic only)",
               "gtcrn:checkpoints/shipped_best.pt": "neural (single mic, shipped)"}.get(
            r["method"], r["method"])
        if name not in rows:
            rows[name] = {"pesq": m["pesq"], "stoi": m["stoi"],
                         "si_sdr_gain": m["si_sdr_gain"], "ms_per_16ms_frame": None}

    classical = _load("classical_multimic.json")
    for r in classical:
        m = r["metrics"]
        rows[r["method"]] = {"pesq": m["pesq"], "stoi": m["stoi"],
                             "si_sdr_gain": m["si_sdr_gain"], "ms_per_16ms_frame": None}

    asr = _load("asr_multimic.json")
    asr_by_file = {
        "primary.wav": "unprocessed (primary mic only)",
        "nlms.wav": "NLMS (reference mic, DSP only)",
        "nlms_gated.wav": "NLMS, VAD-gated adaptation",
        "lms.wav": "LMS (fixed step, DSP only)",
        "rls.wav": "RLS (DSP only)",
        "neural.wav": "neural (single mic, shipped)",
        "hybrid.wav": "NLMS + neural (hybrid)",
    }
    for fname, method_name in asr_by_file.items():
        pct = asr.get("scores", {}).get(fname, {}).get("pct")
        if method_name in rows and pct is not None:
            rows[method_name]["asr_pct"] = pct

    order = ["unprocessed (primary mic only)", "wiener", "specsub",
            "NLMS (reference mic, DSP only)", "NLMS, VAD-gated adaptation",
            "LMS (fixed step, DSP only)", "RLS (DSP only)",
            "rnnoise", "deepfilternet", "neural (single mic, shipped)",
            "NLMS + neural (hybrid)"]
    ordered_rows = [(name, rows[name]) for name in order if name in rows]
    ordered_rows += [(n, v) for n, v in rows.items() if n not in order]

    lines = [
        "# Consolidated method comparison — one synthetic clip",
        "",
        "**Read `scripts/aggregate_results.py`'s docstring before trusting this "
        "table** — every method here ran on the SAME 45 s synthetic clip "
        "(`results/multimic_demo/`), not the frozen 720-clip defence testset. "
        "This is a single data point per method, not a statistically powered "
        "result. The pre-existing per-category testset numbers in "
        "`results/results.md` remain the primary evidence for GTCRN/Wiener/"
        "spectral-subtraction.",
        "",
        "PESQ/STOI/SI-SDR moving the right way is not sufficient evidence of an "
        "intelligibility improvement in this project (see CLAUDE.md). The **ASR "
        "word-recognition column** (whisper-medium, 26 known tokens, "
        "`results/asr_multimic.json`) is the trusted measure here — see how often "
        "it disagrees with PESQ/STOI's ranking below.",
        "",
        "| method | PESQ | STOI | SI-SDR gain (dB) | compute (ms/16ms frame) | ASR word score |",
        "|---|---|---|---|---|---|",
    ]
    for name, v in ordered_rows:
        compute = f"{v['ms_per_16ms_frame']:.3f}" if v["ms_per_16ms_frame"] else "—"
        asr_pct = f"{v['asr_pct']}%" if "asr_pct" in v else "—"
        lines.append(f"| {name} | {v['pesq']:.3f} | {v['stoi']:.3f} | "
                     f"{v['si_sdr_gain']:+.2f} | {compute} | {asr_pct} |")

    out_md = ROOT / "results" / "method_comparison.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, v in ordered_rows:
            x = v["ms_per_16ms_frame"] if v["ms_per_16ms_frame"] is not None else 0.01
            ax.scatter(x, v["pesq"], s=60)
            ax.annotate(name, (x, v["pesq"]), fontsize=7,
                       xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("compute cost (ms per 16ms frame, log scale; DSP-only "
                      "methods measured, GTCRN/RNNoise/DeepFilterNet/classical "
                      "not timed on this run and shown near the axis)")
        ax.set_ylabel("PESQ (higher is better)")
        ax.set_title("Latency/compute vs PESQ — one synthetic 45s clip, not the "
                     "frozen testset")
        ax.axhline(2.5, color="gray", linestyle="--", linewidth=0.8)
        ax.annotate("PESQ > 2.5 target", (ax.get_xlim()[0], 2.5), fontsize=7,
                   color="gray")
        fig.tight_layout()
        out_png = ROOT / "results" / "pareto_latency_quality.png"
        fig.savefig(out_png, dpi=150)
        print(f"wrote {out_png}")
    except ImportError:
        print("matplotlib not available, skipped the plot")


if __name__ == "__main__":
    main()
