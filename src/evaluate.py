"""Evaluation harness.

The roadmap doc puts this before any training, and is right to:

    Until this exists, no later number means anything.

Sweeps one or more methods over the frozen test set and writes BOTH a per-clip
CSV and a per-(method, category) aggregate. Per-category is the point: an
overall mean can look perfectly healthy while gunshots quietly fail, and that is
the one case this project cannot afford to miss.

    python -m src.evaluate --methods unprocessed wiener specsub gtcrn_dns3
    python -m src.evaluate --methods gtcrn:checkpoints/ft_best.pt --tag ft
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import audio as A          # noqa: E402
from src import methods as MET      # noqa: E402
from src import metrics as M        # noqa: E402

# Targets from the problem statement, carried here so the table can mark
# pass/fail directly rather than leaving the reader to compare by eye.
TARGETS = {"snr_gain": 15.0, "stoi": 0.85, "pesq": 2.5, "rtf": 0.5}


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or "not-a-repo"
    except Exception:  # noqa: BLE001
        return "unknown"


def evaluate_method(name: str, testset: Path, device: str = "cpu",
                    limit: int | None = None,
                    ref_cache: dict | None = None) -> pd.DataFrame:
    with open(testset / "index.json", encoding="utf-8") as f:
        index = json.load(f)
    items = index["items"][:limit] if limit else index["items"]
    fn = MET.get(name, device=device)
    sr = index["sr"]

    rows = []
    for it in tqdm(items, desc=f"{name:22s}", ncols=88, unit="clip"):
        noisy = A.load_audio(testset / it["noisy"], sr)
        clean = A.load_audio(testset / it["clean"], sr)

        t0 = time.perf_counter()
        enh = fn(noisy, sr)
        dt = time.perf_counter() - t0

        # Unprocessed-clip metrics depend only on the clip, so they are shared
        # across every method in the sweep rather than recomputed each time.
        if ref_cache is None:
            ref = None
        elif it["id"] in ref_cache:
            ref = ref_cache[it["id"]]
        else:
            ref = M.reference_metrics(clean, noisy, sr)
            ref_cache[it["id"]] = ref

        r = M.evaluate_pair(clean, noisy, enh, sr, ref=ref)

        # Burst-local metrics: the whole-clip score is diluted by the ~88% of a
        # clip that contains no gunfire, so it can look healthy while the
        # gunshots themselves are untouched.
        if it.get("mask"):
            mpath = testset / it["mask"]
            if mpath.exists():
                r.update(M.masked_metrics(clean, noisy, enh,
                                          np.load(mpath)["mask"]))

        r.update({
            "method": name, "id": it["id"], "category": it["category"],
            "speaker": it.get("speaker"), "bg_snr_db": it.get("bg_snr_db"),
            "n_events": it.get("n_events", 0), "reverb": it.get("reverb"),
            "proc_s": dt, "rtf": dt / (len(noisy) / sr),
        })
        rows.append(r)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["pesq", "stoi", "estoi", "si_sdr", "snr", "segsnr",
            "snr_gain", "si_sdr_gain", "pesq_gain", "stoi_gain", "rtf"]
    cols += [c for c in ("burst_si_sdr_gain", "burst_snr_gain", "burst_frac")
             if c in df.columns]
    g = (df.groupby(["method", "category"])[cols]
           .agg(["mean", "std", "count"]))
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def _fmt_table(agg: pd.DataFrame) -> str:
    """Markdown table, per category, with pass/fail against the PS targets."""
    has_burst = "burst_si_sdr_gain_mean" in agg.columns
    lines = []
    if has_burst:
        b = agg[agg["burst_si_sdr_gain_count"] > 0]
        if len(b):
            lines.append("\n## Burst-local performance (impulsive categories)\n")
            lines.append("SI-SDR gain measured INSIDE the gunfire/explosion bursts "
                         "only. A whole-clip score is diluted by the ~88% of each "
                         "clip containing no transient, so a model that removes "
                         "none of the gunfire can still look acceptable overall. "
                         "This column is the one that answers the question.\n")
            lines.append("| category | method | burst SI-SDR gain | whole-clip SI-SDR gain | burst % of clip |")
            lines.append("|---|---|---|---|---|")
            for cat in sorted(b["category"].unique()):
                for _, r in b[b["category"] == cat].sort_values(
                        "burst_si_sdr_gain_mean", ascending=False).iterrows():
                    lines.append(
                        f"| {cat} | `{r['method']}` | "
                        f"**{r['burst_si_sdr_gain_mean']:+.2f}** | "
                        f"{r['si_sdr_gain_mean']:+.2f} | "
                        f"{100*r['burst_frac_mean']:.1f}% |")

    for cat in sorted(agg["category"].unique()):
        sub = agg[agg["category"] == cat]
        lines.append(f"\n### {cat}\n")
        lines.append("| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            ok = []
            if not np.isnan(r["pesq_mean"]):
                ok.append(("PESQ", r["pesq_mean"] >= TARGETS["pesq"]))
            ok.append(("STOI", r["stoi_mean"] >= TARGETS["stoi"]))
            ok.append(("SNRg", r["snr_gain_mean"] >= TARGETS["snr_gain"]))
            ok.append(("RTF", r["rtf_mean"] <= TARGETS["rtf"]))
            flag = " ".join(f"{k}{'PASS' if v else 'FAIL'}" for k, v in ok)
            lines.append(
                f"| `{r['method']}` | {r['pesq_mean']:.3f} | {r['stoi_mean']:.3f} "
                f"| {r['si_sdr_gain_mean']:+.2f} | {r['snr_gain_mean']:+.2f} "
                f"| {r['rtf_mean']:.4f} | {flag} |")
    return "\n".join(lines)


SNR_BINS = [-np.inf, 0, 5, 10, 15, np.inf]
SNR_LABELS = ["<0", "0-5", "5-10", "10-15", ">15"]


def _fmt_snr_table(df: pd.DataFrame) -> str:
    """SNR gain stratified by INPUT SNR.

    A single averaged SNR-gain number cannot be compared against the 15 dB
    target, because the achievable gain is bounded by how much noise was there
    to begin with - measured correlation between input SNR and gain is about
    -0.78. Averaging over a wide input-SNR range therefore either buries a good
    result or flatters a poor one, depending only on how the test set was drawn.
    """
    d = df.copy()
    d["in_snr"] = pd.cut(d["snr_noisy"], SNR_BINS, labels=SNR_LABELS)
    piv = d.pivot_table(index="method", columns="in_snr",
                        values="snr_gain", aggfunc="mean", observed=True)
    cnt = d.pivot_table(index="method", columns="in_snr",
                        values="snr_gain", aggfunc="count", observed=True)
    lines = ["\n## SNR gain by INPUT SNR (dB)\n",
             "The >15 dB target is only reachable in the low-input-SNR regime; "
             "at high input SNR there is little noise left to remove.\n",
             "| method | " + " | ".join(f"{c} dB" for c in piv.columns) + " |",
             "|---" * (len(piv.columns) + 1) + "|"]
    for m in piv.index:
        cells = []
        for c in piv.columns:
            v, n = piv.loc[m, c], cnt.loc[m, c]
            cells.append("-" if pd.isna(v) else f"{v:+.2f} (n={int(n)})")
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--device", default="cpu",
                    help="cpu is the honest setting: the target is an embedded chip")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--testset", default=None,
                    help="override the frozen test set (e.g. the "
                         "VoiceBank-DEMAND benchmark directory)")
    a = ap.parse_args()

    with open(ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    testset = Path(a.testset) if a.testset else Path(cfg["paths"]["testset"])
    if not (testset / "index.json").exists():
        raise SystemExit(f"no frozen test set at {testset} - "
                         "run scripts/make_testset.py first")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    frames, ref_cache = [], {}
    for name in a.methods:
        frames.append(evaluate_method(name, testset, a.device, a.limit, ref_cache))
    df = pd.concat(frames, ignore_index=True)

    if not M.PESQ_AVAILABLE:
        print("\n!! PESQ unavailable - those columns are NaN, not zero !!\n")

    suffix = f"_{a.tag}" if a.tag else ""
    per_clip = results_dir / f"per_clip{suffix}.csv"
    agg_csv = results_dir / f"results{suffix}.csv"
    md = results_dir / f"results{suffix}.md"

    # Append rather than clobber, so a later run of one method does not erase
    # earlier methods' measurements.
    if per_clip.exists():
        old = pd.read_csv(per_clip)
        old = old[~old["method"].isin(df["method"].unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(per_clip, index=False)

    agg = aggregate(df)
    agg.to_csv(agg_csv, index=False)

    header = (
        f"# Results{(' - ' + a.tag) if a.tag else ''}\n\n"
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"- git: `{_git_rev()}`\n"
        f"- test set: `{testset}` (frozen, seed "
        f"{json.load(open(testset / 'index.json', encoding='utf-8'))['seed']})\n"
        f"- device: `{a.device}`\n"
        f"- PESQ available: {M.PESQ_AVAILABLE}\n"
        f"- clips: {len(df) // max(df['method'].nunique(), 1)} per method\n\n"
        f"Targets: SNR gain > {TARGETS['snr_gain']} dB, STOI > {TARGETS['stoi']}, "
        f"PESQ > {TARGETS['pesq']}, RTF < {TARGETS['rtf']}.\n"
    )
    md.write_text(header + _fmt_table(agg) + "\n" + _fmt_snr_table(df),
                  encoding="utf-8")

    print(f"\nper-clip -> {per_clip}\naggregate -> {agg_csv}\ntable -> {md}")
    print(_fmt_table(agg))
    print(_fmt_snr_table(df))


if __name__ == "__main__":
    main()
