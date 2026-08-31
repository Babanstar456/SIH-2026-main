"""QA the mixer: render samples, then CHECK them rather than trusting them.

The plan's Phase-2 verification. Two things can silently ruin training here and
neither raises an exception:

  1. bursts blended into a wash instead of landing as discrete events, and
  2. a transient mask misaligned with the audio it is supposed to mark.

(2) is the nastier one. A mask off by a few frames still trains, still shows a
falling loss, and quietly teaches the model to protect the wrong frames. So this
script measures the energy ratio inside vs outside the mask - if the mask is
right, masked regions must be dramatically louder.

    python scripts/qa_mixtures.py --n 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import audio as A                          # noqa: E402
from src import framing as S                    # noqa: E402
from src.dataset import load_manifest               # noqa: E402
from src.mixer import Mixer                         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="results/qa_mixtures")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    with open(ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    man = load_manifest(ROOT / "manifests" / "manifest.json")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    speech = man["speech"].get(a.split) or man["speech"].get("val") or []
    if not speech:
        raise SystemExit("no speech in manifest")
    background = [r["path"] for r in
                  man.get("noise", {}).get("_background", {}).get(a.split, [])]
    rirs = [r["path"] for r in man.get("rir", {}).get(a.split, [])]
    mixer = Mixer(cfg)
    rng = np.random.default_rng(a.seed)

    cats = {c: man["noise"].get(c, {}).get(a.split, [])
            for c in cfg["categories"]}
    cats = {c: v for c, v in cats.items() if v}
    print(f"speech={len(speech)}  background={len(background)}  rirs={len(rirs)}")
    print(f"categories available: {list(cats)}\n")

    ratios, results, loudest_inside = [], [], []
    for i in range(a.n):
        cat = list(cats)[i % len(cats)]
        impulsive = bool(cfg["categories"][cat].get("impulsive"))
        sp = speech[int(rng.integers(0, len(speech)))]
        wav = A.load_audio(sp["path"])

        bg = [A.load_random_window(p, mixer.n, rng) for p in
              (rng.choice(background, size=min(2, len(background)), replace=False)
               if background else [])]
        bursts = []
        if impulsive:
            for i in rng.choice(len(cats[cat]), size=min(2, len(cats[cat])),
                                replace=False):
                bursts.append(A.load_burst(cats[cat][i], rng))
        else:
            lo, hi = cfg["categories"][cat].get("layers", [1, 1])
            bg.append(A.load_layered([r["path"] for r in cats[cat]], mixer.n,
                                     rng, layers=int(rng.integers(lo, hi + 1))))

        rs = rn = None
        if rirs:
            rs = A.load_audio(rirs[int(rng.integers(0, len(rirs)))])
            rn = A.load_audio(rirs[int(rng.integers(0, len(rirs)))])

        res = mixer.build(rng, wav, bg, bursts, rs, rn, category=cat,
                          force_burst=impulsive)
        if res is None:
            continue

        stem = f"{i:02d}_{cat}"
        A.save_audio(out / f"{stem}_noisy.wav", res.noisy)
        A.save_audio(out / f"{stem}_clean.wav", res.target)

        m = res.transient_mask
        # Burst prominence: the injected transient's peak against the steady
        # background's peak, both measured before they are summed. Comparing
        # inside-vs-outside the mask on the SUMMED residual does not work -
        # MUSAN's noise set contains impulsive content of its own, so the
        # background can out-peak the gunshot somewhere else in the clip and
        # tell you nothing about the burst.
        bp, gp = res.meta["burst_peak"], res.meta["bg_peak"]
        if bp > 0 and gp > 0:
            ratio_db = 20 * np.log10(bp / gp)
            ratios.append(ratio_db)
            loudest_inside.append(bp > gp)
        else:
            ratio_db = float("nan")

        nf = S.n_frames(len(res.noisy))
        fm = S.samples_to_frame_mask(m, nf)
        results.append((stem, res.meta["n_bursts"], float(m.mean()),
                        float(fm.mean()), ratio_db, res.meta["bg_snr_db"]))

    print(f"{'clip':22s} {'bursts':>6s} {'mask%':>6s} {'frm%':>6s} "
          f"{'in/out dB':>10s} {'bgSNR':>6s}")
    for r in results:
        rd = "   n/a" if np.isnan(r[4]) else f"{r[4]:+9.1f}"
        print(f"{r[0]:22s} {r[1]:6d} {r[2]*100:5.1f}% {r[3]*100:5.1f}% "
              f"{rd} {r[5]:6.1f}")

    print(f"\nrendered {len(results)} mixtures -> {out}")
    # Frame-mask coverage must track the sample mask; a gross mismatch means
    # samples_to_frame_mask is broken (it is also unit-tested directly).
    cov = [(r[2], r[3]) for r in results if r[2] > 0]
    if cov:
        drift = max(abs(f - s) for s, f in cov)
        print(f"\nframe-mask coverage: max |frame% - sample%| = {100*drift:.1f} pp"
              f"  {'OK' if drift < 0.05 else '!! sample->frame mapping looks wrong'}")

    if ratios:
        med = float(np.median(ratios))
        frac = float(np.mean(loudest_inside)) if loudest_inside else 0.0
        print(f"\nburst prominence, over {len(ratios)} clips with bursts:")
        print(f"  median burst peak / background peak : {med:+.1f} dB")
        print(f"  burst out-peaks background          : {100*frac:.0f}% of clips")
        if not background:
            print("  DEGENERATE: no background pool loaded - re-run once MUSAN is "
                  "available.")
        elif med < 0.0:
            print("  !! FAIL: gunfire is typically quieter than the background.\n"
                  "     That inverts the problem - a stationary filter would\n"
                  "     handle it. Raise bursts.peak_snr_db magnitude.")
        else:
            print("  OK: injected transients dominate the steady background,\n"
                  "      which is the situation the model has to solve.")
    else:
        print("\n!! no impulsive clips produced - check burst pools")

    if not background or not rirs:
        missing = [n for n, v in (("MUSAN background", background), ("RIRs", rirs)) if not v]
        print(f"\nNOTE: {', '.join(missing)} not yet available - these mixtures are\n"
              f"      not representative of final training data.")


if __name__ == "__main__":
    main()
