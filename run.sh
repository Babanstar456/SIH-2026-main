#!/usr/bin/env bash
# SIH 26052 pipeline driver.
#
#   ./run.sh status      what is downloaded / built so far
#   ./run.sh data        extract + resample + manifests   (safe to re-run)
#   ./run.sh testset     freeze the evaluation set        (refuses to overwrite)
#   ./run.sh baseline    the table that must exist BEFORE training
#   ./run.sh train       fine-tune
#   ./run.sh ablate      same run with the transient term disabled
#   ./run.sh finish      evaluate + bench + export + handoff bundle
#   ./run.sh test        unit tests
#
# Stages are ordered deliberately: `baseline` before `train`, because a result
# with nothing to compare it against is not a result.

set -e

STAGE="${1:-status}"

PY=${SIH_PY:-.venv/bin/python}
RAW="$HOME/SIH26052_data/raw"

cd "$(dirname "${BASH_SOURCE[0]}")"

head() { printf '\n=== %s ===\n' "$1"; }

case "$STAGE" in

  status)
    head "downloads"
    if [ -d "$RAW" ]; then
      for f in "$RAW"/*.gz "$RAW"/*.zip; do
        [ -e "$f" ] || continue
        size_mb=$(( $(stat -c%s "$f" 2>/dev/null || stat -f%z "$f") / 1048576 ))
        if [ -f "$f.done" ]; then done="DONE"; else done="..."; fi
        printf '%-28s %8d MB  %s\n' "$(basename "$f")" "$size_mb" "$done"
      done
    fi
    head "built"
    for p in "manifests/manifest.json" "$HOME/SIH26052_data/testset/index.json" \
             "$HOME/SIH26052_data/voicebank_demand/index.json" \
             "checkpoints/ft_best.pt" "artifacts/model.onnx" "results/results.md"; do
      if [ -f "$p" ]; then yn="yes"; else yn="-"; fi
      printf '%-46s %s\n' "$p" "$yn"
    done
    ;;

  data)
    head "extract + resample";  "$PY" scripts/prepare_data.py --step all
    head "manifests";           "$PY" scripts/build_manifests.py
    head "mixture QA";          "$PY" scripts/qa_mixtures.py --n 20
    ;;

  testset)
    head "frozen defence-noise test set"; "$PY" scripts/make_testset.py --per-category 120
    head "VoiceBank-DEMAND benchmark";    "$PY" scripts/make_vbd_testset.py
    ;;

  baseline)
    head "baselines + pretrained (defence-noise set)"
    "$PY" -m src.evaluate --methods unprocessed wiener specsub gtcrn_dns3 gtcrn_vctk
    head "baselines + pretrained (VoiceBank-DEMAND)"
    "$PY" -m src.evaluate --testset "$HOME/SIH26052_data/voicebank_demand" \
      --methods unprocessed gtcrn_vctk --tag vbd
    ;;

  train)
    head "fine-tune"; "$PY" -m src.train --tag ft --resume
    ;;

  ablate)
    head "ablation: transient weighting disabled"
    "$PY" -m src.train --tag ablation_no_transient --w-transient 0.0 --resume
    ;;

  finish)
    head "evaluate fine-tuned"
    "$PY" -m src.evaluate --methods gtcrn:checkpoints/ft_best.pt \
      gtcrn:checkpoints/ablation_no_transient_best.pt
    head "export streaming ONNX"
    "$PY" -m src.export_onnx --ckpt checkpoints/ft_best.pt --out artifacts/model.onnx
    head "benchmark"
    echo "close other work first - background load inflates these numbers"
    "$PY" -m src.bench --onnx artifacts/model_simple.onnx
    head "handoff bundle"
    "$PY" scripts/make_handoff.py --model artifacts/model_simple.onnx
    ;;

  test)
    head "unit tests"; "$PY" -m pytest tests -q
    ;;

  *)
    echo "unknown stage '$STAGE'. See the header of this file." >&2
    exit 1
    ;;

esac
