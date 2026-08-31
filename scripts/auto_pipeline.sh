#!/usr/bin/env bash
# Wait for MUSAN, then run the remaining pipeline through to a shippable
# deliverable, unattended.
#
# Ordering decisions that matter:
#   * the baseline table is produced BEFORE training - a trained result with
#     nothing to compare it against is not a result;
#   * export/bench/handoff/report run BEFORE the ablation, so a complete,
#     shippable bundle exists even if the ablation is interrupted;
#   * nothing else is downloaded during the run - background traffic inflates
#     the RTF benchmark several-fold (see CLAUDE.md invariant 10).
#
# Every stage skips work already done, so re-running this is the normal way to
# continue. A failing stage aborts the chain rather than letting later stages
# run on bad inputs and produce plausible-looking wrong numbers.
#
# Progress: tail -f /c/SIH26052_data/auto_pipeline.log

set -o pipefail
PY="C:/SIH26052_data/.venv/Scripts/python.exe"
PROJ="/c/Users/Ayushi Kundu/OneDrive/Desktop/SIH_2026"
LOG=/c/SIH26052_data/auto_pipeline.log
cd "$PROJ" || exit 1

log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

stage () {           # stage <name> <cmd...>
  local name="$1"; shift
  log "── $name"
  if "$@" >>"$LOG" 2>&1; then
    log "   ok: $name"
  else
    local rc=$?
    log "   FAILED: $name (exit $rc) - chain stopped, see $LOG"
    exit 1
  fi
}

# ---------------------------------------------------------------- wait
if [ ! -f /c/SIH26052_data/raw/musan.tar.gz.done ]; then
  log "waiting for musan.tar.gz ..."
  for i in $(seq 1 720); do        # up to 6 h
    [ -f /c/SIH26052_data/raw/musan.tar.gz.done ] && break
    sleep 30
  done
fi
if [ ! -f /c/SIH26052_data/raw/musan.tar.gz.done ]; then
  log "musan never completed - aborting"; exit 1
fi
log "musan present ($(du -h /c/SIH26052_data/raw/musan.tar.gz | cut -f1))"

# ---------------------------------------------------------------- data
stage "extract + resample"        "$PY" scripts/prepare_data.py --step all
stage "manifests + disjointness"  "$PY" scripts/build_manifests.py
stage "mixture QA"                "$PY" scripts/qa_mixtures.py --n 24
stage "freeze test set"           "$PY" scripts/make_testset.py --per-category 120

# ------------------------------------------------------------ baselines
stage "baseline table" "$PY" -m src.evaluate \
      --methods unprocessed wiener specsub gtcrn_dns3 gtcrn_vctk

# -------------------------------------------------------------- training
stage "fine-tune (transient-weighted)" "$PY" -m src.train --tag ft --resume
stage "evaluate fine-tuned" "$PY" -m src.evaluate \
      --methods gtcrn:checkpoints/ft_best.pt

# ------------------------------------------------- deliverable (complete here)
stage "export streaming ONNX" "$PY" -m src.export_onnx \
      --ckpt checkpoints/ft_best.pt --out artifacts/model.onnx
stage "benchmark (machine is idle at this point)" "$PY" -m src.bench \
      --onnx artifacts/model_simple.onnx
stage "handoff bundle" "$PY" scripts/make_handoff.py --model artifacts/model_simple.onnx
stage "report" "$PY" scripts/make_report.py
log "=== SHIPPABLE: trained model, results, ONNX, spec sheet, report ==="

# -------------------------------------------------------------- ablation
# Proves the transient term earns its place: identical run, w_transient = 0,
# which recovers GTCRN's upstream objective exactly.
stage "ablation (transient term disabled)" "$PY" -m src.train \
      --tag ablation_no_transient --w-transient 0.0 --resume
stage "evaluate ablation" "$PY" -m src.evaluate \
      --methods gtcrn:checkpoints/ablation_no_transient_best.pt
stage "regenerate report with ablation" "$PY" scripts/make_report.py

log "=== pipeline complete ==="
