# SIH 26052 pipeline driver.
#
#   .\run.ps1 status      what is downloaded / built so far
#   .\run.ps1 data        extract + resample + manifests   (safe to re-run)
#   .\run.ps1 testset     freeze the evaluation set        (refuses to overwrite)
#   .\run.ps1 baseline    the table that must exist BEFORE training
#   .\run.ps1 train       fine-tune
#   .\run.ps1 ablate      same run with the transient term disabled
#   .\run.ps1 finish      evaluate + bench + export + handoff bundle
#   .\run.ps1 test        unit tests
#
# Stages are ordered deliberately: `baseline` before `train`, because a result
# with nothing to compare it against is not a result.

param([Parameter(Position = 0)][string]$Stage = "status")

$PY = "C:\SIH26052_data\.venv\Scripts\python.exe"
$RAW = "C:\SIH26052_data\raw"
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Head($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

switch ($Stage) {

  "status" {
    Head "downloads"
    Get-ChildItem $RAW -File -Include *.gz, *.zip -ErrorAction SilentlyContinue | ForEach-Object {
      $done = Test-Path "$($_.FullName).done"
      "{0,-28} {1,8:N0} MB  {2}" -f $_.Name, ($_.Length / 1MB), $(if ($done) { "DONE" } else { "..." })
    }
    Head "built"
    foreach ($p in @("manifests\manifest.json", "C:\SIH26052_data\testset\index.json",
        "C:\SIH26052_data\voicebank_demand\index.json",
        "checkpoints\ft_best.pt", "artifacts\model.onnx", "results\results.md")) {
      "{0,-46} {1}" -f $p, $(if (Test-Path $p) { "yes" } else { "-" })
    }
  }

  "data" {
    Head "extract + resample";  & $PY scripts\prepare_data.py --step all
    Head "manifests";           & $PY scripts\build_manifests.py
    Head "mixture QA";          & $PY scripts\qa_mixtures.py --n 20
  }

  "testset" {
    Head "frozen defence-noise test set"; & $PY scripts\make_testset.py --per-category 120
    Head "VoiceBank-DEMAND benchmark";    & $PY scripts\make_vbd_testset.py
  }

  "baseline" {
    Head "baselines + pretrained (defence-noise set)"
    & $PY -m src.evaluate --methods unprocessed wiener specsub gtcrn_dns3 gtcrn_vctk
    Head "baselines + pretrained (VoiceBank-DEMAND)"
    & $PY -m src.evaluate --testset C:\SIH26052_data\voicebank_demand `
      --methods unprocessed gtcrn_vctk --tag vbd
  }

  "train"  { Head "fine-tune"; & $PY -m src.train --tag ft --resume }

  "ablate" {
    Head "ablation: transient weighting disabled"
    & $PY -m src.train --tag ablation_no_transient --w-transient 0.0 --resume
  }

  "finish" {
    Head "evaluate fine-tuned"
    & $PY -m src.evaluate --methods gtcrn:checkpoints/ft_best.pt `
      gtcrn:checkpoints/ablation_no_transient_best.pt
    Head "export streaming ONNX"
    & $PY -m src.export_onnx --ckpt checkpoints/ft_best.pt --out artifacts/model.onnx
    Head "benchmark"
    Write-Host "close other work first - background load inflates these numbers" -ForegroundColor Yellow
    & $PY -m src.bench --onnx artifacts/model_simple.onnx
    Head "handoff bundle"
    & $PY scripts\make_handoff.py --model artifacts/model_simple.onnx
  }

  "test" { Head "unit tests"; & $PY -m pytest tests -q }

  default { Write-Host "unknown stage '$Stage'. See the header of this file." -ForegroundColor Red }
}
