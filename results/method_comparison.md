# Consolidated method comparison — one synthetic clip

**Read `scripts/aggregate_results.py`'s docstring before trusting this table** — every method here ran on the SAME 45 s synthetic clip (`results/multimic_demo/`), not the frozen 720-clip defence testset. This is a single data point per method, not a statistically powered result. The pre-existing per-category testset numbers in `results/results.md` remain the primary evidence for GTCRN/Wiener/spectral-subtraction.

PESQ/STOI/SI-SDR moving the right way is not sufficient evidence of an intelligibility improvement in this project (see CLAUDE.md). The **ASR word-recognition column** (whisper-medium, 26 known tokens, `results/asr_multimic.json`) is the trusted measure here — see how often it disagrees with PESQ/STOI's ranking below.

| method | PESQ | STOI | SI-SDR gain (dB) | compute (ms/16ms frame) | ASR word score |
|---|---|---|---|---|---|
| unprocessed (primary mic only) | 1.333 | 0.794 | +0.00 | — | 62% |
| wiener | 1.246 | 0.762 | -0.61 | — | — |
| specsub | 1.301 | 0.791 | -0.03 | — | — |
| NLMS (reference mic, DSP only) | 1.196 | 0.658 | -9.98 | 1.431 | 54% |
| NLMS, VAD-gated adaptation | 1.333 | 0.794 | +0.00 | — | 62% |
| LMS (fixed step, DSP only) | 1.322 | 0.743 | -3.58 | 1.157 | 62% |
| RLS (DSP only) | 1.019 | 0.048 | -71.66 | 5.564 | 0% |
| rnnoise | 1.038 | 0.272 | -32.77 | — | — |
| deepfilternet | 1.049 | 0.480 | -9.38 | — | — |
| neural (single mic, shipped) | 1.100 | 0.539 | -7.99 | — | 54% |
| NLMS + neural (hybrid) | 1.101 | 0.444 | -14.06 | — | 4% |
