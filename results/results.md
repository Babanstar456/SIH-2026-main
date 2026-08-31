# Results

- generated: 2026-08-28T16:33:03+00:00
- git: `not-a-repo`
- test set: `C:\SIH26052_data\testset` (frozen, seed 20260827)
- device: `cpu`
- PESQ available: True
- clips: 720 per method

Targets: SNR gain > 15.0 dB, STOI > 0.85, PESQ > 2.5, RTF < 0.5.

## Burst-local performance (impulsive categories)

SI-SDR gain measured INSIDE the gunfire/explosion bursts only. A whole-clip score is diluted by the ~88% of each clip containing no transient, so a model that removes none of the gunfire can still look acceptable overall. This column is the one that answers the question.

| category | method | burst SI-SDR gain | whole-clip SI-SDR gain | burst % of clip |
|---|---|---|---|---|
| artillery | `gtcrn:checkpoints/ablation_no_transient_best.pt` | **+7.87** | +8.05 | 20.3% |
| artillery | `gtcrn:checkpoints/ft_best.pt` | **+7.83** | +7.85 | 20.3% |
| artillery | `gtcrn_dns3` | **+7.03** | +7.27 | 20.3% |
| artillery | `wiener` | **+0.16** | +0.32 | 20.3% |
| artillery | `specsub` | **+0.09** | +0.20 | 20.3% |
| artillery | `unprocessed` | **+0.00** | +0.00 | 20.3% |
| artillery | `gtcrn_vctk` | **-0.12** | +0.16 | 20.3% |
| gunshot | `gtcrn:checkpoints/ft_best.pt` | **+7.21** | +6.71 | 18.7% |
| gunshot | `gtcrn:checkpoints/ablation_no_transient_best.pt` | **+7.16** | +6.98 | 18.7% |
| gunshot | `gtcrn_dns3` | **+5.84** | +5.74 | 18.7% |
| gunshot | `wiener` | **+0.19** | +0.48 | 18.7% |
| gunshot | `gtcrn_vctk` | **+0.18** | -0.19 | 18.7% |
| gunshot | `specsub` | **+0.11** | +0.25 | 18.7% |
| gunshot | `unprocessed` | **+0.00** | +0.00 | 18.7% |

### artillery

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn:checkpoints/ablation_no_transient_best.pt` | 1.722 | 0.835 | +8.05 | +8.77 | 0.0659 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `gtcrn:checkpoints/ft_best.pt` | 1.721 | 0.832 | +7.85 | +8.57 | 0.0514 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `gtcrn_dns3` | 1.628 | 0.832 | +7.27 | +8.12 | 0.0477 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `gtcrn_vctk` | 1.210 | 0.666 | +0.16 | +3.70 | 0.0530 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `specsub` | 1.197 | 0.766 | +0.20 | +0.27 | 0.0028 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `unprocessed` | 1.188 | 0.765 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `wiener` | 1.205 | 0.764 | +0.32 | +0.53 | 0.0041 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |

### babble

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn:checkpoints/ablation_no_transient_best.pt` | 1.833 | 0.827 | +4.55 | +5.47 | 0.0474 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `gtcrn:checkpoints/ft_best.pt` | 1.802 | 0.822 | +4.19 | +5.17 | 0.0511 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `gtcrn_dns3` | 1.702 | 0.818 | +3.82 | +4.79 | 0.0575 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `gtcrn_vctk` | 1.320 | 0.702 | -1.79 | +0.82 | 0.0625 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `specsub` | 1.352 | 0.786 | +0.46 | +0.54 | 0.0028 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `unprocessed` | 1.333 | 0.785 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `wiener` | 1.374 | 0.783 | +0.72 | +0.94 | 0.0040 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |

### engine

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn:checkpoints/ablation_no_transient_best.pt` | 2.066 | 0.873 | +5.39 | +5.81 | 0.0595 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `gtcrn:checkpoints/ft_best.pt` | 2.018 | 0.868 | +5.07 | +5.50 | 0.0501 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `gtcrn_dns3` | 1.967 | 0.870 | +4.83 | +5.34 | 0.0486 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `gtcrn_vctk` | 1.357 | 0.723 | -1.45 | +0.58 | 0.0472 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `specsub` | 1.395 | 0.818 | +0.62 | +0.67 | 0.0028 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `unprocessed` | 1.380 | 0.816 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `wiener` | 1.422 | 0.819 | +1.09 | +1.29 | 0.0042 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |

### gunshot

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn:checkpoints/ablation_no_transient_best.pt` | 1.901 | 0.854 | +6.98 | +7.59 | 0.0639 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `gtcrn:checkpoints/ft_best.pt` | 1.898 | 0.850 | +6.71 | +7.35 | 0.0530 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `gtcrn_dns3` | 1.793 | 0.849 | +5.74 | +6.44 | 0.0499 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `gtcrn_vctk` | 1.311 | 0.697 | -0.19 | +2.72 | 0.0493 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `specsub` | 1.303 | 0.783 | +0.25 | +0.31 | 0.0029 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `unprocessed` | 1.297 | 0.782 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `wiener` | 1.316 | 0.781 | +0.48 | +0.66 | 0.0043 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |

### rotor

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn:checkpoints/ablation_no_transient_best.pt` | 1.995 | 0.875 | +4.76 | +5.12 | 0.1120 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `gtcrn:checkpoints/ft_best.pt` | 1.955 | 0.870 | +4.49 | +4.83 | 0.0494 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `gtcrn_dns3` | 1.868 | 0.870 | +4.02 | +4.46 | 0.0480 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `gtcrn_vctk` | 1.354 | 0.757 | -1.31 | +0.26 | 0.0484 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `specsub` | 1.372 | 0.825 | +0.38 | +0.43 | 0.0028 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `unprocessed` | 1.354 | 0.823 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `wiener` | 1.404 | 0.824 | +0.74 | +0.90 | 0.0042 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |

### siren

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn:checkpoints/ablation_no_transient_best.pt` | 2.080 | 0.892 | +5.26 | +5.51 | 0.0483 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `gtcrn:checkpoints/ft_best.pt` | 2.042 | 0.886 | +4.83 | +5.11 | 0.0505 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `gtcrn_dns3` | 1.934 | 0.885 | +4.46 | +4.82 | 1.0748 | PESQFAIL STOIPASS SNRgFAIL RTFFAIL |
| `gtcrn_vctk` | 1.351 | 0.757 | -1.98 | -0.27 | 0.0628 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `specsub` | 1.375 | 0.838 | +0.30 | +0.35 | 0.0028 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `unprocessed` | 1.364 | 0.837 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `wiener` | 1.408 | 0.837 | +0.58 | +0.73 | 0.0042 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |

## SNR gain by INPUT SNR (dB)

The >15 dB target is only reachable in the low-input-SNR regime; at high input SNR there is little noise left to remove.

| method | <0 dB | 0-5 dB | 5-10 dB | 10-15 dB | >15 dB |
|---|---|---|---|---|---|
| `gtcrn:checkpoints/ablation_no_transient_best.pt` | +10.08 (n=170) | +6.92 (n=232) | +4.81 (n=169) | +4.04 (n=95) | +1.44 (n=54) |
| `gtcrn:checkpoints/ft_best.pt` | +9.93 (n=170) | +6.72 (n=232) | +4.43 (n=169) | +3.58 (n=95) | +0.83 (n=54) |
| `gtcrn_dns3` | +8.66 (n=170) | +6.42 (n=232) | +4.54 (n=169) | +3.37 (n=95) | +0.49 (n=54) |
| `gtcrn_vctk` | +6.27 (n=170) | +2.50 (n=232) | -0.28 (n=169) | -3.08 (n=95) | -6.80 (n=54) |
| `specsub` | +0.35 (n=170) | +0.56 (n=232) | +0.38 (n=169) | +0.39 (n=95) | +0.33 (n=54) |
| `unprocessed` | +0.00 (n=170) | +0.00 (n=232) | +0.00 (n=169) | +0.00 (n=95) | +0.00 (n=54) |
| `wiener` | +0.76 (n=170) | +1.14 (n=232) | +0.72 (n=169) | +0.74 (n=95) | +0.35 (n=54) |