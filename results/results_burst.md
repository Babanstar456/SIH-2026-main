# Results - burst

- generated: 2026-08-27T17:58:14+00:00
- git: `not-a-repo`
- test set: `C:\SIH26052_data\testset_prelim` (frozen, seed 20260827)
- device: `cpu`
- PESQ available: True
- clips: 60 per method

Targets: SNR gain > 15.0 dB, STOI > 0.85, PESQ > 2.5, RTF < 0.5.

## Burst-local performance (impulsive categories)

SI-SDR gain measured INSIDE the gunfire/explosion bursts only. A whole-clip score is diluted by the ~88% of each clip containing no transient, so a model that removes none of the gunfire can still look acceptable overall. This column is the one that answers the question.

| category | method | burst SI-SDR gain | whole-clip SI-SDR gain | burst % of clip |
|---|---|---|---|---|
| artillery | `gtcrn_dns3` | **+3.75** | +1.69 | 18.4% |
| artillery | `unprocessed` | **+0.00** | +0.00 | 18.4% |
| artillery | `specsub` | **-0.08** | -0.30 | 18.4% |
| artillery | `wiener` | **-0.30** | -0.72 | 18.4% |
| gunshot | `gtcrn_dns3` | **+4.07** | +1.27 | 17.7% |
| gunshot | `unprocessed` | **+0.00** | +0.00 | 17.7% |
| gunshot | `specsub` | **-0.04** | -0.08 | 17.7% |
| gunshot | `wiener` | **-0.06** | -0.19 | 17.7% |

### artillery

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn_dns3` | 2.503 | 0.937 | +1.69 | +1.96 | 0.0493 | PESQPASS STOIPASS SNRgFAIL RTFPASS |
| `specsub` | 2.155 | 0.928 | -0.30 | -0.28 | 0.0032 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `unprocessed` | 2.195 | 0.928 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `wiener` | 2.113 | 0.925 | -0.72 | -0.69 | 0.0052 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |

### gunshot

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn_dns3` | 3.005 | 0.956 | +1.27 | +1.06 | 0.0527 | PESQPASS STOIPASS SNRgFAIL RTFPASS |
| `specsub` | 2.540 | 0.943 | -0.08 | -0.22 | 0.0035 | PESQPASS STOIPASS SNRgFAIL RTFPASS |
| `unprocessed` | 2.569 | 0.944 | +0.00 | +0.00 | 0.0000 | PESQPASS STOIPASS SNRgFAIL RTFPASS |
| `wiener` | 2.489 | 0.941 | -0.19 | -0.38 | 0.0050 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |

## SNR gain by INPUT SNR (dB)

The >15 dB target is only reachable in the low-input-SNR regime; at high input SNR there is little noise left to remove.

| method | <0 dB | 0-5 dB | 5-10 dB | 10-15 dB | >15 dB |
|---|---|---|---|---|---|
| `gtcrn_dns3` | +6.34 (n=1) | +4.61 (n=3) | +3.61 (n=15) | +3.40 (n=14) | -1.50 (n=27) |
| `specsub` | +0.08 (n=1) | +0.01 (n=3) | +0.03 (n=15) | -0.02 (n=14) | -0.54 (n=27) |
| `unprocessed` | +0.00 (n=1) | +0.00 (n=3) | +0.00 (n=15) | +0.00 (n=14) | +0.00 (n=27) |
| `wiener` | +0.16 (n=1) | +0.03 (n=3) | +0.05 (n=15) | -0.11 (n=14) | -1.05 (n=27) |