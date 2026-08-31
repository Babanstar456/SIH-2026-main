# Results - prelim

- generated: 2026-08-27T17:22:44+00:00
- git: `not-a-repo`
- test set: `C:\SIH26052_data\testset_prelim` (frozen, seed 20260827)
- device: `cpu`
- PESQ available: True
- clips: 200 per method

Targets: SNR gain > 15.0 dB, STOI > 0.85, PESQ > 2.5, RTF < 0.5.

### artillery

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn_dns3` | 2.590 | 0.938 | +3.10 | +3.28 | 0.0512 | PESQPASS STOIPASS SNRgFAIL RTFPASS |
| `specsub` | 2.227 | 0.926 | -0.15 | -0.13 | 0.0031 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `unprocessed` | 2.265 | 0.926 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `wiener` | 2.188 | 0.922 | -0.39 | -0.35 | 0.0043 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |

### engine

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn_dns3` | 1.698 | 0.797 | +5.99 | +7.10 | 0.0505 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `specsub` | 1.190 | 0.743 | +0.39 | +0.50 | 0.0030 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `unprocessed` | 1.185 | 0.742 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `wiener` | 1.231 | 0.743 | +1.10 | +1.42 | 0.0039 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |

### gunshot

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn_dns3` | 3.005 | 0.956 | +1.27 | +1.06 | 0.0529 | PESQPASS STOIPASS SNRgFAIL RTFPASS |
| `specsub` | 2.540 | 0.943 | -0.08 | -0.22 | 0.0030 | PESQPASS STOIPASS SNRgFAIL RTFPASS |
| `unprocessed` | 2.569 | 0.944 | +0.00 | +0.00 | 0.0000 | PESQPASS STOIPASS SNRgFAIL RTFPASS |
| `wiener` | 2.489 | 0.941 | -0.19 | -0.38 | 0.0040 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |

### rotor

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn_dns3` | 1.681 | 0.840 | +6.12 | +7.05 | 0.0531 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `specsub` | 1.273 | 0.788 | +0.39 | +0.49 | 0.0030 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `unprocessed` | 1.263 | 0.785 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |
| `wiener` | 1.312 | 0.793 | +1.14 | +1.45 | 0.0039 | PESQFAIL STOIFAIL SNRgFAIL RTFPASS |

### siren

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn_dns3` | 2.223 | 0.902 | +8.46 | +9.00 | 0.0528 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `specsub` | 1.551 | 0.868 | +0.05 | +0.08 | 0.0029 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `unprocessed` | 1.562 | 0.869 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `wiener` | 1.536 | 0.864 | +0.10 | +0.18 | 0.0041 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |