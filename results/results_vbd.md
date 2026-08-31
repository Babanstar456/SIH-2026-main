# Results - vbd

- generated: 2026-08-27T17:51:06+00:00
- git: `not-a-repo`
- test set: `C:\SIH26052_data\voicebank_demand` (frozen, seed None)
- device: `cpu`
- PESQ available: True
- clips: 824 per method

Targets: SNR gain > 15.0 dB, STOI > 0.85, PESQ > 2.5, RTF < 0.5.

### voicebank_demand

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|
| `gtcrn_dns3` | 2.508 | 0.915 | +7.26 | +7.36 | 0.0625 | PESQPASS STOIPASS SNRgFAIL RTFPASS |
| `gtcrn_vctk` | 2.868 | 0.940 | +10.35 | +10.41 | 0.0605 | PESQPASS STOIPASS SNRgFAIL RTFPASS |
| `specsub` | 2.029 | 0.922 | +0.58 | +0.60 | 0.0035 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `unprocessed` | 1.968 | 0.921 | +0.00 | +0.00 | 0.0000 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |
| `wiener` | 2.120 | 0.920 | +1.49 | +1.55 | 0.0047 | PESQFAIL STOIPASS SNRgFAIL RTFPASS |