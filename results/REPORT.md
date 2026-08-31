# SIH 26052 — Smart Noise Cancellation for Defence Communication

## AI/ML workstream results

_Generated 2026-08-28T16:33:06+00:00 from measured results only._


## 1. Data

Splits are disjoint by *recording*, asserted in code — not by file. The gunshot corpus ships each shot as several per-channel files plus a channel-mean file sharing one uuid, so a per-file split would leak the same physical gunshot into train and test roughly eightfold.

| pool | train | val | test |
|---|---|---|---|
| speech (LibriSpeech) | 27,269 | 1,940 | 1,850 |
| noise / gunshot | 2,165 | 132 | 265 |
| noise / artillery | 66 | 4 | 10 |
| noise / rotor | 72 | 2 | 6 |
| noise / engine | 2,617 | 188 | 315 |
| noise / siren | 1,122 | 41 | 275 |
| noise / babble | 362 | 21 | 43 |
| noise / _background | 1,352 | 80 | 158 |
| room impulse responses | 51,112 | 3,101 | 6,005 |

## 2. Per-category results

Reported per category, never as a single average: an overall mean can look healthy while gunshots quietly fail.


### artillery

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF |
|---|---|---|---|---|---|
| gtcrn:checkpoints/ablation_no_transient_best.pt | 1.722 | 0.835 | +8.05 | +8.77 | 0.0659 |
| gtcrn:checkpoints/ft_best.pt | 1.721 | 0.832 | +7.85 | +8.57 | 0.0514 |
| gtcrn_dns3 | 1.628 | 0.832 | +7.27 | +8.12 | 0.0477 |
| wiener | 1.205 | 0.764 | +0.32 | +0.53 | 0.0041 |
| specsub | 1.197 | 0.766 | +0.20 | +0.27 | 0.0028 |
| gtcrn_vctk | 1.210 | 0.666 | +0.16 | +3.70 | 0.0530 |
| unprocessed | 1.188 | 0.765 | +0.00 | +0.00 | 0.0000 |

### babble

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF |
|---|---|---|---|---|---|
| gtcrn:checkpoints/ablation_no_transient_best.pt | 1.833 | 0.827 | +4.55 | +5.47 | 0.0474 |
| gtcrn:checkpoints/ft_best.pt | 1.802 | 0.822 | +4.19 | +5.17 | 0.0511 |
| gtcrn_dns3 | 1.702 | 0.818 | +3.82 | +4.79 | 0.0575 |
| wiener | 1.374 | 0.783 | +0.72 | +0.94 | 0.0040 |
| specsub | 1.352 | 0.786 | +0.46 | +0.54 | 0.0028 |
| unprocessed | 1.333 | 0.785 | +0.00 | +0.00 | 0.0000 |
| gtcrn_vctk | 1.320 | 0.702 | -1.79 | +0.82 | 0.0625 |

### engine

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF |
|---|---|---|---|---|---|
| gtcrn:checkpoints/ablation_no_transient_best.pt | 2.066 | 0.873 | +5.39 | +5.81 | 0.0595 |
| gtcrn:checkpoints/ft_best.pt | 2.018 | 0.868 | +5.07 | +5.50 | 0.0501 |
| gtcrn_dns3 | 1.967 | 0.870 | +4.83 | +5.34 | 0.0486 |
| wiener | 1.422 | 0.819 | +1.09 | +1.29 | 0.0042 |
| specsub | 1.395 | 0.818 | +0.62 | +0.67 | 0.0028 |
| unprocessed | 1.380 | 0.816 | +0.00 | +0.00 | 0.0000 |
| gtcrn_vctk | 1.357 | 0.723 | -1.45 | +0.58 | 0.0472 |

### gunshot

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF |
|---|---|---|---|---|---|
| gtcrn:checkpoints/ablation_no_transient_best.pt | 1.901 | 0.854 | +6.98 | +7.59 | 0.0639 |
| gtcrn:checkpoints/ft_best.pt | 1.898 | 0.850 | +6.71 | +7.35 | 0.0530 |
| gtcrn_dns3 | 1.793 | 0.849 | +5.74 | +6.44 | 0.0499 |
| wiener | 1.316 | 0.781 | +0.48 | +0.66 | 0.0043 |
| specsub | 1.303 | 0.783 | +0.25 | +0.31 | 0.0029 |
| unprocessed | 1.297 | 0.782 | +0.00 | +0.00 | 0.0000 |
| gtcrn_vctk | 1.311 | 0.697 | -0.19 | +2.72 | 0.0493 |

### rotor

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF |
|---|---|---|---|---|---|
| gtcrn:checkpoints/ablation_no_transient_best.pt | 1.995 | 0.875 | +4.76 | +5.12 | 0.1120 |
| gtcrn:checkpoints/ft_best.pt | 1.955 | 0.870 | +4.49 | +4.83 | 0.0494 |
| gtcrn_dns3 | 1.868 | 0.870 | +4.02 | +4.46 | 0.0480 |
| wiener | 1.404 | 0.824 | +0.74 | +0.90 | 0.0042 |
| specsub | 1.372 | 0.825 | +0.38 | +0.43 | 0.0028 |
| unprocessed | 1.354 | 0.823 | +0.00 | +0.00 | 0.0000 |
| gtcrn_vctk | 1.354 | 0.757 | -1.31 | +0.26 | 0.0484 |

### siren

| method | PESQ | STOI | SI-SDR gain | SNR gain | RTF |
|---|---|---|---|---|---|
| gtcrn:checkpoints/ablation_no_transient_best.pt | 2.080 | 0.892 | +5.26 | +5.51 | 0.0483 |
| gtcrn:checkpoints/ft_best.pt | 2.042 | 0.886 | +4.83 | +5.11 | 0.0505 |
| gtcrn_dns3 | 1.934 | 0.885 | +4.46 | +4.82 | 1.0748 |
| wiener | 1.408 | 0.837 | +0.58 | +0.73 | 0.0042 |
| specsub | 1.375 | 0.838 | +0.30 | +0.35 | 0.0028 |
| unprocessed | 1.364 | 0.837 | +0.00 | +0.00 | 0.0000 |
| gtcrn_vctk | 1.351 | 0.757 | -1.98 | -0.27 | 0.0628 |

## 3. Burst-local performance

SI-SDR gain measured **inside the gunfire/explosion bursts only**. Bursts occupy roughly 12–18% of a clip, so a whole-clip score is diluted by the majority of the signal that contains no transient — a model that removes none of the gunfire can still post an acceptable overall number. This is the column that answers the question the problem statement actually asks.

| category | method | burst SI-SDR gain | whole-clip SI-SDR gain |
|---|---|---|---|
| artillery | gtcrn:checkpoints/ablation_no_transient_best.pt | +7.87 | +8.05 |
| artillery | gtcrn:checkpoints/ft_best.pt | +7.83 | +7.85 |
| gunshot | gtcrn:checkpoints/ft_best.pt | +7.21 | +6.71 |
| gunshot | gtcrn:checkpoints/ablation_no_transient_best.pt | +7.16 | +6.98 |
| artillery | gtcrn_dns3 | +7.03 | +7.27 |
| gunshot | gtcrn_dns3 | +5.84 | +5.74 |
| gunshot | wiener | +0.19 | +0.48 |
| gunshot | gtcrn_vctk | +0.18 | -0.19 |
| artillery | wiener | +0.16 | +0.32 |
| gunshot | specsub | +0.11 | +0.25 |
| artillery | specsub | +0.09 | +0.20 |
| gunshot | unprocessed | +0.00 | +0.00 |
| artillery | unprocessed | +0.00 | +0.00 |
| artillery | gtcrn_vctk | -0.12 | +0.16 |

## 4. External benchmark — VoiceBank-DEMAND

Included so our numbers can be checked against published results rather than only against a test set we built ourselves.

| method | PESQ | STOI | SI-SDR gain |
|---|---|---|---|
| gtcrn_vctk | 2.868 | 0.940 | +10.35 |
| gtcrn_dns3 | 2.508 | 0.915 | +7.26 |
| wiener | 2.120 | 0.920 | +1.49 |
| specsub | 2.029 | 0.922 | +0.58 |
| unprocessed | 1.968 | 0.921 | +0.00 |

## 5. Latency and compute

Model: **48,245 parameters**.

| stage | ms |
|---|---|
| chunk buffering (hop) | 16.00 |
| overlap-add delay (measured) | 16.00 |
| model compute (p95, 1 thread) | 6.01 |
| **total** | **38.01** (**OVER by 6.0 ms** vs 32 ms target) |

RTF (mean, 1 thread): **0.2955** — target < 0.5 (PASS).


**On the latency target.** Algorithmic delay is fixed by the framing: a 16 ms chunk must be collected, and overlap-add holds each sample until every frame covering it is computed — measured at 16 ms, exactly `win − hop`. That is 32 ms before any arithmetic. The problem statement asks for 16 ms chunks *and* under 32 ms delay; both follow from `n_fft=512`, so they cannot both hold. Reducing it requires 320/160 framing (20 ms window, ~21 ms total) and a retrain.


**Streaming export fidelity.** The exported ONNX was run frame by frame and compared against the offline PyTorch model: max absolute difference `2.87e-07`, `4.61e-06` relative to signal RMS (match).


## 6. Limitations

- Latency exceeds the 32 ms target (see §5); this is structural, not a tuning issue.
- `SNR gain > 15 dB` is bounded by input SNR (measured correlation −0.78), so it is reported stratified by input-SNR band rather than as one average.
- ESC-50 and UrbanSound8K are CC BY-NC; fine for research and competition, but they must be replaced with CC-BY sources if this moves toward procurement. The firearm corpus is already CC BY 4.0.
- Drone/quadcopter rotor audio is thinly covered by available open corpora; helicopter is well covered, drones are not.
