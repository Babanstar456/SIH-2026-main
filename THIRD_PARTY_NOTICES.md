# Third-party notices

## GTCRN

`src/models/gtcrn.py` and `third_party/gtcrn/` are taken from
<https://github.com/Xiaobin-Rong/gtcrn>, MIT licence, copyright (c) Xiaobin Rong.
The full licence text is preserved at `src/models/LICENSE.gtcrn`.

Rong et al., *"GTCRN: A Speech Enhancement Model Requiring Ultralow
Computational Resources"*, ICASSP 2024.

`checkpoints/pretrained/` contains the author's released weights
(`model_trained_on_dns3.tar`, `model_trained_on_vctk.tar`), used as the
initialisation for fine-tuning and as evaluation baselines.

## Datasets

Not redistributed here — downloaded by `scripts/download_*.sh`.

| Dataset | Licence |
|---|---|
| LibriSpeech (OpenSLR 12) | CC BY 4.0 |
| MUSAN (OpenSLR 17) | CC BY 4.0 |
| Room Impulse Responses (OpenSLR 28) | Apache 2.0 |
| Gunshot/Gunfire Audio Dataset (Zenodo 7004819) | CC BY 4.0 |
| UrbanSound8K (Zenodo 1203745) | CC BY-NC 4.0 |
| ESC-50 | CC BY-NC 3.0 |
| VoiceBank-DEMAND (Edinburgh DataShare 10283/2791) | see record EULA |

**ESC-50 and UrbanSound8K are non-commercial.** Suitable for research and
competition use; they must be replaced with CC-BY sources before any commercial
or procurement use.

## Metrics

- `pesq` — ITU-T P.862 reference implementation
- `pystoi` — reference STOI/ESTOI implementation
