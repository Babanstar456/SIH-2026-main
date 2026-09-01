# Resume here

Current state, what is blocking, and exactly how to pick each item up.
`CLAUDE.md` has the architecture and the invariants; `README.md` has results,
setup, and where to hear every audio file.

---

## Read this first: the project's status changed

The deliverable is built. **It does not yet do its job.**

Every artefact the problem statement asks for exists and is measured — model,
ONNX, spec sheet, results table, ablation, external calibration. On the frozen
synthetic test set the numbers are respectable. On **real recordings made with a
real microphone and real gunfire, the model makes speech HARDER to understand
than doing nothing at all.**

Measured with a speech recogniser standing in for a listener (whisper-medium,
26 known tokens per recording — see `scripts/asr_score.py`):

| recording | input SNR | unprocessed | model + floor −18 dB | model, full depth |
|---|---|---|---|---|
| `voice_noisy.wav`  (take 1) | +8 dB | **85%** | 73% | 12% |
| `voice_noisy2.wav` (take 2) | −12 dB | **50%** | 4% | 4% |
| `voice_noisy3.wav` (take 3) | −5.9 dB | **85%** | 77% | 69% |

Doing nothing wins on all three, and the ordering is monotonic: **the more the
model suppresses, the fewer words survive.** The gunfire is removed and the
speech goes with it.

That is the problem to solve. Everything below is organised around it.

### Why this was not caught earlier

The frozen test set says the model is good, because it measures PESQ, STOI and
SI-SDR against a clean reference on synthetic mixtures at **positive** SNR. None
of those is a measure of whether a listener can make out the words, and the test
set's SNR range does not cover the deployment case. Three separate proxy metrics
(STOI against a degraded reference, consonant-to-vowel ratio, band energy share)
all indicated improvement while word recognition fell. See
"Negative results" below — several plausible fixes were built, measured, and
found harmful.

### One large caveat, stated honestly

ASR is not an ear. Recognisers are trained on enormous quantities of noisy speech
and are far more robust to additive noise than humans, while being *less* robust
to processing artefacts they have never seen. Enhancement hurting ASR while
helping humans is a known effect. So the table above proves the chain hurts
**Whisper**; it does not by itself prove it hurts a person.

What makes it credible anyway: the human who recorded these files independently
reported the same thing repeatedly — gunfire gone, words unintelligible. Two
independent signals agreeing is much stronger than either alone.

**A scored human listening test is still the missing measurement.** The kit is
ready at `test-result/listening_test/` (answer sheet, key, scoring bands). It
needs three people who have not seen the script.

---

## What is running / nothing is running

No background jobs. GPU idle. Repo is in a clean, working state:
`19 passed` on the test suite.

---

## Where to pick up, in priority order

### 1. Decide whether the model belongs in the signal path at all

This is the honest first question, not a defeatist one. At +8 dB the full model
scores 12% against 85% unprocessed. If that reproduces under a human listening
test, then shipping it as-is is worse than shipping nothing, and the correct
engineering answer is either a better enhancer or a much shallower one.

The cheapest resolution is the listening test above.

### 2. Push the low-SNR training further

The one durable gain of the last session. The training mixture SNR was
`[0, +20] dB` — the steady background was **always quieter than the voice**, so
the model never saw the case the product exists for. `configs/data_lowsnr.yaml`
lowers it to `[-12, +12]`, verified by rendering actual mixtures:

| config | median realised SNR | below 0 dB | below −5 dB |
|---|---|---|---|
| `data.yaml` | +5.3 dB | 21% | 9% |
| `data_lowsnr.yaml` | **−3.8 dB** | **62%** | **41%** |

That produced `checkpoints/lowsnr_best.pt`, which beats the shipped model
substantially on real audio (take 3: 69% vs the shipped model's far worse
performance at the same depth). Next steps:

**Windows (PowerShell):**

```powershell
$PY = "C:\SIH26052_data\.venv\Scripts\python.exe"

# push lower still
& $PY -m src.train --tag lowsnr18 --data-config configs\data_lowsnr.yaml `
      --w-transient 0.0 --w-consonant 1.0     # then edit snr_db to [-18, 6]

# isolate the consonant term - the lowsnr run changed TWO things at once
& $PY -m src.train --tag lowsnr_noconsonant --data-config configs\data_lowsnr.yaml `
      --w-transient 0.0 --w-consonant 0.0
```

**Linux (bash):**

```bash
PY=~/SIH26052_data/.venv/bin/python

# push lower still
$PY -m src.train --tag lowsnr18 --data-config configs/data_lowsnr.yaml \
    --w-transient 0.0 --w-consonant 1.0     # then edit snr_db to [-18, 6]

# isolate the consonant term - the lowsnr run changed TWO things at once
$PY -m src.train --tag lowsnr_noconsonant --data-config configs/data_lowsnr.yaml \
    --w-transient 0.0 --w-consonant 0.0
```

**The `lowsnr` run changed both the SNR range and the loss, so attribution
between them is unknown.** The second command above settles it.

### 3. Stop selecting models on val PESQ

`checkpoint.monitor: val_pesq` in `configs/train.yaml`. The `lowsnr` run reached
its best val PESQ at **epoch 5** and then flatlined for 12 epochs until early
stopping — while PESQ is demonstrably not tracking word recognition. Selecting on
an ASR word score would optimise the thing that matters. This is a real change to
`src/train.py` and needs care: ASR scoring is slow, so it cannot run every epoch.

### 4. Understand why deep suppression destroys words

Nobody has looked at *what* the model removes when it over-suppresses. The floor
sweep shows the tradeoff but not the mechanism. A spectrogram diff between
`floor_18dB` and `floor_full_model` on take 3, focused on consonant regions,
would say whether it is over-gating brief high-frequency events (likely) or
something else.

### 5. The two older decisions, still open

- **Latency: 38.01 ms against a 32 ms target.** Unchanged and unchangeable with
  512/256 framing — 32 ms is the floor before any arithmetic. Fix is a 320/160
  retrain from scratch (~21 ms total). Now more attractive than before, because a
  from-scratch retrain is on the table anyway.
- **Artillery (66 clips) and rotor (72) rest on thin evidence.**
  `scripts/download_fsd50k_eval.sh` fills the gap, implies a full retrain, and
  `make_testset.py --force` invalidates every reported number.

---

## Negative results — do not rebuild these

Each was built as a plausible fix, measured, and found harmful or void. They are
kept (with warnings in their docstrings) because the measurements are the result.

| what | verdict |
|---|---|
| `scripts/post_enhance.py` | **Deleted.** Built on the belief the model was over-suppressing speech that was present. On the recording it was written for, the speech above 1 kHz genuinely was not there — it re-admitted noise, not voice. Premise disproved; do not rebuild a mask-floor post-processor without first checking the speech is actually in the band. |
| `scripts/voice_eq.py` — presence EQ toward a reference spectrum | Sounds better, measures worse. Costs 1.6 dB of consonant-to-vowel ratio; word score 58% → 42%. |
| `scripts/intelligibility.py` — consonant boost | Restores CVR to clean-speech parity (−16.8 → −10.9 dB) and still lowers word score. CVR is not intelligibility. |
| Multiband upward compression | The textbook move, and wrong here. Lifts every quiet frame, and most quiet frames are pauses and vowel tails rather than consonants: CVR −19.65 → −22.92 dB. Removed from `intelligibility.py`; do not reintroduce without measuring CVR. |
| Transient-weighted loss (earlier session) | No effect where intended (+0.05 dB on gunshot bursts, p = 0.72), significant cost elsewhere (PESQ −0.027, p < 0.001). `--w-transient` retained so the ablation is repeatable. |

---

## Measurement traps found the hard way

The instruments lie in specific, reproducible ways. All are now guarded in code.

1. **Whisper's temperature fallback is non-deterministic.** Left at its default
   tuple, a decode that trips an internal quality check is silently retried with
   *sampling*. The same unchanged file scored **69% and then 23%**. Fixed by
   passing `temperature=0.0` as a scalar. Always sanity-check with `--repeats 3`.
2. **Whisper drops or loops on hard audio.** Two distinct failures, both scoring
   ~4% on perfectly usable audio: a repetition loop (`"...A.M.A.M.A.M..."`), and
   silently skipping a segment — one file transcribed parts 1 and 4 correctly and
   omitted the alphabet and digits entirely, which is exactly the scored
   material. `looks_degenerate()` flags these now; **treat any flagged row as no
   measurement at all.**
3. **Chunked decoding is not a free fix.** It contains repetition loops but costs
   Whisper its context and destabilises other files — one file's *unprocessed*
   audio went from 69% to 0%. `--chunk` is opt-in; whole-file with a larger model
   is more reliable. Use `--model medium` for anything conclusive.
4. **STOI against a degraded reference is not intelligibility.** Scoring the
   output against the speaker's own muffled recording gave 0.84 and meant only
   "faithfully muffled".
5. **SI-SDR gain rises as SNR falls** (+6.3 dB at +10 dB SNR, +10.0 dB at −5 dB)
   because there is more noise to remove — it flatters exactly where
   intelligibility is collapsing.

---

## Recording quality: the capture chain matters more than expected

Two capture defects were found and fixed, both invisible until measured.

**Bluetooth headset mics are unusable.** In call mode they collapse to narrowband:
30 dB down by **1312 Hz**, against 4406 Hz for reference speech. Everything above
1.3 kHz — every consonant cue — was simply never recorded. No model recovers it.

**Phone VIDEO capture scoops the formant region.** Recording with the camera app
applies its own noise suppression and AGC. Switching to a plain **voice recorder**
app recovered **+19.2 dB at 1–2 kHz** and took consonant-to-vowel ratio from
−16.07 dB to −3.79 dB, at zero cost.

**For any future test recording:** voice recorder app, Bluetooth off, phone
10–15 cm from the mouth, gunfire from a speaker in the room (never through
earphones — the mic must hear it acoustically), and **leave ~10 s of gunfire
before speaking** so a clean noise bed can be extracted.

Verify any new recording before trusting a test built on it:

**Windows (PowerShell):**

```powershell
& $PY scripts\asr_score.py --model medium --inputs your_dry_take.wav
```

**Linux (bash):**

```bash
$PY scripts/asr_score.py --model medium --inputs your_dry_take.wav
```

A good dry take should score well above 70%. If it does not, fix the capture
before anything else.

---

## Environment: PyTorch is no longer blocked

RESUME.md previously documented Windows Smart App Control blocking PyTorch's
unsigned DLLs. **This is stale.** Torch 2.13.0+cu126 loads and trains normally on
this machine, GPU included, even with the policy still reporting `1` (enforcing).
Windows-only check — Smart App Control has no Linux equivalent:

```powershell
(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" `
  -Name VerifiedAndReputablePolicyState).VerifiedAndReputablePolicyState
```

The torch-free inference split in `src/framing.py` remains correct and worth
keeping — it is what lets a teammate run the model without a 2 GB CUDA download —
but it is no longer load-bearing for *this* machine.

**The test suite reports `19 passed`.** It previously claimed "17 pass + 2 skip
without PyTorch"; in fact those two tests were broken — `tests/test_core.py`
bound `S` to `src.framing` (NumPy-only, no `stft`/`istft`) while two tests called
`S.stft`. They errored rather than skipping. Fixed.

---

## Files added in the last session

| path | what |
|---|---|
| `checkpoints/lowsnr_best.pt` | model trained on deployment-matched SNR |
| `artifacts/model_lowsnr_simple.onnx` | 390 KB streaming export, verified at 1.05e-06 |
| `configs/data_lowsnr.yaml` | mixture config with `snr_db: [-12, 12]` |
| `scripts/asr_score.py` | **the measurement rig** — word recognition, no listeners needed |
| `scripts/floor_sweep.py` | suppression-depth vs word-survival curve |
| `scripts/robustness_sweep.py` | operating envelope with real PESQ/STOI |
| `scripts/snr_sweep.py` | behaviour across input SNR on a given noise bed |
| `test-result/` | every real recording, every processed variant, all scores |
| `src/stream_demo.py` | gained `--floor-db` (live and file paths) |
| `src/losses.py` | gained `w_consonant` (1–4 kHz band term), default 0.0 |

---

## Rebuilding from nothing

**Windows (PowerShell — run from Git Bash / WSL, or PowerShell if `bash` is on
PATH) / Linux (bash) — identical:**

```bash
# 1. environment - see README.md "Full setup from scratch"
# 2. data
bash scripts/download_tier1.sh
bash scripts/download_transients.sh

# 3. everything else, unattended and resumable
bash scripts/auto_pipeline.sh
```

`manifests/manifest.json` is not committed (22 MB of machine-specific absolute
paths); rebuild with `scripts/build_manifests.py`, about a minute. **Check the
group COUNTS it prints, not just that the assertions passed** — degenerate splits
are trivially disjoint. Healthy output has `background val=80`, `babble val=21`.

If you stop the pipeline mid-run, reap the workers:

**Windows (PowerShell):**

```powershell
Get-Process python | Stop-Process -Force
```

**Linux (bash):**

```bash
pkill -f python
```

(kills python processes broadly, matching the intent of the PowerShell
one-liner it replaces)

---

## Where things live

| what | where |
|---|---|
| code | `C:\dev\SIH-2026` |
| venv | `C:\SIH26052_data\.venv` (Python 3.12, **not** 3.14) |
| datasets | `C:\SIH26052_data\{raw,prepared}` (~64 GB) |
| frozen test set | `C:\SIH26052_data\testset` |
| VoiceBank-DEMAND | `C:\SIH26052_data\voicebank_demand` |
| real test recordings | `test-result/voice/` |
| processed variants + scores | `test-result/` |
