# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SIH 2026 problem 26052: a **streaming speech-enhancement model** that strips
gunfire, artillery, rotor and engine noise off a soldier's outgoing microphone
feed, small enough for an embedded chip. Base model is **GTCRN** (48,245
parameters, ICASSP 2024, MIT), vendored at `src/models/gtcrn.py` and fine-tuned
here on defence-noise mixtures.

**Scope boundary — this matters.** This repo is *Path 1* only: the DNN that
cleans the **outgoing transmitted voice**. Earcup rumble cancellation, the
DAC/driver chain, and the analog limiter in front of the ADC belong to the
hardware team. The model performs **suppression only — it does not apply gain**.
Requests to "amplify the voice" belong to the analog path downstream, not here.

**Contract with the hardware team:** 16 kHz mono in, 16 ms chunks, same out.

Read **`RESUME.md`** for current state and what to do next. Read **`README.md`**
for results, setup and the full command reference.

## Status, as of the last session — read before changing anything

The deliverable is built and **it does not yet do its job.** On real recordings
the model removes gunfire and takes the intelligibility of the speech with it.
Measured by ASR word recognition (`scripts/asr_score.py`, whisper-medium, 26
known tokens):

| recording | input SNR | unprocessed | + floor -18 dB | model, full depth |
|---|---|---|---|---|
| take 1 | +8 dB | **85%** | 73% | 12% |
| take 2 | -12 dB | **50%** | 4% | 4% |
| take 3 | -5.9 dB | **85%** | 77% | 69% |

Doing nothing wins on all three; the more the model suppresses, the fewer words
survive. The frozen test set does not show this because it measures PESQ/STOI
against a clean reference on synthetic mixtures at POSITIVE SNR - none of which
is a measure of whether a listener can make out the words.

**Consequence for anyone working here: a change is not an improvement until it
raises the ASR word score.** PESQ, STOI, SI-SDR, consonant-to-vowel ratio and
band-energy share have each, in this project, moved the right way while word
recognition moved the wrong way.

## Commands

The interpreter is **not** on PATH and lives outside the project directory:

**Windows (PowerShell):**

```powershell
$PY = "C:\SIH26052_data\.venv\Scripts\python.exe"
```

**Linux (bash):**

```bash
PY=~/SIH26052_data/.venv/bin/python
```

Python 3.12, not the system 3.14 — PyTorch ships CPU-only wheels for 3.14, so
3.14 silently gives you no GPU and a ~50x slowdown with no error.

**Windows (PowerShell):**

```powershell
.\run.ps1 status      # what is downloaded / built
.\run.ps1 data        # extract + resample + manifests + mixture QA
.\run.ps1 testset     # freeze evaluation set + VoiceBank-DEMAND benchmark
.\run.ps1 baseline    # the comparison table - run BEFORE training
.\run.ps1 train       # fine-tune
.\run.ps1 ablate      # same run with the transient term disabled
.\run.ps1 finish      # evaluate + bench + export + handoff bundle
.\run.ps1 test        # unit tests

bash scripts/auto_pipeline.sh   # unattended chain, resumable, skips done work
```

**Linux (bash):**

```bash
./run.sh status      # what is downloaded / built
./run.sh data        # extract + resample + manifests + mixture QA
./run.sh testset     # freeze evaluation set + VoiceBank-DEMAND benchmark
./run.sh baseline    # the comparison table - run BEFORE training
./run.sh train       # fine-tune
./run.sh ablate      # same run with the transient term disabled
./run.sh finish      # evaluate + bench + export + handoff bundle
./run.sh test        # unit tests

bash scripts/auto_pipeline.sh   # unattended chain, resumable, skips done work
```

### Measuring intelligibility - do this before claiming any improvement

**Windows (PowerShell):**

```powershell
# Word recognition, with a recogniser standing in for a listener. USE medium for
# anything conclusive: whisper-small is too weak on this audio and fails
# unpredictably in ways that look like results.
& $PY scriptssr_score.py --model medium --inputs a.wav b.wav
& $PY scriptssr_score.py --model medium --repeats 3 --inputs a.wav  # stability
& $PY scriptssr_score.py --model medium --show-transcript --inputs a.wav

# Suppression depth vs word survival - the tradeoff curve. Optimise the PAIR,
# never suppression alone; that is exactly how this went wrong.
& $PY scriptsloor_sweep.py --input noisy.wav --ckpt checkpoints\lowsnr_best.pt `
      --out-dir test-resultloors
```

**Linux (bash):**

```bash
# Word recognition, with a recogniser standing in for a listener. USE medium for
# anything conclusive: whisper-small is too weak on this audio and fails
# unpredictably in ways that look like results.
$PY scripts/asr_score.py --model medium --inputs a.wav b.wav
$PY scripts/asr_score.py --model medium --repeats 3 --inputs a.wav  # stability
$PY scripts/asr_score.py --model medium --show-transcript --inputs a.wav

# Suppression depth vs word survival - the tradeoff curve. Optimise the PAIR,
# never suppression alone; that is exactly how this went wrong.
$PY scripts/floor_sweep.py --input noisy.wav --ckpt checkpoints/lowsnr_best.pt \
    --out-dir test-result/floors
```

A row flagged `DECODER GLITCH` is **not a measurement** - discard it, never
average it in. See "Measurement traps" below.

Tests:

**Windows (PowerShell):**

```powershell
& $PY -m pytest tests -q
& $PY -m pytest tests/test_core.py::test_wola_roundtrip_is_exact -q   # single test
```

**Linux (bash):**

```bash
$PY -m pytest tests -q
$PY -m pytest tests/test_core.py::test_wola_roundtrip_is_exact -q   # single test
```

Fast checks that catch most breakage in under a minute:

**Windows (PowerShell):**

```powershell
& $PY scripts\smoke_train.py --steps 20 --batch 24 --workers 8   # data->loss->backward
& $PY -m src.train --tag smoke --epochs 2 --epoch-size 480 --val-size 96
& $PY scripts\qa_mixtures.py --n 24                              # renders AND checks
```

**Linux (bash):**

```bash
$PY scripts/smoke_train.py --steps 20 --batch 24 --workers 8    # data->loss->backward
$PY -m src.train --tag smoke --epochs 2 --epoch-size 480 --val-size 96
$PY scripts/qa_mixtures.py --n 24                                # renders AND checks
```

Training resumes from `checkpoints/<tag>_last.pt` with `--resume`; an
interrupted run costs one epoch.

## Where things live

| what | where |
|---|---|
| code | this repository (`C:\dev\SIH-2026`) |
| venv | `C:\SIH26052_data\.venv` (~4.7 GB, 47k files) |
| raw + prepared datasets | `C:\SIH26052_data\{raw,prepared}` (~59 GB) |
| frozen test set | `C:\SIH26052_data\testset` |
| VoiceBank-DEMAND benchmark | `C:\SIH26052_data\voicebank_demand` |

Data and venv sit **outside the repo on purpose** — 64 GB of audio and a
50,000-file `site-packages` tree cannot go in git, and were originally kept out
of OneDrive because it would sync them continuously. Everything there is
regenerable from `scripts/download_*.sh`. `manifests/manifest.json` is also not
committed: it is 21 MB of machine-specific absolute paths, rebuilt by
`build_manifests.py`.

## Architecture

Pipeline, in dependency order:

```
scripts/download_*.sh   → raw corpora (resumable; hosts drop long transfers)
scripts/prepare_data.py → extract, resample ONLY what needs it
scripts/build_manifests.py → splits + disjointness ASSERTIONS → manifests/manifest.json
scripts/qa_mixtures.py  → render mixtures and CHECK them
scripts/make_testset.py → frozen evaluation set, written once
src/evaluate.py         → comparison table  (run BEFORE training)
src/train.py            → fine-tune
src/bench.py            → latency + RTF
src/export_onnx.py      → streaming ONNX + faithfulness check
scripts/make_handoff.py → the bundle the hardware team receives
scripts/make_report.py  → assembles results/REPORT.md from measured files only
```

### The mixer is the centre of gravity (`src/mixer.py`)

Mixtures are built in **two distinct layers**, and this distinction is the whole
project:

- **Steady background** (MUSAN) mixed at a global SNR measured over
  *active-speech frames only* — whole-clip SNR is skewed by silence, so two
  clips nominally at "0 dB" differ audibly.
- **Impulsive bursts** (gunfire, explosions) placed at random offsets as
  discrete events, cut at their **acoustic onset**, scaled by **peak** level, and
  routinely louder than the speech.

Blending gunfire evenly through a clip instead of injecting discrete bursts
teaches only the easy stationary case and the model then fails on exactly the
sound the problem statement cares about. Do not "simplify" this.

The mixer also models the **analog limiter** the hardware team places before the
ADC (soft-knee, threshold 0.6–0.95). The model must train on limited audio
because that is what it receives in the field.

Where a corpus provides ground-truth event timestamps (the firearm corpus ships
6,212 annotated shots), `audio.load_burst` cuts at the annotation rather than at
an energy-detected onset.

### The transient mask threads through the codebase

`mixer.build()` returns a sample-level `transient_mask` marking every burst.
`src/stft.py::samples_to_frame_mask` converts it to STFT frames;
`src/dataset.py` passes it through the dataloader; `src/losses.py` can use it to
weight those frames harder; `src/metrics.py::masked_metrics` uses it to measure
*inside* the bursts.

**The loss weighting was tested and dropped — see "Findings" below.** The mask is
still load-bearing for the burst-local metrics, which are how gunshot
performance is honestly reported.

### STFT constants are not free parameters

`src/framing.py` defines `n_fft=512, hop=256, win=hann(512)**0.5` at 16 kHz.
These must match `src/models/gtcrn.py` exactly or the pretrained weights are
meaningless. `src/baselines/classical.py` uses the same framing so the
comparison is like-for-like.

### The inference path must not import PyTorch

`framing.py` is **NumPy only** and holds the constants, the sample→frame
mapping, and the streaming cache shapes. `stft.py` adds the torch STFT/iSTFT and
re-exports everything from `framing`, so older imports keep working.

This split is load-bearing, not tidiness. Running the shipped model used to pull
in PyTorch transitively (`stream_demo` → `stft` → `torch`), which meant the
documented "no PyTorch needed" setup failed with ImportError on a clean machine,
and the whole project stopped working when Windows Smart App Control began
blocking torch's unsigned DLLs. Anything on the deployment path — `stream_demo`,
`baselines/classical`, `make_demo`, `make_handoff`, `qa_mixtures` — imports from
`framing`, and `methods.py` imports torch lazily inside the neural methods only.

**When adding code, check which side of that line it belongs on.** Verified with
torch unloadable: the ONNX model runs on real audio, the classical baselines
evaluate, and the NumPy invariants pass (the 2 torch-specific tests skip).
Training and ONNX export legitimately require PyTorch.

### Method registry (`src/methods.py`)

Every enhancement method has the identical signature
`f(noisy: np.ndarray, sr: int) -> np.ndarray`, returning audio of the same
length. That uniformity is what lets `src/evaluate.py` sweep classical and
neural methods over identical audio without special-casing — which is what makes
the comparison table trustworthy. Names: `unprocessed`, `wiener`, `specsub`,
`noisereduce`, `gtcrn_dns3`, `gtcrn_vctk`, or `gtcrn:<path-to-checkpoint>`.

## Invariants — these fail SILENTLY if broken

Everything here was either learned the hard way or is enforced in code. None of
it raises an exception when violated; it just produces wrong numbers that look
plausible.

1. **Splits are disjoint by RECORDING, never by file.** `build_manifests.py`
   *raises* on overlap. The firearm corpus ships each shot as up to 8
   per-channel files plus a channel-mean file sharing one `uuid`; ESC-50 and
   UrbanSound8K cut many clips from one Freesound upload (`src_file` / `fsID`);
   OpenSLR-28 room IDs repeat across small/medium/large. Group on the
   *recording*, not the filename.

   **But the group key must not be too COARSE either.** MUSAN's subdirectories
   are provenance labels (`free-sound`, `librivox`, `fma`, …) — only 2 to 5 per
   type — and grouping on them made an 85/5/10 split put *everything* in train:
   babble vanished from val and test entirely, and the background pool lost its
   whole validation set, so every validation mixture silently had no steady
   noise in it. The disjointness assertion passed the whole time, because
   degenerate splits are trivially disjoint. MUSAN is grouped **per file** —
   each of its wavs is an independent recording. When adding a corpus, check the
   resulting group COUNT, not just that the assertion passes.

2. **The test set is frozen.** `make_testset.py` refuses to overwrite without
   `--force`. Regenerating it means later comparisons measure the test set, not
   the model. Use `--out` for scratch sets.

3. **Impulsive categories must contain their event.** `force_burst=True` is set
   for impulsive categories; without it Poisson(1.5) leaves ~30% of "gunshot"
   clips containing no gunshot, inflating precisely the category that must be
   honest.

4. **Report per category, never one average.** An overall mean looks healthy
   while gunshots fail.

5. **The test set must be able to MEASURE, and its difficulty is not what the
   config says.** Burst energy is not counted in `background.snr_db`, so the
   realised whole-clip SNR lands well below it — a nominal `[-5, 15]` produced a
   median of **−0.4 dB** with 63% of unprocessed clips on the PESQ floor
   (< 1.15), where PESQ stops discriminating and no method can approach the
   PESQ 2.5 / STOI 0.85 targets. Every row of that table read FAIL, which is a
   broken instrument, not a result. `background.snr_db` is now `[0, 20]` and
   `bursts.peak_snr_db` `[-20, 0]`, giving a median of +4.1 dB spanning −7 to
   +17 with a clean PESQ progression per band. **Render a scratch set with
   `make_testset.py --out <scratch>` and measure the realised SNR distribution
   before freezing** — do not infer difficulty from the config.

6. **Bursts must out-peak the background.** MUSAN's noise set contains impulsive
   material of its own with a 10–14 dB crest factor. When burst peak SNR reached
   +5 dB the injected "gunshot" peaked *below* the background, and the loudest
   thing in the clip was not the transient 40% of the time — which inverts the
   problem into one a stationary filter could solve. Keep `peak_snr_db` at or
   below 0.

7. **Burst-local metrics are the real answer.** Bursts occupy ~12–20% of a clip,
   so whole-clip SI-SDR is diluted by the majority that contains no transient.
   `metrics.masked_metrics` measures inside the bursts.

   Note the *converse* trap in `qa_mixtures.py`: comparing mean energy inside vs
   outside the mask on the SUMMED residual does not test mask alignment. The
   mask is aligned by construction (`mask[start:end] = True` at the placed
   samples; the sample→frame mapping is unit-tested separately), and mean energy
   is dominated by the continuous background rather than a millisecond-scale
   crack. QA therefore reports two separate things: frame-mask coverage drift,
   and burst prominence measured on the burst and background tracks *before*
   they are summed (`meta["burst_peak"]` / `meta["bg_peak"]`).

8. **`SNR gain > 15 dB` is not well-posed alone.** Measured correlation with
   input SNR is **−0.78**; you cannot remove noise that is not there. Results
   are stratified by input-SNR band.

9. **Reference metric implementations only** — `pesq` (ITU-T P.862), `pystoi`.
   Never hand-roll a perceptual metric. If PESQ is unavailable the value is NaN
   and a `pesq_available=False` column is written; NaN must never be read as 0.

10. **Align before measuring anything time-domain against the streaming path.**
    The streaming enhancer lags its input by exactly `win - hop` (256 samples,
    16 ms). PESQ time-aligns internally and hides this; SI-SDR and STOI do not.
    An unaligned comparison scored a perfectly good model at **−32.55 dB
    SI-SDR**. `scripts/make_demo.py` compensates; `src/evaluate.py` is unaffected
    because it uses the centre-padded offline path.

11. **The shipped ONNX must be self-contained.** `torch.onnx` splits weights into
    a `.onnx.data` sidecar; `export_onnx.py` folds them inline and *proves* it by
    loading the file from an empty scratch directory. A sidecar dependency loads
    fine here and fails on the hardware team's machine.

12. **Verify the streaming export against the offline model.** Wrong cache wiring
    still loads, still runs, and sounds subtly worse. Current fidelity: max abs
    diff 3.99e-07.

13. **Benchmark on an idle machine** — though note the measured difference was
    small (37.45 ms busy vs 38.01 ms idle); per-frame ONNX Runtime dispatch
    dominates, not contention.

14. **The only intentional placeholder** is `artifacts/passthrough_stub.onnx` — a
    Day-1 identity model with the real interface so hardware integration could
    proceed. It is clearly labelled and produces no metrics. Everything else must
    be measured; no hardcoded numbers anywhere in results.


### The suppression floor is a first-class control (`--floor-db`)

`StreamingEnhancer` accepts `floor_db`, which caps how deep any bin may be cut:

    G_final = max(G_model, 10 ** (floor_db / 20))

`None` reproduces the model untouched. The floor exists because **suppression
depth is a dial to be set, not maximised**. At negative SNR - the deployment
case - the unconstrained mask cuts speech away along with the noise, and measured
word recognition falls monotonically as depth rises. On take 3:

| setting | gunfire suppressed | word score |
|---|---|---|
| unprocessed | 0.0 dB | 85% |
| floor -18 dB | 15.1 dB | 77% |
| full model | 27.5 dB | 69% |

The live path and the offline `floor_sweep.py` use the identical construction -
recover the model's implied gain, clamp it, reapply to the NOISY spectrum so the
noisy phase is kept - so the two cannot silently diverge.

## Measurement traps - these produce confident wrong numbers

Additional to the silent-failure list above, and all learned the hard way in one
session. Every one of them produced a number that looked like a result.

15. **Whisper's temperature fallback is non-deterministic.** Left at its default
    tuple, a decode that trips an internal quality check is silently retried with
    SAMPLING. The same unchanged file scored **69% and then 23%**. `asr_score.py`
    passes `temperature=0.0` as a scalar to disable it. Sanity-check anything
    important with `--repeats 3`.

16. **Whisper drops or loops on hard audio, and both score ~4%.** Two distinct
    failures observed on perfectly usable audio: a repetition loop
    (`"...A.M.A.M.A.M..."` for the rest of the clip), and silently skipping a
    segment - one file transcribed parts 1 and 4 correctly and omitted the
    alphabet and the digits entirely, which is exactly the scored material.
    `looks_degenerate()` flags these. **A flagged row is not a measurement.**

17. **Chunked decoding is not a free fix.** It contains repetition loops but
    costs Whisper its context and destabilises other files - one file's
    UNPROCESSED audio went from 69% to 0%. `--chunk` is opt-in; whole-file with
    `--model medium` is the reliable route. Degenerate chunks are FLAGGED, never
    dropped: dropping cannot raise a score (garbage matches no target) but does
    discard the real words the decoder did get.

18. **STOI against a degraded reference is not intelligibility.** Scoring output
    against the speaker's own muffled recording returned 0.84 and meant only
    "faithfully muffled". A reference metric is only as meaningful as its
    reference.

19. **The training SNR range must cover deployment, and it did not.**
    `configs/data.yaml` sets `background.snr_db: [0, 20]` - the steady background
    is ALWAYS quieter than the voice, so the model never saw the case the product
    exists for. That range was chosen for a sound reason (the frozen TEST SET has
    to discriminate; at [-5, 15] it put 63% of clips on the PESQ floor) and then
    applied to TRAINING, where the requirement is the opposite. Test sets need to
    discriminate; training sets need to match reality.
    `configs/data_lowsnr.yaml` uses `[-12, 12]`. **Verify by rendering mixtures,
    never by reading the config** - realised median SNR is +5.3 dB for
    `data.yaml` and -3.8 dB for `data_lowsnr.yaml`.

20. **The capture chain silently destroys what no model can recover.** A
    Bluetooth headset mic in call mode is 30 dB down by **1312 Hz** against
    4406 Hz for reference speech - every consonant cue simply never recorded.
    Phone VIDEO capture applies its own noise suppression and AGC and scoops the
    formant region; switching to a plain voice-recorder app recovered **+19.2 dB
    at 1-2 kHz**. Measure any new recording before building a test on it.

## Findings from the build

Recorded because they are results, not anecdotes, and because several contradict
what the roadmap assumed.

### Suppression and intelligibility are in direct tension

The single most important result in the project, and it was invisible until word
recognition was measured directly. Across three real recordings, word score falls
monotonically as suppression depth rises, and **doing nothing wins on all three**
(see the status table at the top of this file).

The gunfire suppression is genuine - 27.5 dB measured on take 3, against +0.19 dB
for a Wiener filter on the frozen set. The speech does not survive it. Both are
true simultaneously, and any future work has to hold both numbers in view.

### Training on the wrong SNR regime was a root cause

`background.snr_db: [0, 20]` meant the model never trained on the case the
product exists for. Retraining on `[-12, 12]` produced `checkpoints/lowsnr_best.pt`,
which is substantially better on real audio at the same suppression depth. That
run changed the SNR range AND enabled the consonant loss term simultaneously, so
**attribution between the two is unknown** - `--w-consonant 0.0` on the same
config settles it and has not been run.

Training early-stopped at epoch 17 with best val PESQ at **epoch 5**, then flat.
The recipe, not the training length, is the limit - and `checkpoint.monitor:
val_pesq` is selecting on a metric that does not track word recognition.

### Restoring the consonant-to-vowel ratio does not restore intelligibility

Consonants are 20-30 dB below the vowels and are what distinguishes one word from
another, so the consonant-to-vowel ratio looks like the right target. It is not
sufficient. Measured on real audio: clean speech sits at -10.68 dB, the speaker's
dry recording at -16.07 dB, the model output at -18.07 dB. A tilt-ranked
consonant boost restores CVR to -10.9 dB - matching clean speech exactly - and
**word score still falls** (58% -> 42%).

Multiband upward compression, the textbook move, is worse: it lifts every quiet
frame, and most quiet frames are pauses and vowel tails rather than consonants
(CVR -19.65 -> -22.92 dB). Built, measured, removed.

### The transient-weighted loss does not work

The model was built with an extra loss term weighting gunfire frames harder —
the intuitive answer to the roadmap's number-one risk. A paired ablation
(`--w-transient 0.0`, identical otherwise) over the same 720 clips found:

| effect of the transient term | result |
|---|---|
| Gunshot burst SI-SDR gain | +0.048 dB, p = 0.72 — **no effect** |
| Artillery burst SI-SDR gain | −0.044 dB, p = 0.79 — **no effect** |
| Overall PESQ | −0.027, p < 0.001 — **significantly worse** |
| Overall STOI | −0.005, p < 0.001 — **significantly worse** |
| Overall SI-SDR gain | −0.31 dB, p < 0.001 — **significantly worse** |

**The shipped model is the version without it** (`checkpoints/shipped_best.pt`,
copied from `ablation_no_transient_best.pt`). The improvement over the pretrained
baseline is real — +1.376 dB on gunshot bursts (p = 0.0004), +0.798 dB on
artillery (p = 0.0099) — but it comes from **the training data, not the
objective**. Keep `--w-transient` so the experiment stays repeatable.

### Domain transfer is strongly asymmetric

`gtcrn_vctk` scores PESQ **2.868** on VoiceBank-DEMAND (better than DNS3's
2.508) and **1.317** on our defence-noise set — statistically indistinguishable
from unprocessed (1.319), and it *degrades* SNR above 5 dB input. The checkpoint
that wins on café noise is the worst on gunfire. This is the strongest single
argument in the project for domain-specific training.

### The latency target cannot be met

Measured: 16 ms chunk buffering + 16 ms overlap-add delay (= `win − hop`,
measured by cross-correlation in `src/bench.py`, not assumed) + 6.01 ms compute
= **38.01 ms**. The 32 ms floor exists before any arithmetic. The problem
statement asks for 16 ms chunks *and* under 32 ms delay; both derive from
`n_fft=512`, so they cannot both hold. Fix, if the ceiling is hard: retrain at
320/160 (20 ms window, ~21 ms total). RTF passes comfortably either way
(0.2955 vs < 0.5 target).

Most of the compute term is ONNX Runtime per-call dispatch across many small
graph nodes rather than arithmetic — at 33 MMACs/s the actual maths is well under
a millisecond per frame.

### Calibration against published literature

On the standard VoiceBank-DEMAND test set the evaluator reproduces the published
unprocessed baseline **PESQ 1.968 / STOI 0.921** (published: 1.97 / 0.921). Keep
this check working — it is what makes numbers on our own test set credible.

## Known constraints

- **PyTorch exports at opset 18** regardless of the requested version — check
  before any TensorRT/Jetson port.
- **ESC-50 and UrbanSound8K are CC BY-NC.** Fine for research and the
  competition; must be replaced with CC-BY sources if this moves toward
  procurement. The firearm corpus (Zenodo 7004819) is already CC BY 4.0.
- **`us.openslr.org` does not resolve.** Use `www.openslr.org`,
  `openslr.trmal.net`, or `openslr.elda.org`.
- **Zenodo and datashare.ed.ac.uk drop long transfers.** Use
  `scripts/fetch_resume.sh`, or `scripts/fetch_zip_until_valid.sh` for zips —
  datashare answers HEAD with a ~4 KB HTML interstitial, so a size-based
  completion check will happily declare a 10%-downloaded file finished. This bug
  actually occurred: a 15 MB fragment of a 148 MB zip was marked complete.
- **ESC-50 has no `gun_shot` class** (only `fireworks` / `thunderstorm` as
  transient proxies). Real firearm audio comes from Zenodo 7004819 and
  UrbanSound8K.
- **Thin pools:** artillery (~66 source clips) and rotor (~72) versus gunshot
  (~2,165) and engine (~2,617). `configs/data.yaml` holds artillery's sampling
  weight down accordingly; `scripts/download_fsd50k_eval.sh` fills the gap.
- **Drone/quadcopter audio is poorly covered** by open corpora. Helicopter is
  fine; drones are not.
- **Windows Smart App Control** can block PyTorch's unsigned DLLs
  (`WinError 4551` on `c10.dll`). **Currently NOT blocking on this machine** —
  torch 2.13.0+cu126 loads and trains with GPU even while the policy still
  reports `1` (enforcing), so treat older notes to the contrary as stale. The
  torch-free inference split remains correct and worth keeping. Note also the
  suite is `19 passed`, not "17 pass + 2 skip": those two tests were BROKEN, not
  skipping — `tests/test_core.py` bound `S` to `src.framing` (NumPy-only, no
  stft/istft) while two tests called `S.stft`, so they errored instead of
  skipping. Fixed. Check the policy with (Windows-only — Smart App Control has
  no Linux equivalent, so there is nothing to check or port on that platform):
  ```powershell
  (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" `
    -Name VerifiedAndReputablePolicyState).VerifiedAndReputablePolicyState
  # 0 = off, 1 = ON (enforcing), 2 = evaluation
  ```
  Turning it off is irreversible without a Windows reset. See RESUME.md.
