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

```bash
PY=.venv/bin/python

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

### 4. Understand why deep suppression destroys words — ANSWERED

`scripts/spectrogram_diff.py`, run on take 3's `floor_18dB.wav` vs
`floor_full_model.wav` (sample-aligned to each other — same input, same
pipeline, no timing correction needed). The original hypothesis here was
"over-gating brief high-frequency events" — **that is not what the data
shows.**

Measured, using this project's own CVR bands (`scripts/intelligibility.py`:
vowel 200–800 Hz, fricative/stop 2–6 kHz):

| | CVR |
|---|---|
| clean-speech reference | −10.68 dB |
| floor-capped (−18 dB) | −12.77 dB |
| full model | **−23.79 dB** |

Going from floor-capped to full model, the fricative band loses a median
**21.2 dB** more than it already had, against **11.1 dB** more in the vowel
band — the model disproportionately attacks the exact band that carries
consonant identity, as expected. What was NOT expected: the excess-kurtosis
of that extra suppression across time is **−0.59**, i.e. close to zero /
slightly *below* Gaussian — meaning it is **not concentrated in a few loud
transient frames**. The spectrogram diff plot
(`results/spectrogram_diff.png`) shows why directly: a near-continuous
suppression band from roughly 1–7 kHz runs through almost the ENTIRE 62 s
clip, not just around gunshots. **The full-depth model is not selectively
gating loud events — it is applying a broad, near-constant, aggressive
high-frequency rolloff for the whole recording,** and word loss is the
predictable result of doing that to a band speech identity lives in. This
also explains why the floor cap works as well as it does: capping
suppression DEPTH uniformly is a reasonably well-matched fix for a
uniformly-applied problem, not a event-detection problem.

**Implication for item 1 (training objective):** this is not a "the model
needs to detect transients better" problem — the transient detection isn't
the mechanism. It is producing an over-aggressive mask across the whole
signal, all the time, in exactly the band that matters. An ASR/word-loss
term in training should therefore penalize broadband high-frequency
suppression generally, not specifically penalize behavior at burst
boundaries.

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
| Dynamic INT8 quantization of the ONNX model | 5.5% *slower* on this CPU (compute here is ONNX Runtime dispatch-bound across 445 graph nodes, not arithmetic-bound — quantizing adds dequant/quant nodes rather than removing work) and 7.3% RMS output error vs fp32. Not shipped; see `scripts/bench_edge.py`. |
| NLMS/LMS/RLS reference-mic adaptive noise cancellation (`src/baselines/{nlms,lms,rls}.py`) | Every one of them makes real audio **worse than doing nothing**, and the ranking is the OPPOSITE of algorithmic sophistication: RLS (fastest, most complete convergence) is catastrophic, NLMS is bad, the "worst" algorithm — plain LMS with a conservative fixed step — does the *least* damage, purely because it adapts too slowly to fully exploit the problem below. Root cause: the (synthetic) reference-mic channel leaks some of the talker's own speech, and every one of these filters cannot distinguish "correlated because noise" from "correlated because leaked speech" — the more thoroughly an algorithm converges, the more speech it also removes. VAD-gated adaptation (the standard real-headset fix) does not help either in this regime: an energy-based VAD on the primary mic can't tell speech from noise when the noise is this loud and impulsive, so it freezes adaptation almost entirely and the result is indistinguishable from doing nothing. **Confirmed by real ASR word-recognition, not just PESQ/STOI** (whisper-medium, 26 known tokens, `results/asr_multimic.json`): unprocessed 62%, LMS ties it at 62%, NLMS drops to 54%, **RLS scores 0% — total destruction, every single word lost.** |
| RNNoise / DeepFilterNet as drop-in replacements | Both are real, working, pretrained-weight integrations (not stubs — see `src/methods.py`), and both **also** lose to unprocessed on this project's audio (SI-SDR gain −32.8 dB and −9.4 dB respectively on the same clip NLMS/LMS/RLS were measured on). The checkpoint that wins on generic noise keeps losing to doing nothing on gunfire — this is now confirmed by four independent published/pretrained systems (RNNoise, DeepFilterNet, `gtcrn_vctk`, and classical Wiener/spectral-subtraction), not just this project's own model. |
| NLMS + this project's own GTCRN in series (the "hybrid" architecture the DSP-frontend idea points toward) | ASR word score **4%**, worse than either failure alone (NLMS 54%, GTCRN-alone 54%) — stacking two degradations compounds rather than cancels. The GTCRN model alone, single-mic, on this same clip: ASR score **54%**, again below unprocessed's 62%, consistent with the project's central finding on a fourth independent recording. |

All of the above are measured on ONE synthetic 45 s clip (`results/multimic_demo/`,
real dry speech + real gunfire, not the frozen 720-clip testset) — see
`results/method_comparison.md` for the full consolidated table (now including the
ASR column) and `results/pareto_latency_quality.png` for the compute-cost-vs-PESQ
plot. Single data point per method, but the direction (everything loses to
unprocessed, confirmed by ASR not just PESQ/STOI) is now consistent across ten
different methods spanning classical DSP, adaptive filtering, and three
independent neural architectures.

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

```bash
$PY scripts/asr_score.py --model medium --inputs your_dry_take.wav
```

A good dry take should score well above 70%. If it does not, fix the capture
before anything else.

---

## Environment: PyTorch availability

Torch 2.13.0+cu126 loads and trains normally on this machine, GPU included.

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

## Files added in the Linux-migration / multi-mic session

| path | what |
|---|---|
| `main.py` | live mic→model→speaker entry point, wraps `src.stream_demo`, crash-proof fallback |
| `requirements.txt` | grouped, version-pinned dependencies (core/training/metrics/baselines) |
| `dhwanik.service`, `scripts/run_service.sh`, `dhwanik.env.example` | systemd **user** service — self-bootstraps `.venv`, installs deps only when `requirements.txt` changes, execs `main.py` |
| `scripts/bench_edge.py` | torch-free latency/RTF benchmark, safe to copy to an actual embedded target |
| `src/baselines/{nlms,lms,rls}.py`, `src/baselines/reference_mic.py` | reference-mic adaptive filters + synthetic second-channel model — see negative results above |
| `scripts/eval_multimic.py` | builds the synthetic two-mic mixture, runs all DSP/neural/hybrid methods, scores them |
| `scripts/_verify_new_baselines.py`, `src/methods.py` additions | RNNoise + DeepFilterNet, real pretrained weights (DeepFilterNet lives in an isolated `.venv-dfn` — numpy version conflict, see `requirements.txt`) |
| `scripts/aggregate_results.py` | consolidates every method measured on the same clip into `results/method_comparison.md` + `results/pareto_latency_quality.png` |
| `results/asr_multimic.json` | the ASR numbers in the table above |
| `scripts/make_listening_test.py`, `test-result/listening_test_v2/` | replacement listening-test kit — the original `test-result/listening_test/TEST_A.wav` has no recorded provenance anywhere in this repo (checked against every real recording and floor-sweep variant, no match), so a score against it can't be interpreted. This version uses the take-3 floor-sweep files (known provenance), independently randomized per listener. **Caveat:** those source files are also what README's own demo section asks teammates to listen to, so a listener who's used this repo already isn't blind. |
| `tests/test_nlms.py`, `tests/test_lms_rls.py`, `tests/test_streaming_safety.py` | 10 new tests, all passing (30/30 total, up from 19) |
| `scripts/spectrogram_diff.py`, `results/spectrogram_diff.png` | answers "where to pick up" item 4 below — the suppression mechanism |

---

## Rebuilding from nothing

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

```bash
pkill -f python
```

---

## Where things live

| what | where |
|---|---|
| code | this repository |
| venv | `.venv` at the repo root (Python 3.12, **not** 3.14) |
| datasets | `~/SIH26052_data/{raw,prepared}` (~64 GB) |
| frozen test set | `~/SIH26052_data/testset` |
| VoiceBank-DEMAND | `~/SIH26052_data/voicebank_demand` |
| real test recordings | `test-result/voice/` |
| processed variants + scores | `test-result/` |
