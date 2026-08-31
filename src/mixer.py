"""Mixture synthesis.

This module is the single most important piece of the project. The roadmap doc
is emphatic on one point, and it is correct:

    Gunshots must be dropped in as sudden, separate bursts - not blended evenly
    through the clip. Even blending teaches the model only the easy cases, and
    it will then fail on exactly the sound DRDO cares about most.

So the mixture is built in two distinct layers:

  * a STEADY background (engine, wind, babble, music) mixed at a global SNR
    measured over active-speech frames only, and
  * IMPULSIVE bursts (gunfire, explosions) placed at random offsets as discrete
    events, scaled by PEAK level and routinely louder than the speech itself.

Every burst's sample range is recorded in a transient mask, which the training
loss uses to weight those frames harder. That mask is the structural answer to
the doc's number-one risk ("gunshots still get through").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import fftconvolve, lfilter

from . import audio as A


@dataclass
class MixResult:
    noisy: np.ndarray           # what the model sees
    target: np.ndarray          # what it should produce
    transient_mask: np.ndarray  # sample-level bool, True inside burst events
    meta: dict                  # provenance: snr, n_bursts, category, gains...


class Mixer:
    def __init__(self, cfg: dict, sr: int = A.SR):
        self.sr = sr
        self.cfg = cfg
        m = cfg["mixture"]
        self.n = int(round(m["dur_s"] * sr))
        self.min_active = float(m["min_active_speech"])
        self.out_peak = tuple(m["out_peak"])
        self.bg = cfg["background"]
        self.burst = cfg["bursts"]
        self.rev = cfg["reverb"]
        self.lim = cfg["limiter"]

    # ------------------------------------------------------------------ parts

    def _crop_speech(self, rng, speech: np.ndarray, tries: int = 8):
        """Random crop with enough voice in it to be worth training on."""
        if len(speech) < self.n:
            speech = A.fit_length(speech, self.n, rng, loop=False)
        best, best_ratio = None, -1.0
        for _ in range(tries):
            s = int(rng.integers(0, max(1, len(speech) - self.n + 1)))
            c = speech[s:s + self.n]
            ratio = float(A.active_speech_mask(c, self.sr).mean())
            if ratio > best_ratio:
                best, best_ratio = c, ratio
            if ratio >= self.min_active:
                return np.ascontiguousarray(c)
        return np.ascontiguousarray(best) if best_ratio > 0.25 else None

    def _reverb(self, x: np.ndarray, rir: np.ndarray):
        """Return (wet, early). `early` is the training target.

        We denoise, we do not dereverberate: asking a 48k-parameter causal model
        to also remove late reverb is a different and much harder problem, and
        the doc does not ask for it. The target keeps direct path + early
        reflections (first `early_ms`), the DNS-challenge convention.

        Both are aligned on the direct-path peak so the pair introduces no
        artificial delay - a misalignment here would teach the model to
        time-shift its output, which then shows up as extra latency on hardware.
        """
        rir = np.asarray(rir, dtype=np.float32)
        peak = int(np.argmax(np.abs(rir)))
        rir = rir / (np.max(np.abs(rir)) + 1e-12)
        early_n = peak + int(self.sr * float(self.rev["early_ms"]) / 1000.0)
        wet = fftconvolve(x, rir)[peak:peak + len(x)]
        early = fftconvolve(x, rir[:early_n])[peak:peak + len(x)]
        wet = A.fit_length(wet.astype(np.float32), len(x), None, loop=False)
        early = A.fit_length(early.astype(np.float32), len(x), None, loop=False)
        return wet, early

    def _background(self, rng, clips: Sequence[np.ndarray]) -> np.ndarray:
        if not len(clips):
            return np.zeros(self.n, dtype=np.float32)
        lo, hi = self.bg["n_sources"]
        k = min(len(clips), int(rng.integers(lo, hi + 1)))
        idx = rng.choice(len(clips), size=k, replace=False)
        out = np.zeros(self.n, dtype=np.float32)
        for i in idx:
            out += A.fit_length(clips[i], self.n, rng, loop=True)
        return out

    def _augment_event(self, rng, ev: np.ndarray) -> np.ndarray:
        """Vary a transient without blunting its attack.

        Time-stretch is done by plain resampling, which shifts pitch too - for a
        gunshot that is physically reasonable (different calibre, different
        distance) and it is what keeps ~70 artillery clips from being memorised.

        Deliberately NO fade-in and no envelope smoothing: the near-instant
        attack is the feature that defeats a stationary-noise filter, and
        softening it would quietly make the training data easier than reality.
        """
        aug = self.burst.get("augment") or {}
        if not aug.get("enabled"):
            return ev

        if aug.get("polarity_flip") and rng.random() < 0.5:
            ev = -ev

        lo, hi = aug.get("time_stretch", (1.0, 1.0))
        if hi > lo:
            rate = float(rng.uniform(lo, hi))
            n_new = max(16, int(round(ev.size / rate)))
            ev = np.interp(np.linspace(0, ev.size - 1, n_new),
                           np.arange(ev.size), ev).astype(np.float32)

        if rng.random() < float(aug.get("lowpass_prob", 0.0)):
            f_lo, f_hi = aug.get("lowpass_hz", (3000, 7000))
            cutoff = float(rng.uniform(f_lo, f_hi))
            # One-pole low-pass via lfilter, not a Python loop: this runs on
            # every burst of every training sample, and the loop version costs
            # more than the entire rest of the mixer.
            alpha = float(np.exp(-2.0 * np.pi * cutoff / self.sr))
            ev = lfilter([1.0 - alpha], [1.0, -alpha], ev).astype(np.float32)

        g_lo, g_hi = aug.get("gain_db", (0.0, 0.0))
        if g_hi > g_lo:
            ev = ev * float(10.0 ** (rng.uniform(g_lo, g_hi) / 20.0))
        return ev.astype(np.float32)

    def _bursts(self, rng, clips: Sequence[np.ndarray], speech_rms: float,
                force: bool = False):
        """Place discrete impulsive events. Returns (track, sample_mask, events).

        Note there is deliberately NO fade-in on an event: a gunshot's attack is
        the part the model has to learn, and tapering it would sand off exactly
        the feature that defeats a conventional filter.

        `force` guarantees at least one event. It is set for impulsive
        CATEGORIES, because a clip filed under "gunshot" that happens to contain
        no gunshot would be scored as a gunshot result - inflating exactly the
        category we most need to be honest about.
        """
        track = np.zeros(self.n, dtype=np.float32)
        mask = np.zeros(self.n, dtype=bool)
        events: list[dict] = []
        if not len(clips):
            return track, mask, events

        k = int(rng.poisson(float(self.burst["lambda"])))
        if k == 0 and (force or rng.random() < float(self.burst["force_min_prob"])):
            k = 1
        max_len = int(self.sr * float(self.burst["max_event_s"]))
        pad = int(self.sr * float(self.burst["onset_pad_ms"]) / 1000.0)
        lo_snr, hi_snr = self.burst["peak_snr_db"]

        for _ in range(k):
            clip = clips[int(rng.integers(0, len(clips)))]
            # Crop from the actual acoustic onset, not the file start: transient
            # recordings routinely carry seconds of leading silence, and placing
            # by file offset would scatter the impulse anywhere in the mixture.
            on = max(0, A.find_onset(clip) - pad)
            ev = clip[on:on + max_len]
            if ev.size < 16:
                continue
            ev = self._augment_event(rng, ev)
            if ev.size < 16:
                continue
            snr_db = float(rng.uniform(lo_snr, hi_snr))
            g = A.scale_burst_for_peak_snr(speech_rms, ev, snr_db)
            ev = (ev * g).astype(np.float32)

            start = int(rng.integers(0, max(1, self.n - 1)))
            end = min(self.n, start + ev.size)
            track[start:end] += ev[:end - start]
            mask[start:end] = True
            events.append({"start": start, "end": end, "peak_snr_db": round(snr_db, 2)})
        return track, mask, events

    # ------------------------------------------------------------------ build

    def build(
        self,
        rng: np.random.Generator,
        speech: np.ndarray,
        bg_clips: Sequence[np.ndarray] = (),
        burst_clips: Sequence[np.ndarray] = (),
        rir_speech=None,
        rir_noise=None,
        category: str = "mixed",
        force_burst: bool = False,
    ):
        crop = self._crop_speech(rng, speech)
        if crop is None:
            return None

        # 1. room
        use_rev = rir_speech is not None and rng.random() < float(self.rev["prob"])
        if use_rev:
            wet, target = self._reverb(crop, rir_speech)
        else:
            wet, target = crop, crop.copy()

        speech_rms = A.active_rms(wet, self.sr)
        if speech_rms <= 1e-9:
            return None

        # 2. steady background at a global, active-speech-referenced SNR
        bg = self._background(rng, bg_clips)
        snr_db = float(rng.uniform(*self.bg["snr_db"]))
        if bg.any():
            bg = bg * A.scale_noise_for_snr(speech_rms, bg, snr_db)

        # 3. impulsive bursts, scaled by PEAK - these are meant to be loud
        bursts, mask, events = self._bursts(rng, burst_clips, speech_rms,
                                            force=force_burst)

        # Recorded so QA can measure burst prominence against the background
        # directly, rather than inferring it from the summed residual (which the
        # background dominates, since MUSAN's noise set has impulsive content of
        # its own).
        burst_peak = float(np.abs(bursts).max()) if bursts.any() else 0.0
        bg_peak = float(np.abs(bg).max()) if bg.any() else 0.0

        noise = bg + bursts
        if use_rev and rir_noise is not None and noise.any():
            # Different RIR for the noise: the rifle is not at the mouth.
            noise, _ = self._reverb(noise, rir_noise)

        noisy = wet + noise

        # 4. analog limiter in front of the ADC (hardware team's front end)
        applied_lim = None
        if rng.random() < float(self.lim["prob"]):
            applied_lim = float(rng.uniform(*self.lim["threshold"]))
            noisy = A.soft_limit(noisy, applied_lim)

        # 5. common gain. The target is scaled by the SAME factor so the pair
        #    stays consistent - scaling independently would teach a gain offset,
        #    and this model is not supposed to change level at all.
        _, g = A.peak_normalise(noisy, float(rng.uniform(*self.out_peak)))
        noisy = np.clip(noisy * g, -1.0, 1.0).astype(np.float32)
        target = np.clip(target * g, -1.0, 1.0).astype(np.float32)

        return MixResult(
            noisy=noisy,
            target=target,
            transient_mask=mask,
            meta={
                "category": category,
                "bg_snr_db": round(snr_db, 2),
                "n_bursts": len(events),
                "events": events,
                "reverb": bool(use_rev),
                "limiter": applied_lim,
                "gain": round(float(g), 5),
                "burst_peak": burst_peak,
                "bg_peak": bg_peak,
            },
        )
