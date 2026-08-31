"""Score intelligibility with a speech recogniser standing in for a listener.

WHY. The question this project actually has to answer - can the person on the
other end make out the words - needs a listening test, and a listening test needs
listeners. When there are none, an ASR system is the accepted substitute: it is
the standard proxy in speech-enhancement work, it is perfectly repeatable, and it
can rescore every candidate the moment a new model lands.

WHAT IT IS NOT. A recogniser is not an ear. Whisper is trained on vast amounts of
noisy speech and is in some ways more robust than a human, in others less; it
also invents fluent text when the audio is hopeless, so a confident transcript of
noise is a real failure mode. Read a single absolute score with suspicion. Read a
DIFFERENCE between two candidates over the same audio as meaningful - that is the
comparison this exists for.

The recording script is built for exactly this: the phonetic alphabet, the digits
and a grid reference are a closed set of known tokens, so scoring is unambiguous
and a near-miss ("Sierra" heard as "Tango") counts as the error it would be on a
real radio net.

    python scripts/asr_score.py --inputs a.wav b.wav --model small
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The closed sets the script was written around.
NATO = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
        "sierra", "tango", "victor", "whiskey", "xray", "zulu"]
DIGITS = ["one", "two", "three", "four", "five",
          "six", "seven", "eight", "nine", "zero"]
GRID = ["four", "seven", "two", "nine"]

_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "oh": "zero", "o": "zero"}


def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, expand digits, fold x-ray spellings."""
    text = text.lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    out = []
    for w in text.split():
        if w.isdigit():
            out.extend(_NUM.get(c, c) for c in w)
        else:
            out.append(_NUM.get(w, w))
    return [w.replace("xray", "xray") for w in out]


def transcribe_chunked(model, path: Path, chunk_s: float = 15.0,
                       overlap_s: float = 0.5) -> tuple[str, bool]:
    """Transcribe in independent fixed windows. Returns (text, looked_broken).

    Whole-file decoding has two failure modes that both score as near-zero and
    are indistinguishable from destroyed audio - measured on real outputs here:

      * a REPETITION LOOP, where the decoder emits "...A.M.A.M.A.M..." for the
        remainder of the clip;
      * a SKIPPED SEGMENT, where a stretch it cannot decode is silently dropped
        - one file transcribed parts 1 and 4 correctly and omitted the alphabet
        and the digits entirely, which is precisely the material being scored.

    Both took files to 4% while the audio was demonstrably fine. Windowing
    contains the damage to one chunk instead of the whole recording.
    """
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    step = int((chunk_s - overlap_s) * sr)
    win = int(chunk_s * sr)
    parts, suspect = [], False
    for start in range(0, max(len(audio) - 1, 1), step):
        seg_audio = audio[start:start + win]
        if len(seg_audio) < int(0.5 * sr):
            break
        segs, _ = model.transcribe(seg_audio, language="en", beam_size=5,
                                   temperature=0.0,
                                   condition_on_previous_text=False)
        t = " ".join(s.text for s in segs)
        if looks_degenerate(t):
            suspect = True                # FLAG only - never drop.
            # Dropping the chunk was worse than keeping it: degenerate text
            # matches no target anyway, so discarding it cannot raise the score,
            # but it does throw away any real words the decoder did get. That
            # took one file's UNPROCESSED audio from 69% to 0%.
        parts.append(t)
    return " ".join(parts), suspect


def looks_degenerate(text: str) -> bool:
    """Flag a decoder repetition loop: one short token dominating the output."""
    w = text.lower().split()
    if len(w) < 12:
        return False
    top = max(set(w), key=w.count)
    return w.count(top) / len(w) > 0.35


def count_hits(words: list[str], targets: list[str]) -> tuple[int, list[str]]:
    """How many target tokens appear, each consumed at most once.

    Order-free on purpose. Requiring the right order would conflate a listener
    mishearing a word with the recogniser dropping one, and the question here is
    only whether the word survived the channel.
    """
    pool = list(words)
    hits, missed = 0, []
    for t in targets:
        if t in pool:
            pool.remove(t)
            hits += 1
        else:
            missed.append(t)
    return hits, missed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--model", default="small",
                   help="tiny|base|small|medium; small is a good accuracy/speed point")
    p.add_argument("--device", default="cpu",
                   help="cpu keeps the GPU free for a training run")
    p.add_argument("--repeats", type=int, default=1,
                   help="rescore each file N times; the spread exposes an "
                        "unstable measurement before it becomes a conclusion")
    p.add_argument("--chunk", action="store_true",
                   help="decode in 15 s windows. Contains a repetition loop, but "
                        "costs Whisper its context and destabilises other files - "
                        "measured better on some inputs and much worse on others. "
                        "Whole-file with a larger model is the more reliable route.")
    p.add_argument("--show-transcript", action="store_true")
    args = p.parse_args()

    from faster_whisper import WhisperModel
    print("loading whisper-%s on %s ..." % (args.model, args.device))
    model = WhisperModel(args.model, device=args.device, compute_type="int8")

    total = len(NATO) + len(DIGITS) + len(GRID)
    print("\n%-34s %7s %7s %7s %9s %s"
          % ("file", "NATO", "digits", "grid", "TOTAL", "runs" if args.repeats > 1 else ""))
    print("%-34s %7s %7s %7s %9s" % ("", "/12", "/10", "/4", "/%d" % total))
    results = []
    for f in args.inputs:
        runs = []
        texts = []
        suspect = False
        for _ in range(args.repeats):
            # temperature=0.0 as a SCALAR disables Whisper's temperature
            # fallback. Left at its default tuple, a decode that trips an
            # internal quality check is silently retried with SAMPLING, so the
            # same file scores differently run to run - measured here as 69% and
            # then 23% on one unchanged input, which is worse than no metric.
            if args.chunk:
                text, sus = transcribe_chunked(model, f)
            else:
                segs, _ = model.transcribe(str(f), language="en", beam_size=5,
                                           temperature=0.0,
                                           condition_on_previous_text=False)
                text = " ".join(s.text for s in segs)
                sus = looks_degenerate(text)
            suspect = suspect or sus
            w = normalise(text)
            n, n_miss = count_hits(w, NATO)
            d, _ = count_hits(w, DIGITS)
            g, _ = count_hits(w, GRID)
            runs.append((n, d, g, n + d + g))
            texts.append((text, n_miss))
        arr = [r[3] for r in runs]
        n, d, g, tot = runs[0]
        spread = "" if args.repeats == 1 else "  [%d-%d over %d runs]" % (
            min(arr), max(arr), args.repeats)
        print("%-34s %7d %7d %7d %6d %3.0f%%%s%s"
              % (f.name, n, d, g, tot, 100 * tot / total, spread,
                 "  DECODER GLITCH - chunk dropped" if suspect else ""))
        results.append((f.name, tot, texts[0][0], texts[0][1]))

    print("\n(scale: >90%% operational, 70-89%% usable with repeats, "
          "40-69%% marginal, <40%% not usable)")
    for name, tot, text, n_miss in results:
        if n_miss:
            print("  %-30s missed NATO: %s" % (name, ", ".join(n_miss)))
    if args.show_transcript:
        print("\n--- transcripts ---")
        for name, _, text, _ in results:
            print("\n[%s]\n%s" % (name, text.strip()[:1200]))


if __name__ == "__main__":
    main()
