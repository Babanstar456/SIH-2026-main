"""Build a proper blind A/B/C listening-test kit.

Why this exists: the previous kit (`test-result/listening_test/TEST_A.wav`)
has no recorded provenance anywhere in this repo or its scripts - nothing
says which processing condition produced it (unprocessed? floor-capped?
full model?). A score against it cannot be interpreted, which is exactly the
kind of measurement trap CLAUDE.md warns about elsewhere in this project. It
is left in place untouched (its origin is unknown, not necessarily wrong),
but this script builds a REPLACEMENT with fully known provenance instead.

Uses the three take-3 floor-sweep recordings, which already have documented
provenance throughout this project (README/RESUME): real gunfire, real
microphone, -5.9 dB input SNR.

  unprocessed       <- test-result/floors/floor_none_unprocessed.wav
  floor -18 dB       <- test-result/floors/floor_18dB.wav      (current best)
  full model         <- test-result/floors/floor_full_model.wav

CAVEAT, read before administering: these three files are the exact ones the
README's "test this in 10 minutes" section asks teammates to listen to. A
listener who has already gone through that section has heard these clips and
is no longer blind. Pick listeners who have not used this repo, or record a
fresh take-4 through the same three settings first.

Generates one packet per listener with the three conditions independently
shuffled (a different random order per listener, so listeners cannot compare
notes to reverse-engineer which is which), plus ONE master key file - never
shown to a listener - that maps every listener's shuffled labels back to the
real condition and holds the scoring key.

    python scripts/make_listening_test.py --listeners 3 --seed 20260901
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "test-result" / "floors"
CONDITIONS = {
    "unprocessed": "floor_none_unprocessed.wav",
    "floor_-18dB": "floor_18dB.wav",
    "full_model": "floor_full_model.wav",
}

ANSWER_SHEET_TEMPLATE = """\
LISTENING TEST - ANSWER SHEET
=============================
Listener name: ______________________   Date: ____________

RULES
  * Do NOT read the recording script beforehand. If you know the answers
    the test measures nothing.
  * Headphones. Play each file ONCE, straight through, in the order below.
  * Write what you HEAR. Guessing is fine. Leave blanks if you hear nothing.
  * Do not replay, pause or scrub.
  * The three recordings are the SAME speech under different processing.
    You are not told which is which - that is the point.

{sections}
PART D - after all three, rank them
   Which recording let you follow the most words?  1st: ___  2nd: ___  3rd: ___
"""

SECTION_TEMPLATE = """\
=== Recording {label} ({filename}) ===

PART A - phonetic alphabet (12 words, in order)
   1. ____________     5. ____________     9. ____________
   2. ____________     6. ____________    10. ____________
   3. ____________     7. ____________    11. ____________
   4. ____________     8. ____________    12. ____________

PART B - digits (10, in order)
   ___  ___  ___  ___  ___  ___  ___  ___  ___  ___

PART C - the grid reference (4 digits)
   "Position secured at grid  ___  ___  ___  ___"

"""

KEY_ANSWERS = """\
PART A  1 Alpha   2 Bravo   3 Charlie  4 Delta
        5 Echo    6 Foxtrot 7 Sierra   8 Tango
        9 Victor 10 Whiskey 11 X-ray  12 Zulu
PART B  1 2 3 4 5 6 7 8 9 0     (spoken "one ... nine, zero")
PART C  4 7 2 9

SCORING
  One mark per item, 26 total per recording. Spelling does not matter; the
  word does. Near-misses count as WRONG - "Sierra" heard as "Tango" is
  exactly the confusion this test exists to detect.

INTERPRETING THE TOTAL
   90-100%   operational
   70-89%    usable with repeats
   40-69%    marginal - this is the "I catch some of it" band
   under 40% not usable
"""


def build_listener_packet(listener_id: int, rng: random.Random, out_root: Path) -> dict:
    conditions = list(CONDITIONS.items())
    rng.shuffle(conditions)
    labels = ["1", "2", "3"]

    listener_dir = out_root / f"listener_{listener_id}"
    listener_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    sections = []
    for label, (condition_name, src_filename) in zip(labels, conditions):
        src = SOURCE_DIR / src_filename
        dst_name = f"Recording_{label}.wav"
        shutil.copyfile(src, listener_dir / dst_name)
        mapping[label] = condition_name
        sections.append(SECTION_TEMPLATE.format(label=label, filename=dst_name))

    answer_sheet = ANSWER_SHEET_TEMPLATE.format(sections="\n".join(sections))
    (listener_dir / "ANSWER_SHEET.txt").write_text(answer_sheet, encoding="utf-8")
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listeners", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="test-result/listening_test_v2")
    a = ap.parse_args()

    for name, fname in CONDITIONS.items():
        src = SOURCE_DIR / fname
        if not src.exists():
            raise SystemExit(f"missing source file: {src}")

    out_root = ROOT / a.out
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    key_lines = [
        "MASTER KEY - tester only. Never show this to a listener.",
        "=" * 60,
        f"Source: {SOURCE_DIR.relative_to(ROOT)} (take 3, real gunfire, -5.9 dB input SNR)",
        f"Seed: {a.seed}",
        "",
    ]
    for i in range(1, a.listeners + 1):
        mapping = build_listener_packet(i, rng, out_root)
        key_lines.append(f"listener_{i}/:")
        for label in ["1", "2", "3"]:
            key_lines.append(f"    Recording_{label}.wav -> {mapping[label]}")
        key_lines.append("")

    key_lines.append(KEY_ANSWERS)
    (out_root / "MASTER_KEY_tester_only.txt").write_text("\n".join(key_lines), encoding="utf-8")

    print(f"built {a.listeners} listener packet(s) in {out_root}")
    print(f"each has 3 shuffled recordings + a blank ANSWER_SHEET.txt")
    print(f"master key (mapping + scoring): {out_root / 'MASTER_KEY_tester_only.txt'}")
    print("\nHand each listener_N/ folder to a different listener. Do not let them "
          "see MASTER_KEY_tester_only.txt or each other's folders beforehand.")


if __name__ == "__main__":
    main()
