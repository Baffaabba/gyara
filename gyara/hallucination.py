"""Detect Whisper repetition loops and runaway output.

Whisper-family models sometimes loop ("na gode na gode na gode ...") on noise,
music or long silence. Two signals, both cheap and model-free:

* gzip compression ratio of the text. Whisper itself treats > 2.4 as a failed
  decode; repetitive text compresses unusually well.
* the longest run of a repeated n-gram (n = 1..4).
* a run of the same short character unit inside a "word". Small models loop
  without spaces ("21.5.5.5.5...", "Gb2.2.2.2...", "ggbhpppppp..."), which is
  one whitespace token, so the word-level checks alone never see it. Found in
  the whisper-tiny FLEURS smoke run (9 Oct 2026): three such outputs went
  through unflagged, and the standard normaliser then read every digit as a
  Hausa number word, turning one loop into ~90 word errors.

A flagged segment is retried with temperature fallback by the transcriber and,
if still flagged, is trimmed and marked for human review. It is never silently
accepted.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

COMPRESSION_RATIO_THRESHOLD = 2.4
MAX_NGRAM_REPEATS = 4  # the same n-gram 4+ times in a row is a loop
MAX_WORDS_PER_SECOND = 6.0  # Hausa speech is ~2-3.5 words/s; 6 is implausible
# Compression ratio is meaningless on very short text; gate on words OR characters.
MIN_WORDS_FOR_RATIO = 8
MIN_CHARS_FOR_RATIO = 40
# A 1-10 character unit repeated this many times in a row is a loop, whitespace or not.
CHAR_UNIT_MAX = 10
CHAR_REPEATS = 10
_CHAR_RUN = re.compile(r"(.{1,%d}?)\1{%d,}" % (CHAR_UNIT_MAX, CHAR_REPEATS - 1), re.DOTALL)


def compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(raw) / len(zlib.compress(raw))


def longest_repeat(words: list[str], max_n: int = 4) -> tuple[int, int, int]:
    """Return (count, n, start) of the longest consecutive n-gram repetition."""
    best = (1, 1, 0)
    for n in range(1, max_n + 1):
        i = 0
        while i + 2 * n <= len(words):
            gram = words[i : i + n]
            reps, j = 1, i + n
            while words[j : j + n] == gram:
                reps += 1
                j += n
            if reps > best[0]:
                best = (reps, n, i)
            i += 1 if reps == 1 else n * (reps - 1)
    return best


def longest_char_run(text: str) -> tuple[int, str]:
    """(repeats, unit) of the longest run of one short character unit."""
    best = (1, "")
    for m in _CHAR_RUN.finditer(text):
        unit = m.group(1)
        reps = len(m.group(0)) // len(unit)
        if reps > best[0]:
            best = (reps, unit)
    return best


@dataclass
class LoopCheck:
    flagged: bool
    compression_ratio: float
    repeat_count: int
    repeat_n: int
    words_per_second: float | None
    reason: str = ""


def check(text: str, duration_s: float | None = None) -> LoopCheck:
    words = text.split()
    cr = compression_ratio(text)
    reps, n, _ = longest_repeat([w.lower() for w in words])
    wps = len(words) / duration_s if duration_s and duration_s > 0 else None
    reasons = []
    if cr > COMPRESSION_RATIO_THRESHOLD and (
        len(words) >= MIN_WORDS_FOR_RATIO or len(text.strip()) >= MIN_CHARS_FOR_RATIO
    ):
        reasons.append(f"compression ratio {cr:.2f}")
    if reps >= MAX_NGRAM_REPEATS:
        reasons.append(f"'{' '.join(words[:n])}...' repeated {reps}x" if n else "repeat")
    c_reps, unit = longest_char_run(text)
    if c_reps >= CHAR_REPEATS:
        reasons.append(f"characters {unit!r} repeated {c_reps}x")
    if wps is not None and wps > MAX_WORDS_PER_SECOND and len(words) >= 8:
        reasons.append(f"{wps:.1f} words/s")
    return LoopCheck(bool(reasons), cr, reps, n, wps, "; ".join(reasons))


def trim_loop(text: str, keep: int = 2) -> str:
    """Collapse the longest repeated n-gram run, and any character-unit run,
    to ``keep`` repetitions."""
    text = _CHAR_RUN.sub(lambda m: m.group(1) * keep, text)
    words = text.split()
    reps, n, start = longest_repeat([w.lower() for w in words])
    if reps < MAX_NGRAM_REPEATS:
        return text
    head = words[: start + n * keep]
    tail = words[start + n * reps :]
    return " ".join(head + tail)
