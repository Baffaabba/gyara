"""SRT and WebVTT export.

Segments from the transcriber can be up to 30 s long, which is far too long for
a subtitle. Cues are split to broadcast-friendly limits (42 characters per line,
2 lines, 7 seconds), timing words proportionally to their length when real
word timestamps are not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CUE_SECONDS = 7.0
MIN_CUE_SECONDS = 0.7


@dataclass
class Cue:
    start: float
    end: float
    text: str


@dataclass
class Word:
    text: str
    start: float
    end: float


def estimate_word_times(text: str, start: float, end: float) -> list[Word]:
    """Spread a segment's duration over its words, weighted by length."""
    words = text.split()
    if not words:
        return []
    weights = [len(w) + 1 for w in words]
    total = sum(weights)
    span = max(end - start, 0.0)
    out, t = [], start
    for w, wt in zip(words, weights):
        d = span * wt / total
        out.append(Word(w, t, t + d))
        t += d
    out[-1].end = end
    return out


def _wrap(words: Sequence[str], width: int = MAX_CHARS_PER_LINE) -> list[str]:
    lines, cur = [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if len(cand) <= width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def build_cues(segments: Iterable[dict]) -> list[Cue]:
    """Turn segments ``{start, end, text, words?}`` into subtitle cues."""
    cues: list[Cue] = []
    max_chars = MAX_CHARS_PER_LINE * MAX_LINES
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        words = [Word(**w) for w in seg["words"]] if seg.get("words") else estimate_word_times(
            text, float(seg["start"]), float(seg["end"])
        )
        group: list[Word] = []
        for w in words:
            cand = " ".join(x.text for x in group + [w])
            too_long = len(cand) > max_chars or (group and w.end - group[0].start > MAX_CUE_SECONDS)
            if group and too_long:
                cues.append(_cue(group))
                group = []
            group.append(w)
        if group:
            cues.append(_cue(group))
    # Enforce a minimum display time without overlapping the next cue.
    for i, c in enumerate(cues):
        if c.end - c.start < MIN_CUE_SECONDS:
            limit = cues[i + 1].start if i + 1 < len(cues) else c.start + MIN_CUE_SECONDS
            c.end = max(c.end, min(c.start + MIN_CUE_SECONDS, limit))
    return cues


def _cue(words: list[Word]) -> Cue:
    lines = _wrap([w.text for w in words])
    if len(lines) > MAX_LINES:  # very long single words; keep everything visible
        lines = [" ".join(lines[:-1]), lines[-1]]
    return Cue(words[0].start, words[-1].end, "\n".join(lines))


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(cues: Iterable[Cue]) -> str:
    blocks = [
        f"{i}\n{_ts(c.start, ',')} --> {_ts(c.end, ',')}\n{c.text}"
        for i, c in enumerate(cues, start=1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(cues: Iterable[Cue]) -> str:
    blocks = [f"{_ts(c.start, '.')} --> {_ts(c.end, '.')}\n{c.text}" for c in cues]
    return "WEBVTT\n\n" + "\n\n".join(blocks) + ("\n" if blocks else "")
