"""Hausa text normalisation for fair ASR scoring.

Every WER/CER number Gyara reports is computed three ways, and this module
defines the two non-raw ones:

``standard``  Removes differences that are only *encoding or formatting*:
              Unicode forms, the several ways people type hooked letters and
              apostrophes, casing, punctuation, digits vs number words,
              whitespace. A hooked-letter mistake (``kasa`` vs ``ƙasa``) still
              counts as an error, because those are different words.

``lenient``   ``standard`` plus folding hooked letters to plain ones
              (ɓ→b, ɗ→d, ƙ→k, 'y→y). Many keyboards and many ASR models cannot
              produce hooks. The gap between ``standard`` and ``lenient`` WER counts the
              hook-only disagreements. It does not say which side is wrong:
              on FLEURS the reference often omits hooks the model wrote.

The rules are versioned. Any change to them changes every benchmark number, so
bump ``RULES_VERSION`` and re-run the baselines. The written rules, with the
reason for each, are in ``docs/NORMALISATION.md``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

RULES_VERSION = "1.0.0"

# --- Hooked letters ---------------------------------------------------------

# Canonical forms (Nigerian standard orthography). Capitals are lowercased
# before this map is applied, but are listed so a case-preserving caller works.
# ƴ (Niger orthography) is written 'y in Nigeria; we canonicalise to 'y.
_HOOK_CANON: Mapping[str, str] = {
    "Ɓ": "ɓ",  # Ɓ -> ɓ
    "Ɗ": "ɗ",  # Ɗ -> ɗ
    "Ƙ": "ƙ",  # Ƙ -> ƙ
    "Ƴ": "'y",      # Ƴ -> 'y
    "ƴ": "'y",      # ƴ -> 'y
    # Look-alikes seen in the wild (copy-paste from other orthographies).
    "Ɖ": "ɗ",  # Đ (African D) -> ɗ
    "ɖ": "ɗ",  # ɖ (d with tail) -> ɗ
    "ĸ": "ƙ",  # ĸ (kra) -> ƙ
    "ǩ": "ƙ",  # ǩ -> ƙ  (rare; seen in some keyboard layouts)
}

_HOOK_FOLD: Mapping[str, str] = {
    "ɓ": "b",  # ɓ
    "ɗ": "d",  # ɗ
    "ƙ": "k",  # ƙ
}

# Apostrophe look-alikes all become ASCII '.
_APOSTROPHES = "’‘ʼʻ`´′ʹꞌ"
_APOS_TABLE = str.maketrans({c: "'" for c in _APOSTROPHES})

# --- Numbers ----------------------------------------------------------------
# Standard Hausa cardinals. Owned by the Hausa normalisation owner: see
# docs/NORMALISATION.md before changing.
_UNITS = {
    0: "sifiri", 1: "ɗaya", 2: "biyu", 3: "uku", 4: "huɗu", 5: "biyar",
    6: "shida", 7: "bakwai", 8: "takwas", 9: "tara", 10: "goma",
}
_TENS = {
    20: "ashirin", 30: "talatin", 40: "arba'in", 50: "hamsin",
    60: "sittin", 70: "saba'in", 80: "tamanin", 90: "casa'in",
}


def hausa_number(n: int) -> str:
    """Spell a non-negative integer in Hausa words (0 .. 999,999,999)."""
    if n < 0:
        raise ValueError("negative numbers are not supported")
    if n <= 10:
        return _UNITS[n]
    if n < 20:
        return f"goma sha {_UNITS[n - 10]}"
    if n < 100:
        tens, unit = divmod(n, 10)
        word = _TENS[tens * 10]
        return word if unit == 0 else f"{word} da {_UNITS[unit]}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "ɗari" if hundreds == 1 else f"ɗari {_UNITS[hundreds]}"
        return head if rest == 0 else f"{head} da {hausa_number(rest)}"
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        head = "dubu" if thousands == 1 else f"dubu {hausa_number(thousands)}"
        return head if rest == 0 else f"{head} da {hausa_number(rest)}"
    if n < 1_000_000_000:
        millions, rest = divmod(n, 1_000_000)
        head = "miliyan" if millions == 1 else f"miliyan {hausa_number(millions)}"
        return head if rest == 0 else f"{head} da {hausa_number(rest)}"
    raise ValueError("number too large to verbalise")


_NUM_RE = re.compile(r"\d+(?:[.,]\d{3})*")


def _verbalise_numbers(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        digits = re.sub(r"[.,]", "", m.group(0))
        try:
            return f" {hausa_number(int(digits))} "
        except ValueError:
            return m.group(0)

    return _NUM_RE.sub(repl, text)


# --- Punctuation ------------------------------------------------------------

# Apostrophe survives only where it is part of a word: before y at the start of
# a word ('ya'ya) or between two letters (arba'in). Everywhere else it is a
# quote mark and goes.
_APOS_KEEP = re.compile(r"(?<![\w'])'(?=y)|(?<=\w)'(?=\w)", re.UNICODE)
_SENTINEL = ""


def _strip_punctuation(text: str) -> str:
    text = _APOS_KEEP.sub(_SENTINEL, text)
    text = text.replace("'", " ")
    # Hyphens and slashes join words; treat them as spaces so "ɗan-uwa" and
    # "ɗan uwa" score the same.
    out = []
    for ch in text:
        if ch == _SENTINEL:
            out.append("'")
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _fold(word: str) -> str:
    for src, dst in _HOOK_FOLD.items():
        word = word.replace(src, dst)
    return word.replace("'y", "y")


# --- Public API -------------------------------------------------------------


@dataclass(frozen=True)
class HausaNormalizer:
    """Configurable normaliser. ``HausaNormalizer()`` is the ``standard`` mode."""

    fold_hooks: bool = False
    lowercase: bool = True
    remove_punctuation: bool = True
    verbalise_numbers: bool = True
    # Optional whole-word spelling variants -> canonical, e.g. loanwords.
    # Applied after everything else, on the normalised form.
    variants: Mapping[str, str] = field(default_factory=dict)

    def __call__(self, text: str | None) -> str:
        if not text:
            return ""
        t = unicodedata.normalize("NFC", text)
        t = t.translate(_APOS_TABLE)
        if self.lowercase:
            t = t.lower()
        for src, dst in _HOOK_CANON.items():
            t = t.replace(src, dst)
        if self.verbalise_numbers:
            t = _verbalise_numbers(t)
        if self.remove_punctuation:
            t = _strip_punctuation(t)
        words = t.split()
        if self.variants:
            # Variants are applied before folding so one map serves both modes.
            words = " ".join(self.variants.get(w, w) for w in words).split()
        if self.fold_hooks:
            words = [_fold(w) for w in words]
        return " ".join(words)

    def many(self, texts: Iterable[str | None]) -> list[str]:
        return [self(t) for t in texts]


def load_variants(path: str | Path | None) -> dict[str, str]:
    """Load a YAML map of spelling variant -> canonical form."""
    if path is None:
        return {}
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    base = HausaNormalizer()
    # Keys and values are normalised so the map works on normalised text.
    return {base(k): base(v) for k, v in (data.get("variants") or {}).items()}


def get_normalizer(mode: str, variants: Mapping[str, str] | None = None):
    """Return a callable for ``raw``, ``standard`` or ``lenient``."""
    if mode == "raw":
        return lambda s: " ".join((s or "").split())
    if mode == "standard":
        return HausaNormalizer(variants=variants or {})
    if mode == "lenient":
        return HausaNormalizer(fold_hooks=True, variants=variants or {})
    raise ValueError(f"unknown normalisation mode: {mode!r}")


MODES = ("raw", "standard", "lenient")
