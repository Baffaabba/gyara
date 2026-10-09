"""Word and character error rates, with the counts needed for honest statistics.

Corpus WER is *pooled*: total edits divided by total reference words. It is not
the mean of per-utterance WERs, which overweights short clips (a 2-word clip with
one error counts as much as a 40-word clip with twenty). Per-utterance counts are
kept so ``gyara.stats`` can bootstrap confidence intervals from them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Sequence

from rapidfuzz.distance import Levenshtein


@dataclass
class EditCounts:
    """Edit counts for one reference/hypothesis pair at one unit (word or char)."""

    ref_len: int = 0
    hyp_len: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Error rate. ``nan`` when the reference is empty (undefined)."""
        return self.errors / self.ref_len if self.ref_len else float("nan")

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            self.ref_len + other.ref_len,
            self.hyp_len + other.hyp_len,
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
        )


def _counts(ref: Sequence, hyp: Sequence) -> tuple[EditCounts, list[tuple[str, str, str]]]:
    ops = Levenshtein.editops(ref, hyp)
    c = EditCounts(ref_len=len(ref), hyp_len=len(hyp))
    pairs: list[tuple[str, str, str]] = []
    for op in ops:
        if op.tag == "replace":
            c.substitutions += 1
            pairs.append(("S", ref[op.src_pos], hyp[op.dest_pos]))
        elif op.tag == "delete":
            c.deletions += 1
            pairs.append(("D", ref[op.src_pos], ""))
        elif op.tag == "insert":
            c.insertions += 1
            pairs.append(("I", "", hyp[op.dest_pos]))
    return c, pairs


def word_counts(ref: str, hyp: str) -> tuple[EditCounts, list[tuple[str, str, str]]]:
    return _counts(ref.split(), hyp.split())


def char_counts(ref: str, hyp: str) -> EditCounts:
    # Characters of the space-collapsed string, spaces included: the usual CER.
    r, h = " ".join(ref.split()), " ".join(hyp.split())
    return _counts(r, h)[0]


def wer(ref: str, hyp: str) -> float:
    return word_counts(ref, hyp)[0].rate


def cer(ref: str, hyp: str) -> float:
    return char_counts(ref, hyp).rate


@dataclass
class UtteranceScore:
    id: str
    ref: str
    hyp: str
    words: EditCounts
    chars: EditCounts
    group: str | None = None  # speaker / sentence / dialect: the bootstrap cluster

    def to_dict(self) -> dict:
        d = asdict(self)
        d["wer"] = self.words.rate
        d["cer"] = self.chars.rate
        return d


@dataclass
class CorpusScore:
    utterances: list[UtteranceScore] = field(default_factory=list)
    confusions: Counter = field(default_factory=Counter)

    @property
    def words(self) -> EditCounts:
        total = EditCounts()
        for u in self.utterances:
            total = total + u.words
        return total

    @property
    def chars(self) -> EditCounts:
        total = EditCounts()
        for u in self.utterances:
            total = total + u.chars
        return total

    @property
    def wer(self) -> float:
        return self.words.rate

    @property
    def cer(self) -> float:
        return self.chars.rate

    @property
    def n_scored(self) -> int:
        return sum(1 for u in self.utterances if u.words.ref_len > 0)

    def summary(self) -> dict:
        w, c = self.words, self.chars
        return {
            "utterances": len(self.utterances),
            "utterances_scored": self.n_scored,
            "ref_words": w.ref_len,
            "wer": w.rate,
            "cer": c.rate,
            "substitutions": w.substitutions,
            "deletions": w.deletions,
            "insertions": w.insertions,
        }

    def top_confusions(self, n: int = 25) -> list[dict]:
        return [
            {"op": op, "ref": r, "hyp": h, "count": k}
            for (op, r, h), k in self.confusions.most_common(n)
        ]


def score_corpus(
    ids: Iterable[str],
    refs: Iterable[str],
    hyps: Iterable[str],
    normalizer: Callable[[str], str] | None = None,
    groups: Iterable[str | None] | None = None,
) -> CorpusScore:
    """Score aligned lists of references and hypotheses.

    Utterances whose normalised reference is empty are kept (so insertions on
    silence are visible) but contribute 0 reference words.
    """
    norm = normalizer or (lambda s: " ".join((s or "").split()))
    ids, refs, hyps = list(ids), list(refs), list(hyps)
    if not (len(ids) == len(refs) == len(hyps)):
        raise ValueError("ids, refs and hyps must have the same length")
    groups = list(groups) if groups is not None else [None] * len(ids)
    if len(groups) != len(ids):
        # zip() would silently drop utterances and change the score.
        raise ValueError(f"groups has {len(groups)} entries but there are {len(ids)} "
                         "utterances; give one group per utterance")
    out = CorpusScore()
    for uid, r, h, g in zip(ids, refs, hyps, groups):
        rn, hn = norm(r or ""), norm(h or "")
        wc, pairs = word_counts(rn, hn)
        out.utterances.append(UtteranceScore(uid, rn, hn, wc, char_counts(rn, hn), g))
        out.confusions.update(pairs)
    return out
