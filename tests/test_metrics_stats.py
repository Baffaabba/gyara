import math

import numpy as np
import pytest

from gyara import hallucination, metrics, stats


def test_word_counts_basic():
    c, pairs = metrics.word_counts("a b c d", "a x c")
    assert (c.substitutions, c.deletions, c.insertions) == (1, 1, 0)
    assert c.rate == pytest.approx(0.5)
    assert ("S", "b", "x") in pairs


def test_empty_reference_is_nan_not_crash():
    c, _ = metrics.word_counts("", "na gode")
    assert c.insertions == 2 and math.isnan(c.rate)


def test_corpus_wer_is_pooled_not_averaged():
    # One short utterance fully wrong, one long one perfect.
    s = metrics.score_corpus(["1", "2"], ["a b", "c d e f g h i j"], ["x y", "c d e f g h i j"])
    assert s.wer == pytest.approx(2 / 10)  # pooled, not (1.0 + 0.0) / 2


def test_cer():
    assert metrics.cer("abcd", "abxd") == pytest.approx(0.25)


def test_bootstrap_interval_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    n = rng.integers(5, 30, 300)
    e = rng.binomial(n, 0.2)
    a = stats.bootstrap_rate(e, n, b=2000)
    b = stats.bootstrap_rate(e, n, b=2000)
    assert a.low < a.estimate < a.high
    assert (a.low, a.high) == (b.low, b.high)
    assert a.estimate == pytest.approx(e.sum() / n.sum())


def test_cluster_bootstrap_is_wider_for_correlated_data():
    rng = np.random.default_rng(1)
    speakers = np.repeat(np.arange(10), 30)
    spk_rate = rng.uniform(0.05, 0.6, 10)[speakers]  # strong speaker effect
    n = np.full(300, 20)
    e = rng.binomial(n, spk_rate)
    naive = stats.bootstrap_rate(e, n, b=3000)
    clustered = stats.bootstrap_rate(e, n, groups=[str(s) for s in speakers], b=3000)
    assert clustered.unit == "cluster" and clustered.n_units == 10
    assert (clustered.high - clustered.low) > 1.5 * (naive.high - naive.low)


def test_paired_bootstrap_detects_real_gain_and_not_noise():
    rng = np.random.default_rng(2)
    n = rng.integers(10, 30, 400)
    eb = rng.binomial(n, 0.30)
    ea = np.maximum(eb - rng.binomial(n, 0.10), 0)  # A strictly better
    r = stats.paired_bootstrap(ea, eb, n, b=3000)
    assert r.delta < 0 and r.significant and r.p_value < 0.01
    assert "fewer errors" in r.sentence("tuned", "base")

    same = stats.paired_bootstrap(eb, eb, n, b=1000)
    assert not same.significant and same.p_value == 1.0
    assert "No reliable difference" in same.sentence()


def test_p_value_is_never_printed_as_zero():
    # Bug: a huge effect printed "p=0.0000". With B resamples the smallest
    # non-zero two-sided p is 2/B, so 0 means "below the bootstrap's resolution".
    rng = np.random.default_rng(3)
    n = rng.integers(10, 30, 300)
    eb = rng.binomial(n, 0.6)
    ea = rng.binomial(n, 0.1)
    r = stats.paired_bootstrap(ea, eb, n, b=10_000)
    assert r.p_value == 0.0
    assert r.p_str() == "p < 0.001"
    assert "p < 0.001" in r.sentence("A", "B") and "0.0000" not in r.sentence("A", "B")
    # With few resamples the bootstrap cannot resolve 0.001: say what it can resolve.
    assert stats.fmt_p(0.0, 500) == "p < 0.004"
    assert stats.fmt_p(0.0, 10_000) == "p < 0.001"
    assert stats.fmt_p(0.0004, 10_000) == "p < 0.001"
    assert stats.fmt_p(0.0016, 10_000) == "p = 0.0016"
    assert stats.fmt_p(0.7, 10_000) == "p = 0.7000"
    assert stats.fmt_p(float("nan"), 0) == "p n/a"


def test_spearman():
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(stats.spearman([1, 2], [1, 2]))


def test_loop_detection():
    loop = "na gode " * 12
    chk = hallucination.check(loop, duration_s=5)
    assert chk.flagged
    trimmed = hallucination.trim_loop(loop)
    assert len(trimmed.split()) == 4
    ok = hallucination.check("Sannu da zuwa, yaya aiki yau da safe?", duration_s=3)
    assert not ok.flagged


# Real whisper-tiny outputs from the FLEURS Hausa smoke run (9 Oct 2026). Each
# is one whitespace token looping, so word-level checks missed all three, and
# the standard normaliser then expanded every digit into a Hausa number word.
_NOSPACE_LOOPS = [
    "msa mamma 21" + ".5" * 90,
    "Gb2" + ".2" * 90,
    "2.2.1 naiki 2.4 ggbh" + "p" * 140,
]


@pytest.mark.parametrize("hyp", _NOSPACE_LOOPS)
def test_loop_without_spaces_is_flagged_and_trimmed(hyp):
    chk = hallucination.check(hyp, duration_s=10)
    assert chk.flagged, chk
    trimmed = hallucination.trim_loop(hyp)
    assert len(trimmed) < 40
    assert not hallucination.check(trimmed, duration_s=10).flagged


@pytest.mark.parametrize("text", [
    # Real FLEURS Hausa references: long, numbers, punctuation, hooked letters.
    "Cibiyar Tabbatar da Adalci da Demokradiyya ta ƙasar Haiti ta nakalto waɗansu "
    "ayyukan nazari masu zaman kansu waɗanda ke tsammanin bataliyar kiyaye zaman lafiya",
    "Kwatancin 802.11n na aiki duk akan mita 2.4Ghz da 5.0Ghz.",
    "Mintuna biyar da fara nunin wata iska ya fara shigowa ciki, kusan minti daya "
    "daga baya, iskar ta doshi 70km/h... daga baya ruwan sama yazo",
    "Ya ce: \"Haka ne.\" ... Na gode.",
])
def test_normal_hausa_is_not_flagged(text):
    assert not hallucination.check(text, duration_s=10).flagged
