import pytest

from gyara.normalize import HausaNormalizer, get_normalizer, hausa_number

std = get_normalizer("standard")
len_ = get_normalizer("lenient")


@pytest.mark.parametrize(
    "a,b",
    [
        ("Ƙasa", "ƙasa"),  # capital hook
        ("ƴaƴa", "'ya'ya"),  # Niger orthography -> Nigerian
        ("’ya’ya", "'ya'ya"),  # curly apostrophes
        ("ʼyaʼya", "'ya'ya"),  # modifier letter apostrophe
        ("Sannu, yaya kake?", "sannu yaya kake"),
        ("ɗan-uwa", "ɗan uwa"),
        ("  na   gode  ", "na gode"),
        ("arba'in", "arba'in"),  # mid-word glottal stop kept
        ("'Kai' ya ce", "kai ya ce"),  # quote marks removed
        ("Ká", "Ká"),  # NFC: decomposed accent equals composed one
    ],
)
def test_standard_equivalences(a, b):
    assert std(a) == std(b)


def test_hook_errors_still_count_in_standard():
    # ƙasa (land) and kasa (below/fail) are different words.
    assert std("ƙasa") != std("kasa")


def test_lenient_folds_hooks():
    assert len_("ƙasa ɓera ɗaki 'ya'ya") == "kasa bera daki yaya"
    assert len_("ƙasa") == len_("kasa")


@pytest.mark.parametrize(
    "n,words",
    [
        (0, "sifiri"),
        (1, "ɗaya"),
        (10, "goma"),
        (11, "goma sha ɗaya"),
        (20, "ashirin"),
        (25, "ashirin da biyar"),
        (40, "arba'in"),
        (100, "ɗari"),
        (125, "ɗari da ashirin da biyar"),
        (300, "ɗari uku"),
        (1000, "dubu"),
        (2026, "dubu biyu da ashirin da shida"),
        (1_000_000, "miliyan"),
    ],
)
def test_numbers(n, words):
    assert hausa_number(n) == words


def test_digits_match_words():
    assert std("Mutane 25 sun zo") == std("mutane ashirin da biyar sun zo")
    assert std("Naira 1,000") == std("naira dubu")


def test_variants_apply_in_both_modes():
    n = HausaNormalizer(variants={"computer": "kwamfuta"})
    assert n("Computer") == "kwamfuta"
    nl = HausaNormalizer(fold_hooks=True, variants={"ɗaki": "gida"})
    assert nl("ɗaki") == "gida"


def test_empty_and_none():
    assert std("") == ""
    assert std(None) == ""
    assert std("?!.") == ""


def test_raw_mode_only_collapses_space():
    raw = get_normalizer("raw")
    assert raw(" Ƙasa,  ta ") == "Ƙasa, ta"
