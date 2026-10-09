# Hausa normalisation rules (v1.0.0)

Status: **draft, awaiting sign-off by the Hausa normalisation owner.**
Owner: `[[name]]`. Code: `gyara/normalize.py` (`RULES_VERSION = "1.0.0"`).
Tests: `tests/test_normalize.py`.

## Why this exists

A word error rate is only as fair as the comparison behind it. If the
reference says `’ya’ya` and the model writes `'ya'ya`, a plain string match
counts an error where there is none. On Hausa this happens all the time, and it
is the main reason published Hausa accuracy numbers cannot be compared.

These rules remove differences that are only **how the text was typed**, and
keep differences that are **different words**. They are applied identically to
the reference and to the model output.

## The three modes

| Mode | What it does | Use it for |
| --- | --- | --- |
| `raw` | Collapses whitespace only | Showing how much the rules matter |
| `standard` | All rules below. Hook errors still count | **The headline number** |
| `lenient` | `standard`, plus ɓ→b, ɗ→d, ƙ→k, 'y→y | Seeing how many disagreements are hook-only (either side) |

`standard − lenient` is the share of word errors caused only by a missing or
wrong hooked letter. That is a useful diagnostic on its own: it tells a
developer whether their model's problem is hearing or spelling.

## Rules in `standard`, in order

1. **Unicode NFC.** A letter typed as base + combining mark equals the
   precomposed letter.
2. **Apostrophes unified.** ’ ‘ ʼ ʻ ` ´ ′ ꞌ all become `'`.
3. **Lowercase.**
4. **Hooked letters canonical.** Ɓ→ɓ, Ɗ→ɗ, Ƙ→ƙ. The Niger-orthography ƴ/Ƴ
   becomes `'y`, the Nigerian standard. Look-alikes copied from other
   orthographies are mapped: Đ and ɖ → ɗ; ĸ and ǩ → ƙ.
5. **Digits become Hausa number words.** `25` → `ashirin da biyar`,
   `1,000` → `dubu`, `2026` → `dubu biyu da ashirin da shida`. Pattern:
   units ɗaya…goma; 11–19 `goma sha X`; tens ashirin, talatin, arba'in, hamsin,
   sittin, saba'in, tamanin, casa'in; `da` joins parts; ɗari (100), dubu
   (1,000), miliyan (1,000,000) with a multiplier after them (`ɗari uku`).
6. **Punctuation and symbols removed**, with two exceptions for the
   apostrophe, which is part of Hausa spelling:
   - kept at the start of a word before `y` (`'ya'ya`), and
   - kept between two letters (`arba'in`, `ya'ya`).
   Anywhere else it is a quotation mark and is removed.
   Hyphens and slashes become spaces, so `ɗan-uwa` = `ɗan uwa`.
7. **Whitespace collapsed.**
8. **Optional spelling-variant map**, whole words only, from a YAML file
   (`variants:` key). For loanwords with several accepted spellings. Empty by
   default; every entry must be approved by the owner.

## What the rules deliberately do *not* do

- **They do not fold hooks in `standard`.** `ƙasa` (land) and `kasa` (below,
  to fail) are different words.
- **They do not map ASCII digraphs** like `'b` or `b'` to ɓ. In real text these
  are ambiguous with quotation marks and with the glottal stop in `arba'in`.
  If the owner finds a source that uses them consistently, add a variant map.
- **They do not correct spelling.** Word-boundary disagreements (`a kan` vs
  `akan`) count as errors unless added to the variant map. CER is reported
  next to WER because it is less sensitive to these.
- **They do not translate English loanwords.** Code-switched English is scored
  as written.

## Questions for the owner to settle before sign-off

1. Number words: is `dubu biyu da ashirin da shida` the form our references
   will use for 2026? Are there regional variants we should map?
2. Common word-boundary variants worth a variant entry (e.g. `a kan`/`akan`,
   `ba za`/`baza`)? List them with a source.
3. Do any of our audio sources use Niger orthography (ƴ) consistently?

## Changing a rule

1. Write a failing test in `tests/test_normalize.py` showing the case.
2. Change `gyara/normalize.py`, bump `RULES_VERSION` (minor for an added case,
   major for a change in meaning).
3. Re-score every run without re-running the model: `gyara rescore runs/<name>`.
4. Add an entry to `docs/DECISIONS.md` if the owner made the call.

Numbers produced under different rule versions are not comparable, and
`gyara compare` refuses to compare them.
