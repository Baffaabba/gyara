# Benchmark: Hausa speech-to-text on FLEURS

Reviewed by gyara-stats on 9 Oct 2026. Every number on this page comes from a
file under `runs/`. Each one was re-scored from `predictions.jsonl` and
matched to the last digit.

> N-ATLAS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

## Results

Test set: FLEURS Hausa (`google/fleurs`, `ha_ng`, test split). 621 utterances,
331 distinct sentences, 16,242 reference words (standard), 3.34 hours.
Normalisation rules `1.0.0`. Free Colab T4, fp16, greedy decoding, Hausa
language token forced. Code at commit `f7351d6`.

Numbers are word error rate (WER) and character error rate (CER), in percent,
with 95% confidence intervals. Lower is better.

| Model | WER standard | WER lenient | WER raw | CER standard |
|---|---|---|---|---|
| `NCAIR1/Hausa-ASR` | **30.19** [28.63, 31.84] | 27.33 [25.77, 28.98] | 44.48 [42.15, 47.03] | 10.08 [9.00, 11.35] |
| `openai/whisper-small` | **89.81** [88.51, 91.28] | 89.38 [88.06, 90.85] | 94.69 [93.11, 96.72] | 30.88 [29.60, 32.38] |

`standard` is the headline. It ignores case, punctuation, apostrophe and
hook encodings, and digits vs number words. `lenient` also treats ɓ ɗ ƙ 'y as
b d k y. `raw` only collapses spaces. See `docs/NORMALISATION.md`.

### Hausa-ASR vs its base model

`NCAIR1/Hausa-ASR` is a fine-tune of `openai/whisper-small`. Comparing the two
on the same 621 clips shows what the fine-tune adds.

| Metric (standard) | Δ points (Hausa-ASR − whisper-small) | 95% CI | p | MDE |
|---|---|---|---|---|
| WER | −59.62 | −61.00 to −58.17 | < 0.001 | 2.01 |
| CER | −20.80 | −21.67 to −19.95 | < 0.001 | 1.21 |

The gap holds in every mode: raw WER −50.21 [−51.81, −48.57], lenient WER
−62.05 [−63.41, −60.60]. That is about two-thirds fewer word errors.

### N-ATLaS spelling suggestions

We passed each Hausa-ASR transcript to `NCAIR1/N-ATLaS` (4-bit, the
`transformers` backend default) through Gyara's guard. Then we accepted every
suggestion the guard let through, with no human review. The speech model's
output was identical in both runs (621 of 621), so the difference comes from
the suggestions alone.

| Mode | Metric | ASR alone | With suggestions | Δ points | 95% CI | p | MDE |
|---|---|---|---|---|---|---|---|
| standard | WER | 30.19 | 30.00 | **−0.18** | −0.30 to −0.07 | 0.002 | 0.17 |
| standard | CER | 10.08 | 10.09 | +0.01 | −0.03 to +0.05 | 0.70 | 0.06 |
| lenient | WER | 27.33 | 27.13 | −0.20 | −0.33 to −0.09 | 0.001 | 0.17 |
| raw | WER | 44.48 | 42.47 | −2.01 | −2.35 to −1.70 | < 0.001 | 0.46 |

What this means:

- **The real effect is small.** Under the standard score, accepting every
  suggestion removed 30 of 4,903 word errors. CER showed no reliable
  difference.
- **The raw gain is mostly capitals and punctuation.** Hausa-ASR writes in
  lower case. FLEURS references use sentence case. N-ATLaS put capitals and
  full stops back, and the raw score counts those. The standard score ignores
  them, by design.
- **It is not hook fixes.** The lenient effect (hooks ignored) is the same
  size as the standard one.

## Method

- **Pooled rates.** Total edits ÷ total reference words, not an average of
  per-clip rates. The per-clip average would read 30.50, not 30.19.
- **Same rules for both sides.** The normaliser is applied to reference and
  transcript alike. Rules version `1.0.0` is stamped in every `metrics.json`.
- **Clustered confidence intervals.** 10,000 bootstrap resamples, seed 1234.
  We resample FLEURS sentence ids (331), not clips. Several speakers read the
  same sentence, so their clips are not independent.
- **Paired comparisons.** Same 621 clip ids, same references, same
  normaliser. Paired cluster bootstrap (Bisani & Ney, 2004). We report the
  delta in absolute points, its 95% CI, a two-sided p-value and the minimum
  detectable effect (MDE: the smallest difference this test set detects 80%
  of the time).
- **No tuning on test.** We chose one setting, forced Hausa vs automatic
  language detection, on the first 100 FLEURS validation clips. Both gave
  identical output (dev WER 30.18). No clip, sentence or text is shared
  between those 100 and the test set. We ran the N-ATLaS prompt once, with
  no prompt variants tried.

### Multiple comparisons

There are 12 paired tests on this page: 2 comparisons × 3 modes × 2 metrics.
The headline for each comparison was fixed in advance: standard WER.

- Hausa-ASR vs whisper-small: all 6 tests have p < 0.001. They survive any
  correction.
- N-ATLaS: standard WER p = 0.0016. Bonferroni over all 12 tests gives
  0.0016 × 12 = 0.019, which is still under 0.05. Holm over the 6 N-ATLaS
  tests gives the same verdict.
- The p-values are bootstrap estimates with a floor of 2/10,000 = 0.0002.
  The run files print anything under the resolution as "p < 0.001".

The N-ATLaS effect (0.18 points) is barely above the MDE (0.17). A result this
close to the detection limit tends to overstate the true size, so the real
effect may be smaller. FLEURS Hausa has only 331 test sentences. Detecting
half this effect would need about four times as many, which FLEURS does not
have.

## Caveats

1. **FLEURS may not be unseen by Hausa-ASR.** Its model card (checked 9 Oct
   2026) names Langeasy recordings and "publicly available datasets" but does
   not list them. If FLEURS was in its training data, its FLEURS score may be
   optimistic. One weak signal: validation WER (30.18, first 100 clips) and
   test WER (30.19) are nearly equal. A second probe found no sign that the
   model memorised FLEURS training data: on the first 100 FLEURS train clips
   it scores 31.92% [29.15, 34.79], against 30.01% [27.22, 32.90] on 100
   validation clips, same CPU settings (`runs/probe/`). That cannot show the
   test split was unseen. Our own held-out audio from new speakers
   is the number that settles this. It is not measured yet.
2. **The intervals account for repeated sentences, not repeated speakers.**
   FLEURS publishes no speaker ids. If a few speakers read many clips, the
   true interval is wider than shown.
3. **FLEURS references often leave out hooks.** 432 word errors in the
   standard score are hook-only differences. In 301 of them (70%), Hausa-ASR
   wrote the hooked letter and the reference did not, for example *daya* /
   *ɗaya*, *hudu* / *huɗu*, *wadanda* / *waɗanda*. Only 4.0% of reference
   words carry a hook (ɓ ɗ ƙ 'y), against 5.5% of Hausa-ASR's words. 128
   go the other way (the reference has the hook, the model missed it). So on FLEURS the
   standard − lenient gap (2.86 points) is mostly the reference, not the
   model. For FLEURS, the lenient score (27.33) is arguably the fairer one. Standard
   stays the headline because that rule was fixed before the results came in
   (DECISIONS, 9 Oct).
4. **Numbers score worse.** Clips whose reference contains digits score
   32.62% WER, against 29.32% for the rest. Part of that gap is likely the
   normaliser (it writes 1,000 as *dubu*; speakers often say *alif* for
   years). It is a candidate for rules `1.1.0`.
5. **Clips over 30 seconds** (65) are split by voice-activity detection
   before decoding. They score 33.90% [28.24, 40.23].
6. **Accepting every suggestion is not an upper bound.** Of the clips whose
   standard score changed, 34 improved and 11 got worse. In the worse ones,
   N-ATLaS swapped an unusual but correct word for a common one (*domin* →
   *don*). A reviewer who rejects the bad suggestions does better than
   accept-all. This is why Gyara never applies them automatically.
7. **One test set, read speech.** FLEURS is read Wikipedia sentences.
   Conversational, broadcast or noisy Hausa will score differently.

Caveats 3, 4 and 6 and the 30/34/11 counts come from re-scoring
`predictions.jsonl`. Those files are currently git-ignored. Commit them, or
add these counts to `evaluate.py`'s report, before quoting them publicly.

## Other measured numbers

- **Confidence tracks errors.** Spearman ρ between Hausa-ASR's average token
  log-probability and clip WER: −0.57 (n = 621 clips). This is measured per
  clip, not per segment. For whisper-small it is −0.23 (weak).
- **Speed.** Hausa-ASR transcribed 3.34 h of audio in 304 s on a T4: a
  real-time factor of 0.025, about 40 times faster than real time.
- **N-ATLaS speed.** Adding suggestions took the run from 304 s to 3,168 s,
  about 4.6 s per sentence on a T4. This figure is derived; the run does not
  record per-suggestion timing.

## Not measured yet

- Our own held-out Hausa audio (`runs/hausa-asr-own/`).
- Before vs after fine-tuning on human-corrected data.
- Share of N-ATLaS suggestions the guard rejected. The run does not record it.

## Files

| What | File |
|---|---|
| Hausa-ASR report and metrics | [`runs/hausa-asr-fleurs/report.md`](../runs/hausa-asr-fleurs/report.md), [`metrics.json`](../runs/hausa-asr-fleurs/metrics.json), [`meta.json`](../runs/hausa-asr-fleurs/meta.json) |
| whisper-small report and metrics | [`runs/whisper-small-fleurs/report.md`](../runs/whisper-small-fleurs/report.md), [`metrics.json`](../runs/whisper-small-fleurs/metrics.json) |
| Paired comparison | [`runs/hausa-asr-fleurs/compare.md`](../runs/hausa-asr-fleurs/compare.md), [`compare.json`](../runs/hausa-asr-fleurs/compare.json) |
| N-ATLaS suggestion effect | [`runs/hausa-asr-fleurs-natlas/report.md`](../runs/hausa-asr-fleurs-natlas/report.md), [`metrics.json`](../runs/hausa-asr-fleurs-natlas/metrics.json) |
| Language setting chosen on dev | [`runs/g0-smoke/decoding_choice.json`](../runs/g0-smoke/decoding_choice.json), `runs/dev/` |
| Leaderboard | [`runs/leaderboard.md`](../runs/leaderboard.md) |
| Test manifest | `data/fleurs/test.jsonl`, sha256 `05eecae508dc…` |

Reproduce on a GPU with `notebooks/benchmark_colab.ipynb`, or:

```bash
gyara fetch-fleurs --split test
gyara eval --model NCAIR1/Hausa-ASR --manifest data/fleurs/test.jsonl --name hausa-asr-fleurs --language hausa
gyara eval --model openai/whisper-small --manifest data/fleurs/test.jsonl --name whisper-small-fleurs --language hausa
gyara compare runs/hausa-asr-fleurs runs/whisper-small-fleurs
```
