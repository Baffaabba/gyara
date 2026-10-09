---
name: eval-integrity
description: The statistical rules every Gyara number must pass — pooled WER, three normalisation modes, cluster bootstrap CIs, paired tests for every comparison, minimum detectable effect, speaker-disjoint held-out sets, and how to word a result so it cannot over-claim. Load before running, reporting, reviewing or writing about any benchmark, fine-tune result or N-ATLaS suggestion effect.
---

# Evaluation integrity

Judges in a technical challenge, and the developers who adopt Gyara, will trust
the toolkit exactly as far as they trust its numbers. One inflated claim costs
more than a modest honest one. These rules are what "statistically correct"
means in this repo.

## 1. What a WER is

- **Pooled**: total word edits ÷ total reference words. Never the mean of
  per-utterance WERs (overweights short clips). `CorpusScore.wer` does this.
- Always three modes: `raw` (whitespace only), `standard` (orthography and
  formatting normalised, hook errors still count), `lenient` (hooks folded).
  `standard` is the headline. `standard − lenient` = hook-only disagreements;
  it does not say which side is wrong (FLEURS references often omit hooks), so
  report its direction split (`hook_substitutions` in metrics.json). Report CER alongside; for Hausa, CER is less sensitive to word
  segmentation disagreements.
- The normaliser is applied identically to reference and hypothesis.
  `RULES_VERSION` is stamped into every `metrics.json`. Numbers from different
  rule versions are not comparable.

## 2. Uncertainty, always

- Every corpus rate carries a **95% bootstrap CI** (`stats.bootstrap_rate`,
  10,000 resamples, fixed seed).
- **Resample clusters, not utterances**: speakers for our own audio; FLEURS
  sentence ids (the same sentence is read by several speakers). Independent
  resampling of correlated utterances gives intervals that are too narrow.
- State n: utterances, clusters, reference words, hours.

## 3. Comparisons

- Any "A beats B" — tuned vs base, Hausa-ASR vs whisper-small, with vs
  without N-ATLaS suggestions — uses `stats.paired_bootstrap` on **the same
  utterances, same references, same normaliser**.
- Report delta in percentage points, its 95% CI, p-value, and the **minimum
  detectable effect** (MDE). If the CI includes 0, the sentence is "no
  reliable difference", and you say the MDE. `PairedResult.sentence()`
  writes this for you; use it.
- Do not run many comparisons and report the best one. If you tried several
  fine-tune configs, say how many, and evaluate the chosen one on the test set
  **once**. Choose configs on a dev split, never on test.

## 4. Data splits

- **Test (held-out)**: FLEURS Hausa test + our own held-out audio
  (≥ 30 min). Frozen manifests in `data/heldout/`. Never trained on, never used
  to pick hyperparameters.
- **Speaker-disjoint**: no speaker in train/dev appears in test. A fine-tune
  measured on a voice it trained on is a leak, not a gain.
  `manifest.leakage_check` enforces audio and speaker disjointness; fine-tuning
  refuses to start otherwise.
- **Dev**: a speaker-disjoint slice of the training pool, for early stopping.
- Shared sentence *text* between train and test (common in FLEURS) is
  reported as a warning, not blocked.

## 5. Breakdown, not just a headline

Report by dialect, by speaker, by source (FLEURS vs own audio), by clip
length, and the top confusions (`CorpusScore.top_confusions`). A single
headline hides where the model fails, and "where it fails" is the evidence
NCAIR asked for.

## 6. Confidence is a claim too

If the UI ranks segments by model confidence ("least confident first"), back
it with a number: Spearman correlation between segment `avg_logprob` and
segment WER on the test set, with its n. If the correlation is weak, say so
and do not market the feature.

## 7. Wording

- "Reduced WER from 41.2% [38.9, 43.6] to 37.8% [35.5, 40.1], −3.4 points
  (95% CI −4.6 to −2.2, p < 0.001, n = 743 utterances, 92 speakers)." Good.
- "Improved accuracy by 8%." Bad: relative or absolute? CI? n? test set?
- Never "state of the art", "perfect", "guaranteed".
- Small or zero gains are reported as found. The concept document promises
  this, and judges reward it.
