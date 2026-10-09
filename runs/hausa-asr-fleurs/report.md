# Gyara evaluation: `NCAIR1/Hausa-ASR` on fleurs ha_ng test

- **n** = 621 utterances, 331 clusters (group), 16242 reference words (standard), 3.34 h of audio
- Manifest: `data/fleurs/test.jsonl` (sha256 `05eecae508dc`)
- Device: cuda, language forcing: 'hausa' (used: 'hausa'), real-time factor: 0.0253
- Normalisation rules 1.0.0, gyara 0.1.0, run started 2026-10-09T10:58:08+00:00
- 95% CIs: cluster bootstrap, 10000 resamples. Clusters: `group` (FLEURS sentence id) for 621 utterances.

## Headline

`standard` is the headline. `raw` only collapses whitespace; `lenient` also folds hooked letters.

| mode | WER % [95% CI] | CER % [95% CI] | S | D | I | ref words |
|---|---|---|---|---|---|---|
| raw | 44.48 [42.15, 47.03] | 16.45 [14.60, 18.60] | 5231 | 453 | 1269 | 15632 |
| standard | 30.19 [28.63, 31.84] | 10.08 [9.00, 11.35] | 3539 | 497 | 867 | 16242 |
| lenient | 27.33 [25.77, 28.98] | 9.34 [8.26, 10.60] | 3071 | 499 | 869 | 16242 |

Standard minus lenient WER = **2.86 points**. This is how much of the error rate disappears when hooked letters (ɓ ɗ ƙ 'y) are folded. It measures disagreement between transcript and reference; it does not say which side is wrong.

Of the word substitutions under `standard`, 432 differ only in hooked letters:

| direction | substitutions |
|---|---|
| hook in the transcript, not in the reference | 301 |
| hook in the reference, not in the transcript | 128 |
| hooks on both sides, in different places | 3 |

Words carrying a hook: 4.0% of reference words, 5.5% of transcript words.

## Where it fails (standard)

### By clip length

| clip length | utterances | ref words | WER % [95% CI] | CER % [95% CI] |
|---|---|---|---|---|
| 15-30s | 336 | 9382 | 28.57 [26.81, 30.38] | 8.34 [7.53, 9.24] |
| 5-15s | 220 | 4137 | 31.42 [29.03, 33.91] | 10.47 [9.24, 11.98] |
| >30s | 65 | 2723 | 33.90 [28.24, 40.23] | 15.56 [10.34, 21.87] |

### By source

| source | utterances | ref words | WER % [95% CI] | CER % [95% CI] |
|---|---|---|---|---|
| fleurs | 621 | 16242 | 30.19 [28.63, 31.84] | 10.08 [9.00, 11.35] |

### Top 25 word confusions

| op | reference | hypothesis | count |
|---|---|---|---|
| I | ∅ | da | 73 |
| I | ∅ | a | 40 |
| D | da | ∅ | 26 |
| I | ∅ | ɗaya | 22 |
| D | su | ∅ | 21 |
| S | daya | ɗaya | 19 |
| I | ∅ | ba | 19 |
| I | ∅ | na | 17 |
| S | shi | shine | 16 |
| D | yi | ∅ | 16 |
| D | goma | ∅ | 15 |
| D | ne | ∅ | 15 |
| S | da | ta | 14 |
| D | na | ∅ | 14 |
| D | a | ∅ | 14 |
| D | ka | ∅ | 13 |
| D | ke | ∅ | 13 |
| S | na | da | 13 |
| S | hada | haɗa | 13 |
| I | ∅ | ya | 12 |
| I | ∅ | alif | 12 |
| S | wadanda | waɗanda | 11 |
| S | ashirin | ishirin | 10 |
| D | ba | ∅ | 10 |
| I | ∅ | za | 10 |

## Is the confidence score useful?

Spearman correlation between `avg_logprob` and utterance WER: **ρ = -0.574** (n = 621), strong; as hoped, lower confidence goes with more errors.

## Loops and flags

| flag | utterances |
|---|---|
| long_clip_vad | 64 |

`loop` = still repetitive after temperature fallback; trimmed and left for a human (0 utterance(s)).

---

*N-ATLAS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.*
