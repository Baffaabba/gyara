# Gyara evaluation: `NCAIR1/Hausa-ASR` on fleurs ha_ng validation (first 100)

- **n** = 100 utterances, 81 clusters (group), 2396 reference words (standard), 0.49 h of audio
- Manifest: `data\fleurs\probe-dev\validation.jsonl` (sha256 `4d545a27ef13`)
- Device: cpu, language forcing: 'hausa' (used: 'hausa'), real-time factor: 0.7651
- Normalisation rules 1.0.0, gyara 0.1.0, run started 2026-10-09T13:21:55+00:00
- 95% CIs: cluster bootstrap, 10000 resamples. Clusters: `group` (FLEURS sentence id) for 100 utterances.

## Headline

`standard` is the headline. `raw` only collapses whitespace; `lenient` also folds hooked letters.

| mode | WER % [95% CI] | CER % [95% CI] | S | D | I | ref words |
|---|---|---|---|---|---|---|
| raw | 42.80 [38.83, 47.01] | 13.53 [11.30, 16.17] | 791 | 67 | 141 | 2334 |
| standard | 30.01 [27.22, 32.90] | 8.67 [7.50, 9.87] | 542 | 75 | 102 | 2396 |
| lenient | 26.54 [23.82, 29.39] | 7.75 [6.64, 8.91] | 459 | 75 | 102 | 2396 |

Standard minus lenient WER = **3.46 points**. This is how much of the error rate disappears when hooked letters (ɓ ɗ ƙ 'y) are folded. It measures disagreement between transcript and reference; it does not say which side is wrong.

Of the word substitutions under `standard`, 77 differ only in hooked letters:

| direction | substitutions |
|---|---|
| hook in the transcript, not in the reference | 56 |
| hook in the reference, not in the transcript | 21 |
| hooks on both sides, in different places | 0 |

Words carrying a hook: 3.2% of reference words, 5.4% of transcript words.

## Where it fails (standard)

### By clip length

| clip length | utterances | ref words | WER % [95% CI] | CER % [95% CI] |
|---|---|---|---|---|
| 15-30s | 51 | 1429 | 30.09 [26.25, 33.80] | 8.76 [7.16, 10.32] |
| 5-15s | 42 | 724 | 32.04 [27.33, 37.21] | 8.99 [7.34, 10.80] |
| >30s | 7 | 243 | 23.46 [19.47, 28.44] | 7.07 [4.53, 11.42] |

### By source

| source | utterances | ref words | WER % [95% CI] | CER % [95% CI] |
|---|---|---|---|---|
| fleurs | 100 | 2396 | 30.01 [27.22, 32.90] | 8.67 [7.50, 9.87] |

### Top 25 word confusions

| op | reference | hypothesis | count |
|---|---|---|---|
| I | ∅ | da | 9 |
| D | ka | ∅ | 8 |
| I | ∅ | a | 5 |
| I | ∅ | na | 4 |
| S | kasar | ƙasar | 4 |
| S | su | suka | 4 |
| I | ∅ | ko | 4 |
| D | da | ∅ | 4 |
| S | abincin | abinci | 3 |
| S | daya | ɗaya | 3 |
| I | ∅ | ke | 3 |
| I | ∅ | gaba | 2 |
| S | gabashi | shi | 2 |
| I | ∅ | ya | 2 |
| S | dokta | dr | 2 |
| S | dan | ɗin | 2 |
| S | za | zaka | 2 |
| I | ∅ | ba | 2 |
| S | karfe | ƙarfe | 2 |
| I | ∅ | alif | 2 |
| I | ∅ | ɗaya | 2 |
| D | ne | ∅ | 2 |
| S | binciken | bincike | 2 |
| S | wadanda | da | 2 |
| S | amurka | america | 2 |

## Is the confidence score useful?

Spearman correlation between `avg_logprob` and utterance WER: **ρ = -0.551** (n = 100), strong; as hoped, lower confidence goes with more errors.

## Loops and flags

| flag | utterances |
|---|---|
| long_clip_vad | 7 |

`loop` = still repetitive after temperature fallback; trimmed and left for a human (0 utterance(s)).

---

*N-ATLAS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.*
