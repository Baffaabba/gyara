# Gyara evaluation: `openai/whisper-small` on fleurs ha_ng test

- **n** = 621 utterances, 331 clusters (group), 16242 reference words (standard), 3.34 h of audio
- Manifest: `data/fleurs/test.jsonl` (sha256 `05eecae508dc`)
- Device: cuda, language forcing: 'hausa' (used: 'hausa'), real-time factor: 0.0294
- Normalisation rules 1.0.0, gyara 0.1.0, run started 2026-10-09T11:03:17+00:00
- 95% CIs: cluster bootstrap, 10000 resamples. Clusters: `group` (FLEURS sentence id) for 621 utterances.

## Headline

`standard` is the headline. `raw` only collapses whitespace; `lenient` also folds hooked letters.

| mode | WER % [95% CI] | CER % [95% CI] | S | D | I | ref words |
|---|---|---|---|---|---|---|
| raw | 94.69 [93.11, 96.72] | 36.06 [34.35, 38.06] | 11845 | 2053 | 904 | 15632 |
| standard | 89.81 [88.51, 91.28] | 30.88 [29.60, 32.38] | 11514 | 2378 | 695 | 16242 |
| lenient | 89.38 [88.06, 90.85] | 30.39 [29.11, 31.90] | 11424 | 2388 | 705 | 16242 |

Standard minus lenient WER = **0.43 points**. This is how much of the error rate disappears when hooked letters (ɓ ɗ ƙ 'y) are folded. It measures disagreement between transcript and reference; it does not say which side is wrong.

Of the word substitutions under `standard`, 32 differ only in hooked letters:

| direction | substitutions |
|---|---|
| hook in the transcript, not in the reference | 0 |
| hook in the reference, not in the transcript | 32 |
| hooks on both sides, in different places | 0 |

Words carrying a hook: 4.0% of reference words, 0.2% of transcript words.

## Where it fails (standard)

### By clip length

| clip length | utterances | ref words | WER % [95% CI] | CER % [95% CI] |
|---|---|---|---|---|
| 15-30s | 336 | 9382 | 89.23 [88.10, 90.37] | 29.45 [28.28, 30.76] |
| 5-15s | 220 | 4137 | 89.63 [87.70, 91.48] | 31.52 [30.20, 32.99] |
| >30s | 65 | 2723 | 92.07 [86.28, 99.29] | 34.91 [29.06, 42.46] |

### By source

| source | utterances | ref words | WER % [95% CI] | CER % [95% CI] |
|---|---|---|---|---|
| fleurs | 621 | 16242 | 89.81 [88.51, 91.28] | 30.88 [29.60, 32.38] |

### Top 25 word confusions

| op | reference | hypothesis | count |
|---|---|---|---|
| D | da | ∅ | 178 |
| S | da | de | 63 |
| D | a | ∅ | 59 |
| D | ya | ∅ | 45 |
| D | ta | ∅ | 40 |
| D | ba | ∅ | 40 |
| D | su | ∅ | 36 |
| D | ke | ∅ | 35 |
| D | na | ∅ | 33 |
| D | yi | ∅ | 29 |
| D | ne | ∅ | 27 |
| D | cikin | ∅ | 26 |
| D | kuma | ∅ | 22 |
| D | shi | ∅ | 18 |
| S | yana | yena | 17 |
| D | biyu | ∅ | 17 |
| S | cikin | chicken | 15 |
| D | zai | ∅ | 14 |
| D | ce | ∅ | 14 |
| D | iya | ∅ | 14 |
| S | cewa | wa | 14 |
| D | ɗari | ∅ | 14 |
| D | sha | ∅ | 14 |
| D | ka | ∅ | 13 |
| S | da | dha | 13 |

## Is the confidence score useful?

Spearman correlation between `avg_logprob` and utterance WER: **ρ = -0.227** (n = 621), weak; as hoped, lower confidence goes with more errors.
Do not market "least confident first" on this evidence.

## Loops and flags

| flag | utterances |
|---|---|
| long_clip_vad | 64 |
| loop | 1 |
| retried_t0.2 | 10 |

`loop` = still repetitive after temperature fallback; trimmed and left for a human (1 utterance(s)).

---

*N-ATLAS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.*
