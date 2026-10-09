# Gyara evaluation: `NCAIR1/Hausa-ASR` on fleurs ha_ng train (first 100)

- **n** = 100 utterances, 98 clusters (group), 2484 reference words (standard), 0.41 h of audio
- Manifest: `data\fleurs\probe-train\train.jsonl` (sha256 `97b9db7828e4`)
- Device: cpu, language forcing: 'hausa' (used: 'hausa'), real-time factor: 0.6565
- Normalisation rules 1.0.0, gyara 0.1.0, run started 2026-10-09T13:05:45+00:00
- 95% CIs: cluster bootstrap, 10000 resamples. Clusters: `group` (FLEURS sentence id) for 100 utterances.

## Headline

`standard` is the headline. `raw` only collapses whitespace; `lenient` also folds hooked letters.

| mode | WER % [95% CI] | CER % [95% CI] | S | D | I | ref words |
|---|---|---|---|---|---|---|
| raw | 45.55 [42.02, 49.36] | 16.84 [14.48, 19.39] | 810 | 86 | 169 | 2338 |
| standard | 31.92 [29.15, 34.79] | 10.99 [9.48, 12.69] | 574 | 115 | 104 | 2484 |
| lenient | 29.11 [26.20, 32.11] | 10.20 [8.66, 11.95] | 502 | 116 | 105 | 2484 |

Standard minus lenient WER = **2.82 points**. This is how much of the error rate disappears when hooked letters (ɓ ɗ ƙ 'y) are folded. It measures disagreement between transcript and reference; it does not say which side is wrong.

Of the word substitutions under `standard`, 66 differ only in hooked letters:

| direction | substitutions |
|---|---|
| hook in the transcript, not in the reference | 37 |
| hook in the reference, not in the transcript | 28 |
| hooks on both sides, in different places | 1 |

Words carrying a hook: 4.5% of reference words, 5.5% of transcript words.

## Where it fails (standard)

### By clip length

| clip length | utterances | ref words | WER % [95% CI] | CER % [95% CI] |
|---|---|---|---|---|
| 15-30s | 41 | 1180 | 30.34 [26.58, 34.39] | 10.73 [8.35, 13.75] |
| 5-15s | 57 | 1247 | 33.92 [29.81, 38.10] | 11.49 [9.64, 13.43] |
| <5s | 1 | 9 | 22.22 [CI n/a] | 5.88 [CI n/a] |
| >30s | 1 | 48 | 20.83 [CI n/a] | 5.47 [CI n/a] |

### By source

| source | utterances | ref words | WER % [95% CI] | CER % [95% CI] |
|---|---|---|---|---|
| fleurs | 100 | 2484 | 31.92 [29.15, 34.79] | 10.99 [9.48, 12.69] |

### Top 25 word confusions

| op | reference | hypothesis | count |
|---|---|---|---|
| D | da | ∅ | 15 |
| I | ∅ | da | 9 |
| D | su | ∅ | 7 |
| D | yi | ∅ | 7 |
| D | a | ∅ | 6 |
| D | ne | ∅ | 5 |
| I | ∅ | ta | 4 |
| S | daya | ɗaya | 4 |
| D | cewa | ∅ | 4 |
| S | huɗu | hudu | 4 |
| I | ∅ | ba | 3 |
| S | kasar | ƙasar | 3 |
| S | ƙasa | kasa | 3 |
| S | shi | shine | 3 |
| S | ba | basu | 3 |
| S | dubu | alif | 3 |
| I | ∅ | a | 3 |
| D | ɗari | ∅ | 3 |
| D | takwas | ∅ | 3 |
| S | hada | haɗa | 2 |
| I | ∅ | alif | 2 |
| I | ∅ | ɗaya | 2 |
| S | su | sukan | 2 |
| D | ba | ∅ | 2 |
| S | zamanin | zamani | 2 |

## Is the confidence score useful?

Spearman correlation between `avg_logprob` and utterance WER: **ρ = -0.629** (n = 100), strong; as hoped, lower confidence goes with more errors.

## Loops and flags

| flag | utterances |
|---|---|
| long_clip_vad | 1 |

`loop` = still repetitive after temperature fallback; trimmed and left for a human (0 utterance(s)).

---

*N-ATLAS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.*
