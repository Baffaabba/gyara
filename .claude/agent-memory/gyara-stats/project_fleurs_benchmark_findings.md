---
name: fleurs-benchmark-findings
description: Findings from the first FLEURS Hausa benchmark review (2026-10-09) - FLEURS refs omit hooks, N-ATLaS effect anatomy, clustering limits, what was blocked
metadata:
  type: project
---

First real runs (Colab T4, commit f7351d6, rules 1.0.0) reviewed 2026-10-09; approved numbers live in docs/BENCHMARK.md.

- FLEURS ha_ng references under-use hooks: of 432 hook-only word errors (standard), 300 are model-hooked / ref-plain (daya/ɗaya, hudu/huɗu). The standard−lenient gap on FLEURS is mostly the reference, not the model. Blocked the "2.86 points are hook mistakes" wording.
- N-ATLaS accept-all effect: standard WER −0.18 [−0.30,−0.07], p=0.0016, MDE 0.17. It is 30 of 4,903 errors; 34 clips better, 11 worse; only 1 of 126 word edits was hook-only. Raw −2.01 is capitals/punctuation. "Upper bound" wording blocked (accept-all is not an upper bound).
- Survives Bonferroni over 12 tests (0.019). Effect is about the size of the MDE, so expect winner's-curse shrinkage.
- FLEURS has no speaker ids: clusters are sentence ids (331, sizes 1-3). The CIs ignore speaker correlation; say so.
- whisper-small output is phonetic Hausa, not English and not empty; fair as "base model", not as "beats Whisper".
- Hausa-ASR FLEURS contamination is unknown. Validation and test WER are nearly equal (30.18 vs 30.19). The proposed probe is to score 100 FLEURS *train* clips.

**Why:** these took re-scoring predictions.jsonl to find, and they recur whenever FLEURS numbers are worded.
**How to apply:** check any new public FLEURS sentence against these. If rules change past 1.0.0 or new runs land, re-verify rather than reuse. See [[eval-integrity]].
