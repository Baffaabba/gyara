# Compare: `hausa-asr-dev100-cpu` (A) vs `hausa-asr-dev100-lang-hausa` (B)

- Same 100 utterances, same references, rules 1.0.0. Dropped (not in both runs): 0 from A, 0 from B.
- Paired cluster bootstrap, 10000 resamples over 81 clusters (`group` (FLEURS sentence id) for 100 utterances). Δ = A − B; negative means A makes fewer errors.
- Headline, fixed in advance: **standard WER**. The table shows all 6 tests (3 normalisation modes × WER/CER); read the others as supporting evidence, not as extra chances to find a difference.

| mode | metric | A % | B % | Δ points | 95% CI | p | MDE (points) |
|---|---|---|---|---|---|---|---|
| raw | WER | 42.80 | 42.97 | -0.17 | [-0.53, +0.12] | 0.3928 | 0.47 |
| raw | CER | 13.53 | 13.56 | -0.03 | [-0.12, +0.04] | 0.5636 | 0.12 |
| **standard** | WER | 30.01 | 30.18 | -0.17 | [-0.44, +0.05] | 0.2538 | 0.36 |
| standard | CER | 8.67 | 8.69 | -0.02 | [-0.08, +0.02] | 0.4658 | 0.07 |
| lenient | WER | 26.54 | 26.67 | -0.13 | [-0.40, +0.08] | 0.4260 | 0.34 |
| lenient | CER | 7.75 | 7.76 | -0.02 | [-0.07, +0.03] | 0.6862 | 0.07 |

**WER (standard):** No reliable difference: hausa-asr-dev100-cpu vs hausa-asr-dev100-lang-hausa differ by 0.17 points, inside the noise (95% CI -0.44 to +0.05). With this test set, differences under about 0.36 points cannot be detected.

**CER (standard):** No reliable difference: hausa-asr-dev100-cpu vs hausa-asr-dev100-lang-hausa differ by 0.02 points, inside the noise (95% CI -0.08 to +0.02). With this test set, differences under about 0.07 points cannot be detected.

---

*N-ATLAS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.*
