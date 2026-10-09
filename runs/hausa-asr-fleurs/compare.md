# Compare: `NCAIR1/Hausa-ASR` (A) vs `openai/whisper-small` (B)

- Same 621 utterances, same references, rules 1.0.0. Dropped (not in both runs): 0 from A, 0 from B.
- Paired cluster bootstrap, 10000 resamples over 331 clusters (`group` (FLEURS sentence id) for 621 utterances). Δ = A − B; negative means A makes fewer errors.
- Headline, fixed in advance: **standard WER**. The table shows all 6 tests (3 normalisation modes × WER/CER); read the others as supporting evidence, not as extra chances to find a difference.

| mode | metric | A % | B % | Δ points | 95% CI | p | MDE (points) |
|---|---|---|---|---|---|---|---|
| raw | WER | 44.48 | 94.69 | -50.21 | [-51.81, -48.57] | < 0.001 | 2.32 |
| raw | CER | 16.45 | 36.06 | -19.61 | [-20.43, -18.79] | < 0.001 | 1.17 |
| **standard** | WER | 30.19 | 89.81 | -59.62 | [-61.00, -58.17] | < 0.001 | 2.01 |
| standard | CER | 10.08 | 30.88 | -20.80 | [-21.67, -19.95] | < 0.001 | 1.21 |
| lenient | WER | 27.33 | 89.38 | -62.05 | [-63.41, -60.60] | < 0.001 | 2.00 |
| lenient | CER | 9.34 | 30.39 | -21.06 | [-21.91, -20.23] | < 0.001 | 1.19 |

**WER (standard):** NCAIR1/Hausa-ASR makes fewer errors than openai/whisper-small: 59.62 points (95% CI -61.00 to -58.17, p < 0.001).

**CER (standard):** NCAIR1/Hausa-ASR makes fewer errors than openai/whisper-small: 20.80 points (95% CI -21.67 to -19.95, p < 0.001).

---

*N-ATLAS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.*
