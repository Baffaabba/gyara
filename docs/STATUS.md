# Status

Last check: **Thu 8 Oct 2026, night.** Owner of this file: PM (Baffa).
Submit by **Sun 11 Oct, 6 PM WAT** (gate G3). Monday is buffer only.

Rule: a box with no evidence is red. Green means a file, command or URL proves it.

## Gates

| Gate | Due | Test | State | What happens now |
| --- | --- | --- | --- | --- |
| G0 | Tue 6 Oct | Both models run on the team GPU | **Red: not verified.** The dev laptop has no GPU. No `nvidia-smi` record, no smoke-test output. | Plan fallback: 4-bit / GGUF and Colab or Kaggle. Run one 30 s clip through `NCAIR1/Hausa-ASR` and one N-ATLaS prompt on Colab **Fri morning**, save the output under `runs/`. |
| G1 | Wed 7 Oct | Baseline WER with normalisation | **Red: overdue.** `runs/` is empty. | Plan says freeze to Layer A. Recommendation: do not freeze yet. Get the FLEURS baseline on Colab by **Fri 12:00**. If there is no `runs/hausa-asr-fleurs/metrics.json` by then, freeze to Layer A and cut the fine-tune. |
| G2 | Fri 9 Oct | Demo reachable by testers | Red: not started. No Space, no `app.py`. | If the Space is not up by Fri 6 PM, testers use the local-install guide (`gyara ui --share`). |
| G3 | Sun 11 Oct, 6 PM | Every DoD box green | Red | Submit. |

## Definition of Done

| Item | Owner | Status | Evidence |
| --- | --- | --- | --- |
| **Build** | | | |
| Core scoring: normalisation, pooled WER/CER, bootstrap CIs, paired test, loop detection, manifests, leakage check | Dev-ML | Green | `gyara/{normalize,metrics,stats,hallucination,manifest}.py`; `tests/` (93 pass, 1 fail in the eval test being built, Thu night) |
| Public repo, tag `v0.1.0`, installs from README on a fresh machine | Dev-App | Red | No remote, no tag, no README yet. Fresh-machine install not tried. |
| Transcribes ≥ 20 min Hausa with timestamps, no repetition loops | Dev-ML | Amber | `gyara/asr.py`, `vad.py`, `hallucination.py` written. No 20-min run saved. |
| Correction app with live WER/CER per segment | Dev-App | Amber | `gyara/ui/` being built. No `app.py` yet. |
| Exports `.srt` and an HF-loadable dataset | Dev-App / Dev-ML | Amber | `gyara/subtitles.py`, `export.py`, `tests/test_store_export.py`. `load_dataset()` on a real export not shown. |
| N-ATLaS used for real; measured effect documented | Dev-ML | Amber | `gyara/suggest.py` (guard + backends), `tests/test_suggest_finetune.py`. No measured run. |
| Fine-tune script runs end to end on exported data | Dev-ML | Amber | `gyara/finetune/train.py`, `config.yaml`. No run. |
| **Validation** | | | |
| Baseline WER/CER (raw, standard, lenient) on FLEURS Hausa test | Dev-ML | Red | `[[RESULT: runs/hausa-asr-fleurs/metrics.json]]` |
| Same on own held-out audio (≥ 30 min, speakers not in training) | Dev-ML + Baffa | Red | `data/heldout/` is empty. Needs Baffa's audio + references. |
| Honest before/after fine-tune on held-out data | Dev-ML | Red | `[[RESULT: runs/<tuned>-vs-base/compare.json]]` |
| ≥ 2 external beta testers, written feedback + screenshots | Baffa | Red | `docs/validation/` holds the kit only. No testers named. |
| **Docs and submission** | | | |
| Architecture, setup, usage | Devs | Amber | Being written (README). |
| Normalisation rules documented and signed off | Hausa owner | Amber | `docs/NORMALISATION.md` exists. No owner named, no sign-off. |
| Licence and attribution documented | PM | Amber | `gyara.ATTRIBUTION`, `gyara about`, `docs/N-ATLAS-INTEGRATION.md`. UI footer pending. |
| N-ATLAS integration evidence | PM | Amber | `docs/N-ATLAS-INTEGRATION.md` drafted; needs result numbers. |
| Business page | PM | Amber | `docs/BUSINESS.md` drafted. |
| 3–5 min end-to-end video | Baffa + Dev-App | Red | Script in `docs/VIDEO-SCRIPT.md`. Not recorded. |
| Team profile | Baffa | Red | Template in `docs/TEAM.md`. |
| CAC certificate uploaded | Baffa | Red | Not in hand. |
| Portal filled, confirmation screenshot | Baffa | Red | — |
| **Compliance** | | | |
| Attribution on every public surface | All | Amber | In code, CLI and fine-tune model card. UI footer, dataset card, Space, video pending. |
| Fine-tuned models keep the N-ATLAS licence, "Powered by Awarri" | Dev-ML | Amber | `gyara/finetune/train.py` writes it into the model card. No model published. |
| Written consent for every non-public clip | Baffa | Red | `data/REGISTER.csv` has the header only. Form: `docs/validation/CONSENT-FORM.md`. |

## Baffa's to-do (human-only)

Agents cannot do these. Ordered by deadline.

| # | Task | Deadline | Done when |
| --- | --- | --- | --- |
| 1 | HF accounts for both devs; accept the gated licences for `NCAIR1/Hausa-ASR` and `NCAIR1/N-ATLaS`; set `HF_TOKEN` | Thu tonight | Both model pages say "access granted" |
| 2 | GPU access: Colab (Pro if possible) or Kaggle for the devs | Thu tonight | A dev has run one cell on a GPU |
| 3 | Name 2–3 external testers (outside the team), with email and role | **Fri 9 Oct, 10 AM** | Names written in `docs/validation/README.md` |
| 4 | Name the Hausa normalisation owner | Fri 10 AM | Name written in `docs/NORMALISATION.md` header (ask a dev to add it) |
| 5 | Print or send consent forms (`docs/validation/CONSENT-FORM.md`); get the Hausa section translated by a fluent speaker | Fri 12:00 | Signed forms for every speaker before recording |
| 6 | Collect ≥ 2 h own Hausa audio: several speakers, several dialects; log each clip in `data/REGISTER.csv` | Fri 6 PM (first 30 min by Fri 12:00) | Register filled; audio on the team drive, **not in git** |
| 7 | Of that, ≥ 30 min held out, from speakers who are not in the rest; references typed by the Hausa owner | Fri 6 PM | Held-out folder handed to Dev-ML |
| 8 | Hausa owner signs off `docs/NORMALISATION.md` | Fri 6 PM | Name + date in the file |
| 9 | Send testers the kit (`docs/validation/TESTER-KIT.md`) | Fri 6 PM | Sent |
| 10 | Chase tester feedback + screenshots | Sat 6 PM | ≥ 2 filled forms in `docs/validation/` |
| 11 | CAC certificate as PDF | Sat 12:00 | File in hand |
| 12 | Team profile filled (`docs/TEAM.md`) | Sat 6 PM | No placeholders left |
| 13 | Record the video (`docs/VIDEO-SCRIPT.md`) | Sun 12:00 | Uploaded, link works logged out |
| 14 | Fill the portal; screenshot the confirmation | Sun 6 PM | Screenshot saved |

## Decisions only Baffa can make

1. **G1 is overdue.** Do we keep building the loop, or freeze to Layer A now?
   Recommendation: keep going until Fri 12:00. If there is no FLEURS baseline
   by then, freeze to Layer A and cut the fine-tune.
2. **Who are the testers and the Hausa owner?** Without names by Fri morning,
   the validation criterion is at risk and cannot be recovered on Sunday.
