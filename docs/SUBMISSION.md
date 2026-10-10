# Submission package

NAIC 2026, Problem Statement 01 (Developer Infrastructure), Innovation &
Enterprise track. Internal deadline **Sun 11 Oct, 6 PM WAT**. Official
deadline Mon 12 Oct, 11:59 PM WAT. Late entries are not accepted.

Re-check https://ncair.nitda.gov.ng/naic/ on Sunday morning. The organisers
edit the page.

## The seven required components

| # | Component | What we submit | Where | Status |
| --- | --- | --- | --- | --- |
| 1 | Working artefact | Public repo tagged `v0.1.0` that installs from the README; the correction app runs locally (`gyara ui`), and testers reach it through a temporary `gyara ui --share` link. No hosted demo: Gradio Spaces now need HF PRO (DECISIONS, 9 Oct). Benchmark results. Dataset and fine-tuned model are not published. | Repo https://github.com/Baffaabba/gyara. Tag `v0.1.0`: pending (due Sun 12:00). Results `docs/BENCHMARK.md`, `runs/*/report.md` | Amber |
| 2 | N-ATLAS integration evidence | Every place each N-ATLAS model is used, with module names, what is measured, and licence compliance | `docs/N-ATLAS-INTEGRATION.md` | Amber |
| 3 | Real-world validation | (a) WER for `NCAIR1/Hausa-ASR` with CIs on FLEURS Hausa test (done; the model card publishes none, and we found none) and on our own held-out audio (pending). (b) ≥ 2 external testers, written feedback and screenshots (pending). (c) Hours of human-corrected audio collected (pending). | `docs/BENCHMARK.md`, `runs/`, `docs/validation/` | Amber (FLEURS done; own audio and testers red) |
| 4 | Technical documentation | Architecture, setup, usage, normalisation rules, licence and attribution | `README.md`, `docs/NORMALISATION.md`, `docs/N-ATLAS-INTEGRATION.md` | Amber |
| 5 | Video (3–5 min, end to end) | Local run, the loop in one take, numbers with CIs on screen | Video: pending, not recorded. Script `docs/VIDEO-SCRIPT.md`: not written yet | Red |
| 6 | Team profile | Names, roles, what each person shipped | `docs/TEAM.md` | Red |
| 7 | Endorsement / registration | CAC certificate | CAC certificate PDF: pending (Baffa) | Red |

All materials in English.

## The six criteria: our strongest evidence

No weights are published. The page says judging "focuses heavily on the
quality of the working build and the depth of its integration with N-ATLAS".
Lead with those two.

| Criterion | Strongest evidence | Show it in |
| --- | --- | --- |
| Working artefact & technical rigour | Installs from the README into a fresh venv (QA check, 9 Oct; a clean-clone re-check is pending). Tests pass. Every WER has a 95% cluster-bootstrap CI; every comparison is a paired test. The fine-tune refuses to start if a held-out speaker is in the training data. | README, `tests/`, `docs/validation/QA-2026-10-09-install.md`, video |
| N-ATLAS integration | Both N-ATLAS models do real work. `NCAIR1/Hausa-ASR` is the engine we benchmark and fine-tune. `NCAIR1/N-ATLaS` gives guarded spelling suggestions, and a person accepts or rejects each one. Its effect is measured and small: accepting every guarded suggestion, with no human review, changed standard WER by −0.18 points (95% CI −0.30 to −0.07, p = 0.002, MDE 0.17, n = 621 FLEURS test clips). English subtitles through N-ATLaS are built but not yet tested on real audio. A no-GPU route through community GGUF builds is supported in code but not yet tested. | `docs/N-ATLAS-INTEGRATION.md`, `docs/BENCHMARK.md`, `runs/hausa-asr-fleurs-natlas/`, video |
| Real-world validation | On the FLEURS Hausa test set, `NCAIR1/Hausa-ASR` gets 30.2% of words wrong (WER 30.19%, 95% CI 28.63–31.84; 621 clips, 331 sentences). Character error rate is 10.1%. Caveat: its model card does not list its training data, so if FLEURS was in it this score may be optimistic. Own held-out audio: pending; if not measured by Sat 10 Oct 12:00 we say "not measured". Fine-tune before/after (FLEURS train, measured once on FLEURS test): pending. Testers: pending, none named yet. Hours of human-corrected audio: pending, none yet. | `docs/BENCHMARK.md`, `runs/`, `docs/validation/` |
| Impact potential | Any Hausa speech developer, and NCAIR itself, gets a trustworthy benchmark and a way to turn corrections into training data. Hausa media get subtitles. | `docs/BUSINESS.md` |
| Scalability & sustainability | Open-source core; hosted subtitles and data services as revenue; licence-cap plan (1,000 users, then an Awarri commercial licence); other languages by changing the model id. | `docs/BUSINESS.md` |
| Team capability | Who shipped what, the gates we held, the decisions log. | `docs/TEAM.md`, `docs/STATUS.md`, `docs/DECISIONS.md` |

Quote the problem statement: it lists a "fine-tuning starter kit with
training scripts and evaluation tools" as an example. That is Gyara.

## Pre-submit checklist (Sun, before 2 PM)

Every line needs a tick and a person.

- [ ] Fresh machine (or fresh venv): clone, follow README word for word, `gyara --help` works.
- [ ] `pytest -q` green. Paste the last line here: (not run yet)
- [ ] Tag `v0.1.0` pushed. Repo is public. Open it logged out.
- [ ] A tester outside the team opened a `gyara ui --share` link and transcribed a clip. (No hosted demo; DECISIONS, 9 Oct.)
- [ ] No double-bracket placeholder left in README, `docs/SUBMISSION.md`, `docs/N-ATLAS-INTEGRATION.md`, `docs/TEAM.md`, `docs/BUSINESS.md`.
- [ ] Every number in docs and video matches a `metrics.json` file and carries its CI and n.
- [ ] No sentence says "better" without a paired test behind it. Non-significant results say "no reliable difference".
- [ ] Attribution line on README, UI footer, `gyara about`, dataset card, model card, video end card.
- [ ] No audio, tokens or `.env` in git (`git ls-files | grep -Ei "\.(wav|mp3|m4a|env)$"` returns nothing).
- [ ] `data/REGISTER.csv`: every non-public clip has a consent form and date.
- [ ] ≥ 2 tester feedback forms and screenshots in `docs/validation/`.
- [ ] Video 3:00–5:00, plays logged out, attribution at the end.
- [ ] CAC PDF opens. Team profile has no placeholders.

## Portal-day runbook (Sun 11 Oct)

| Time (WAT) | Who | Do |
| --- | --- | --- |
| 9:00 | All | Stand-up. Read this checklist. Any red item: cut it now, not at 5 PM. |
| 9:00–12:00 | Devs | Code freeze at 12:00. Last fixes only. |
| 10:00 | Baffa | Re-read the NAIC page for rule changes. Log in to the portal and check every field it asks for. |
| 12:00 | Dev-App | Tag `v0.1.0`, push. Re-run the README install from a clean clone of the tag. |
| 12:00–14:00 | Baffa + Dev-App | Final video cut (if not already done); upload; test the link logged out. |
| 14:00 | Baffa | Run the pre-submit checklist above. |
| 15:00 | Baffa | Fill the portal. Paste URLs from this file. Upload CAC and team profile. |
| 16:00 | Baffa | Submit. Screenshot the confirmation. Save it in `docs/validation/submission-confirmation_20261011.png`. |
| 16:00–18:00 | — | Slack for portal trouble. G3 is 18:00. |
| Mon 12 Oct | — | Buffer for emergencies only. No new features. |

If the portal fails: screenshot the error with the time, email the organisers
from the registered address with all links, and retry every hour. Keep the
email as proof.
