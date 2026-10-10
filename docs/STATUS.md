# Status

Last check: **Fri 9 Oct 2026, 16:00 WAT.** Owner of this file: gyara-pm (for Baffa).
Submit by **Sun 11 Oct, 6 PM WAT** (gate G3). Monday is buffer only.

Rule: a box with no evidence is red. Green means a file, command or URL proves it.
"Not re-run" means the evidence is a report from earlier today, not a check made
at the time written above.

## Gates

| Gate | Due | Test | State | What happens now |
| --- | --- | --- | --- | --- |
| G0 | Tue 6 Oct | Both models run on a GPU | **Green (late, via the plan's fallback).** Free Colab T4, not a team GPU. Hausa-ASR fp16 and N-ATLaS 4-bit both ran (7.58 GB peak). | Nothing. Evidence: `runs/g0-smoke/nvidia-smi.txt`, `asr.json`, `natlas.json`, `env.json`. |
| G1 | Wed 7 Oct | Baseline WER with normalisation | **Green for FLEURS (2 days late). Own audio: missing.** Hausa-ASR 30.19% standard WER [28.63, 31.84], n = 621. | Nothing for FLEURS. Own audio: see decision 2 below. Evidence: `runs/hausa-asr-fleurs/metrics.json`, `docs/BENCHMARK.md`. |
| G2 | Fri 9 Oct | Demo reachable by testers | **Red.** The Hugging Face Space could not be deployed: Gradio Spaces now need a paid HF PRO account (reported by the main session, not re-checked here). | The plan's fallback applies: testers get the local-install guide. For the correction screen, a dev runs `gyara ui --share` (on a laptop or Colab) and sends the link during a set test window. The plan also allowed "or team server". |
| G3 | Sun 11 Oct, 6 PM | Every DoD box ticked | Red | Submit what is green. Cut, don't stretch. |

## Definition of Done (Execution Plan, section 5)

| Item | Owner | Status | Evidence |
| --- | --- | --- | --- |
| **Build** | | | |
| Public repo, tag `v0.1.0`, installs from README on a fresh machine | gyara-app | **Amber** | Public at github.com/Baffaabba/gyara (branch `feat/foundation`). **No tag yet** (`.git/refs/tags/` is empty). Fresh-venv install passed on Windows earlier today (`docs/validation/QA-2026-10-09-install.md`), but that was before the repo had commits. A real `git clone` install has not been re-run. |
| Transcribes ≥ 20 min Hausa with timestamps, no repetition loops | gyara-ml | **Amber** | Code: `gyara/asr.py`, `vad.py`, `hallucination.py`. Ran on 621 FLEURS clips; the 65 clips over 30 s were split by VAD and scored (`docs/BENCHMARK.md`, caveat 5). **No 20-minute file run saved.** |
| Correction app with live WER/CER per segment | gyara-app | **Amber** | `gyara/ui/app.py`, `tests/test_ui.py`. Opens without a token (QA report). Nobody outside the team has corrected a full file in it. |
| Exports `.srt` and an HF-loadable dataset | gyara-app / gyara-ml | **Amber** | `gyara/subtitles.py`, `export.py`; `tests/test_store_export.py:216` runs `load_dataset("audiofolder")` on a test export. `runs/g0-smoke/asr.srt` exists. Not shown: an export of real corrected audio, or the `.srt` playing in a video player. |
| N-ATLaS used for real; measured effect documented | gyara-ml | **Green** | `runs/hausa-asr-fleurs-natlas/metrics.json`; `docs/BENCHMARK.md`: accepting every guarded suggestion changed standard WER by −0.18 points (95% CI −0.30 to −0.07, p = 0.002, MDE 0.17, n = 621). Small, and reported as small. |
| Fine-tune script runs end to end on exported data | gyara-ml | **Red** | `gyara/finetune/train.py`, `config.yaml`, `notebooks/finetune_colab.ipynb`, unit tests. **Never run.** The notebook needs an export zip and stops without one (cell 8). |
| **Validation** | | | |
| Baseline WER/CER (raw, standard, lenient) on FLEURS Hausa test | gyara-ml | **Green** | `runs/hausa-asr-fleurs/`, `runs/whisper-small-fleurs/`, `runs/leaderboard.md`, `docs/BENCHMARK.md` (reviewed by gyara-stats). |
| Same on own held-out audio (≥ 30 min, speakers not in training) | gyara-ml + Baffa | **Red** | `data/heldout/` is empty. `data/REGISTER.csv` has the header only. |
| Honest before/after fine-tune on held-out data | gyara-ml | **Red** | No run. Proposal: fine-tune on FLEURS train, measure once on FLEURS test (decision 1). |
| ≥ 2 external beta testers, written feedback + screenshots | Baffa | **Red** | `docs/validation/README.md` still has `[[TESTER 1]]` placeholders. No names. **This one cannot be rescued on Sunday.** |
| **Docs and submission** | | | |
| Architecture, setup, usage | gyara-app | **Amber** | `README.md`, `docs/ARCHITECTURE.md`. **README "Results" still says "Pending"**, though `docs/BENCHMARK.md` has the numbers. README still has a "Deploy to a Space" section that no longer works on a free account. |
| Normalisation rules documented and signed off | Hausa owner | **Amber** | `docs/NORMALISATION.md` (rules 1.0.0). Header says `Owner: [[name]]`, "awaiting sign-off". |
| Licence and attribution documented | gyara-pm | **Green** | `LICENSE` (Apache-2.0), README "Licence and attribution", `gyara about`, UI footer (QA report), `space/README.md`, model-card code in `train.py`. |
| N-ATLAS integration evidence | gyara-pm | **Amber** | `docs/N-ATLAS-INTEGRATION.md`. Still full of `[[RESULT: ...]]` placeholders, though the FLEURS and N-ATLaS numbers now exist. |
| Business page | gyara-pm | **Green (draft)** | `docs/BUSINESS.md`. |
| 3–5 min end-to-end video | Baffa + gyara-app | **Red** | **`docs/VIDEO-SCRIPT.md` does not exist** (the previous status said it did). Not recorded. |
| Team profile | Baffa | **Red** | **`docs/TEAM.md` does not exist** (the previous status said a template did). |
| CAC certificate uploaded | Baffa | **Red** | Not in hand. |
| Portal filled, confirmation screenshot | Baffa | **Red** | — |
| **Compliance** | | | |
| Attribution on every public surface | All | **Amber** | README, BENCHMARK, Space README, UI footer, `gyara about`, model-card code. Missing until made: video end card, a real dataset card. |
| Fine-tuned models keep the N-ATLAS licence, "Powered by Awarri" | gyara-ml | **Amber** | Written into the model card by `train.py` (`write_model_card`). No model trained yet. |
| Written consent for every non-public clip | Baffa | **Green while we use only FLEURS; red the moment own audio is used.** | `data/REGISTER.csv` header only. Form: `docs/validation/CONSENT-FORM.md`. No own audio exists yet, so nothing is unconsented. |

## Traceability to the source documents

Source: the two PDFs Baffa sent on 9 Oct (`Gyara Concept Document.pdf` v1.0,
6 Oct; `Naic Execution Plan.pdf`, draft). They are the same versions as
`docs/source/`, but the Markdown copies leave out each task's "done when" line.
Those lines are included here, because they are the test.

### Concept Document

| # | Promise | Status | Evidence / gap |
| --- | --- | --- | --- |
| C1 | Transcription with word/segment timestamps, long files in ≤ 30 s chunks | Partial | `asr.py`, `vad.py`. No long real recording run. |
| C2 | N-ATLaS suggestions, never automatic | Done | `suggest.py` guard; `tests/test_suggest_finetune.py`; effect measured (BENCHMARK). |
| C3 | Correction screen: listen, edit, live score | Partial | `ui/app.py`, `tests/test_ui.py`. No outside user yet. |
| C4 | WER and CER, raw and normalised | Done | `runs/hausa-asr-fleurs/metrics.json`. |
| C5 | Documented rules: hooks, apostrophes, numbers, casing, punctuation | Partial | `docs/NORMALISATION.md`. No owner, no sign-off. |
| C6 | `.srt` and `.vtt`, ready for YouTube / CapCut / Premiere | Partial | `subtitles.py` + tests. Not opened in any of those. |
| C7 | Dataset export, HF format | Partial | `export.py`; `load_dataset` test on a fixture only. |
| C8 | Fine-tuning starter kit: script + config | Partial | Exists; never run. |
| C9 | Benchmark on FLEURS test **and our own audio** | Partial | FLEURS done. Own audio missing. |
| C10 | Optional Hausa → English subtitles | Partial (optional) | `suggest.translate`, `export srt --english`, unit tests. Never run on the real model. First to cut. |
| C11 | Python package + CLI (`transcribe`, `eval`, `export`) | Done | `gyara/cli.py`; QA fresh-venv install. |
| C12 | Live web demo | **Missing** | Space blocked (needs HF PRO). `--share` link is a stop-gap, not a lasting demo. |
| C13 | Setup, usage, architecture docs | Partial | README "Results: Pending" is out of date. |
| C14 | ≥ 2 external developer beta testers, written feedback | **Missing** | No names. |
| C15 | Honest before/after, even if small | Partial | N-ATLaS before/after done. Fine-tune before/after missing. |
| C16 | Out-of-scope list respected | Done | No dubbing, live, diarisation, OCR, other languages. The `openai` suggestion backend is a client, not an API we ship. |
| C17 | Honest limitations stated | Done | `docs/BENCHMARK.md` caveats 1–7. |
| C18 | Attribution on all public use | Partial | See compliance rows above. |
| C19 | Fine-tunes keep the licence; renamed ones say "Powered by Awarri" | Done in code | `train.py` `write_model_card`. Nothing published. |
| C20 | Written speaker consent (NDPA) | Not yet needed | Becomes a blocker the moment own audio is used. |
| C21 | Business model | Done (draft) | `docs/BUSINESS.md`. |

### Execution Plan tasks, with their "done when" lines

| Task | Done when (PDF) | Status | Evidence / gap |
| --- | --- | --- | --- |
| 0.1 Licences accepted | Both models download without 403 | Done | `runs/g0-smoke/asr.json`, `natlas.json` |
| 0.2 `nvidia-smi` recorded | Written in team notes | Done | `runs/g0-smoke/nvidia-smi.txt` (Colab T4) |
| 0.3 Smoke test 30 s clip + one N-ATLaS prompt | Output text saved | Done | `runs/g0-smoke/asr.json`, `natlas.json` |
| 0.4 Repo, README, task board | Repo link shared | Done | github.com/Baffaabba/gyara; this file serves as the board |
| 0.5 2–3 external testers agreed | Names + contacts in notes | **Missing** | `docs/validation/README.md` |
| 0.6 Audio + signed consent, source and licence per file | Data register exists | Partial | `data/REGISTER.csv` exists, empty |
| 0.7 Hausa normalisation owner | Name confirmed | **Missing** | `docs/NORMALISATION.md` header |
| 0.8 Out-of-scope list frozen | Team acknowledged | Done | `.claude/skills/gyara-context/SKILL.md` |
| 1.1 Normalisation rules v1 (incl. English loanwords) | Rules doc + code with tests | Partial | Code + `tests/test_normalize.py` done. Loanwords: documented as "not translated". Owner sign-off missing. |
| 1.2 Held-out set: FLEURS test + ≥ 30 min own audio | Manifest, never used for training | Partial | FLEURS test manifest (sha in BENCHMARK). Own part missing. |
| 1.3 `gyara eval` CLI | Runs end to end | Done | `runs/hausa-asr-fleurs/` |
| 1.4 Baselines Hausa-ASR + whisper-small | Results table committed | Done | `runs/leaderboard.md`, commit `4da947d` |
| 1.5 Error analysis: top misrecognised words, long-audio failures | Short report in `docs/` | Partial | Top 25 confusions in `runs/hausa-asr-fleurs/report.md`; BENCHMARK caveats 3–5. Long audio = FLEURS clips > 30 s only. |
| 1.6 Business page | Draft v1 | Done | `docs/BUSINESS.md` |
| 2.1 VAD chunking + timestamps | 20-min file, no loops | Partial | No 20-min run |
| 2.2 `.srt` / `.vtt` | Opens in a video player / YouTube | Partial | Not shown |
| 2.3 Gradio correction UI | A tester corrects a full file | Partial | No tester |
| 2.4 SQLite storage | Data survives restart | Partial | `store.py`; not shown on a real restart |
| 2.5 Dataset export | `load_dataset()` works on export | Partial | Fixture test only |
| 2.6 N-ATLAS integration draft | Draft v1 | Partial | Drafted; placeholders not filled |
| 3.1 N-ATLaS suggestions, 4-bit, accept/reject | Suggestions visible in UI | Partial | 4-bit run on Colab. Not shown in the UI with the real model. |
| 3.2 Measure suggestion impact | Reported honestly, even if no gain | Done | BENCHMARK, "N-ATLaS spelling suggestions" |
| 3.3 Optional English subtitles | English `.srt` export works | Partial (optional) | Unit tests only |
| 3.4 `pip install -e .` + CLI | Fresh install from README works | Partial | QA passed on a copied tree; re-run on a real clone |
| 3.5 Demo on HF Space (or team server) | Public URL works | **Missing** | Blocked by HF PRO. Fallback: `--share` + local install |
| 3.6 Build + script + form to testers | Testers confirm receipt | **Missing** | Kit ready: `docs/validation/TESTER-KIT.md`, `FEEDBACK-FORM.md` |
| 4.1 Fine-tune script + config | Runs end to end | Partial | Never run |
| 4.2 Small run, held-out only | Before/after table | **Missing** | See decision 1 |
| 4.3 Tester evidence | Folder complete, ≥ 2 testers | **Missing** | — |
| 4.4 Fix top tester bugs | Issues closed | Not started | Depends on 4.3 |
| 4.5 Video script + shot list | Approved by team | **Missing** | gyara-pm writes it by Sat 12:00 |
| 5.1 Code freeze, tag `v0.1.0` | Tag exists | Missing | Due Sun 12:00 |
| 5.2 Technical docs | Reviewed by PM | Partial | See docs rows above |
| 5.3 Video, local run | 3–5 min, end to end | Missing | Due Sun 12:00 |
| 5.4 Team profile (names, roles, affiliations) + CAC | Files ready | Missing | — |
| 5.5 Portal filled | Confirmation before midnight Sunday | Missing | We aim for 4 PM Sunday |

### Where the repo and the documents disagree

1. **README says results are "Pending".** `docs/BENCHMARK.md` and `runs/` have them. Fix the README before anyone outside sees it.
2. **`docs/N-ATLAS-INTEGRATION.md` and `docs/SUBMISSION.md` still hold `[[RESULT]]` placeholders** for numbers that now exist.
3. **"First public WER for Hausa-ASR"** (`docs/N-ATLAS-INTEGRATION.md`, the rubric notes). We cannot prove nobody else published one. Say "the model card publishes none, and we found none".
4. **"Runs on a laptop via GGUF"** (`docs/SUBMISSION.md`) has no run behind it. Either time one suggestion on a laptop and save it under `runs/`, or say "can run via GGUF (not measured)".
5. **English subtitles** are listed as an N-ATLaS role in README and SUBMISSION. They were never run on the real model. Say "optional, not yet tested on real audio", or run once.
6. **`docs/BENCHMARK.md` lines 147–149** say `predictions.jsonl` is git-ignored. `.gitignore` now keeps FLEURS predictions (DECISIONS, 9 Oct). The sentence is out of date; gyara-stats should confirm the counts can now be quoted.
7. **DECISIONS (9 Oct) says the demo Space is public** at `huggingface.co/spaces/Baffaabba/gyara`. It was never deployed. README and SUBMISSION still point to a Space.
8. **The previous status claimed `docs/VIDEO-SCRIPT.md` and `docs/TEAM.md` exist.** They do not.
9. The Execution Plan PDF puts the deadline in three ways: G3 at Sun 6 PM, task 5.5 "before midnight Sunday", DoD "by Sun 11 Oct". Our target stays Sun 4 PM submit, 6 PM gate.

## What is left, ranked

| # | Task | Owner | Deadline |
| --- | --- | --- | --- |
| 1 | Name 2 external developer testers (outside the team) and get a yes | **Baffa** | **Fri 9 Oct, 21:00** |
| 2 | Re-point tester kit, README and SUBMISSION from the Space to local install + `gyara ui --share`; fix README "Results" from BENCHMARK | gyara-app | Fri 21:00 |
| 3 | Make the fine-tune notebook run with no export (FLEURS train only); start the run on Colab T4 | gyara-ml | Start Fri 22:00; results zip Sat 10:00 |
| 4 | Send testers the kit; book a 1-hour `--share` window with each | Baffa (+ gyara-app runs the link) | Sat 09:00 |
| 5 | CAC certificate PDF | Baffa | Sat 12:00 |
| 6 | Video script and shot list (`docs/VIDEO-SCRIPT.md`) | gyara-pm | Sat 12:00 |
| 7 | Own audio, if decision 2 is yes: signed consent, register filled, recordings | Baffa | Sat 12:00 |
| 8 | Name the Hausa owner; owner signs `docs/NORMALISATION.md` | Baffa | Sat 18:00 |
| 9 | 20-min transcription run (own recording if consented, else stitched FLEURS test clips, labelled as such); a small real export → `load_dataset` → fine-tune smoke run with `eval.after: false` | gyara-ml | Sat 18:00 |
| 10 | Fine-tune before/after reviewed; fill N-ATLAS-INTEGRATION and SUBMISSION placeholders; fix claims 3–6 above | gyara-stats + gyara-pm | Sat 18:00 |
| 11 | Team profile (`docs/TEAM.md`: names, roles, affiliations, what each shipped) | Baffa | Sat 18:00 |
| 12 | Tester feedback forms + screenshots in `docs/validation/` | Baffa (chase) | Sun 10:00 |
| 13 | Fresh-clone install check; QA bugs S1–S5 re-checked | gyara-qa | Sun 10:00 |
| 14 | Record the video (local run, loop in one take, numbers with CIs) | Baffa + gyara-app | Sun 12:00 |
| 15 | Code freeze, tag `v0.1.0` | gyara-app | Sun 12:00 |
| 16 | Gate G3 review | gyara-judge | Sun 14:00 |
| 17 | Fill the portal, submit, screenshot | Baffa | Sun 16:00 |

## Cuts already proposed

- **English subtitles (optional):** cut from claims unless tested once by Sat 18:00. Saves no build time; removes an untested promise. Rubric cost: none.
- **Published dataset and published model:** cut. Saves Saturday time and any consent risk. Rubric cost: small; the repo and benchmark are the artefact.
- **Hosted demo:** replaced by the plan's fallback (local install + `--share`). Rubric cost: judges can't click a link after Sunday; they must install. The README path must work.

## Decisions only Baffa can make

1. **Fine-tune.** Recommendation: run it now on FLEURS train and measure once
   on FLEURS test. FLEURS train and test have different speakers and
   different sentences by design, so the test is fair. Cut the own-data
   fine-tune for this submission. Wording: "We fine-tuned NCAIR1/Hausa-ASR on
   the FLEURS Hausa train split with Gyara's kit and measured it once on the
   FLEURS test split: [result as found]. This shows the kit works end to end.
   It does not yet show the value of human corrections; we have no corrected
   own audio yet."
2. **Own held-out audio.** Recommendation: try a small set with a hard stop.
   If by **Sat 12:00** you have signed consent and at least 15 minutes from at
   least 3 new speakers, gyara-ml measures it and we report it with its wide
   interval. If not, we cut it and say plainly "own-audio benchmark: not
   measured". Either way, testers (decision-free, task 1) come first.
