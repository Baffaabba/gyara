# NAIC 2026: Execution Plan (Phase 3 blueprint)

Transcribed from `Naic Execution Plan.pdf`. The PDF is the original.
Submit by **Sun 11 Oct 2026**; official deadline Mon 12 Oct 11:59 PM WAT
(Monday is buffer only).

## 0. Decisions and assumptions

| Item | Decision |
| --- | --- |
| Problem statement | 01 Developer Infrastructure |
| Track | Innovation & Enterprise (startup, CAC registered) |
| Primary user | Developers & researchers building Hausa speech apps |
| Paying customer (later) | Hausa media houses & content creators (subtitles) |
| Direction | Layer A: Hausa benchmark/eval kit (also the fallback submission). Layer B: correction → measure → dataset → retrain loop on top of A |
| Differentiator | The loop. Transcription alone is not new |
| Spelling normalisation | Rules fixed on Day 1, owned by a strong written-Hausa speaker |
| External testers | 2–3 named by the PM (outside the team) |
| GPU | Team says sufficient; verified at gate G0 |

Assumptions: 2 developers (Dev-ML: models, eval, fine-tuning; Dev-App: UI,
SDK, deployment), PM = Baffa; full-time until 11 Oct; consented,
rights-cleared Hausa audio, target ≥ 2 hours, multiple speakers/dialects.

## 1. Milestones and gates

| Milestone | Date | Outcome |
| --- | --- | --- |
| M0 Setup & gates | Tue 6 Oct | Access, hardware, repo, roles, testers, consent |
| M1 Benchmark kit (Layer A) | Wed 7 Oct | Normalisation rules + eval CLI + baseline WER on FLEURS & own audio |
| M2 Core loop | Thu 8 Oct | Transcribe → correct → score → export `.srt` + dataset |
| M3 N-ATLaS + SDK + deploy | Fri 9 Oct | Measured LLM suggestions, Python package, live demo; testers get the build |
| M4 Fine-tune kit + validation | Sat 10 Oct | Fine-tune script + before/after on held-out; tester feedback fixed |
| M5 Submission | Sun 11 Oct | All 7 components submitted |
| Buffer | Mon 12 Oct | Emergencies only |

Gates: **G0** (Tue) both models run on the team GPU, else 4-bit / GGUF and
Colab/Kaggle. **G1** (Wed) baseline WER with normalisation, else freeze to
Layer A. **G2** (Fri) demo reachable by testers, else local-install guide.
**G3** (Sun 6 PM) every DoD item ticked → submit.

## 2. Tasks (abridged; owners in brackets)

- **M0**: HF accounts + licences accepted (Devs); `nvidia-smi` recorded (ML);
  smoke test one 30 s clip + one N-ATLaS prompt (ML); repo + board (App);
  2–3 testers agreed (PM); audio + signed consent + data register (PM);
  normalisation owner named (PM); out-of-scope list frozen (PM).
- **M1**: normalisation rules v1 + tests (Hausa owner + ML); held-out test set:
  FLEURS test + ≥ 30 min own audio, never trained on (ML + PM); `gyara eval`
  CLI (ML); baselines for Hausa-ASR and whisper-small (ML); error analysis in
  `docs/` (ML); business page draft (PM).
- **M2**: VAD chunking ≤ 30 s + timestamps, a 20-min file with no loops (ML);
  `.srt`/`.vtt` export (App); Gradio correction UI with live WER/CER (App);
  SQLite storage that survives restart (App); JSONL + audio dataset export,
  `load_dataset()` works (ML); N-ATLAS integration evidence draft (PM).
- **M3**: N-ATLaS suggestions (4-bit), accept/reject, never auto-overwrite
  (ML); measured suggestion effect on the test set, reported honestly (ML);
  optional English subtitles (ML); `pip install -e .` + CLI works on a fresh
  machine (App); demo on HF Space (App); build + test script + feedback form
  to testers (PM).
- **M4**: fine-tune script + config (ML); small run, held-out evaluation only,
  honest before/after (ML); tester evidence ≥ 2 (PM); fix top tester bugs
  (Devs); video script + shot list (PM).
- **M5**: code freeze, tag `v0.1.0` (App); docs: architecture, setup, usage,
  licence & attribution (Devs + PM); 3–5 min video, local run (PM + App); team
  profile + CAC (PM); portal filled, confirmation screenshot before midnight
  Sunday (PM).

## 3. Rituals

Daily 9:00 AM stand-up (15 min). Daily 9:00 PM check: any red item triggers a
scope cut, not overtime. One source of truth: the board + repo.

## 4. Out of scope (frozen)

Dubbing / TTS; Postgres, Redis, Celery, React, object storage; an
OpenAI-compatible API as a headline feature; claims of a "perfect" model or
guaranteed gains; Yoruba / Igbo (roadmap only).

## 5. Definition of Done

**Build**: public repo with tag `v0.1.0`, installable from README on a fresh
machine; transcribes ≥ 20 min Hausa audio with timestamps, no repetition loops;
correction UI with live WER/CER per segment; exports `.srt` and an HF-loadable
dataset; N-ATLaS genuinely used, its measured effect documented; fine-tuning
script runs end to end on exported data.

**Validation**: baseline WER/CER (raw + normalised) on FLEURS Hausa + own
held-out audio; honest before/after fine-tune on held-out data; ≥ 2 external
beta testers with written feedback + screenshots.

**Documentation & submission**: architecture, setup, usage, normalisation
rules, licence/attribution documented; 3–5 min end-to-end video; team profile +
CAC uploaded; submission confirmation by Sun 11 Oct.

**Compliance**: attribution line; fine-tuned models under the same N-ATLAS
licence; speaker consent on file for every non-public clip.
