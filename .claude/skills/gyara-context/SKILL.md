---
name: gyara-context
description: Ground truth for the Gyara repo — what it is, the NAIC 2026 deadline and Definition of Done, settled decisions, the module map with its interface contracts, and the hard rules (licence, consent, honest numbers, no training on held-out data). Load before any Gyara product, ML, app, QA, review or submission work.
---

# Gyara: project ground truth

Gyara (Hausa: "correction") is an open-source toolkit that helps developers
**measure, correct and improve Hausa speech-to-text**, turning every human
correction into training data that makes N-ATLAS better.

Entered in **NAIC 2026 (National AI Innovation Challenge)**, Problem Statement
**01 Developer Infrastructure**, track **Innovation & Enterprise** (registered
startup). Source documents: `docs/source/CONCEPT.md` and
`docs/source/EXECUTION-PLAN.md`. Judging rubric: the `naic-rubric` skill.

## Deadline

**Submit by Sun 11 Oct 2026, 6 PM WAT** (gate G3). The official deadline is
Mon 12 Oct 11:59 PM WAT; Monday is buffer only. Every scope question is
answered against this date. A smaller thing that works beats a bigger thing
that half works.

## The loop (the differentiator)

transcribe → suggest (N-ATLaS) → human corrects → measure (WER/CER) →
export (`.srt` + HF dataset) → fine-tune → re-measure on held-out data.

Transcription alone is not new. **The loop is.** Layer A (benchmark/eval kit)
is the fallback submission if Layer B (the loop) slips.

## Settled decisions (do not reopen without new evidence)

- Python package `gyara` + Typer CLI (`gyara transcribe | eval | compare | export | finetune | ui | fetch-fleurs`).
- Correction UI is **Gradio**. Storage is **SQLite**. No Postgres, Redis,
  Celery, React or object storage.
- Speech model: `NCAIR1/Hausa-ASR` (Whisper-small fine-tune). Comparison
  baseline: `openai/whisper-small`.
- LLM: `NCAIR1/N-ATLaS` (Llama-3 8B fine-tune), suggestions only, **never
  auto-applied**. Optional Hausa→English subtitles.
- Hausa only. Uploaded files only (no live). No dubbing, no diarisation.
- Demo hosted on a Hugging Face Space (fallback: local-install guide).
- Normalisation rules are versioned in `gyara/normalize.py`
  (`RULES_VERSION`) and written in `docs/NORMALISATION.md`; a strong
  written-Hausa speaker owns them.

## Module map and contracts (verify before citing; files move)

| Module | Contract |
| --- | --- |
| `gyara/normalize.py` | `get_normalizer(mode)` for `raw`, `standard`, `lenient`. `standard` keeps hook errors; `lenient` folds hooks. |
| `gyara/metrics.py` | `score_corpus(ids, refs, hyps, normalizer, groups) -> CorpusScore`. Pooled WER (sum errors / sum ref words). |
| `gyara/stats.py` | `bootstrap_rate` (cluster CI), `paired_bootstrap` (A vs B, p, MDE), `spearman`. |
| `gyara/hallucination.py` | `check(text, duration)` loop detection; `trim_loop`. |
| `gyara/subtitles.py` | `build_cues(segments)`, `to_srt`, `to_vtt`. |
| `gyara/manifest.py` | JSONL manifest `Row`; `read`, `write`, `leakage_check(train, heldout)`. |
| `gyara/audio.py` | `load(path, sr=16000) -> np.float32 mono`. Any format, ffmpeg via imageio-ffmpeg. |
| `gyara/vad.py` | `chunk(audio, sr, max_s=30) -> list[(start_s, end_s)]`. Silero, energy fallback. |
| `gyara/asr.py` | `Transcriber(model_id, device=None, language="hausa")`; `.transcribe(path or array) -> list[Segment]`; `Segment(start, end, text, avg_logprob, compression_ratio, flags, words)`. |
| `gyara/evaluate.py` | Runs a model over a manifest → `runs/<name>/{predictions.jsonl, metrics.json, report.md}`. `compare(run_a, run_b)` paired. |
| `gyara/data.py` | `fetch_fleurs(split)` → manifest + wavs under `data/fleurs/`. |
| `gyara/suggest.py` | `Suggester(backend)`; `.suggest(text) -> Suggestion(original, suggested, edits, accepted_by_guard, reason)`. Backends: `transformers`, `openai` (any OpenAI-compatible server), `none`. |
| `gyara/store.py` | SQLite: `assets`, `segments`, `revisions`, `suggestions`. Only `verified` segments export to training. |
| `gyara/export.py` | Dataset export as HF `audiofolder`: `audio/*.wav` + `metadata.jsonl` (`file_name, transcription, speaker, dialect, duration, asset_sha256, draft`) + `README.md` dataset card. |
| `gyara/finetune/` | `train.py` + `config.yaml`. Reads an export dir; refuses to start if `leakage_check` against the held-out manifest is not clean. |
| `gyara/ui/app.py` | Gradio correction screen. Root `app.py` launches it for HF Spaces. |
| `gyara/cli.py` | Typer entry point. |

## Hard rules

1. **Honest numbers.** Every WER/CER is reported raw, `standard` and
   `lenient`, with a 95% bootstrap CI. Every "better than" claim comes from
   `paired_bootstrap`. A non-significant result is reported as "no reliable
   difference", never as a gain. See the `eval-integrity` skill.
2. **Held-out is sacred.** Nothing in a held-out manifest, and no speaker in
   it, is ever trained on. `leakage_check` enforces it.
3. **Humans decide.** N-ATLaS output is a suggestion; it never overwrites
   text. Only human-verified segments become training data.
4. **Licence and attribution.** Every public surface (README, UI footer,
   model card, dataset card, video) carries: *"N-ATLAS is an initiative of the
   Federal Ministry of Communications, Innovation and Digital Economy, and
   powered by Awarri Technologies."* (`gyara.ATTRIBUTION`). Fine-tuned models
   keep the N-ATLAS licence; renamed versions say "Powered by Awarri". Free use
   is capped at 1,000 active users.
5. **Consent.** Every non-public audio clip has written speaker consent
   (NDPA). The data register is `data/REGISTER.csv`. No surveillance,
   profiling or impersonation uses.
6. **Never commit audio, tokens or `.env`.**

## Environment

- Dev laptop: Windows, **no usable GPU**, no system ffmpeg. Venv at
  `.venv/` (`.venv/Scripts/python`). Code must run on CPU (slowly) and on
  CUDA (team GPU, Colab, Kaggle).
- Models are gated on Hugging Face: `HF_TOKEN` env var or `huggingface-cli login`.
- Tests: `.venv/Scripts/python -m pytest -q`. Tests marked `slow` download
  models/data and are skipped by default.

## Git

Never commit to `main` or `dev`. Work on feature branches. Ask Baffa before
every commit.
