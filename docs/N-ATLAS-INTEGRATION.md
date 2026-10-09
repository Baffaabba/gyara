# N-ATLAS integration

NAIC component 2. This page shows exactly where and how Gyara uses N-ATLAS.

> N-ATLAS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

## Summary

Gyara uses both N-ATLAS models for real work:

- **`NCAIR1/Hausa-ASR`** is the speech engine. We benchmark it, correct its
  output, and fine-tune it on the corrections.
- **`NCAIR1/N-ATLaS`** proofreads the transcripts (suggestions only, behind a
  guard) and translates Hausa subtitles into English.

`openai/whisper-small` is a comparison baseline only. It is never the engine.

## Where each model is used

| Model | Used for | Module | What we measure | Evidence |
| --- | --- | --- | --- | --- |
| `NCAIR1/Hausa-ASR` | Transcribe Hausa audio into timed segments, with a confidence score per segment and loop protection | `gyara/asr.py` (`Transcriber`) | — | `gyara transcribe` |
| `NCAIR1/Hausa-ASR` | Benchmark on FLEURS Hausa test and our own held-out audio: WER and CER, raw / standard / lenient, 95% cluster-bootstrap CI | `gyara/evaluate.py` | FLEURS: `[[RESULT: WER standard % [95% CI], n utts, from runs/hausa-asr-fleurs/metrics.json]]`. Own audio: `[[RESULT: runs/hausa-asr-own/metrics.json]]` | `runs/*/report.md` |
| `NCAIR1/Hausa-ASR` vs `openai/whisper-small` | Show what the N-ATLAS fine-tune adds over its base model, on the same clips | `gyara/evaluate.py` (`compare`, paired bootstrap) | `[[RESULT: delta points [CI], p, MDE, from runs/compare-hausa-asr-vs-whisper-small]]` | `gyara compare` |
| `NCAIR1/Hausa-ASR` | Fine-tune on human-verified corrections; measure before vs after on held-out speakers only | `gyara/finetune/train.py`, `config.yaml` | `[[RESULT: base vs tuned, delta points [CI], p, MDE, n, held-out set]]` | `gyara finetune` |
| `NCAIR1/N-ATLaS` | Suggest spelling and punctuation fixes (hooked letters ɓ ɗ ƙ ƴ, apostrophes, capitals). A human accepts or rejects. Never applied automatically. | `gyara/suggest.py` (`Suggester`, `guard`) | Effect of accepting every guarded suggestion, paired bootstrap on the same utterances: `[[RESULT: suggestion effect, delta points [CI], p, MDE, from runs/<name>/metrics.json suggestion_effect]]`. Also: share of suggestions the guard rejected `[[RESULT]]`. | `gyara eval --suggest openai --llm-url ...` |
| `NCAIR1/N-ATLaS` | Hausa → English subtitles | `gyara/suggest.py` (`translate`), `gyara/export.py` | Not scored (no English references). Shown as optional. | `gyara export srt --english` |
| N-ATLaS GGUF builds | Run N-ATLaS on a laptop CPU, no GPU: community GGUF builds served by llama.cpp, vLLM or Ollama behind an OpenAI-compatible endpoint | `gyara/suggest.py` (`openai` backend) | Seconds per suggestion on a laptop: `[[RESULT: timing from runs/<name>/metrics.json]]` | `llama-server -m N-ATLaS-Q4_K_M.gguf --port 8080` |

## Why N-ATLaS sits behind a guard

N-ATLaS reads text, not audio. It cannot know what was said. An LLM will also
"improve" a correct but rare word into a common one, or reply in English. So
`guard()` in `gyara/suggest.py` rejects a suggestion when:

- it is chatter or English, not a corrected Hausa sentence;
- it adds or removes words (one split/merge like *a kan* → *akan* is allowed);
- it rewrites a word beyond a small spelling distance, unless the change is
  only hooks, apostrophes, case or punctuation;
- it changes too many words.

A rejected suggestion equals the original text. So even a careless caller
changes nothing. Only human-verified segments become training data.

We report the measured effect as found. If the CI includes zero, we say "no
reliable difference" and give the minimum detectable effect.

## Licence compliance

| Rule (N-ATLAS licence) | How Gyara meets it |
| --- | --- |
| Attribution on public use | `gyara.ATTRIBUTION` holds the exact line. It appears in `gyara about`, the README, the app footer, the Space, dataset cards, model cards and the video. |
| Free use capped at 1,000 active end-users | Stated in `gyara about` and the README. Above the cap we need an Awarri / Ministry commercial licence (see `docs/BUSINESS.md`). |
| Derivatives keep the same licence | `gyara/finetune/train.py` writes the N-ATLAS licence into every fine-tuned model card. |
| Renamed derivatives say "Powered by Awarri" | The same model card names a renamed model "<name> (Powered by Awarri)". |
| No surveillance, profiling or impersonation | Out of scope by design: uploaded files only, no speaker identification, no voice cloning. |

Gyara's own code is Apache-2.0. The models keep their own licences.

## What we found, as feedback for NCAIR

These are useful to the N-ATLAS team. Gyara documents and works around each.

1. **No published WER.** The `NCAIR1/Hausa-ASR` model card publishes no
   accuracy figure. Gyara produces the first public one, with CIs and a fixed,
   versioned Hausa normalisation (`docs/NORMALISATION.md`).
2. **Processor does not load.** `WhisperProcessor.from_pretrained` fails on
   the repo. Gyara falls back to the `openai/whisper-small` processor, which is
   identical for every whisper-small fine-tune, and logs a warning
   (`gyara/asr.py`, `_load_processor`).
3. **No language token set.** The card calls `generate()` with no language.
   Gyara forces Hausa by default, allows `--language auto`, and records the
   setting in every run, because it can move WER a lot.
   `[[RESULT: WER with language=hausa vs auto, if run]]`

## The baseline in context

The Whisper paper (Radford et al., 2022, "Robust Speech Recognition via
Large-Scale Weak Supervision", FLEURS results table) reports **90.1% WER for
whisper-small on FLEURS Hausa**. That is the starting point
`NCAIR1/Hausa-ASR` was fine-tuned from. Our own whisper-small run on the same
FLEURS test set, under Gyara's normaliser, is
`[[RESULT: runs/whisper-small-fleurs/metrics.json]]`. Numbers from different
normalisers are not directly comparable; we compare only our own runs with a
paired test.

We do not know whether `NCAIR1/Hausa-ASR` was trained on any part of FLEURS.
Its model card (checked 9 Oct 2026) lists Langeasy platform recordings and
"publicly available datasets" without naming them, and we found no public NCAIR
source that does. If FLEURS was in its training data, its FLEURS test number
may be optimistic. That is why our own held-out audio, from speakers no model
has heard, is the number that matters.
