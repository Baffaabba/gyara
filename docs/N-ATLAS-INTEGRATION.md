# N-ATLAS integration

NAIC component 2. This page shows exactly where and how Gyara uses N-ATLAS.

> N-ATLAS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

## Summary

Gyara uses both N-ATLAS models for real work:

- **`NCAIR1/Hausa-ASR`** is the speech engine. We benchmark it, correct its
  output, and fine-tune it on the corrections.
- **`NCAIR1/N-ATLaS`** proofreads the transcripts (suggestions only, behind a
  guard). It can also translate Hausa subtitles into English; that option is
  built but not yet tested on real audio.

`openai/whisper-small` is a comparison baseline only. It is never the engine.

## Where each model is used

| Model | Used for | Module | What we measure | Evidence |
| --- | --- | --- | --- | --- |
| `NCAIR1/Hausa-ASR` | Transcribe Hausa audio into timed segments, with a confidence score per segment and loop protection | `gyara/asr.py` (`Transcriber`) | — | `gyara transcribe` |
| `NCAIR1/Hausa-ASR` | Benchmark on FLEURS Hausa test and our own held-out audio: WER and CER, raw / standard / lenient, 95% cluster-bootstrap CI | `gyara/evaluate.py` | FLEURS Hausa test: 30.19% standard WER (95% CI 28.63–31.84), lenient 27.33% (25.77–28.98), CER 10.08% (9.00–11.35); 621 clips, 331 sentences. May be optimistic if FLEURS was in its training data (see below). Own held-out audio: pending, not measured yet. | `runs/hausa-asr-fleurs/report.md`, `docs/BENCHMARK.md` |
| `NCAIR1/Hausa-ASR` vs `openai/whisper-small` | Show what the N-ATLAS fine-tune adds over its base model, on the same clips | `gyara/evaluate.py` (`compare`, paired bootstrap) | Standard WER −59.62 points (95% CI −61.00 to −58.17, p < 0.001, MDE 2.01, same 621 FLEURS test clips). CER −20.80 points (−21.67 to −19.95). | `runs/hausa-asr-fleurs/compare.md`, `gyara compare` |
| `NCAIR1/Hausa-ASR` | Fine-tune on human-verified corrections; measure before vs after on held-out speakers only | `gyara/finetune/train.py`, `config.yaml` | Pending. For this submission we fine-tune on FLEURS train and measure once on FLEURS test. That shows the kit works end to end, not what human corrections add; we have no corrected audio yet. | `gyara finetune` |
| `NCAIR1/N-ATLaS` | Suggest spelling and punctuation fixes. A human accepts or rejects. Never applied automatically. | `gyara/suggest.py` (`Suggester`, `guard`) | Accepting every guarded suggestion, with no human review, on the 621 FLEURS test clips (4-bit, Colab T4, paired bootstrap): standard WER −0.18 points (95% CI −0.30 to −0.07, p = 0.002, MDE 0.17). Small, and close to what this test set can detect. Raw WER fell 2.01 points, mostly from restored capitals and full stops. It was not hook fixes: the lenient effect is the same size. Some suggestions made a correct word worse, so accept-all is not an upper bound. About 4.6 s per sentence on a T4 (derived). Share of suggestions the guard rejected: not recorded by this run. | `runs/hausa-asr-fleurs-natlas/report.md`, `docs/BENCHMARK.md`; `gyara eval --suggest transformers` |
| `NCAIR1/N-ATLaS` | Hausa → English subtitles | `gyara/suggest.py` (`translate`), `gyara/export.py` | Not scored (no English references). Optional. Built and unit-tested, not yet tested with the real model on real audio. | `gyara export srt --english` |
| N-ATLaS GGUF builds | Run N-ATLaS without a GPU (supported in code, untested): community GGUF builds served by llama.cpp, vLLM or Ollama behind an OpenAI-compatible endpoint | `gyara/suggest.py` (`openai` backend) | Not yet tested or timed. Our measured suggestions ran on a Colab T4 GPU, not through GGUF. | `llama-server -m N-ATLaS-Q4_K_M.gguf --port 8080` |

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
| Attribution on public use | `gyara.ATTRIBUTION` holds the exact line. It appears in `gyara about`, the README, the app footer, dataset cards, model cards and the video. |
| Free use capped at 1,000 active end-users | Stated in `gyara about` and the README. Above the cap we need an Awarri / Ministry commercial licence (see `docs/BUSINESS.md`). |
| Derivatives keep the same licence | `gyara/finetune/train.py` writes the N-ATLAS licence into every fine-tuned model card. |
| Renamed derivatives say "Powered by Awarri" | The same model card names a renamed model "<name> (Powered by Awarri)". |
| No surveillance, profiling or impersonation | Out of scope by design: uploaded files only, no speaker identification, no voice cloning. |

Gyara's own code is Apache-2.0. The models keep their own licences.

## What we found, as feedback for NCAIR

These are useful to the N-ATLAS team. Gyara documents and works around each.

1. **No published WER.** The `NCAIR1/Hausa-ASR` model card publishes no
   accuracy figure, and we found none elsewhere. Gyara publishes one, with
   CIs and a fixed, versioned Hausa normalisation (`docs/NORMALISATION.md`):
   30.19% standard WER on FLEURS Hausa test (95% CI 28.63–31.84, 621 clips),
   with the training-data caveat below.
2. **Processor does not load.** `WhisperProcessor.from_pretrained` fails on
   the repo. Gyara falls back to the `openai/whisper-small` processor, which is
   identical for every whisper-small fine-tune, and logs a warning
   (`gyara/asr.py`, `_load_processor`).
3. **No language token set.** The card calls `generate()` with no language.
   Gyara forces Hausa by default, allows `--language auto`, and records the
   setting in every run. On the first 100 FLEURS validation clips, forced
   Hausa and automatic detection gave identical output (WER 30.18% both,
   `runs/dev/`, `runs/g0-smoke/decoding_choice.json`). We chose the setting
   there, never on test.

## The baseline in context

`NCAIR1/Hausa-ASR` was fine-tuned from `openai/whisper-small`. On the same
621 FLEURS test clips, under Gyara's normaliser, whisper-small gets 89.81%
standard WER (95% CI 88.51–91.28) and Hausa-ASR 30.19% (28.63–31.84).
Published figures from other papers use other normalisers and are not
directly comparable, so we compare only our own runs, with a paired test.

We do not know whether `NCAIR1/Hausa-ASR` was trained on any part of FLEURS.
Its model card (checked 9 Oct 2026) lists Langeasy platform recordings and
"publicly available datasets" without naming them, and we found no public NCAIR
source that does. If FLEURS was in its training data, its FLEURS test number
may be optimistic. That is why our own held-out audio, from speakers no model
has heard, is the number that matters.
