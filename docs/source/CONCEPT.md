# Gyara: Concept Document (v1.0, 6 Oct 2026)

Transcribed from `Gyara Concept Document.pdf`. The PDF is the original.

**Gyara** (Hausa: "correction"). Working name. **Challenge:** National AI
Innovation Challenge (NAIC) 2026. **Problem statement:** 01 Developer
Infrastructure. **Track:** Innovation & Enterprise (registered startup).

## 1. One-line summary

Gyara is an open-source toolkit that helps developers measure, correct and
improve Hausa speech-to-text, turning every human correction into training
data that makes N-ATLAS better over time.

## 2. The problem

1. **Hausa transcription is still weak.** Open speech models make frequent
   mistakes on Hausa, especially with regional accents, background noise,
   everyday speech and Hausa–English mixing.
2. **Nobody can say how weak.** There is no simple, shared way for Nigerian
   developers to measure Hausa transcription accuracy consistently. Spelling
   differences (e.g. *ƙasa* vs *kasa*) make most accuracy numbers unreliable.
3. **Corrections are wasted.** When people fix AI transcripts today, the fixed
   text is thrown away. It never goes back into improving the model.
4. **Developers start from zero.** Anyone who wants to test, evaluate or
   fine-tune `NCAIR1/Hausa-ASR` has to write the whole pipeline themselves.

## 3. The solution

| # | Step | What happens |
| --- | --- | --- |
| 1 | Transcribe | Hausa audio → text with timestamps, using `NCAIR1/Hausa-ASR` |
| 2 | Suggest | `NCAIR1/N-ATLaS` suggests spelling/punctuation fixes. A human decides |
| 3 | Correct & measure | A person corrects the text; Gyara scores the AI (WER/CER) with fixed Hausa spelling rules |
| 4 | Export | `.srt` subtitles and a clean training dataset |
| 5 | Improve | A starter script fine-tunes the speech model on corrected data and re-measures it |

**What makes Gyara new:** others have built Hausa transcription and N-ATLAS
APIs. Nobody has built the loop that connects correction → measurement →
dataset → retraining.

## 4. Who it is for

| User | What they get |
| --- | --- |
| Primary: developers & researchers building Hausa speech apps | A ready pipeline, a trustworthy benchmark, a fine-tuning starter kit |
| Secondary: transcribers & linguists | A simple screen to correct transcripts quickly |
| Future paying customers: Hausa media & creators | Fast, accurate Hausa subtitles |
| NCAIR / Awarri | Measured evidence of where Hausa-ASR fails, plus verified training data |

## 5. In scope

- Hausa transcription with word/segment timestamps, long files split into ≤ 30 s chunks
- N-ATLaS suggestions for spelling and punctuation; suggestions only
- Human correction screen: listen per segment, edit, see accuracy live
- WER and CER, raw and with Hausa spelling normalisation
- Documented Hausa normalisation rules (hooked letters ɓ ɗ ƙ ƴ, apostrophes, numbers, casing, punctuation)
- `.srt` and `.vtt` export
- Dataset export: corrected audio + text, Hugging Face format
- Fine-tuning starter kit: script + config
- Benchmark report: `NCAIR1/Hausa-ASR` on FLEURS Hausa test and own audio
- Optional: Hausa → English subtitle translation with N-ATLaS
- Python package and CLI (`transcribe`, `eval`, `export`); live web demo; docs
- Validation: ≥ 2 external developer beta testers with written feedback; honest before/after results

## 6. Out of scope

Dubbing / voice-over; a "perfect" model; self-training without humans;
training from scratch; Yoruba, Igbo, Nigerian English; real-time transcription;
speaker identification; OCR; Ajami; an OpenAI-compatible API as a headline
feature; a full production platform (accounts, payments, mobile); commercial
service above 1,000 users (needs an Awarri / Ministry licence).

## 7. Honest limitations

- Fine-tuning gains will be small at first. The challenge proves the loop.
- LLM suggestions are not guaranteed to help. N-ATLaS reads text, not audio. We
  measure its real effect and report it.
- Dialects vary (Kano, Sokoto, Zaria, Niger…). The benchmark shows where it is weakest.
- CPU-only use is slow for the 8B LLM; it needs a GPU or a compressed version.

## 8. How it uses N-ATLAS

| Component | Use |
| --- | --- |
| `NCAIR1/Hausa-ASR` (Whisper-small fine-tune) | Core transcription engine; the model we benchmark and fine-tune |
| `NCAIR1/N-ATLaS` (Llama-3 8B fine-tune) | Correction suggestions and optional Hausa → English translation |
| N-ATLAS training pipeline | The fine-tuning kit retrains `NCAIR1/Hausa-ASR` directly |

## 9–12. Fit, business model, compliance, roadmap

- Fits the PS01 example "fine-tuning starter kit with training scripts and evaluation tools".
- Business: open-source core (free) → hosted subtitle service (per minute or
  subscription) → data services (project contracts) → commercial N-ATLAS licence
  partnership beyond 1,000 users.
- Compliance: attribution on all public use; same licence for fine-tuned models,
  renamed versions carry "Powered by Awarri"; written speaker consent (NDPA); no
  surveillance, profiling or impersonation.
- Roadmap: more data → stronger model; Yoruba, Igbo, Nigerian English; speaker
  ID and real-time; dubbing; hosted service; public Hausa speech leaderboard.
