# Architecture

Gyara is one Python package with three front doors: a CLI for developers, a
Gradio app for people correcting transcripts, and Colab notebooks for anyone
without a GPU. Everything is local files and one SQLite database. There is no
server to run, no queue and no external service.

```
                      ┌────────────── gyara/cli.py (Typer) ──────────────┐
                      │   transcribe · eval · score · compare · export   │
                      │   finetune · ui · fetch-fleurs · check-leakage   │
                      └──────────────────────────────────────────────────┘
                                   │                     │
     audio / video                 ▼                     ▼
  ──────────────► audio.py ─► vad.py ─► asr.py ─────► evaluate.py ──► runs/<name>/
                  any format  ≤30 s     Hausa-ASR      over a           predictions.jsonl
                  16 kHz mono chunks    + loop guard   manifest.py      metrics.json
                                         │                  │           report.md
                                         │                  ▼
                                         │      normalize.py → metrics.py → stats.py
                                         │      Hausa rules    pooled WER   bootstrap CI,
                                         ▼                     /CER        paired test, MDE
   ui/app.py (Gradio) ◄──► store.py (SQLite) ◄── suggest.py (N-ATLaS + guard)
   listen · edit · accept/reject   assets, segments,
   live WER/CER · verify           revisions, suggestions
                                         │
                                         ▼
                                   export.py ──► .srt / .vtt
                                         │   ──► HF audiofolder dataset
                                         ▼        (verified + consented only)
                                finetune/train.py ──► tuned Hausa-ASR
                                 leakage check first   + before/after via evaluate.compare
```

## Design choices and why

| Choice | Why |
| --- | --- |
| One package, local files, SQLite | A developer clones and runs it. Judges run an integration check. Nothing to deploy beyond an optional HF Space |
| JSONL manifests for every test set | One format for FLEURS, own audio and exports. Diffable, hashable, easy to freeze |
| Predictions saved before scoring | Rules can change (`RULES_VERSION`) and every run is re-scored without re-running the model (`gyara rescore`) |
| VAD chunks ≤ 30 s | Whisper's window is 30 s. Chunking at pauses avoids cutting words, and short windows give the loop detector a clear signal |
| Loop detection + temperature fallback + flag | Whisper loops on noise. A flagged segment is retried, then trimmed and shown to a human first, never silently accepted |
| Append-only revisions | Every correction is kept, so the AI's draft is always there to score against the final text |
| Suggestions stored separately with a status | Accept/reject is auditable, and the measured effect of N-ATLaS uses exactly what people decided |
| Guard on N-ATLaS output | It reads text, not audio. Word count and per-word edit limits stop it rewriting what was said |
| Leakage check before training | A gain measured on voices the model trained on is not a gain |
| Cluster bootstrap | Clips from one speaker, or FLEURS readings of one sentence, are correlated. Treating them as independent overstates certainty |

## Data that leaves the machine

None, by default. Models download from Hugging Face. The optional `openai`
suggestion backend talks only to the URL you give it (normally a local
`llama-server`). Exported datasets go where you point them.

## Extending to another language

The pipeline is language-agnostic except `normalize.py`. A Yoruba or Igbo
version needs a normaliser with its own rules document, and the matching
`NCAIR1/*-ASR` model id.
