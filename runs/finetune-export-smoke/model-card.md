---
language: [ha]
base_model: NCAIR1/Hausa-ASR
tags: [automatic-speech-recognition, whisper, hausa, gyara]
---

# hausa-asr-gyara-export-smoke (Powered by Awarri)

Fine-tuned from [`NCAIR1/Hausa-ASR`](https://huggingface.co/NCAIR1/Hausa-ASR) with
[Gyara](https://github.com/Baffaabba/gyara) on human-verified Gyara exports.

## Training data

- 4 utterances, 0.01 h
  (0.01 h own audio, 0 FLEURS ha_ng train utterances)
- Training speakers: speakers not labelled
- Dev set (checkpoint choice only): random split of the training pool by utterance (2 utterances): no speaker ids, so the same voice can be in train and dev and dev WER is optimistic
- Leakage check passed for audio: no file is shared (sha256 of every file). Speaker overlap could only be checked where both sides carry speaker ids: 6 of 6 training/dev and 621 of 621 held-out utterances have none. FLEURS publishes no speaker ids; its dataset card states that train speakers differ from dev/test speakers, which we rely on but cannot check.

## Training

- Language/task tokens: `hausa` / `transcribe`
- Steps: 2, LR 1e-05, encoder frozen: True
- Seed 42; configs tried so far (`attempt`): 1
- Best dev result: 7.50% WER (standard)
- Notes: language/task tokens: <|hausa|> <|transcribe|>

## Held-out results (paired bootstrap vs base, `standard` normalisation v1.0.0)

- Not evaluated on held-out data yet.

## Licence

This model is a fine-tune of an N-ATLAS model and is released under the **N-ATLAS licence** of its base model. Free use is capped at 1,000 active end-users (rolling 30 days); derivatives keep the same licence; renamed derivatives must carry the suffix "Powered by Awarri".

N-ATLAS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.
