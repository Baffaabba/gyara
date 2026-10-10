---
name: project-no-hf-token
description: As of 2026-10-09 the Windows dev laptop has no Hugging Face token, so gated NCAIR1 models cannot load here; real-model runs happen on Colab
metadata:
  type: project
---

On 2026-10-09 (G2 day) the dev laptop had no `HF_TOKEN` and no saved HF login.
`NCAIR1/Hausa-ASR` returns 401 (gated). Agents must not create HF repos.

**Why:** gated licences and accounts are on Baffa's to-do list (docs/STATUS.md), and agents must not create HF repos.

**How to apply:** verify the app's no-token path (startup notice plus a plain error on Transcribe) rather than a real transcription. Check `huggingface_hub.get_token()` before assuming this is still true. Real-model runs happen on Colab/Kaggle.
