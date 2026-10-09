---
name: laptop-env-gotchas
description: Baffa's Windows dev laptop quirks that cost time in ML work (no GPU/token, flaky HF CDN, CRLF from write_text, slow-test duration)
metadata:
  type: project
---

Dev laptop facts as of 2026-10-09:

- No GPU, no HF_TOKEN: NCAIR1/Hausa-ASR and NCAIR1/N-ATLaS cannot be loaded here. Real numbers come from Colab via `notebooks/benchmark_colab.ipynb`. Stand-in for local smoke runs: `openai/whisper-tiny`.
- The HF xet CDN (us.aws.cdn.hf.co) is flaky from this network: `gyara fetch-fleurs --limit 15` took 9 min of retries but succeeded. Budget for it. Don't assume the code is broken.
- `Path.write_text` on Windows writes CRLF. When patching repo files with a Python script, pass `newline="\n"` or use the Edit tool. Bash heredocs also mangle `\1`/`\n` in Python regex/string patches, so use Edit/Write for those.
- `pytest -m slow` (whisper-tiny FLEURS eval, audiofolder export, 2-step train) takes about 11 min on CPU, and all 3 passed on 2026-10-09.

**Why:** each of these cost a retry cycle in the 9 Oct smoke session.
**How to apply:** plan long runs in the background. Verify line endings with `file` after scripted edits.
