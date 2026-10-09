---
name: laptop-env-gotchas
description: Baffa's Windows dev laptop quirks that cost time in ML work (CPU only, flaky HF CDN, CRLF from write_text, slow-test and CPU-eval durations)
metadata:
  type: project
---

Dev laptop facts (updated 2026-10-09):

- No GPU. Since 2026-10-09 the HF token in `.env` can load NCAIR1/Hausa-ASR (cached locally). On CPU fp32 it runs at RTF ~0.66-0.77: 100 FLEURS clips (0.4-0.5 h audio) took 16-23 min. CPU output matched the Colab T4 fp16 run on 93 of 100 validation clips, with no reliable WER difference. N-ATLaS (8B) is still impractical here.
- The HF xet CDN (us.aws.cdn.hf.co) is flaky from this network. `fetch-fleurs --limit 100` needed several read-timeout retries, but both splits finished within ~10 min. Run fetches in the background with a retry loop.
- `Path.write_text` on Windows writes CRLF. evaluate.py now passes `newline=LF`, but `gyara leaderboard --out` (cli.py) still writes CRLF. Bash heredocs and `sed` mangle `\n` inside Python string literals (twice on 9 Oct), so use the Edit tool for any edit containing `\n`.
- `pytest -m slow` takes about 11 min on CPU. The fast suite takes about 40-60 s.

**Why:** each of these cost a retry cycle in the 9 Oct sessions.
**How to apply:** plan CPU evals and fetches as background jobs. Verify line endings with `file` after scripted edits or rescoring.
