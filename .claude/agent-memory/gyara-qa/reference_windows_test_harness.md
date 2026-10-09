---
name: windows-test-harness
description: How to run Gyara install/CLI QA on Baffa's Windows laptop - the Bash tool's stdout is a cp1252 pipe, real-console checks, fake clone when uncommitted, install time
metadata:
  type: reference
---

- The Bash tool (Git Bash) gives child processes a **piped cp1252 stdout**. Any CLI printing ɓ ɗ ƙ ƴ via rich can crash there but work in a real console. To check the real-console behaviour, launch with `subprocess.Popen([...], creationflags=CREATE_NEW_CONSOLE)` and strip `PYTHONUTF8` from the child env; read the exit code. Use `PYTHONUTF8=1` only for your own harness output.
- When the repo has no commits, simulate a clone by copying `git ls-files --others --exclude-standard` (minus `.claude/`) into scratch.
- `pip install -e ".[all]"` took ~22 min on this connection (2026-10-09). Run it in the background and poll the log.
- The HF cache on this laptop already holds openai/whisper-small, so it is not a truly clean machine.
- Never `taskkill /IM gyara.exe` — it kills other agents' processes. Kill by PID.

First used: install QA, report `docs/validation/QA-2026-10-09-install.md`. Related: [[gyara-context]]
