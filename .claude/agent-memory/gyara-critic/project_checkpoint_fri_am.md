---
name: checkpoint-fri-9-oct-am
description: Critic findings at the Fri 9 Oct 2026 morning checkpoint (code done, nothing run/committed); items to re-verify at the next checkpoint
metadata:
  type: project
---

Fri 9 Oct 2026 AM: code done (106 fast tests pass), but zero git commits, runs/ empty, no Space, no testers, no held-out audio, no consent.

Findings raised (check these were acted on at the next checkpoint):
1. Critical path: both Colab notebooks `pip install` from a GitHub URL (`<OWNER>` assert). No commit + public push = no baseline = G1 can't close. Push first.
2. Notebooks don't save runs/ off Colab (no download/commit cell). metrics.json lost when the runtime dies.
3. Benchmark notebook never runs `evaluate.run(..., suggester=...)`, so the DoD item "N-ATLaS measured effect" has no path to a number.
4. No contamination caveat: nobody checked whether NCAIR1/Hausa-ASR was trained on FLEURS. The "first public WER" claim depends on it.
5. README links docs/BENCHMARK.md, which doesn't exist.
6. Space will run with GYARA_LLM_BACKEND=none on a free CPU, so a judge sees no N-ATLaS in the demo. Needs a stated answer.
7. Fine-tune fallback: if own audio is late, fine-tune on FLEURS train (`fleurs_fraction`) and evaluate on FLEURS test, so a before/after exists anyway.
8. Freeze candidates: English translation, `openai` backend polish, leaderboard/rescore/normalize CLI, `speaker_dev_split`, UI styling.

**Why:** the next checkpoint should measure progress against these, not start from nothing.
**How to apply:** at the next review, check each item against the repo before repeating it. Drop the ones that are fixed.
