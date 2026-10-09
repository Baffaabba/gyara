# Decisions

Baffa's calls on this project, newest first. Every Gyara agent reads this
before it starts, and adds to it when Baffa decides something or overrules one
of them.

A decision here beats a default in an agent file, a skill, or the plan. If two
entries disagree, the newer one wins. Say so in the new entry rather than
editing the old one, so the history of a change of mind survives.

## How to add one

One entry per decision, under today's date:

- **The decision**, in one sentence.
  *Why:* what Baffa said, or what made it necessary.
  *Applies to:* who has to act differently because of it.

Write down only what Baffa decided or confirmed. Your own preference, or a
choice you made because nobody asked, is not a decision. When you are unsure
whether something counts, ask him.

Also record it when he overrules you. "Agent proposed X, Baffa chose Y because
Z" is the most useful kind of entry, because it is the one most likely to be
proposed again.

---

## 2026-10-09

- **FLEURS headline stays standard WER (30.19%), with lenient (27.33%) shown
  beside it and the reason.** *Why:* gyara-stats found the FLEURS references
  often omit hooked letters the model writes; Baffa kept the rule fixed
  before results rather than switch to the nicer number after seeing them.
  *Applies to:* README, BENCHMARK.md, N-ATLAS-INTEGRATION.md, video, pitch.

- **Commit per-utterance predictions for runs on public FLEURS data; own-audio
  predictions stay out of git.** *Why:* lets anyone re-score our numbers
  (FLEURS is CC-BY-4.0); own audio may carry consent limits.
  *Applies to:* gyara-ml (run naming: FLEURS runs live in `runs/*fleurs*/`,
  `runs/dev/` or `runs/probe/`), whoever commits runs.

- **The demo Space is public, at `huggingface.co/spaces/Baffaabba/gyara`.**
  *Why:* Baffa chose public over private so testers need no HF account. The
  Space warns that uploads are shared.
  *Applies to:* gyara-app (Space copy), anyone sending the tester kit.

- **Teammates may keep `HF_TOKEN` in a git-ignored `.env`; Gyara reads it on
  start-up.** *Why:* Baffa logged in from WSL, which Windows can't see, and
  asked for setup steps everyone can follow. *Applies to:* README, gyara-app.

- **Commit the foundation to `feat/foundation` once the agents finish and the
  tests pass.**
  *Why:* nothing was committed, and the Colab baseline needs the code on GitHub.
  *Applies to:* the main session.

- **Publish the repo as a public GitHub repo created with `gh`, and point the
  notebooks at it.**
  *Why:* Baffa chose this over a zip upload, so Colab can install with
  `pip install git+...` and the repo doubles as the public-repo DoD item.
  *Applies to:* the main session, gyara-ml (notebook URLs), gyara-app (README).

## 2026-10-08

- **Build Gyara as specified in the Concept Document v1.0 and the Execution
  Plan (Phase 3 blueprint).** Problem Statement 01, Innovation & Enterprise
  track, Layer A (benchmark kit) first as the fallback, Layer B (the loop) on
  top.
  *Why:* Baffa supplied both documents and said "do what is statistically
  correct… give me the best".
  *Applies to:* everyone. Scope questions are answered against those two
  documents (`docs/source/`).

- **"Statistically correct" is a product requirement, not polish.** Every
  number is pooled, normalised three ways, carries a cluster-bootstrap CI, and
  every comparison is a paired test with its minimum detectable effect.
  *Why:* Baffa's instruction above; the team is up against professionals and
  honest, defensible numbers are the edge. Codified in the `eval-integrity`
  skill.
  *Applies to:* gyara-ml, gyara-stats, anyone writing docs or the video.

- **The agent team mirrors the Dama AI setup** (main session orchestrates; roles
  in `.claude/agents/`; skills; this file; agent memory), adapted for this
  project with a statistician (`gyara-stats`) and a judging panel
  (`gyara-judge`), and no architect.
  *Why:* Baffa asked for "agents and orchestrator… just like the ones i created
  for scholarship-and-job-tracker".
  *Applies to:* the main session.
