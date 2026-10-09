# Gyara

Hausa speech-to-text you can measure, correct and improve, built on N-ATLAS for
NAIC 2026 (Problem Statement 01, Developer Infrastructure). Python package +
Typer CLI + Gradio + SQLite. **Submit by Sun 11 Oct 2026, 6 PM WAT.**

Anything a fresh session needs before touching code is here.

## Read before you work

| When | Read |
| --- | --- |
| **Anything, first** | **`docs/DECISIONS.md`: Baffa's calls. They beat every default below** |
| Anything | `.claude/skills/gyara-context/SKILL.md`: what exists, contracts, hard rules |
| Any number, run or claim | `.claude/skills/eval-integrity/SKILL.md` |
| Scope, docs, video, submission | `.claude/skills/naic-rubric/SKILL.md` |
| Any text a person reads | `.claude/skills/plain-english-copy/SKILL.md` |
| Changing normalisation | `docs/NORMALISATION.md`, then bump `RULES_VERSION` |
| Where things stand | `docs/STATUS.md` (the Definition of Done, ticked with evidence) |
| The original brief | `docs/source/CONCEPT.md`, `docs/source/EXECUTION-PLAN.md` |

## The roles

`gyara-pm` (with `gyara-innovator`) → `gyara-ml` / `gyara-app` →
`gyara-qa` + `gyara-design-review` + `gyara-stats` → `gyara-critic` at every
checkpoint → `gyara-judge` at gates G2 and G3.

There is no architect role: the execution plan fixed the architecture, and a
three-day build cannot afford to reopen it.

### Orchestration: the main session runs the team

There is no orchestrator agent, on purpose. A subagent cannot start or direct
other subagents, so an "orchestrator" agent would only write instructions
nobody had to follow. The main Claude session is the orchestrator, and these
are its rules:

1. **Pick roles from the work, not the habit.** New scope or a cut: pm and
   innovator. Models, eval, fine-tune: ml. App, CLI, packaging: app. Any number
   going public: stats. Any screen: qa and design-review. Every checkpoint:
   critic. Gates and the final package: judge. Say which roles you skipped.
2. **Give each agent a complete brief.** It has not seen this conversation:
   the goal, the files, the relevant `docs/DECISIONS.md` entries, what "done"
   means, what to hand back, and which files it owns. Two builders never own
   the same file at the same time.
3. **Run independent work in parallel.** ml and app build side by side on
   separate files. qa, stats, design-review and critic can review the same
   build at once. A role that needs another's output waits.
4. **Blocking findings block.** A blocker from qa, stats, design-review or
   critic is fixed, or taken to Baffa, before the next gate. Never quietly
   dropped. A claim stats blocks does not go into any public text.
5. **Conflicts go to Baffa.** When two agents disagree and the answer changes
   the work, put both positions to him in a sentence each. Record his answer.
6. **Verify, then report.** Agent reports are claims. Run the tests, open the
   output, before telling Baffa it is done.
7. **The deadline decides.** When a gate is red, cut scope (pm proposes),
   don't stretch hours. Layer A (benchmark kit) always ships.

Every role reads `docs/DECISIONS.md` first and adds to it when Baffa decides
or overrules something. Each keeps notes in `.claude/agent-memory/<agent>/`
(`memory: project`). Both are committed.

## Architecture in one paragraph

`gyara/normalize.py` defines the Hausa scoring rules (raw / standard /
lenient). `metrics.py` scores pooled WER/CER; `stats.py` adds cluster
bootstrap CIs and paired tests. `audio.py` + `vad.py` + `asr.py` turn any
recording into ≤ 30 s segments with text, confidence and loop flags.
`evaluate.py` runs a model over a JSONL manifest (`manifest.py`) into
`runs/<name>/`. `suggest.py` asks N-ATLaS for spelling fixes behind a guard.
`store.py` (SQLite) holds assets, segments, revisions and suggestions for the
Gradio app in `ui/`. `export.py` writes `.srt`/`.vtt` and an HF `audiofolder`
dataset of verified, consented segments. `finetune/train.py` retrains
`NCAIR1/Hausa-ASR` on exports and refuses to run if training data overlaps the
held-out set. `cli.py` exposes all of it.

## Commands

```bash
.venv/Scripts/python -m pytest -q          # fast tests
.venv/Scripts/python -m pytest -q -m slow  # downloads models / FLEURS
.venv/Scripts/gyara --help
```

## Git

Never commit to `main` or `dev`. Feature branches only. Ask Baffa before every
commit. Never commit audio, databases, tokens or `.env`.
