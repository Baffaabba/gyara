---
name: plain-english-copy
description: Rewrite any text a Gyara user or judge will read — app labels, buttons, empty states, errors, CLI messages, README, docs, the video script — into short plain English. Two audiences, transcribers and developers, with rules for each. Apply to every piece of user-facing text before showing it, whichever role wrote it.
---

# Plain English for Gyara

Two kinds of reader.

**Transcribers and linguists** using the correction app. Fluent in written
Hausa, comfortable with a laptop, not developers. Long sessions, repetitive
work. They need to know what to do next and nothing else.

**Developers and judges** reading the CLI, README and docs. Technical, busy,
skeptical. They need the command that works and the number with its evidence.

## Rules for everyone

1. Short sentences. One idea each. Break anything over about 20 words.
2. Say what happened and what to do next. An error without a next step is
   unfinished.
3. No corporate voice: no "leverage", "seamless", "empower", "unlock",
   "revolutionise", "cutting-edge", "robust solution", "journey".
4. Buttons are verbs: "Save and next", "Get suggestion", "Download subtitles".
   Not "Submit", "OK", "Process".
5. Never blame the reader. "We could not read that file. Try an mp3, wav or
   mp4." beats "Invalid file format".
6. Numbers over vagueness, and never a number without its source.
   "Hausa-ASR got 3 in 10 words wrong on FLEURS (WER 30.1%, 95% CI 28.4–31.9)"
   beats "high accuracy".
7. No emoji in functional UI.
8. Sentence case for labels and headings.

## For transcribers (the app)

- No ML vocabulary: not "inference", "hypothesis", "token", "logprob",
  "normalisation", "WER" without a gloss. "AI draft" not "hypothesis".
  "How much the AI got wrong" next to "WER".
- Explain the score once, where it appears: "Word error rate: how many words
  the AI got wrong compared with your version. Lower is better."
- The loop warning is calm: "The AI may have repeated itself here. Listen
  closely before you save."
- Consent is plain: "I have the speaker's written permission to use this
  recording for training."

## For developers and judges (CLI, README, docs)

- Lead with the command, then the explanation.
- Exact names: `NCAIR1/Hausa-ASR`, not "the model".
- Every claim of improvement follows `eval-integrity` wording: absolute points,
  CI, p-value, n, test set.
- Name limitations plainly. "N-ATLaS reads text, not audio, so it can change a
  correct word. We measured that: …"

## Hausa text

- Use the real letters ɓ ɗ ƙ and 'y in examples, never ASCII stand-ins.
- Hausa UI strings are welcome as a secondary line, but only if a fluent
  speaker has checked them. A wrong Hausa label is worse than none.
