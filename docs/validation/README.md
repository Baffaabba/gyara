# Validation evidence

NAIC component 3 asks for "evidence of testing with actual users, real data
or live benchmarks. Not simulated or hypothetical." Problem Statement 01
needs **at least 2 external beta testers**. Everything that proves it goes
here.

## Testers

People outside the team. Named, with permission. None named yet (9 Oct).

There is no hosted demo. Each tester either installs Gyara from the README,
or uses a temporary `gyara ui --share` link during a set window. Both routes
are in `TESTER-KIT.md`.

| Name | Role / organisation | Contacted | Kit sent | Feedback received | Quote by name? |
| --- | --- | --- | --- | --- | --- |
| Pending | | | | | |
| Pending | | | | | |
| Pending (optional third) | | | | | |

## What goes where

| Evidence | File name | Example |
| --- | --- | --- |
| Filled feedback form | `feedback_<tester-name>_YYYYMMDD.md` (or the PDF/email they sent) | `feedback_firstname-lastname_20261010.md` |
| Screenshots | `<tester-name>_YYYYMMDD_<n>.png` | `firstname-lastname_20261010_1.png` |
| Their `gyara score` / `gyara eval` output | `<tester-name>_YYYYMMDD_output.txt` | |
| Bugs they found and what we fixed | Add a line to the table below | |
| Portal confirmation | `submission-confirmation_20261011.png` | |

Lowercase, hyphens in names, no spaces. Remove anything personal they did not
agree to share (emails, phone numbers) before committing.

## Bugs reported by testers

| Tester | What broke | Fixed? (commit / note) |
| --- | --- | --- |
| | | |

## Kit

- `TESTER-KIT.md`: what to send each tester, and how to host a `--share`
  session for them.
- `samples/`: 5 FLEURS sentences and Hausa-ASR's output, for the scoring
  test.
- `FEEDBACK-FORM.md`: the questions.
- `CONSENT-FORM.md`: speaker consent for our own recordings (not for testers).
