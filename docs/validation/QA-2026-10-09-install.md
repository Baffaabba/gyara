# QA: install from README on a fresh venv (Fri 9 Oct 2026)

Tester: gyara-qa. Machine: Windows 11, Python 3.11.0, no GPU, no `HF_TOKEN`,
not logged in to Hugging Face (`hf auth whoami` reports "Not logged in").
DoD item tested: "installs from README on a fresh machine".

## Verdict

The install works: `pip install -e ".[all]"` finished cleanly in a new venv.
`gyara --help`, `about` and `score` work, and the 119 fast tests pass in the
fresh venv. Three things stop this DoD item going green:

1. There is nothing to clone yet, because the repo has no commits (known).
2. On Windows, `gyara normalize` crashes whenever its output is piped or
   redirected, and in Git Bash.
3. Without access to the gated model, `gyara transcribe` prints a traceback of
   about 370 lines. The clear message is at the bottom, after 1.5 to 2 minutes.

Two scoring bugs give silently wrong or incomplete numbers (S1, S2). They
break the "honest numbers" rule, so fix them before any public result.

## What I ran

The repo has no commits, so `git clone` would give an empty folder. I copied
exactly what git would commit (`git ls-files --others --exclude-standard`,
leaving out `.claude/`) into a scratch folder. I re-synced it before testing so
the copy matched the working tree at about 10:50. Then:

```bash
python -m venv .venv
source .venv/bin/activate          # README line, literally
pip install -e ".[all]"
gyara --help ; gyara --version ; gyara about
gyara normalize "Ƴan makaranta sun ɗauki ’yan littattafai a ƙasar Ɓauchi, 25 ga wata!"   # all 3 modes
gyara score --ref refs.txt --hyp hyp.txt --json s.json
gyara transcribe tone.wav          # 3 s generated sine, no token
pip install -e ".[all,dev]" && python -m pytest -q
gyara ui --port 7899               # no token
```

I also made a second venv with `pip install -e .` only (the README's
"scoring only" row), plus the edge cases listed below.

## What passed

- `pip install -e ".[all]"`: exit 0, no resolver errors. It took **22 min 37 s**
  on this connection (torch 2.14.1 CPU, transformers 4.57.6, gradio 5.50.0,
  datasets 3.6.0, numpy 2.2.6, huggingface_hub 0.36.2).
- `gyara --help` (1.7 s), `gyara --version` (prints `gyara 0.1.0`) and
  `gyara about` (prints the full attribution sentence and the 1,000-user cap).
- `hf auth login`, as written in the README, exists in the installed
  huggingface_hub.
- `gyara score` on a 3-line Hausa pair: raw 42.86 % [25.00, 60.00], standard
  21.43 % [0.00, 40.00], lenient 0.00 %. I checked these by hand: the 3
  standard errors are the 3 missing hooks (kasar/ƙasar, dauki/ɗauki,
  boye/ɓoye), and `'yan` against `Ƴan` is correctly not an error.
- `score` on files whose names contain spaces and ɓ ɗ ƙ ƴ: correct.
- `score` with JSONL input works when the hypotheses are in a different order
  (matched by id). It warns and scores a missing id as empty output.
- `score` with a missing file or a different number of lines exits 2 with a
  one-line error.
- `transcribe` on a missing file exits 2 with "Path 'nope.wav' does not exist".
- `transcribe` on a Hausa file name with a quote (`ƙasa ɗan ɓoye ƴa 'quote'.wav`)
  and on a stereo 44.1 kHz wav: the audio loads and the run reaches the model
  step (where it fails on the gated model, as expected).
- **`gyara ui` without a token** opens on http://127.0.0.1:7899 within about
  8 s and shows a clear message: "Transcription is not available on this
  server yet… accept the licence on the model page, then add your token as
  HF_TOKEN… and restart." The attribution footer is present. This matches the
  README.
- Scoring-only install (`pip install -e .`, 1 min 19 s): `score`, `normalize`
  and `about` all work without torch.
- Test suite in the fresh venv: `119 passed, 3 deselected` in 55 s. No test
  needs a file that git ignores.
- `export`, `finetune` and `compare` given missing paths: one-line errors,
  exit 2.

## What broke, by severity

### Blockers (for this DoD item)

**B1. Nothing to clone.** `git log` says "your current branch 'feat/foundation'
does not have any commits yet". The README says "Clone this repository" but
gives no URL or command. This is already planned (DECISIONS 2026-10-09: public
GitHub repo via `gh`). When the repo is published, add the real
`git clone https://github.com/<owner>/gyara && cd gyara` line and re-run this
check from a real clone.

**B2. `gyara normalize` crashes on Hausa letters when output is not an
interactive console (Windows).** This happens with `> file`, with `| more`,
in CI logs, and in Git Bash (mintty uses pipes). Python then writes in cp1252,
and rich raises on ɗ ƙ ɓ ƴ.

```
$ gyara normalize "ɗan ƙasa" > o.txt ; echo $?
1
UnicodeEncodeError: 'charmap' codec can't encode character 'ɗ' in position 0: character maps to <undefined>
```

The same command in a real new console window (cmd or PowerShell, started
with `CREATE_NEW_CONSOLE`) exits 0. With `PYTHONUTF8=1` it also works when
piped. `--mode lenient` works because it folds hooks to ASCII. Any other CLI
path that prints Hausa text is likely affected the same way: `transcribe`
status lines with a Hausa file name, and `eval` reports. `score` survives
because its table prints only numbers. Suggested fix: reconfigure
stdout/stderr to UTF-8 at CLI start-up (`sys.stdout.reconfigure(encoding="utf-8",
errors="replace")`), or build the rich `Console` on a UTF-8 stream. I count it
as a blocker because judges on Windows with Git Bash hit it on the second
command they try.

**B3. `gyara transcribe` without model access: about 370 lines of traceback,
1.5 to 2 minutes later.**

```
$ gyara transcribe tone.wav        # no HF_TOKEN
VAD (silero) found no speech; falling back to fixed 30 s windows
No processor in NCAIR1/Hausa-ASR (You are trying to access a gated repo. ... Please log in.); using openai/whisper-small
`torch_dtype` is deprecated! Use `dtype` instead!
┌──── Traceback (most recent call last) ────┐ ... (~360 lines) ...
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/NCAIR1/Hausa-ASR.
401 Client Error. ... Please log in.
exit=1   real 1m40s  (1m24s and 2m15s on two other files)
```

The facts are right, but they are buried. The line "using openai/whisper-small"
reads as if Gyara switched to a different model. It only fell back for the
processor. The `ui` command already has the right message. Suggested fix:
before loading audio, check that the model is reachable (e.g.
`huggingface_hub.auth_check` or catching `GatedRepoError`/`OSError` in
`cli.transcribe` and `cli.eval_`). Print 3 lines: the model is gated, open the
model page and accept the licence, then `hf auth login` or set `HF_TOKEN`.
Exit 1. Judges without access will see exactly this.

### Should fix

**S1. `gyara score --json` drops every confidence interval.**
`cli.py:154` builds `{"wer": w.to_dict(), "cer": c.to_dict(), **sc.summary(), ...}`.
`summary()` also has `"wer"` and `"cer"` keys (plain floats), and they
overwrite the CI dicts. Run `gyara score --ref refs.txt --hyp hyp.txt --json s.json`
and `s.json` has `"wer": 0.4285…` with no CI anywhere. The table on screen is
correct. Only the JSON loses the CIs. This breaks hard rule 1 for anyone who
uses the JSON. Fix: put the summary first, or rename the keys.

**S2. A `--groups` file of the wrong length silently changes the score.** With
a 1-line groups file for 3 utterances:

```
$ gyara score --ref refs.txt --hyp hyp.txt --groups g1.txt
Gyara score  (n=3, ...)
raw 60.00 | standard 20.00 | lenient 0.00     <- no CI; correct values are 42.86 / 21.43 / 0.00
```

The JSON shows `"utterances": 1, "ref_words": 5`: the scorer zips with the
groups list and drops 2 of the 3 utterances, but the title still says n=3.
Fix: exit 2 with "--groups has 1 line, --ref has 3".

**S3. A UTF-8 BOM silently counts as a word error in every mode.** A reference
saved by Notepad or Excel (`EF BB BF`, CRLF) scored against identical text
gives WER 20 % raw, standard *and* lenient. It should be 0. Fix: read with
`encoding="utf-8-sig"` in `_read_texts` (and in the manifest readers).
UTF-16 files ("Unicode" in old Notepad) fail with a bare `UnicodeDecodeError`
traceback. Catch it and say "save as UTF-8".

**S4. Exporting from a workspace that does not exist "succeeds".**
`gyara export srt --asset 1 --out a.srt` in an empty folder creates
`workspace/gyara.db`, writes a 0-byte `a.srt` and prints "Wrote a.srt",
exit 0. `gyara export dataset --out ds` on an empty workspace writes an empty
dataset folder (with README and empty metadata) and reports success. Fix:
error if the db file is missing or the asset id is unknown; warn when 0
verified segments were exported.

**S5. Non-audio files give a traceback, not a message.** `gyara transcribe
notaudio.wav`, `fake.mp3` (text renamed) and `refs.txt` each print 60 to 115
lines ending in `Error opening input files: Invalid data found when processing
input`, after 15 to 18 s. They should print one line: "Could not read
notaudio.wav as audio or video."

**S6. README: no LICENSE file.** The README says "Gyara's code is Apache-2.0",
but there is no `LICENSE` in the repo. A public repo without one is "all
rights reserved" by default, and judges check this.

**S7. README: install time and size.** The README says "expect a few
minutes". It took 22.5 minutes here. On Linux, `pip install torch` pulls the
CUDA wheels (several GB). Say "10 to 30 minutes, about 1 GB on Windows/macOS,
more on Linux", or point CPU-only users to the PyTorch CPU index.

### Minor

- **M1.** `source .venv/bin/activate` fails in Git Bash on Windows
  (`No such file or directory`). The comment gives `.venv\Scripts\activate`,
  which is right for cmd and PowerShell. Git Bash needs
  `source .venv/Scripts/activate`. PowerShell may also need
  `Set-ExecutionPolicy -Scope Process Bypass` the first time.
- **M2.** `gyara transcribe` in a scoring-only install fails with a bare
  `ModuleNotFoundError: No module named 'torch'`. It should say "Install the
  speech extras: pip install -e \".[asr]\"".
- **M3.** `gyara score` with empty input files, or only blank references,
  prints `nan` in every cell and exits 0. It should say there are no reference
  words to score.
- **M4.** `gyara score` with an empty reference line gives pooled WER 100 %
  (the 2 inserted words over 2 reference words). The arithmetic is right, but
  nothing tells the user that one reference was empty. Add a warning.
- **M5.** The line-count error doesn't give the counts, and a trailing blank
  line in only one file triggers it. Show "--ref has 3 lines, --hyp has 2",
  or ignore trailing blank lines.
- **M6.** `gyara normalize "x" --mode bogus` prints a full traceback
  (`ValueError: unknown normalisation mode`). Use `click.Choice` for `--mode`
  here and in `score`/`compare`.
- **M7.** A binary file passed as `--ref` gives a `UnicodeDecodeError`
  traceback.
- **M8.** A silent or empty wav (`empty.wav`, 0 frames) writes a 0-byte `.srt`
  and exits 0 with "0 segments". It should say "No speech found."
- **M9.** The help shows `Usage: gyara transcribe [OPTIONS] {audio}` (curly
  braces) under typer 0.27.3. This is cosmetic. `typer>=0.12` has no upper
  bound.
- **M10.** The README shows `gyara score --ref refs.txt --hyp model_output.txt`
  but not the file format: UTF-8, one utterance per line, same order, or
  `.jsonl` with `id`, `text`. `--help` has it. Add one sentence to the README.
- **M11.** With only 2 clusters in `--groups`, `score` still prints a
  bootstrap CI ([0.00, 30.00]) with no warning that 2 clusters are too few.
  This is for gyara-stats to decide.

## What I could not test, and why

- **A real `git clone`:** there are no commits and no remote (B1).
- **Transcription output, `eval`, `finetune`, N-ATLaS suggestions:** I have no
  `HF_TOKEN` or licence access, and no GPU. I tested only the failure path
  without access.
- **Python 3.10, 3.12, Linux and macOS:** only Python 3.11.0 on Windows is
  installed here.
- **A truly clean machine:** this laptop's HF cache already holds
  `openai/whisper-small`, so the processor fallback did not download it. On a
  clean machine it would, which adds time to B3.
- **The README's interactive console experience in Windows Terminal:** I
  confirmed the exit code (0) in a new console window, but I couldn't read the
  rendered text.
- **The correction loop and the UI beyond the opening screen:** out of scope
  for this pass.

## Note on process

To stop the `gyara ui` test server I ran `taskkill /F /IM gyara.exe`. That
kills every `gyara.exe` on the machine. If another agent had a `gyara`
process running at about 10:55, it was stopped. Afterwards, `tasklist` showed
no `gyara.exe` running.
