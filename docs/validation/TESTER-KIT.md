# Gyara tester kit

Send this to each tester, with `FEEDBACK-FORM.md`. About 20 minutes of their
time, plus install.

---

## What Gyara is

Gyara measures how well AI transcribes Hausa speech, helps a person correct
the transcript, and turns the corrections into training data.
It uses N-ATLAS: `NCAIR1/Hausa-ASR` for speech and `NCAIR1/N-ATLaS` for
spelling suggestions.
We want to know what breaks and what is confusing. Be blunt.

## Before you start (about 10 minutes)

**Option A: no install.** Open the demo: `[[URL: Hugging Face Space]]`.
You can do task 3 there. Tasks 1 and 2 need option B.

**Option B: install on your machine.** Python 3.10 or newer. No GPU needed;
it is slower without one.

1. Make a free Hugging Face account. Open both pages below and click
   "Agree and access" (they are gated):
   - https://huggingface.co/NCAIR1/Hausa-ASR
   - https://huggingface.co/NCAIR1/N-ATLaS
2. Create a read token at https://huggingface.co/settings/tokens.
3. In a terminal:

   ```
   [[INSTALL: pip install git+https://github.com/<org>/gyara@v0.1.0  — confirm with Dev-App]]
   huggingface-cli login        # paste your token
   gyara --help
   ```

If anything fails here, stop and screenshot it. That is useful feedback.

## The test (20 minutes)

Note the time when you start each task.

### Task 1: score transcripts you already have (5 min)

You need two text files, one sentence per line, same order: the correct
text (`refs.txt`) and an AI's output (`hyps.txt`). Any Hausa ASR output works.
If you have none, use the two sample files we send with this kit
`[[ATTACH: sample refs.txt / hyps.txt]]`.

```
gyara score --ref refs.txt --hyp hyps.txt
```

You should see word and character error rates three ways (raw, standard,
lenient), each with a 95% range. Does the result make sense to you?

### Task 2: benchmark the N-ATLAS speech model on 10 FLEURS clips (8 min)

```
gyara fetch-fleurs --split test --limit 10
gyara eval --manifest data/fleurs/test.jsonl --limit 10
```

The first run downloads the model (about 1 GB). With only 10 clips the range
will be wide. That is expected. Open the `report.md` it writes under `runs/`.

### Task 3: correct a 2-minute Hausa clip (7 min)

Use your own 2-minute Hausa recording (you must have the right to use it),
or our sample `[[ATTACH: 2-min sample clip]]`.

```
gyara ui
```

(or the demo link). Then:

1. Upload the clip and let it transcribe.
2. Correct at least 5 segments. Try "Get suggestion" on one.
3. Download the subtitles (`.srt`). Open them in a video player if you can.
4. Export the dataset.

### Send back

1. The filled feedback form.
2. 2–4 screenshots: one of each task's result, plus anything that broke.
   Name them `yourname_YYYYMMDD_1.png`, `_2`, …
3. Please reply by **`[[DATE: Sat 10 Oct, 4 PM WAT]]`** to `[[EMAIL]]`.

Thank you. With your permission we will name you as a tester in our NAIC
2026 submission.

N-ATLAS is an initiative of the Federal Ministry of Communications,
Innovation and Digital Economy, and powered by Awarri Technologies.
