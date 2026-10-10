# Gyara tester kit

Send this page to each tester, with `FEEDBACK-FORM.md`. The tests take about
15 minutes. Installing takes longer (see below), but you can leave it running.

The last section is for Baffa: how to host a session and what to send.

---

## What Gyara is

Gyara measures how well AI transcribes Hausa speech, helps a person correct
the transcript, and turns the corrections into training data. It uses
N-ATLAS: `NCAIR1/Hausa-ASR` for speech and `NCAIR1/N-ATLaS` for spelling
suggestions.

We want to know what breaks and what is confusing. Be blunt. If a step fails,
take a screenshot and move on. A failure is useful feedback.

## Pick your route

| Route | You need | You can do |
| --- | --- | --- |
| **A. Browser only** | A link from Baffa, during the time he gives you | Test 3 (correct a recording) |
| **B. Install on your computer** | Python 3.10 or newer, a few GB of free disk, 10 to 30 minutes for the install | Tests 1, 2 and 3 |

Route B is the one we most need feedback on. Developers who adopt Gyara will
install it.

## Route A: browser only

1. Baffa sends you a link that ends in `.gradio.live`, and a time window.
2. Open the link in Chrome, Edge or Firefox during that window. Outside it,
   the link does not work.
3. Do **Test 3** below.

Other testers may use the same link at the same time. They can see what you
upload. Only upload a recording you are allowed to share.

## Route B: install on your computer

### Step 1. Get access to the models (5 minutes)

1. Sign in at https://huggingface.co (a free account is fine).
2. Open https://huggingface.co/NCAIR1/Hausa-ASR. Fill in the short form at
   the top and agree.
3. Do the same at https://huggingface.co/NCAIR1/N-ATLaS.
4. Reload both pages. Each should say you have access. If one says
   "pending", wait for the approval email.
5. Open https://huggingface.co/settings/tokens and create a token of type
   **Read**. Copy it. It starts with `hf_`.

Never paste your token into a chat, email or screenshot.

### Step 2. Download Gyara

If you have git:

```
git clone https://github.com/Baffaabba/gyara
cd gyara
```

No git? Open https://github.com/Baffaabba/gyara, click **Code**, then
**Download ZIP**. Unzip it, then open a terminal in the unzipped folder.

### Step 3. Install (10 to 30 minutes, mostly waiting)

**Windows** (PowerShell or Command Prompt):

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]"
```

If PowerShell refuses to activate, run
`Set-ExecutionPolicy -Scope Process Bypass`, then activate again.

**macOS or Linux:**

```
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]"
```

The `torch` line gets the smaller CPU-only build. If you have an NVIDIA GPU,
skip that line.

### Step 4. Give Gyara your token

Create a file called `.env` in the `gyara` folder, with this one line (your
own token):

```
HF_TOKEN=hf_your_token_here
```

In Notepad, choose **Save as type: All files** and type the name `.env`.
Otherwise Notepad saves it as `.env.txt` and Gyara will not find it.

### Step 5. Check it works

```
gyara --help
```

You should see a list of commands. If you see an error instead, screenshot
it. Then send it to us; that is useful on its own.

Every time you open a new terminal, go to the `gyara` folder and activate
again (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on
macOS or Linux) before you run `gyara`.

## The tests (about 15 minutes)

Note the time when you start each test. Run every command from the `gyara`
folder.

### Test 1. Score transcripts (3 minutes, route B)

We include 5 Hausa sentences and what `NCAIR1/Hausa-ASR` wrote for them.

```
gyara score --ref docs/validation/samples/refs.txt --hyp docs/validation/samples/hyps.txt
```

You should see a table with word and character error rates, scored three
ways (raw, standard, lenient), each with a range. With only 5 sentences the
ranges are very wide. That is expected.

Optional: try it on your own files. Each file has one sentence per line, in
the same order: the correct text in one, an AI's output in the other.

Does the table make sense to you? What would you want to see instead?

### Test 2. Benchmark the speech model on 10 clips (5 minutes, route B)

```
gyara fetch-fleurs --split test --limit 10
gyara eval --manifest data/fleurs/test.jsonl --limit 10
```

The first command downloads 10 Hausa clips from FLEURS, a public test set.
The second downloads the speech model once (about 1 GB) and transcribes the
clips. On a laptop without a GPU this takes a few minutes.

When it finishes, open the `report.md` file it names, inside the `runs`
folder. With 10 clips the range will be wide. That is expected.

### Test 3. Correct a recording (7 minutes, routes A and B)

Use a short Hausa recording, under 2 minutes, that you are allowed to use.
No recording? On route B, use any `.wav` file in `data/fleurs/test/` from
Test 2.

On route B, start the app first:

```
gyara ui
```

Then open http://127.0.0.1:7860 in your browser. On route A, open Baffa's
link.

1. Type your name in **Your name**.
2. In **1 · Transcribe**, upload the recording, fill in **Speaker** (any
   code, for example `test-01`) and click **Transcribe**. Wait for the draft.
3. In **2 · Correct**, listen to each piece, fix the text, and click
   **Save & verify, next**. Do at least 5 pieces.
4. Click **Get suggestion from N-ATLaS** on one piece, if the button is
   there. Accept or reject what it proposes. If the page says suggestions are
   off, that is expected on most computers. Skip this step.
5. In **3 · Export**, click **Make subtitles** and download the `.srt`
   file. Open it in a video player (VLC works) if you can.
6. Still in **3 · Export**, click **Build dataset (.zip)**. If you did not
   tick the consent box in step 2, it says "Nothing to export yet". That is
   on purpose: only checked pieces from recordings with the speaker's written
   consent become training data. If you ticked it, you get a `.zip`.

On route B, stop the app with Ctrl+C in the terminal.

## Send back

1. The filled feedback form (`FEEDBACK-FORM.md`).
2. 2 to 4 screenshots: one result from each test you did, plus anything that
   broke. Name them `yourname_YYYYMMDD_1.png`, `_2`, and so on.
3. Reply to the person who sent you this kit by the date they gave you.

Thank you. With your permission we will name you as a tester in our NAIC
2026 submission.

---

## For Baffa: hosting a route A session

Do this on a laptop with Gyara installed and the token set (route B above).

1. Agree a one-hour window with the tester. The link only works while your
   terminal stays open.
2. Start the app with a separate workspace, so testers do not see each
   other's work from earlier sessions:

   ```
   gyara ui --share --db workspace/tester-NAME.db
   ```

3. Wait for the line `Running on public URL: https://….gradio.live`. Open the
   link yourself first to check it loads.
4. Send the tester:
   - the `.gradio.live` link;
   - the window, for example "Sat 10 Oct, 10:00 to 11:00 WAT";
   - this kit and `FEEDBACK-FORM.md`;
   - one line: "Others may use the same link. Only upload a recording you
     are allowed to share."
5. Keep the laptop awake and online for the whole window. When it ends, press
   Ctrl+C. The link stops working.

N-ATLaS suggestions are off in this session unless you start the app with a
suggestion backend (see the README, "Correct"). Tell the tester which.

---

N-ATLAS is an initiative of the Federal Ministry of Communications,
Innovation and Digital Economy, and powered by Awarri Technologies.
