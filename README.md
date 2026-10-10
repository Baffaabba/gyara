# Gyara

**Measure, correct and improve Hausa speech-to-text on N-ATLAS.**

*Gyara* is Hausa for "correction". It is an open-source toolkit for developers
working with [`NCAIR1/Hausa-ASR`](https://huggingface.co/NCAIR1/Hausa-ASR) and
[`NCAIR1/N-ATLaS`](https://huggingface.co/NCAIR1/N-ATLaS). It closes this
loop:

```
transcribe ──► N-ATLaS suggests ──► a person corrects ──► measure WER/CER
    ▲                                                          │
    └──── fine-tune Hausa-ASR ◄──── export dataset + .srt ◄────┘
```

Built for NAIC 2026, Problem Statement 01: Developer Infrastructure.

## Results

On the FLEURS Hausa test set, `NCAIR1/Hausa-ASR` gets 30.2% of words wrong
(WER 30.19%, 95% CI 28.63–31.84; 621 clips, 331 sentences). Character error
rate is 10.1%.

| Model | WER standard | WER lenient | CER standard |
| --- | --- | --- | --- |
| `NCAIR1/Hausa-ASR` | **30.19** [28.63, 31.84] | 27.33 [25.77, 28.98] | 10.08 [9.00, 11.35] |
| `openai/whisper-small` (its base model) | 89.81 [88.51, 91.28] | 89.38 [88.06, 90.85] | 30.88 [29.60, 32.38] |

Percent, lower is better, 95% cluster-bootstrap confidence intervals. FLEURS
Hausa test: 621 clips, 331 sentences, 3.34 hours. Rules version `1.0.0`
([docs/NORMALISATION.md](docs/NORMALISATION.md)).

- **Read these FLEURS numbers with one caveat.** The `NCAIR1/Hausa-ASR`
  model card does not list its training data. If FLEURS was in it, the
  FLEURS score may be optimistic. Our own held-out audio from new speakers
  would settle this. It is not measured yet.
- **Standard is the headline; lenient sits beside it.** Lenient also ignores
  hooked letters (ɓ ɗ ƙ 'y). FLEURS references often leave hooks out where
  Hausa-ASR writes them, so on FLEURS the lenient score is arguably the fairer
  one. We fixed standard as the headline before we saw any results.
- **Against its base model**, Hausa-ASR has 59.62 points lower WER (95% CI
  −61.00 to −58.17, p < 0.001, same 621 clips, paired test).
- **N-ATLaS spelling suggestions** had a small effect. Accepting every
  suggestion that passed Gyara's guard, with no human review, changed WER by
  −0.18 points (95% CI −0.30 to −0.07, p = 0.002, MDE 0.17). Some suggestions
  made the text worse, which is why a person decides.
- **Fine-tuning result: pending.** We are fine-tuning on FLEURS train and
  will measure once on FLEURS test. That shows the kit works end to end, not
  what human corrections add.

Method, all seven caveats and the files behind each number:
[docs/BENCHMARK.md](docs/BENCHMARK.md). Run reports:
[`runs/hausa-asr-fleurs/report.md`](runs/hausa-asr-fleurs/report.md),
[`runs/whisper-small-fleurs/report.md`](runs/whisper-small-fleurs/report.md),
[`runs/hausa-asr-fleurs/compare.md`](runs/hausa-asr-fleurs/compare.md),
[`runs/hausa-asr-fleurs-natlas/report.md`](runs/hausa-asr-fleurs-natlas/report.md),
[`runs/leaderboard.md`](runs/leaderboard.md). The per-clip predictions
(`predictions.jsonl`) are committed, so anyone can re-score them.

## Install

Python 3.10 or newer. No system ffmpeg needed.

```bash
git clone https://github.com/Baffaabba/gyara
cd gyara
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
pip install -e ".[all]"
gyara --help
```

To activate the venv on Windows, use one of these instead of
`source .venv/bin/activate`:

| Shell | Command |
| --- | --- |
| Command Prompt or PowerShell | `.venv\Scripts\activate` |
| Git Bash | `source .venv/Scripts/activate` |

If PowerShell refuses to run the script, run
`Set-ExecutionPolicy -Scope Process Bypass` first, then activate again.

**Windows and WSL don't mix.** A venv made in Windows has a `Scripts` folder
and only works from Windows shells. A venv made in WSL (Ubuntu) has a `bin`
folder and only works from WSL. Don't paste a Windows path such as
`E:\Repos\gyara\.venv\Scripts\activate` into a WSL terminal: bash drops the
backslashes and says `command not found`. Pick one side and stay on it:

| You are in | Make and activate the venv with |
| --- | --- |
| PowerShell or Command Prompt | `python -m venv .venv` then `.venv\Scripts\activate` |
| Git Bash | `python -m venv .venv` then `source .venv/Scripts/activate` |
| WSL (Ubuntu) | `python3 -m venv .venv-wsl` then `source .venv-wsl/bin/activate` |

Use a different folder name in WSL (`.venv-wsl`), so it can't clash with a
Windows `.venv` in the same checkout. In WSL the repo is under `/mnt/e/...`
when it lives on the `E:` drive.

`.[all]` installs everything: the speech model, the correction app, dataset
export and fine-tuning. **Expect 10 to 30 minutes.** On Windows and macOS it
downloads about 1 GB. On Linux, pip fetches the CUDA build of PyTorch, which is
several GB. If you have no NVIDIA GPU, install the CPU build first, which is
much smaller:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]"
```

If you only want one part:

| You want | Install |
| --- | --- |
| Scoring only (`gyara score`, `gyara compare`) | `pip install -e .` |
| Transcribe and benchmark | `pip install -e ".[asr]"` |
| The correction app (`gyara ui`) | `pip install -e ".[ui]"` |
| FLEURS download and fine-tuning | `pip install -e ".[train]"` |
| Run the tests | add `dev`, e.g. `pip install -e ".[all,dev]"` |

### Get access to the N-ATLAS models (once per person)

Both models are gated: Hugging Face won't let you download them until you
accept their licence. Do this before anything else, with your own account.

1. Sign in at [huggingface.co](https://huggingface.co) (make a free account
   if you need one).
2. Open [`NCAIR1/Hausa-ASR`](https://huggingface.co/NCAIR1/Hausa-ASR), the
   speech-to-text model. Fill in the form at the top and click to agree.
3. Open [`NCAIR1/N-ATLaS`](https://huggingface.co/NCAIR1/N-ATLaS), the
   language model that suggests spelling fixes. Do the same.
4. Reload each page. It should say you have been granted access. If it says
   your request is pending, wait for the approval email.
5. Make a token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   A **read** token is enough to use the models. You need a **write** token
   only to deploy a Space (optional, see the end of this page).

Then give Gyara the token in one of these ways:

1. **A `.env` file (simplest, works in every shell).** Create a file called
   `.env` in the repo folder with one line:

   ```
   HF_TOKEN=hf_your_token_here
   ```

   `gyara` and `python app.py` read it on start-up. It is in `.gitignore`, so
   it never gets committed. A variable already set in your shell wins over it.
2. **`hf auth login`.** It asks for the token and saves it for your user
   account. Answer `N` to "Add token as git credential?" unless you plan to
   push to the Hub with git.
3. **An environment variable:** `export HF_TOKEN=hf_...` (macOS, Linux, WSL,
   Git Bash) or `$env:HF_TOKEN="hf_..."` (PowerShell). It lasts only for that
   terminal.

`hf auth login` saves the token per environment. A login in WSL is stored in
WSL's home folder, and Windows can't see it, and the other way round. If you
logged in on one side and run `gyara` on the other, use the `.env` file.

Never paste a token into an issue, a commit or a chat. If one leaks, delete it
on the tokens page and make a new one.

To check that it works:

```bash
gyara transcribe some-short-clip.wav --out clip.srt
```

If access is missing, Gyara says so in three lines within a few seconds and
tells you which licence page to open.

Without a token the app still opens. It says transcription is not available,
and you can still correct and export earlier work.

No GPU? Transcription runs on a laptop CPU, slowly. For N-ATLaS suggestions
without a GPU, you can serve a community GGUF build (for example `tosinamuda/N-ATLaS-GGUF`) with `llama-server` and pass
`--suggest openai --llm-url http://localhost:8080`. We have not yet tested or
timed this route; our measured suggestions ran on a Colab T4 GPU.

### The Colab notebooks (no GPU needed on your side)

| Notebook | What it does | Run it when |
| --- | --- | --- |
| [`benchmark_colab.ipynb`](notebooks/benchmark_colab.ipynb) | Measures how accurate the models are on FLEURS, the public Hausa test set. Hausa-ASR alone, then whisper-small for comparison, then Hausa-ASR with N-ATLaS spelling fixes. Writes the error rates with confidence intervals to `runs/`. About 2 to 2.5 hours on a free T4. | First. These are the numbers in the README and the submission. |
| [`finetune_colab.ipynb`](notebooks/finetune_colab.ipynb) | Retrains Hausa-ASR on corrected transcripts exported from the app, then checks honestly on held-out audio whether the retrained model is better. | After people have corrected recordings in the app and exported a dataset. |

To run one: open it in Colab, choose **Runtime → Change runtime type → T4
GPU**, add your token under **Secrets** (the key icon) as `HF_TOKEN` with
notebook access on, then **Runtime → Run all**. Each notebook downloads a
zip of its results as it goes. Unzip it in the repo folder so the files land
in `runs/`.

## Use it

**Score transcripts you already have.** No model, no audio:

```bash
gyara score --ref refs.txt --hyp model_output.txt
```

Both files are UTF-8 text with one utterance per line, in the same order.
Line 5 of `model_output.txt` is the model's version of line 5 of `refs.txt`.
You can also use `.jsonl` files with an `id` and a `text` on each line. Then
the order does not matter. Add `--groups speakers.txt` (one speaker per line)
so the confidence interval allows for several clips from the same speaker.

**Benchmark a model** on FLEURS Hausa (621 clips) or your own audio:

```bash
gyara fetch-fleurs --split test
gyara eval --model NCAIR1/Hausa-ASR --manifest data/fleurs/test.jsonl --name hausa-asr-fleurs
gyara eval --model openai/whisper-small --manifest data/fleurs/test.jsonl --name whisper-small-fleurs
gyara compare runs/hausa-asr-fleurs runs/whisper-small-fleurs
```

**Transcribe** a recording to subtitles:

```bash
gyara transcribe interview.mp3 --out interview.srt
```

**Correct** in the browser, then export subtitles and a training dataset:

```bash
gyara ui                         # opens http://127.0.0.1:7860
gyara ui --share                 # also prints a public link to send to testers
gyara export dataset --out exports/my-corrections --heldout data/heldout/own.jsonl
```

Your work is kept in `workspace/gyara.db` (change it with `--db`). N-ATLaS
suggestions are off by default. To turn them on without a GPU, start a GGUF
build with `llama-server`, then run
`gyara ui --llm-backend openai --llm-url http://localhost:8080` (not yet
tested by us).

**Fine-tune** Hausa-ASR on your corrections and measure the change honestly:

```bash
gyara finetune --config gyara/finetune/config.yaml
```

Fine-tuning refuses to start if any clip or speaker in the training data also
appears in a held-out test set.

Run `gyara --help` for everything else.

## What makes the numbers trustworthy

- **Hausa-aware scoring.** Curly vs straight apostrophes, ƴ vs 'y, capital
  hooked letters and digits vs number words are not counted as errors. A
  missing hook (`kasa` for `ƙasa`) still is, because those are different words.
  The `lenient` score shows how much of the gap is hook-only disagreement
  (in either direction: test-set references sometimes miss hooks too).
- **Pooled WER**, not an average of per-clip rates.
- **Confidence intervals** from a cluster bootstrap over speakers (or FLEURS
  sentences), so correlated clips don't make results look more certain than
  they are.
- **Paired tests** for every comparison, with the smallest difference the test
  set can detect. If the interval includes zero, Gyara says "no reliable
  difference".
- **Held-out sets are protected** by an audio-hash and speaker check.

## How it uses N-ATLAS

| Model | Role in Gyara |
| --- | --- |
| `NCAIR1/Hausa-ASR` | The transcription engine; the model Gyara benchmarks and fine-tunes |
| `NCAIR1/N-ATLaS` | Spelling and punctuation suggestions behind a guard. Optional Hausa→English subtitles (built, not yet tested on real audio) |

`openai/whisper-small` appears only as a comparison baseline. Details:
[docs/N-ATLAS-INTEGRATION.md](docs/N-ATLAS-INTEGRATION.md).

N-ATLaS reads text, not audio, so it can "fix" a word that was right. Gyara
never applies a suggestion without a person accepting it, rejects suggestions
that add, drop or rewrite words, and measures the real effect of suggestions
on the test set.

## Docs

- [Normalisation rules](docs/NORMALISATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [N-ATLAS integration](docs/N-ATLAS-INTEGRATION.md)
- [Business model](docs/BUSINESS.md)

## Let testers try the app

There is no hosted demo. Testers either install Gyara with the steps above, or
use a temporary link from someone who has:

```bash
gyara ui --share
```

Gradio prints a public `https://….gradio.live` link. Send it to your testers.
It works only while your terminal stays open, and everyone with the link
shares your workspace and can see what others upload. Stop it with Ctrl+C.
The step-by-step guide for testers is
[docs/validation/TESTER-KIT.md](docs/validation/TESTER-KIT.md).

## Deploy to a Hugging Face Space (optional)

Hugging Face now requires a paid PRO account to host a Gradio Space, so we do
not run one. If you have PRO, the files in `space/` deploy the same app. We
have not been able to test this deployment.

From the repository folder, with a Hugging Face token that can write
(`hf auth login`):

```bash
python space/deploy.py --repo YOUR-NAME/gyara
```

This copies `app.py`, the `gyara/` package, `space/README.md` (the Space
settings) and `space/requirements.txt` into `build/space/`, then uploads them.
Leave out `--repo` to build only and check what would be uploaded.

Then, in the Space's **Settings**:

1. Under **Secrets**, add `HF_TOKEN`: a read token from an account that has
   accepted the `NCAIR1/Hausa-ASR` licence. Without it the app opens but
   cannot transcribe.
2. Restart the Space.

Optional Space variables: `GYARA_MAX_MINUTES` (longest upload, default 10 on a
Space), `GYARA_LLM_BACKEND` and `GYARA_LLM_BASE_URL` (N-ATLaS suggestions), and
`GYARA_DB` (set it to `/data/gyara.db` if you add persistent storage). The top
of `app.py` lists them all.

Everyone who opens a Space shares one workspace, and its disk is wiped on
restart. Testers should export their work before they leave.

## Licence and attribution

Gyara's code is Apache-2.0. The N-ATLAS models and any model fine-tuned from
them are under the N-ATLAS licence: free use is limited to 1,000 active users,
derivatives keep the same licence, and renamed derivatives carry "Powered by
Awarri". Speech data collected with Gyara requires each speaker's written
consent ([consent form](docs/validation/CONSENT-FORM.md)).

> N-ATLAS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.

FLEURS is © Google, CC-BY-4.0.
