---
title: Gyara
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Correct Hausa speech-to-text, export subtitles and data
models:
  - NCAIR1/Hausa-ASR
  - NCAIR1/N-ATLaS
---

# Gyara: fix Hausa transcripts

Upload a Hausa recording. `NCAIR1/Hausa-ASR` writes a first draft. You listen
and correct it one short piece at a time, and see how many words the AI got
wrong. Then download subtitles (`.srt` / `.vtt`) or a training dataset made
only of segments a person checked, from speakers who gave written consent.

Source code and install guide: https://github.com/Baffaabba/gyara

## Before you use it

- **This demo is shared.** Everyone using this Space sees the same workspace.
  Do not upload recordings you are not allowed to share.
- **Work is not kept.** The Space's disk is wiped when it restarts or sleeps.
  Export your subtitles and dataset before you leave.
- **It runs on a CPU, so transcription is slow.** Start with a short clip,
  under two minutes.

## Licence and attribution

Gyara's code is Apache-2.0. `NCAIR1/Hausa-ASR` and `NCAIR1/N-ATLaS` are under
the N-ATLAS licence. Free use is limited to 1,000 active users.

> N-ATLAS is an initiative of the Federal Ministry of Communications,
> Innovation and Digital Economy, and powered by Awarri Technologies.
