---
name: naic-rubric
description: The NAIC 2026 rules Gyara is judged against — Problem Statement 01 text, the seven required submission components, the six evaluation criteria, the Real-World Validation rule, deadlines and what winning actually earns — plus the known competing entries. Load before scoping, reviewing the build as a judge would, writing docs or the video script, or packaging the submission.
---

# NAIC 2026: what we are judged on

Source: https://ncair.nitda.gov.ng/naic/ (checked 2026-10-08). Re-check the page
before submission; organisers edit it.

## Problem Statement 01: Developer Infrastructure

> Build the tools that make N-ATLAS easy to build with. N-ATLAS is open-source
> and downloadable, but Nigerian-maintained tooling is needed to help developers
> move from accessing the model to integrating, testing, adapting and deploying it.

Listed examples include a **"fine-tuning starter kit with training scripts and
evaluation tools"**. That is Gyara, almost word for word. Quote it.

**Hard requirement: "must directly integrate with N-ATLAS. Wrapping another
general-purpose model does not qualify."** Gyara uses `NCAIR1/Hausa-ASR` as the
engine it benchmarks and fine-tunes and `NCAIR1/N-ATLaS` for suggestions and
translation. `openai/whisper-small` appears only as a comparison baseline;
never let it look like the engine.

## The seven components (all required)

1. **Working artefact**: repo, deployed app, published model, live API or dataset.
   We ship repo (tagged `v0.1.0`) + local install with `gyara ui --share` for testers (no hosted Space: HF needs PRO; DECISIONS 9 Oct) + benchmark results (+ dataset/model
   if consent and time allow).
2. **N-ATLAS integration evidence**: show exactly where and how. `docs/N-ATLAS-INTEGRATION.md`.
3. **Real-world validation**: "Evidence of testing with actual users, real data
   or live benchmarks. Not simulated or hypothetical." **PS01 requires ≥ 2
   external beta testers.** Keep their written feedback and screenshots in
   `docs/validation/`.
4. **Technical documentation**: architecture, setup, usage.
5. **Video**: **3–5 minutes, end to end.**
6. **Team profile.**
7. **Endorsement / registration**: CAC certificate (Innovation & Enterprise track).

Materials in English. One problem statement per team.

## The six evaluation criteria (no weights published)

The page says evaluation "focuses heavily on the **quality of the working build**
and the **depth of its integration with N-ATLAS**". Weight effort accordingly.

| Criterion | What earns it for Gyara |
| --- | --- |
| Working artefact & technical rigour | Installs from README on a fresh machine; tests green; the loop runs end to end; numbers with CIs |
| N-ATLAS integration | Both N-ATLAS models used for real work; measured effect of N-ATLaS suggestions; fine-tuned Hausa-ASR (FLEURS train, DECISIONS 9 Oct); GGUF laptop path supported in code but untested |
| Real-world validation | First public WER for `NCAIR1/Hausa-ASR` on FLEURS (621 utts) + own held-out audio; ≥ 2 external testers with written feedback; corrected hours collected |
| Impact potential | Every Hausa speech developer and NCAIR itself get a trustworthy benchmark and a data flywheel; media/creators get subtitles |
| Scalability & sustainability | Open-source core + hosted subtitles + data services; licence-cap plan (1,000 users) with Awarri commercial path; more languages by swapping the model id |
| Team capability | Roles, what each person shipped, evidence of disciplined execution (gates, decisions log) |

## Timeline

- Submission: **Mon 12 Oct 2026, 11:59 PM WAT. Late submissions are not accepted.**
  Our internal deadline is Sun 11 Oct, 6 PM.
- N-ATLAS integration check 15–17 Oct; shortlisting 16–20 Oct; mentorship
  25 Oct – 4 Nov; Digital Nigeria Showcase finals 8–10 Nov. The integration
  check means **judges may run it**: the README path must work.

## What winning earns

No cash prize or medals are listed on the page. Innovation & Enterprise winners
get a government procurement pathway review, ONDI incubation, investor
introductions, N-ATLAS API integration support and a national AI startup
directory listing. Shortlisted teams get N-ATLAS API credentials, compute
credits and mentorship. Do not promise the team a cash prize.

## Known competing PS01 entries (as of 2026-10-08)

- **AtlasForge** (github.com/im-aderm/atlasforge): eval/compare with bootstrap
  CIs, 30 s chunking, QLoRA fine-tuning. Pre-alpha; README says it has never run
  against real weights and has published no Hausa WER. **Our edge: real numbers,
  a Hausa normaliser, and the human-correction loop.**
- **N-ATLAS-Forge** (github.com/Greatgeo/N-ATLAS-Forge), `natlas` PyPI /
  N-ATLAS-Kit: SDK clients. "Already built by others" is exactly why an
  OpenAI-compatible API is out of scope for us.

Claims we can make that they cannot: the first published WER for
`NCAIR1/Hausa-ASR` (none on the model card), a documented Hausa normalisation
standard (no maintained library exists), and a closed correction → dataset →
retrain loop with honest before/after statistics.
