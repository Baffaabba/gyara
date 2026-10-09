# Business

## Who uses it, who pays

| Who | What they get | Pays? |
| --- | --- | --- |
| Developers and researchers building Hausa speech apps | A benchmark they can trust, a correction app, a fine-tuning starter kit | No. Open-source core. |
| Transcribers and linguists | A fast screen to correct AI transcripts | No (they are often paid by the customer below) |
| **Hausa media and creators**: Kannywood producers, broadcasters (Freedom Radio, Arewa24, BBC Hausa-type newsrooms), YouTubers, podcasters | Hausa subtitles quickly, English subtitles as an option | **Yes. The paying customer.** |
| NCAIR / Awarri | Measured evidence of where Hausa-ASR fails, and verified training data | Partner, possible data-services customer |

## Why media

- Hausa is one of Africa's largest languages: about 94 million speakers
  (58 m first language, 36 m second), per Ethnologue as cited by
  [Wikipedia](https://en.wikipedia.org/wiki/Hausa_language).
- Kannywood output is large but poorly counted. Estimates run from "not less
  than 300 films per year" ([Blueprint](https://blueprint.ng/?p=282735)) to
  over 500 films and series a year, with rising YouTube and advertising
  revenue ([Africa.com](https://africa.com/top-10-news/inside-northern-nigerias-thriving-film-industry)).
- Broadcasters are moving online. AREWA24 launched an all-Hausa streaming
  service and claims 45 million+ viewers
  ([AREWA24](https://arewa24.com/en/svod-press-release-english/); self-reported).
- Online video needs subtitles for reach: sound-off viewing, search, and
  non-Hausa audiences. Today that work is manual.

We have not yet interviewed a paying customer. That is the first job after
submission.

## Pricing idea (to test, not proven)

| Offer | For | Model |
| --- | --- | --- |
| Open-source toolkit | Developers, researchers | Free (Apache-2.0 code; models under their N-ATLAS licence) |
| Hosted subtitles | Creators, small studios | Pay per audio minute, or a monthly plan with included minutes. Human review as an add-on. |
| Data services | Labs, NCAIR/Awarri, companies building Hausa voice products | Per project: collect consented audio, correct it, deliver a dataset and a measured fine-tune |

Prices will be set after customer interviews. We will not quote a number we
have not tested.

## The licence cap

The N-ATLAS licence allows free use up to **1,000 active end-users**.

- Below the cap: the open-source kit and a hosted pilot run free under the licence.
- Before we reach it: apply to Awarri / the Ministry for a commercial licence.
  Track active users from day one so we know when.
- Every fine-tuned model keeps the N-ATLAS licence and, if renamed, says
  "Powered by Awarri".

## Sustainability

- The core is small (Python, SQLite, Gradio). One developer can maintain it.
- Every correction a customer makes becomes training data (with consent).
  More customers → better model → better subtitles. This is the flywheel.
- Other languages need a different speech model id and normalisation rules,
  not a rewrite.
- Winning NAIC Innovation & Enterprise opens a procurement review, ONDI
  incubation and investor introductions. No cash prize is listed.

## Roadmap

1. More consented, corrected Hausa audio → a stronger fine-tune, re-measured.
2. A public Hausa speech leaderboard on the Gyara benchmark.
3. Hosted subtitle service (pilot with 2–3 creators).
4. Yoruba, Igbo, Nigerian English.
5. Speaker ID and real-time transcription; dubbing later.

## Risks

| Risk | What we do |
| --- | --- |
| Fine-tuning gains are small at first | Report them honestly with CIs. The loop is the product; gains grow with data. |
| N-ATLaS suggestions do not help | Measured and reported. Suggestions stay optional; the guard blocks harmful ones. |
| Licence cap blocks growth | Start the commercial licence talk early. |
| Consent and data protection (NDPA 2023) | Written consent per speaker, a data register, right to withdraw. |
| Dialect gaps (Kano, Sokoto, Zaria, Niger…) | Benchmark by dialect; collect where it is weakest. |
| An 8B LLM is slow without a GPU | GGUF builds on a laptop CPU; GPU for hosted use. |
| Customers will not pay | Interview first; data services as a second revenue line. |
