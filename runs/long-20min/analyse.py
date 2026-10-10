"""Summarise the 20-minute transcription in runs/long-20min/ (text only).

Run from the repo root after `gyara transcribe ... --out runs/long-20min/long-20min.json`:

    .venv/Scripts/python runs/long-20min/analyse.py

Writes long-20min.srt (rendered from the JSON with the same functions
`gyara transcribe --out x.srt` uses) and summary.json.

The content check scores the whole long-file transcript against the 67 joined
FLEURS references, and the published per-clip predictions of the same 67 clips
(runs/hausa-asr-fleurs, Colab T4) against the same joined references, with the
same `standard` normaliser. It is a descriptive check for dropped or repeated
speech on long input, not a benchmark number: one long "utterance", no CI.
"""
import json
from pathlib import Path

import numpy as np

from gyara import hallucination, vad
from gyara.audio import load
from gyara.metrics import score_corpus
from gyara.normalize import RULES_VERSION, get_normalizer
from gyara.subtitles import build_cues, to_srt

HERE = Path("runs/long-20min")
WAV = Path("data/fleurs/long-20min.wav")

tr = json.loads((HERE / "long-20min.json").read_text(encoding="utf-8"))
clips = json.loads((HERE / "clips.json").read_text(encoding="utf-8"))
segs = tr["segments"]

srt = to_srt(build_cues(segs))
(HERE / "long-20min.srt").write_text(srt, encoding="utf-8", newline="\n")

audio = load(str(WAV))
regions, backend = vad.detect(audio, 16000)
durs = [s["end"] - s["start"] for s in segs]
hyp_all = " ".join(s["text"] for s in segs)
whole_check = hallucination.check(hyp_all, len(audio) / 16000)

norm = get_normalizer("standard")
ref_all = " ".join(c["text"] for c in clips["clips"])
long_score = score_corpus(["long"], [ref_all], [hyp_all], normalizer=norm)
preds = {}
for line in Path("runs/hausa-asr-fleurs/predictions.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        preds[r["id"]] = r["hyp"]
per_clip_hyp = " ".join(preds[c["id"]] for c in clips["clips"])
clip_score = score_corpus(["clips"], [ref_all], [per_clip_hyp], normalizer=norm)

wall = None
for line in (HERE / "transcribe.txt").read_text(encoding="utf-8").splitlines():
    if line.startswith("end ") and "wall_s" in line:
        wall = int(line.split("wall_s")[1])

summary = {
    "input": "FLEURS ha_ng test clips concatenated (read speech, not a natural recording)",
    "n_clips": clips["n_clips"], "gap_between_clips_s": clips["gap_s"],
    "audio_seconds": round(len(audio) / 16000, 1),
    "model": tr["model"], "device": "cpu (fp32), Windows laptop",
    "wall_seconds": wall, "rtf": round(wall / (len(audio) / 16000), 3) if wall else None,
    "vad_backend": backend, "vad_regions": len(regions),
    "segments": len(segs),
    "segment_seconds": {"min": round(min(durs), 2), "median": round(float(np.median(durs)), 2),
                        "max": round(max(durs), 2)},
    "segments_over_30s": sum(d > 30.0 for d in durs),
    "segments_flagged": sum(1 for s in segs if s["flags"]),
    "flags": [{"start": s["start"], "flags": s["flags"]} for s in segs if s["flags"]],
    "empty_segments": sum(1 for s in segs if not s["text"].strip()),
    # The per-segment loop check, run once more on the whole joined transcript.
    "whole_text_loop_check": {"flagged": whole_check.flagged, "reason": whole_check.reason,
                              "compression_ratio": round(whole_check.compression_ratio, 3),
                              "longest_consecutive_repeat": {"times": whole_check.repeat_count,
                                                             "ngram_words": whole_check.repeat_n},
                              "words_per_second": round(whole_check.words_per_second or 0, 3)},
    "srt_cues": srt.count(" --> "),
    "content_check_standard_rules_" + RULES_VERSION: {
        "reference_words": long_score.words.ref_len,
        "long_file_wer": round(long_score.wer, 4), "long_file_cer": round(long_score.cer, 4),
        "long_file_hyp_words": long_score.words.hyp_len,
        "same_clips_one_by_one_wer_t4": round(clip_score.wer, 4),
        "same_clips_one_by_one_cer_t4": round(clip_score.cer, 4),
        "same_clips_one_by_one_hyp_words": clip_score.words.hyp_len,
        "long_file_edits": {"sub": long_score.words.substitutions,
                            "del": long_score.words.deletions,
                            "ins": long_score.words.insertions},
        "one_by_one_edits": {"sub": clip_score.words.substitutions,
                             "del": clip_score.words.deletions,
                             "ins": clip_score.words.insertions},
    },
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8", newline="\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
