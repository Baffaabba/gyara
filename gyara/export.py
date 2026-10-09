"""Turn corrected work into subtitles and into a training dataset.

Subtitles can be made from any asset at any time, from the human text or the
raw AI draft. The training dataset is stricter, because it is what makes the
next model better or worse:

* only segments a human has **verified** are exported;
* only recordings with a recorded **consent** reference are exported;
* rows that would leak into a held-out test set (same audio or same speaker,
  ``manifest.leakage_check``) are dropped and listed, so an "after
  fine-tuning" number is never measured on data the model trained on.

The dataset is written in the Hugging Face ``audiofolder`` layout, so
``datasets.load_dataset("audiofolder", data_dir=out)`` loads it directly, plus
a Gyara manifest for ``gyara eval`` and a dataset card.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from gyara import ATTRIBUTION, __version__
from gyara import manifest as mf
from gyara.store import Store
from gyara.subtitles import build_cues, to_srt, to_vtt

SR = 16_000


def export_subtitles(
    store: Store,
    asset_id: int,
    fmt: str = "srt",
    use: str = "current",
    translate: Callable[[str], str] | None = None,
) -> str:
    """Subtitles for one asset.

    ``use="current"`` takes the latest human text (falling back to the draft
    where nobody has edited yet); ``use="asr"`` takes the untouched AI draft.
    Skipped segments are left out. ``translate`` (e.g. ``Suggester.translate``)
    turns each segment into English before cueing.
    """
    if fmt not in ("srt", "vtt"):
        raise ValueError(f"unknown subtitle format: {fmt!r}")
    if use not in ("current", "asr"):
        raise ValueError(f"unknown text source: {use!r}")
    segs = []
    for s in store.segments(asset_id, order="time"):
        if s["status"] == "skipped":
            continue
        text = s["text"] if use == "current" else s["asr_text"]
        if translate and text.strip():
            text = translate(text)
        segs.append({"start": s["start"], "end": s["end"], "text": text})
    cues = build_cues(segs)
    return to_srt(cues) if fmt == "srt" else to_vtt(cues)


# -- audio helpers -------------------------------------------------------------


def _load_audio(path: str | Path) -> np.ndarray:
    """16 kHz mono float32. Uses ``gyara.audio`` (any format via ffmpeg) when present."""
    try:
        from gyara import audio  # noqa: PLC0415  (optional heavy deps)
    except ImportError:
        audio = None
    if audio is not None:
        return audio.load(str(path), sr=SR)
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SR:  # linear resample: good enough as a fallback for wav input
        n = int(round(len(data) * SR / sr))
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
    return data.astype(np.float32)


def _save_wav(path: Path, data: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), SR, subtype="PCM_16")


# -- dataset export ------------------------------------------------------------


@dataclass
class DatasetExport:
    """Result of ``export_dataset``. Behaves like the output ``Path`` too."""

    path: Path
    rows: int = 0
    hours: float = 0.0
    speakers: int = 0
    excluded: list[dict] = field(default_factory=list)
    leakage: str = ""

    def __fspath__(self) -> str:
        return str(self.path)

    def __truediv__(self, other: str) -> Path:
        return self.path / other

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def export_dataset(
    store: Store,
    out_dir: str | Path,
    asset_ids: Iterable[int] | None = None,
    heldout_manifest: str | Path | None = None,
    min_duration: float = 0.5,
    licence: str = "other",
    overwrite: bool = True,
) -> DatasetExport:
    """Write verified, consented segments as an HF ``audiofolder`` dataset.

    Layout::

        out_dir/audio/<asset>_<idx>.wav   16 kHz mono PCM
        out_dir/metadata.jsonl            file_name, transcription, speaker, ...
        out_dir/manifest.jsonl            gyara manifest (for gyara eval / finetune)
        out_dir/README.md                 dataset card

    Returns a ``DatasetExport`` (``.path``, ``.rows``, ``.hours``,
    ``.excluded``); ``os.fspath()`` of it is the output folder.
    """
    out = Path(out_dir)
    if out.exists() and overwrite:
        for name in ("audio",):
            shutil.rmtree(out / name, ignore_errors=True)
        for name in ("metadata.jsonl", "manifest.jsonl", "README.md"):
            (out / name).unlink(missing_ok=True)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    ids = list(asset_ids) if asset_ids is not None else [a["id"] for a in store.assets()]
    excluded: list[dict] = []
    candidates: list[tuple[dict, mf.Row]] = []

    for aid in sorted(ids):
        asset = store.asset(aid)
        if asset is None:
            continue
        verified = [s for s in store.segments(aid) if s["status"] == "verified"]
        if not verified:
            continue
        if not asset["consent_ref"]:
            excluded += [_ex(asset, s, "no recorded speaker consent") for s in verified]
            continue
        audio = _load_audio(asset["path"])
        stem = f"{asset['sha256'][:10]}"
        for s in verified:
            text = (s["text"] or "").strip()
            dur = float(s["end"]) - float(s["start"])
            if not text:
                excluded.append(_ex(asset, s, "empty transcript"))
                continue
            if dur < min_duration:
                excluded.append(_ex(asset, s, f"shorter than {min_duration:g} s"))
                continue
            clip = audio[int(round(s["start"] * SR)): int(round(s["end"] * SR))]
            if len(clip) == 0:
                excluded.append(_ex(asset, s, "segment lies outside the audio"))
                continue
            rel = f"audio/{stem}_{s['idx']:05d}.wav"
            _save_wav(out / rel, clip)
            row = mf.Row(
                id=f"{stem}_{s['idx']:05d}",
                audio=rel,
                text=text,
                speaker=asset["speaker"],
                group=asset["speaker"],
                dialect=asset["dialect"],
                duration=round(len(clip) / SR, 3),
                source="gyara",
                licence=asset["consent_ref"],
                sha256=mf.sha256_file(out / rel),
            )
            meta = {
                "file_name": rel,
                "transcription": text,
                "speaker": asset["speaker"],
                "dialect": asset["dialect"],
                "duration": row.duration,
                "asset_sha256": asset["sha256"],
                "draft": s["asr_text"],
                "source": "gyara",
            }
            candidates.append((meta, row))

    leakage_summary = ""
    if heldout_manifest is not None and candidates:
        held = mf.read(heldout_manifest)
        report = mf.leakage_check([r for _, r in candidates], held)
        held_hashes = set(mf.iter_hashes(held))
        bad_ids = set(report.shared_audio)
        bad_spk = set(report.shared_speakers)
        leakage_summary = report.summary()
        kept = []
        for meta, row in candidates:
            reason = None
            if row.id in bad_ids or meta["asset_sha256"] in held_hashes:
                reason = "audio is in the held-out test set"
            elif row.speaker and row.speaker in bad_spk:
                reason = f"speaker {row.speaker!r} is in the held-out test set"
            if reason:
                excluded.append({"id": row.id, "asset_sha256": meta["asset_sha256"],
                                 "reason": reason})
                (out / row.audio).unlink(missing_ok=True)
            else:
                kept.append((meta, row))
        candidates = kept

    metas = [m for m, _ in candidates]
    rows = [r for _, r in candidates]
    with (out / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    mf.write(out / "manifest.jsonl", rows)

    result = DatasetExport(
        path=out,
        rows=len(rows),
        hours=sum(r.duration or 0 for r in rows) / 3600.0,
        speakers=len({r.speaker for r in rows if r.speaker}),
        excluded=excluded,
        leakage=leakage_summary,
    )
    (out / "README.md").write_text(_card(result, rows, licence), encoding="utf-8")
    return result


def _ex(asset: dict, seg: dict, reason: str) -> dict:
    return {"id": f"{asset['sha256'][:10]}_{seg['idx']:05d}", "asset_sha256": asset["sha256"],
            "reason": reason}


def _card(res: DatasetExport, rows: list[mf.Row], licence: str) -> str:
    dialects: dict[str, int] = {}
    for r in rows:
        dialects[r.dialect or "unspecified"] = dialects.get(r.dialect or "unspecified", 0) + 1
    reasons: dict[str, int] = {}
    for e in res.excluded:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    dialect_lines = "\n".join(f"| {k} | {v} |" for k, v in sorted(dialects.items())) or "| - | 0 |"
    excl_lines = "\n".join(f"- {v} × {k}" for k, v in sorted(reasons.items())) or "- none"
    return f"""---
language:
- ha
license: {licence}
task_categories:
- automatic-speech-recognition
pretty_name: Gyara Hausa corrections
tags:
- hausa
- n-atlas
- gyara
---

# Gyara Hausa speech corrections

Hausa speech clips with transcripts that a person listened to and corrected in
Gyara, starting from an AI draft. Exported
{date.today().isoformat()} by gyara {__version__}.

| | |
| --- | --- |
| Clips | {res.rows} |
| Hours | {res.hours:.3f} |
| Speakers | {res.speakers} |
| Excluded | {len(res.excluded)} |

## Dialects

| Dialect | Clips |
| --- | --- |
{dialect_lines}

## What was left out

{excl_lines}

{('Held-out check: ' + res.leakage) if res.leakage else 'No held-out manifest was given at export time; run `manifest.leakage_check` before training.'}

## Files

- `audio/*.wav`: 16 kHz mono clips cut from the original recordings.
- `metadata.jsonl`: `file_name`, `transcription` (human-verified text), `speaker`,
  `dialect`, `duration`, `asset_sha256` (hash of the source recording), `draft`
  (the AI's original text, for measuring what people changed), `source`.
- `manifest.jsonl`: the same rows in Gyara manifest format (`licence` holds the
  consent reference).

Load it with `datasets.load_dataset("audiofolder", data_dir=".")`.

## Consent and licence

Only recordings for which the uploader confirmed written consent from the
speaker were exported; each row's consent reference is in `manifest.jsonl`.
Licence: `{licence}`. Do not use this data for surveillance, profiling or
impersonation of any speaker.

Models fine-tuned from N-ATLAS on this data keep the N-ATLAS licence. Free use is
limited to 1,000 active users under the N-ATLAS licence.

{ATTRIBUTION}
"""


def zip_export(res: DatasetExport | str | Path) -> Path:
    """Zip an export folder next to itself; returns the .zip path."""
    folder = Path(os.fspath(res))
    return Path(shutil.make_archive(str(folder), "zip", root_dir=folder))
