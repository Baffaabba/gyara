"""Build evaluation manifests: FLEURS Hausa, and your own recordings.

FLEURS (Conneau et al., 2022) is the public Hausa test set everyone can
reproduce, so it is Gyara's first benchmark. Two facts shape this module:

* **Clusters.** FLEURS has no speaker ids, but each sentence (``id``) is read
  by several speakers. Those readings are correlated, so the sentence id is the
  bootstrap cluster (``group``). Gender is kept as an extra field.
* **Text.** We score against ``raw_transcription`` (cased, punctuated) and let
  ``gyara.normalize`` decide what counts as an error, so the same rules apply
  to every test set. Some raw transcriptions carry mojibake (UTF-8 read as
  cp1252, e.g. ``â€œ`` for ``“``); we repair only spans that round-trip
  exactly, and count the repairs in the manifest's sidecar summary.

``google/fleurs`` was converted to Parquet (the loading script is gone), so
``datasets.load_dataset("google/fleurs", "ha_ng")`` works without
``trust_remote_code``. If that ever fails we read the Parquet files directly
from the Hub.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from gyara import audio as audio_mod
from gyara import manifest

log = logging.getLogger(__name__)

FLEURS_REPO = "google/fleurs"
FLEURS_CONFIG = "ha_ng"
FLEURS_LICENCE = "CC-BY-4.0"
_FLEURS_GENDER = {0: "male", 1: "female", 2: "other"}
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".mp4", ".webm", ".aac")

_NON_ASCII_RUN = re.compile(r"[^\x00-\x7f]+")


def fix_mojibake(text: str) -> tuple[str, bool]:
    """Undo UTF-8-decoded-as-cp1252 damage, span by span.

    Each run of non-ASCII characters is re-encoded as cp1252 and decoded as
    UTF-8. Real Hausa letters (ɗ, ƙ, ɓ) are not cp1252-encodable and real
    Latin-1 letters (é) are not valid UTF-8 on their own, so both are left
    alone; only genuine mojibake changes.
    """
    changed = False

    def repl(m: re.Match[str]) -> str:
        nonlocal changed
        s = m.group(0)
        try:
            fixed = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
        if fixed != s:
            changed = True
        return fixed

    return _NON_ASCII_RUN.sub(repl, text), changed


# --- FLEURS loading ------------------------------------------------------------


def _iter_load_dataset(split: str, config: str, limit: int | None) -> Iterator[dict]:
    from datasets import Audio, load_dataset

    # Streaming avoids downloading the whole split when only a few rows are needed.
    ds = load_dataset(FLEURS_REPO, config, split=split, streaming=limit is not None)
    ds = ds.cast_column("audio", Audio(decode=False))  # we decode bytes ourselves
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield ex


def _iter_parquet(split: str, config: str, limit: int | None) -> Iterator[dict]:
    """Read the Hub's Parquet files directly (no datasets loader involved)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    for revision in (None, "refs/convert/parquet"):
        try:
            files = api.list_repo_files(FLEURS_REPO, repo_type="dataset", revision=revision)
        except Exception as e:  # noqa: BLE001
            log.warning("listing %s@%s failed: %s", FLEURS_REPO, revision, e)
            continue
        wanted = sorted(
            f for f in files
            if f.endswith(".parquet") and f"/{config}/" in f"/{f}"
            and (f"/{split}/" in f"/{f}" or Path(f).name.startswith(f"{split}-")
                 or f"/{split}-" in f)
        )
        if not wanted:
            continue
        n = 0
        for f in wanted:
            local = hf_hub_download(FLEURS_REPO, f, repo_type="dataset", revision=revision)
            for batch in pq.ParquetFile(local).iter_batches(batch_size=64):
                for ex in batch.to_pylist():
                    if limit is not None and n >= limit:
                        return
                    n += 1
                    yield ex
        return
    raise RuntimeError(f"no Parquet files for {FLEURS_REPO}/{config}/{split} on the Hub")


def _decode_fleurs_audio(a: Any) -> tuple[Any, str | None]:
    """FLEURS audio cell → (16 kHz array, original file name)."""
    if isinstance(a, dict):
        name = a.get("path")
        if a.get("bytes"):
            return audio_mod.load(a["bytes"]), name
        if a.get("array") is not None:
            import numpy as np

            arr = np.asarray(a["array"], dtype=np.float32)
            return audio_mod.resample(arr, int(a.get("sampling_rate", 16000))), name
        if name and Path(name).exists():
            return audio_mod.load(name), name
    raise ValueError("FLEURS row has no decodable audio")


def fetch_fleurs(
    split: str = "test",
    out_dir: str | Path = "data/fleurs",
    limit: int | None = None,
    config: str = FLEURS_CONFIG,
) -> Path:
    """Download FLEURS Hausa ``split`` → ``out_dir/<split>.jsonl`` + 16 kHz wavs.

    Returns the manifest path. A ``<split>.summary.json`` next to it records
    the route used, row count and how many transcripts were repaired.
    """
    out_dir = Path(out_dir)
    wav_dir = out_dir / split
    wav_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / f"{split}.jsonl"

    route = "datasets.load_dataset"
    try:
        rows_iter: Iterable[dict] = list(_iter_load_dataset(split, config, limit))
    except Exception as e:  # noqa: BLE001 - try the next route, but say why
        log.warning("load_dataset(%s, %s) failed (%s); reading Parquet directly", FLEURS_REPO, config, e)
        route = "hub parquet"
        rows_iter = _iter_parquet(split, config, limit)

    rows: list[manifest.Row] = []
    repaired = 0
    seen: dict[str, int] = {}
    for ex in rows_iter:
        arr, name = _decode_fleurs_audio(ex["audio"])
        sent_id = str(ex.get("id"))
        stem = Path(name or ex.get("path") or "").stem
        if not stem:
            stem = f"{sent_id}_{seen.get(sent_id, 0)}"
        seen[sent_id] = seen.get(sent_id, 0) + 1
        uid = f"fleurs_{config}_{split}_{stem}"
        wav = audio_mod.save_wav(wav_dir / f"{stem}.wav", arr)
        text, fixed = fix_mojibake(ex.get("raw_transcription") or ex.get("transcription") or "")
        repaired += fixed
        gender = ex.get("gender")
        rows.append(manifest.Row(
            id=uid,
            audio=wav.relative_to(out_dir).as_posix(),
            text=text,
            group=sent_id,
            duration=round(len(arr) / audio_mod.SAMPLE_RATE, 3),
            source="fleurs",
            licence=FLEURS_LICENCE,
            extra={k: v for k, v in {
                "gender": _FLEURS_GENDER.get(gender, gender) if gender is not None else None,
                "fleurs_transcription": ex.get("transcription"),
                "split": split,
            }.items() if v is not None},
        ))
    if not rows:
        raise RuntimeError(f"FLEURS {config}/{split}: no rows loaded")
    manifest.write(man_path, rows)
    summary = {
        "repo": FLEURS_REPO, "config": config, "split": split, "route": route,
        "rows": len(rows), "sentence_ids": len({r.group for r in rows}),
        "hours": round(sum(r.duration or 0 for r in rows) / 3600, 3),
        "mojibake_repaired": repaired, "limit": limit, "licence": FLEURS_LICENCE,
    }
    (out_dir / f"{split}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("FLEURS %s: %s", split, summary)
    return man_path


# --- Own recordings ------------------------------------------------------------


def _read_speakers(folder: Path) -> dict[str, dict]:
    """``speakers.csv``: a ``file``/``id`` column plus speaker, dialect, licence..."""
    path = folder / "speakers.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for rec in csv.DictReader(f):
            key = (rec.get("file") or rec.get("id") or "").strip()
            if key:
                out[Path(key).stem] = {k: (v or "").strip() for k, v in rec.items()}
    return out


def build_manifest_from_folder(
    folder: str | Path,
    out: str | Path,
    source: str = "own",
    licence: str | None = None,
) -> Path:
    """Pair ``x.wav`` (or any audio) with ``x.txt`` into a manifest.

    Optional ``speakers.csv`` in the folder maps file stem → speaker, dialect,
    licence (consent reference), group. Audio without a transcript is listed
    in the log and skipped (it cannot be scored), never silently.
    """
    folder, out = Path(folder), Path(out)
    meta = _read_speakers(folder)
    rows, missing = [], []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        txt = p.with_suffix(".txt")
        if not txt.exists():
            missing.append(p.name)
            continue
        m = meta.get(p.stem, {})
        known = {"file", "id", "speaker", "dialect", "licence", "group"}
        rows.append(manifest.Row(
            id=p.stem if p.parent == folder else p.relative_to(folder).with_suffix("").as_posix(),
            audio=Path(os.path.relpath(p.resolve(), out.parent.resolve())).as_posix(),
            text=txt.read_text(encoding="utf-8").strip(),
            speaker=m.get("speaker") or None,
            group=m.get("group") or None,
            dialect=m.get("dialect") or None,
            duration=round(audio_mod.duration(p), 3),
            source=source,
            licence=m.get("licence") or licence,
            extra={k: v for k, v in m.items() if k not in known and v},
        ))
    if missing:
        log.warning("%d audio file(s) without a .txt transcript skipped: %s",
                    len(missing), ", ".join(missing[:20]))
    if not rows:
        raise ValueError(f"no audio+transcript pairs found in {folder}")
    manifest.write(out, rows)
    return out
