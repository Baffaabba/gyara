"""The manifest: one JSON line per utterance. Every eval and every split uses it.

Fields::

    id          unique utterance id                      (required)
    audio       path to audio, relative to the manifest  (required)
    text        human reference transcript               (required for eval)
    speaker     speaker id                               (strongly recommended)
    group       bootstrap cluster; defaults to speaker   (optional)
    dialect     e.g. kano, sokoto, zaria, niger          (optional)
    duration    seconds                                  (optional)
    source      where it came from, e.g. fleurs, own     (optional)
    licence     licence / consent reference              (required for own audio)
    sha256      audio content hash                       (filled by `gyara manifest hash`)

Held-out test manifests are frozen: ``leakage_check`` refuses any training row
that shares an audio hash *or a speaker* with a held-out manifest, so a reported
"after fine-tuning" number is never measured on data the model trained on, or
on a voice it trained on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Row:
    id: str
    audio: str
    text: str = ""
    speaker: str | None = None
    group: str | None = None
    dialect: str | None = None
    duration: float | None = None
    source: str | None = None
    licence: str | None = None
    sha256: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def cluster(self) -> str | None:
        return self.group or self.speaker

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extra" and v is not None}
        d.update(self.extra)
        return d


_KNOWN = set(Row.__dataclass_fields__) - {"extra"}


def read(path: str | Path) -> list[Row]:
    path = Path(path)
    rows: list[Row] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):  # BOM-safe
        if not line.strip():
            continue
        d = json.loads(line)
        if "id" not in d or "audio" not in d:
            raise ValueError(f"{path}:{lineno}: 'id' and 'audio' are required")
        if d["id"] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {d['id']!r}")
        seen.add(d["id"])
        extra = {k: v for k, v in d.items() if k not in _KNOWN}
        row = Row(**{k: v for k, v in d.items() if k in _KNOWN}, extra=extra)
        row.audio = str(resolve_audio(path, row.audio))
        rows.append(row)
    return rows


def write(path: str | Path, rows: Iterable[Row | dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            d = r.to_dict() if isinstance(r, Row) else r
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def resolve_audio(manifest_path: Path, audio: str) -> Path:
    p = Path(audio)
    return p if p.is_absolute() else (manifest_path.parent / p).resolve()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iter_hashes(rows: Iterable[Row]) -> Iterator[str]:
    for r in rows:
        yield r.sha256 or sha256_file(r.audio)


@dataclass
class LeakageReport:
    shared_audio: list[str]
    shared_speakers: list[str]
    shared_text: list[str]

    @property
    def clean(self) -> bool:
        return not (self.shared_audio or self.shared_speakers)

    def summary(self) -> str:
        if self.clean and not self.shared_text:
            return "No overlap with held-out data."
        parts = []
        if self.shared_audio:
            parts.append(f"{len(self.shared_audio)} clip(s) identical to held-out audio")
        if self.shared_speakers:
            parts.append(f"{len(self.shared_speakers)} speaker(s) also in held-out: "
                         + ", ".join(self.shared_speakers[:10]))
        if self.shared_text:
            parts.append(f"{len(self.shared_text)} sentence(s) also in held-out (warning only)")
        return "; ".join(parts)


def leakage_check(train: list[Row], heldout: list[Row], normalizer=None) -> LeakageReport:
    """Compare a training set with a held-out set. Speakers and audio must not overlap.

    Shared sentence text is reported as a warning only: FLEURS reads the same
    sentences with many speakers, and that is a property of the data, not a bug.
    """
    norm = normalizer or (lambda s: " ".join((s or "").lower().split()))
    held_hash = set(iter_hashes(heldout))
    train_hash = {h: r.id for r, h in zip(train, iter_hashes(train))}
    shared_audio = sorted(train_hash[h] for h in train_hash.keys() & held_hash)
    held_spk = {r.speaker for r in heldout if r.speaker}
    shared_spk = sorted({r.speaker for r in train if r.speaker} & held_spk)
    held_txt = {norm(r.text) for r in heldout if r.text}
    shared_txt = sorted({r.id for r in train if r.text and norm(r.text) in held_txt})
    return LeakageReport(shared_audio, shared_spk, shared_txt)
