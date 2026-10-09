"""The correction workspace: one SQLite file holding audio, drafts and every edit.

Why SQLite: a single file a transcriber can copy, back up or send, no server to
run on a laptop or a Hugging Face Space, and transactions so a crash mid-save
never leaves half a correction behind.

Four tables:

``assets``       one row per uploaded recording, deduplicated on the audio's
                 SHA-256 so uploading the same file twice opens the same work.
``segments``     the speech model's draft, one row per segment, with the model's
                 own confidence signals and a review status.
``revisions``    append-only text history. The *current* text of a segment is
                 its latest revision. Nothing is ever overwritten, so we can
                 always say what the model wrote and what a human changed.
``suggestions``  N-ATLaS proposals. They are recorded and decided by a human;
                 a suggestion never changes a segment's text by itself.

Only segments with status ``verified`` count as human ground truth: they are
what ``asset_stats`` scores and what ``gyara.export`` turns into training data.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from gyara.hallucination import check as loop_check
from gyara.manifest import sha256_file
from gyara.metrics import score_corpus
from gyara.normalize import get_normalizer

SCHEMA_VERSION = 1

# Each entry upgrades the schema from version i to i + 1. Append only: never
# edit a migration that has shipped, add a new one.
_MIGRATIONS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS assets (
        id          INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        path        TEXT NOT NULL,
        sha256      TEXT NOT NULL UNIQUE,
        duration    REAL,
        model_id    TEXT,
        speaker     TEXT,
        dialect     TEXT,
        consent_ref TEXT,
        created_at  TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS segments (
        id                INTEGER PRIMARY KEY,
        asset_id          INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
        idx               INTEGER NOT NULL,
        start             REAL NOT NULL,
        "end"             REAL NOT NULL,
        asr_text          TEXT NOT NULL DEFAULT '',
        avg_logprob       REAL,
        compression_ratio REAL,
        flags             TEXT NOT NULL DEFAULT '[]',
        status            TEXT NOT NULL DEFAULT 'draft'
                          CHECK (status IN ('draft', 'verified', 'skipped')),
        verified_by       TEXT,
        verified_at       TEXT,
        UNIQUE (asset_id, idx)
    );
    CREATE TABLE IF NOT EXISTS revisions (
        id          INTEGER PRIMARY KEY,
        segment_id  INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
        text        TEXT NOT NULL,
        source      TEXT NOT NULL CHECK (source IN ('asr', 'human', 'suggestion')),
        author      TEXT,
        created_at  TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS suggestions (
        id          INTEGER PRIMARY KEY,
        segment_id  INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
        model       TEXT,
        original    TEXT NOT NULL,
        suggested   TEXT NOT NULL,
        edits       TEXT NOT NULL DEFAULT '[]',
        guard_ok    INTEGER NOT NULL DEFAULT 1,
        status      TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'accepted', 'rejected')),
        created_at  TEXT NOT NULL,
        decided_at  TEXT
    );
    CREATE INDEX IF NOT EXISTS ix_segments_asset ON segments(asset_id, idx);
    CREATE INDEX IF NOT EXISTS ix_revisions_segment ON revisions(segment_id, id);
    CREATE INDEX IF NOT EXISTS ix_suggestions_segment ON suggestions(segment_id);
    """,
]

# Flag words the transcriber uses for a repetition loop. Any of these in a
# segment's flags, or a positive ``hallucination.check`` on its draft, puts the
# segment at the front of the "least confident first" queue.
_LOOP_WORDS = ("loop", "repeat", "compression", "hallucinat")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a Segment dataclass or a plain dict alike."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def is_loop_flagged(seg: dict) -> bool:
    flags = seg.get("flags") or []
    if any(any(w in str(f).lower() for w in _LOOP_WORDS) for f in flags):
        return True
    dur = (seg.get("end") or 0) - (seg.get("start") or 0)
    return loop_check(seg.get("asr_text") or "", dur if dur > 0 else None).flagged


class Store:
    """A Gyara workspace database. Safe to share between Gradio worker threads."""

    def __init__(self, path: str | Path = "workspace/gyara.db"):
        self.path = Path(path) if str(path) != ":memory:" else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            str(self.path) if self.path else ":memory:",
            check_same_thread=False,
            isolation_level=None,  # we open transactions explicitly
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        if self.path is not None:
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA synchronous = NORMAL")
        self.migrate()

    # -- plumbing ------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Folder next to the database, where uploaded audio is kept."""
        return self.path.parent if self.path else Path("workspace")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _q(self, sql: str, params: Sequence = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def _tx(self):
        store = self

        class _Tx:
            def __enter__(self_inner):
                store._lock.acquire()
                store._db.execute("BEGIN IMMEDIATE")
                return store._db

            def __exit__(self_inner, exc_type, *_):
                try:
                    store._db.execute("ROLLBACK" if exc_type else "COMMIT")
                finally:
                    store._lock.release()

        return _Tx()

    def schema_version(self) -> int:
        rows = self._q("SELECT MAX(version) AS v FROM schema_version")
        return int(rows[0]["v"] or 0)

    def migrate(self) -> int:
        """Bring the schema up to date. Safe to call any number of times."""
        with self._tx() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS schema_version ("
                " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            cur = db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0
            for version in range(cur + 1, len(_MIGRATIONS) + 1):
                for stmt in _MIGRATIONS[version - 1].split(";"):
                    if stmt.strip():
                        db.execute(stmt)
                db.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, _now()),
                )
        return self.schema_version()

    # -- assets --------------------------------------------------------------

    def add_asset(
        self,
        name: str,
        path: str | Path,
        sha256: str | None = None,
        duration: float | None = None,
        model_id: str | None = None,
        speaker: str | None = None,
        dialect: str | None = None,
        consent_ref: str | None = None,
    ) -> int:
        """Register a recording and return its id.

        Re-uploading the same audio (same SHA-256) returns the existing id, so
        work already done on it is not duplicated. Speaker, dialect and consent
        given on the re-upload fill in fields that were empty; they never
        overwrite what is already recorded.
        """
        sha = sha256 or sha256_file(path)
        with self._tx() as db:
            row = db.execute("SELECT id FROM assets WHERE sha256 = ?", (sha,)).fetchone()
            if row:
                db.execute(
                    "UPDATE assets SET speaker = COALESCE(speaker, ?),"
                    " dialect = COALESCE(dialect, ?), consent_ref = COALESCE(consent_ref, ?),"
                    " duration = COALESCE(duration, ?), model_id = COALESCE(model_id, ?)"
                    " WHERE id = ?",
                    (speaker or None, dialect or None, consent_ref or None, duration,
                     model_id, row["id"]),
                )
                return int(row["id"])
            cur = db.execute(
                "INSERT INTO assets (name, path, sha256, duration, model_id, speaker, dialect,"
                " consent_ref, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, str(path), sha, duration, model_id, speaker or None, dialect or None,
                 consent_ref or None, _now()),
            )
            return int(cur.lastrowid)

    def import_file(self, src: str | Path, name: str | None = None, **fields: Any) -> int:
        """Copy an uploaded file into the workspace and register it.

        Upload temp files vanish when the web app restarts, so the audio we
        will later slice for training is kept under ``<workspace>/assets/``.
        """
        src = Path(src)
        sha = sha256_file(src)
        existing = self.asset_by_sha(sha)
        if existing:
            return self.add_asset(existing["name"], existing["path"], sha256=sha, **fields)
        dest_dir = self.root / "assets"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{sha[:16]}{src.suffix.lower()}"
        if not dest.exists():
            shutil.copyfile(src, dest)
        return self.add_asset(name or src.name, dest, sha256=sha, **fields)

    def asset(self, asset_id: int) -> dict | None:
        rows = self._q("SELECT * FROM assets WHERE id = ?", (asset_id,))
        return dict(rows[0]) if rows else None

    def asset_by_sha(self, sha256: str) -> dict | None:
        rows = self._q("SELECT * FROM assets WHERE sha256 = ?", (sha256,))
        return dict(rows[0]) if rows else None

    def assets(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM assets ORDER BY id DESC")]

    def set_consent(self, asset_id: int, consent_ref: str | None) -> None:
        with self._tx() as db:
            db.execute("UPDATE assets SET consent_ref = ? WHERE id = ?",
                       (consent_ref or None, asset_id))

    # -- segments ------------------------------------------------------------

    def add_segments(self, asset_id: int, segments: Iterable[Any]) -> list[int]:
        """Store the speech model's draft segments; each gets an ``asr`` revision.

        Accepts ``gyara.asr.Segment`` objects or dicts with the same fields.
        Replaces nothing: call it once per asset (see ``has_segments``).
        """
        ids: list[int] = []
        with self._tx() as db:
            base = db.execute(
                "SELECT COALESCE(MAX(idx) + 1, 0) FROM segments WHERE asset_id = ?", (asset_id,)
            ).fetchone()[0]
            for i, seg in enumerate(segments):
                text = (_get(seg, "text") or "").strip()
                cur = db.execute(
                    'INSERT INTO segments (asset_id, idx, start, "end", asr_text, avg_logprob,'
                    " compression_ratio, flags) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (asset_id, base + i, float(_get(seg, "start", 0.0)),
                     float(_get(seg, "end", 0.0)), text, _get(seg, "avg_logprob"),
                     _get(seg, "compression_ratio"),
                     json.dumps(list(_get(seg, "flags") or []), ensure_ascii=False)),
                )
                sid = int(cur.lastrowid)
                db.execute(
                    "INSERT INTO revisions (segment_id, text, source, author, created_at)"
                    " VALUES (?, ?, 'asr', ?, ?)",
                    (sid, text, None, _now()),
                )
                ids.append(sid)
        return ids

    def has_segments(self, asset_id: int) -> bool:
        return bool(self._q("SELECT 1 FROM segments WHERE asset_id = ? LIMIT 1", (asset_id,)))

    _SEG_SQL = (
        'SELECT s.*, (SELECT r.text FROM revisions r WHERE r.segment_id = s.id'
        " ORDER BY r.id DESC LIMIT 1) AS text FROM segments s"
    )

    @staticmethod
    def _seg(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["flags"] = json.loads(d.get("flags") or "[]")
        d["loop_flagged"] = is_loop_flagged(d)
        return d

    def segment(self, seg_id: int) -> dict | None:
        rows = self._q(self._SEG_SQL + " WHERE s.id = ?", (seg_id,))
        return self._seg(rows[0]) if rows else None

    def segments(self, asset_id: int, order: str = "time") -> list[dict]:
        """All segments of an asset, each with its current ``text``.

        ``order="confidence"`` puts loop-flagged segments first, then the
        lowest ``avg_logprob`` (least confident) first; segments without a
        confidence score go last. Whether confidence actually predicts errors
        is an empirical claim: see the eval-integrity rules before marketing it.
        """
        rows = [self._seg(r) for r in self._q(
            self._SEG_SQL + " WHERE s.asset_id = ? ORDER BY s.idx", (asset_id,))]
        if order == "time":
            return rows
        if order == "confidence":
            return sorted(rows, key=lambda s: (
                not s["loop_flagged"],
                s["avg_logprob"] is None,
                s["avg_logprob"] if s["avg_logprob"] is not None else 0.0,
                s["idx"],
            ))
        raise ValueError(f"unknown order: {order!r}")

    def current_text(self, seg_id: int) -> str:
        rows = self._q(
            "SELECT text FROM revisions WHERE segment_id = ? ORDER BY id DESC LIMIT 1", (seg_id,))
        if not rows:
            raise KeyError(f"no segment {seg_id}")
        return rows[0]["text"]

    def revisions(self, seg_id: int) -> list[dict]:
        return [dict(r) for r in self._q(
            "SELECT * FROM revisions WHERE segment_id = ? ORDER BY id", (seg_id,))]

    def set_text(self, seg_id: int, text: str, source: str = "human",
                 author: str | None = None) -> bool:
        """Append a revision. Returns False (and writes nothing) if unchanged."""
        text = (text or "").strip()
        with self._tx() as db:
            row = db.execute(
                "SELECT text FROM revisions WHERE segment_id = ? ORDER BY id DESC LIMIT 1",
                (seg_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"no segment {seg_id}")
            if row["text"] == text:
                return False
            db.execute(
                "INSERT INTO revisions (segment_id, text, source, author, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (seg_id, text, source, author, _now()),
            )
            return True

    def verify(self, seg_id: int, author: str | None = None, text: str | None = None,
               source: str = "human") -> None:
        """Mark a segment as checked by a human, optionally saving new text first."""
        if text is not None:
            self.set_text(seg_id, text, source=source, author=author)
        with self._tx() as db:
            db.execute(
                "UPDATE segments SET status = 'verified', verified_by = ?, verified_at = ?"
                " WHERE id = ?",
                (author, _now(), seg_id),
            )

    def skip(self, seg_id: int) -> None:
        """Mark a segment as skipped (e.g. music, unclear speech). Never exported."""
        with self._tx() as db:
            db.execute(
                "UPDATE segments SET status = 'skipped', verified_by = NULL, verified_at = NULL"
                " WHERE id = ?",
                (seg_id,),
            )

    # -- suggestions ---------------------------------------------------------

    def add_suggestion(self, seg_id: int, original: str, suggested: str, model: str | None = None,
                       edits: list[dict] | None = None, guard_ok: bool = True) -> int:
        with self._tx() as db:
            cur = db.execute(
                "INSERT INTO suggestions (segment_id, model, original, suggested, edits,"
                " guard_ok, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (seg_id, model, original, suggested,
                 json.dumps(edits or [], ensure_ascii=False), int(bool(guard_ok)), _now()),
            )
            return int(cur.lastrowid)

    def decide_suggestion(self, sug_id: int, accept: bool) -> dict:
        """Record a human's accept/reject. Does not change the segment's text.

        Accepting puts the suggestion in front of the human; it becomes the
        segment's text only when they save it (``set_text(..., "suggestion")``).
        """
        with self._tx() as db:
            db.execute(
                "UPDATE suggestions SET status = ?, decided_at = ? WHERE id = ?",
                ("accepted" if accept else "rejected", _now(), sug_id),
            )
        return self.suggestion(sug_id)

    def suggestion(self, sug_id: int) -> dict:
        rows = self._q("SELECT * FROM suggestions WHERE id = ?", (sug_id,))
        if not rows:
            raise KeyError(f"no suggestion {sug_id}")
        d = dict(rows[0])
        d["edits"] = json.loads(d["edits"] or "[]")
        d["guard_ok"] = bool(d["guard_ok"])
        return d

    def suggestions(self, seg_id: int) -> list[dict]:
        ids = [r["id"] for r in self._q(
            "SELECT id FROM suggestions WHERE segment_id = ? ORDER BY id", (seg_id,))]
        return [self.suggestion(i) for i in ids]

    # -- numbers -------------------------------------------------------------

    @staticmethod
    def _score(segs: list[dict]) -> dict:
        """Pooled WER/CER of the AI draft against the human text, verified only."""
        out: dict[str, Any] = {}
        ids = [str(s["id"]) for s in segs]
        refs = [s["text"] for s in segs]
        hyps = [s["asr_text"] for s in segs]
        for mode in ("standard", "raw"):
            sc = score_corpus(ids, refs, hyps, get_normalizer(mode))
            out[mode] = {"wer": sc.wer, "cer": sc.cer, "ref_words": sc.words.ref_len,
                         "word_errors": sc.words.errors}
        return out

    def asset_stats(self, asset_id: int) -> dict:
        segs = self.segments(asset_id)
        ver = [s for s in segs if s["status"] == "verified"]
        sc = self._score(ver)
        return {
            "asset_id": asset_id,
            "total": len(segs),
            "verified": len(ver),
            "skipped": sum(1 for s in segs if s["status"] == "skipped"),
            "hours_verified": sum(s["end"] - s["start"] for s in ver) / 3600.0,
            "wer": sc["standard"]["wer"],
            "cer": sc["standard"]["cer"],
            "ref_words": sc["standard"]["ref_words"],
            "raw": sc["raw"],
            "standard": sc["standard"],
        }

    def global_stats(self) -> dict:
        assets = self.assets()
        ver: list[dict] = []
        total = 0
        for a in assets:
            segs = self.segments(a["id"])
            total += len(segs)
            ver += [s for s in segs if s["status"] == "verified"]
        sc = self._score(ver)
        sug = {r["status"]: r["n"] for r in self._q(
            "SELECT status, COUNT(*) AS n FROM suggestions GROUP BY status")}
        return {
            "assets": len(assets),
            "assets_with_consent": sum(1 for a in assets if a["consent_ref"]),
            "segments": total,
            "verified": len(ver),
            "hours_verified": sum(s["end"] - s["start"] for s in ver) / 3600.0,
            "speakers": len({a["speaker"] for a in assets if a["speaker"]}),
            "wer": sc["standard"]["wer"],
            "cer": sc["standard"]["cer"],
            "ref_words": sc["standard"]["ref_words"],
            "raw": sc["raw"],
            "standard": sc["standard"],
            "suggestions": {k: sug.get(k, 0) for k in ("pending", "accepted", "rejected")},
        }
