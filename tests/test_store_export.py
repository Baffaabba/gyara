import json
import math

import numpy as np
import pytest
import soundfile as sf

from gyara import ATTRIBUTION, manifest
from gyara.export import export_dataset, export_subtitles, zip_export
from gyara.store import Store

SR = 16_000


def _wav(path, seconds=6.0, freq=220.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    y = 0.2 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(len(t))
    sf.write(str(path), y.astype(np.float32), SR)
    return path


SEGS = [
    {"start": 0.0, "end": 2.0, "text": "sannu da zuwa", "avg_logprob": -0.2,
     "compression_ratio": 1.1, "flags": []},
    {"start": 2.0, "end": 4.0, "text": "ina kwana", "avg_logprob": -0.9,
     "compression_ratio": 1.2, "flags": []},
    {"start": 4.0, "end": 6.0, "text": "na gode " * 8, "avg_logprob": -0.1,
     "compression_ratio": 3.0, "flags": ["loop"]},
]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "ws" / "gyara.db")
    yield s
    s.close()


def _asset(store, tmp_path, name="a.wav", consent="form-001", speaker="spk1", seed=0, **kw):
    p = _wav(tmp_path / name, seed=seed)
    aid = store.import_file(p, speaker=speaker, dialect="kano", consent_ref=consent, **kw)
    store.add_segments(aid, SEGS)
    return aid


def test_migrations_are_idempotent(tmp_path):
    db = tmp_path / "g.db"
    s = Store(db)
    v = s.schema_version()
    assert v >= 1
    assert s.migrate() == v
    s.close()
    s2 = Store(db)  # reopen: nothing re-applied, no error
    assert s2.schema_version() == v
    rows = s2._q("SELECT COUNT(*) AS n FROM schema_version")
    assert rows[0]["n"] == v
    assert s2._q("PRAGMA foreign_keys")[0][0] == 1
    assert s2._q("PRAGMA journal_mode")[0][0].lower() == "wal"
    s2.close()


def test_asset_dedupe_on_sha(store, tmp_path):
    p = _wav(tmp_path / "x.wav")
    a = store.import_file(p, speaker=None)
    b = store.import_file(p, speaker="spk9", consent_ref="c1")
    assert a == b
    asset = store.asset(a)
    assert asset["speaker"] == "spk9" and asset["consent_ref"] == "c1"  # filled, not duplicated
    assert len(store.assets()) == 1
    assert (store.root / "assets").exists()


def test_revisions_history_and_verify(store, tmp_path):
    aid = _asset(store, tmp_path)
    seg = store.segments(aid)[0]
    assert store.current_text(seg["id"]) == "sannu da zuwa"
    assert store.set_text(seg["id"], "sannu da zuwa", "human", "baffa") is False  # unchanged
    assert store.set_text(seg["id"], "Sannu da zuwa.", "human", "baffa")
    assert store.set_text(seg["id"], "Sannu da zuwa!", "human", "baffa")
    hist = store.revisions(seg["id"])
    assert [h["source"] for h in hist] == ["asr", "human", "human"]
    assert store.current_text(seg["id"]) == "Sannu da zuwa!"
    assert store.segment(seg["id"])["asr_text"] == "sannu da zuwa"  # draft never changes
    store.verify(seg["id"], author="baffa")
    s = store.segment(seg["id"])
    assert s["status"] == "verified" and s["verified_by"] == "baffa" and s["verified_at"]
    store.skip(store.segments(aid)[1]["id"])
    assert store.segments(aid)[1]["status"] == "skipped"
    with pytest.raises(Exception):
        store.set_text(seg["id"], "x", source="robot")  # CHECK constraint


def test_confidence_order_puts_loops_then_low_logprob_first(store, tmp_path):
    aid = _asset(store, tmp_path)
    by_time = [s["idx"] for s in store.segments(aid, order="time")]
    by_conf = [s["idx"] for s in store.segments(aid, order="confidence")]
    assert by_time == [0, 1, 2]
    assert by_conf == [2, 1, 0]
    assert store.segments(aid)[2]["loop_flagged"]


def test_suggestions_are_never_applied_automatically(store, tmp_path):
    aid = _asset(store, tmp_path)
    sid = store.segments(aid)[1]["id"]
    sug = store.add_suggestion(sid, "ina kwana", "ina kwana?", model="n-atlas",
                               edits=[{"op": "replace"}], guard_ok=True)
    d = store.decide_suggestion(sug, accept=True)
    assert d["status"] == "accepted" and d["decided_at"] and d["edits"] == [{"op": "replace"}]
    assert store.current_text(sid) == "ina kwana"  # still the draft
    store.verify(sid, "baffa", text="ina kwana?", source="suggestion")
    assert store.revisions(sid)[-1]["source"] == "suggestion"
    assert store.global_stats()["suggestions"]["accepted"] == 1


def test_asset_stats_pooled_wer(store, tmp_path):
    aid = _asset(store, tmp_path)
    segs = store.segments(aid)
    st = store.asset_stats(aid)
    assert st["verified"] == 0 and st["total"] == 3 and math.isnan(st["wer"])
    store.verify(segs[0]["id"], "a", text="Sannu da zuwa.")  # 0 errors in standard mode
    store.verify(segs[1]["id"], "a", text="ina kwana lafiya")  # 1 deletion / 3 words
    st = store.asset_stats(aid)
    assert st["verified"] == 2
    assert st["ref_words"] == 6
    assert st["wer"] == pytest.approx(1 / 6)
    assert st["raw"]["wer"] == pytest.approx(3 / 6)  # case + punctuation + deletion
    assert st["hours_verified"] == pytest.approx(4 / 3600)
    g = store.global_stats()
    assert g["verified"] == 2 and g["assets"] == 1 and g["wer"] == pytest.approx(1 / 6)


def test_subtitles_export(store, tmp_path):
    aid = _asset(store, tmp_path)
    segs = store.segments(aid)
    store.set_text(segs[0]["id"], "Sannu da zuwa.", "human")
    store.skip(segs[2]["id"])
    srt = export_subtitles(store, aid, "srt")
    assert srt.startswith("1\n00:00:00,000 --> ") and "Sannu da zuwa." in srt
    assert "na gode" not in srt
    assert "sannu da zuwa" in export_subtitles(store, aid, "vtt", use="asr")
    vtt = export_subtitles(store, aid, "vtt", translate=lambda t: f"EN:{t}")
    assert vtt.startswith("WEBVTT") and "EN:Sannu da zuwa." in vtt


def test_export_dataset_only_verified_and_consented(store, tmp_path):
    a = _asset(store, tmp_path, "a.wav", consent="form-001", speaker="spk1", seed=1)
    b = _asset(store, tmp_path, "b.wav", consent=None, speaker="spk2", seed=2)
    for aid in (a, b):
        s0, s1, _ = store.segments(aid)
        store.verify(s0["id"], "t", text="Sannu da zuwa")
        store.verify(s1["id"], "t", text="Ina kwana")
    out = tmp_path / "export"
    res = export_dataset(store, out)
    assert res.rows == 2  # only asset a, only its 2 verified segments
    assert res.hours == pytest.approx(4 / 3600, rel=1e-3)
    assert sum(1 for e in res.excluded if "consent" in e["reason"]) == 2
    meta = [json.loads(x) for x in (out / "metadata.jsonl").read_text("utf-8").splitlines()]
    assert {m["transcription"] for m in meta} == {"Sannu da zuwa", "Ina kwana"}
    for m in meta:
        assert set(m) >= {"file_name", "transcription", "speaker", "dialect", "duration",
                          "asset_sha256", "draft", "source"}
        assert m["file_name"].startswith("audio/") and m["source"] == "gyara"
        y, sr = sf.read(str(out / m["file_name"]))
        assert sr == SR and y.ndim == 1 and abs(len(y) / SR - 2.0) < 0.01
    rows = manifest.read(out / "manifest.jsonl")
    assert len(rows) == 2 and all(r.group == "spk1" and r.sha256 for r in rows)
    card = (out / "README.md").read_text("utf-8")
    assert ATTRIBUTION in card and "1,000 active users" in card and "consent" in card
    z = zip_export(res)
    assert z.exists() and z.suffix == ".zip"


def test_export_dataset_excludes_heldout_leaks(store, tmp_path):
    a = _asset(store, tmp_path, "a.wav", speaker="spk1", seed=1)
    b = _asset(store, tmp_path, "b.wav", speaker="spk_held", seed=2)
    c_path = _wav(tmp_path / "c.wav", seed=3)
    c = store.import_file(c_path, speaker="spk3", consent_ref="f3")
    store.add_segments(c, SEGS)
    for aid in (a, b, c):
        store.verify(store.segments(aid)[0]["id"], "t", text="sannu")
    held = tmp_path / "held.jsonl"
    manifest.write(held, [
        {"id": "h1", "audio": str(tmp_path / "other.wav"), "text": "x", "speaker": "spk_held",
         "sha256": "0" * 64},
        {"id": "h2", "audio": "c.wav", "text": "y", "speaker": "someone",
         "sha256": manifest.sha256_file(c_path)},  # whole recording c is held out
    ])
    res = export_dataset(store, tmp_path / "exp", heldout_manifest=held)
    assert res.rows == 1
    kept = manifest.read(tmp_path / "exp" / "manifest.jsonl")
    assert kept[0].speaker == "spk1"
    reasons = " | ".join(e["reason"] for e in res.excluded)
    assert "spk_held" in reasons and "audio is in the held-out" in reasons
    assert len(list((tmp_path / "exp" / "audio").glob("*.wav"))) == 1


def test_export_respects_min_duration(store, tmp_path):
    p = _wav(tmp_path / "a.wav")
    aid = store.import_file(p, speaker="s", consent_ref="f")
    store.add_segments(aid, [{"start": 0.0, "end": 0.3, "text": "eh"},
                             {"start": 0.3, "end": 2.0, "text": "to shi ke nan"}])
    for s in store.segments(aid):
        store.verify(s["id"], "t")
    res = export_dataset(store, tmp_path / "o", min_duration=0.5)
    assert res.rows == 1 and res.excluded[0]["reason"].startswith("shorter")


@pytest.mark.slow
def test_export_loads_with_datasets_audiofolder(store, tmp_path):
    datasets = pytest.importorskip("datasets")
    aid = _asset(store, tmp_path, speaker="spk1")
    for s in store.segments(aid)[:2]:
        store.verify(s["id"], "t")
    out = export_dataset(store, tmp_path / "hf")
    ds = datasets.load_dataset("audiofolder", data_dir=str(out.path), split="train")
    assert len(ds) == 2
    assert {"audio", "transcription", "speaker", "draft"} <= set(ds.column_names)
    assert ds[0]["audio"]["sampling_rate"] == SR
