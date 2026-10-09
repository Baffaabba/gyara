import json

from gyara import manifest, subtitles


def test_srt_and_vtt_format():
    cues = subtitles.build_cues([{"start": 0.0, "end": 2.5, "text": "Sannu da zuwa"}])
    srt = subtitles.to_srt(cues)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nSannu da zuwa")
    vtt = subtitles.to_vtt(cues)
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.500")


def test_long_segment_is_split_into_readable_cues():
    text = " ".join(["kalma"] * 60)
    cues = subtitles.build_cues([{"start": 0, "end": 28, "text": text}])
    assert len(cues) > 2
    for c in cues:
        lines = c.text.split("\n")
        assert len(lines) <= subtitles.MAX_LINES
        assert all(len(line) <= subtitles.MAX_CHARS_PER_LINE for line in lines)
        assert c.end - c.start <= subtitles.MAX_CUE_SECONDS + 1e-6
    assert cues[0].start == 0 and abs(cues[-1].end - 28) < 1e-6
    assert all(a.end <= b.start + 1e-9 for a, b in zip(cues, cues[1:]))


def test_hours_timestamp():
    assert subtitles._ts(3725.5, ",") == "01:02:05,500"


def _write_audio(p, payload: bytes):
    p.write_bytes(payload)
    return p.name


def test_manifest_roundtrip_and_leakage(tmp_path):
    a = _write_audio(tmp_path / "a.wav", b"aaa")
    b = _write_audio(tmp_path / "b.wav", b"bbb")
    c = _write_audio(tmp_path / "c.wav", b"aaa")  # same bytes as a
    held = tmp_path / "held.jsonl"
    train = tmp_path / "train.jsonl"
    held.write_text(json.dumps({"id": "h1", "audio": a, "text": "x", "speaker": "s1"}) + "\n", encoding="utf-8")
    train.write_text(
        "\n".join(
            json.dumps(d)
            for d in [
                {"id": "t1", "audio": b, "text": "y", "speaker": "s2"},
                {"id": "t2", "audio": c, "text": "z", "speaker": "s3"},
                {"id": "t3", "audio": b, "text": "X", "speaker": "s1", "custom": 1},
            ]
        ),
        encoding="utf-8",
    )
    rows = manifest.read(train)
    assert rows[2].extra == {"custom": 1}
    rep = manifest.leakage_check(rows, manifest.read(held))
    assert rep.shared_audio == ["t2"]
    assert rep.shared_speakers == ["s1"]
    assert rep.shared_text == ["t3"]
    assert not rep.clean


def test_manifest_rejects_duplicates(tmp_path):
    m = tmp_path / "m.jsonl"
    m.write_text('{"id":"1","audio":"a.wav"}\n{"id":"1","audio":"b.wav"}\n', encoding="utf-8")
    try:
        manifest.read(m)
    except ValueError as e:
        assert "duplicate" in str(e)
    else:
        raise AssertionError("duplicate ids must be rejected")
