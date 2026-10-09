from typer.testing import CliRunner

from gyara import cli


def test_compare_prints_both_sentences_not_json(tmp_path, monkeypatch):
    from gyara import evaluate

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    fake = {"sentence_wer": "A makes fewer word errors than B.",
            "sentence_cer": "No reliable difference in character errors.",
            "wer": {"delta": -0.01}, "path": str(a / "compare.md")}
    monkeypatch.setattr(evaluate, "compare", lambda run_a, run_b, mode: fake)
    res = CliRunner().invoke(cli.app, ["compare", str(a), str(b)])
    assert res.exit_code == 0, res.output
    assert "A makes fewer word errors than B." in res.output
    assert "No reliable difference in character errors." in res.output
    assert '"delta"' not in res.output  # no raw JSON dump
    assert "compare.json" in res.output


# --- QA 2026-10-09 install findings -------------------------------------------

import io
import json
import sys

import pytest

runner = CliRunner()


def _w(p, text, enc="utf-8"):
    p.write_bytes(text.encode(enc))
    return str(p)


def test_utf8_output_survives_a_cp1252_pipe(monkeypatch):
    """B2: a redirected Windows stdout is cp1252 and cannot encode ɗ ƙ ɓ."""
    raw = io.BytesIO()
    fake = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", fake)
    cli._utf8_output()
    print("ɗan ƙasa ɓoye", file=sys.stdout)
    sys.stdout.flush()
    assert "ɗan ƙasa ɓoye" in raw.getvalue().decode("utf-8")


def test_normalize_prints_hausa_and_rejects_bad_mode():
    res = runner.invoke(cli.app, ["normalize", "Ɗan ƘASA"])
    assert res.exit_code == 0 and "ɗan ƙasa" in res.output
    res = runner.invoke(cli.app, ["normalize", "x", "--mode", "bogus"])
    assert res.exit_code == 2 and "Unknown --mode 'bogus'" in res.output
    assert "Traceback" not in res.output


def test_score_json_keeps_confidence_intervals(tmp_path):
    """S1: summary() floats must not overwrite the CI dicts."""
    ref = _w(tmp_path / "r.txt", "ina kwana\nsannu da zuwa\nna gode\n")
    hyp = _w(tmp_path / "h.txt", "ina kwana\nsannu zuwa\nna gode sosai\n")
    out = tmp_path / "s.json"
    res = runner.invoke(cli.app, ["score", "--ref", ref, "--hyp", hyp, "--json", str(out)])
    assert res.exit_code == 0, res.output
    m = json.loads(out.read_text(encoding="utf-8"))["modes"]["standard"]
    assert isinstance(m["wer"], dict) and {"low", "high"} <= set(m["wer"])
    assert isinstance(m["cer"], dict) and m["utterances"] == 3


def test_score_refuses_groups_of_the_wrong_length(tmp_path):
    """S2: a short --groups file used to drop utterances silently."""
    ref = _w(tmp_path / "r.txt", "a b\nc d\ne f\n")
    grp = _w(tmp_path / "g.txt", "spk1\n")
    res = runner.invoke(cli.app, ["score", "--ref", ref, "--hyp", ref, "--groups", grp])
    assert res.exit_code == 2 and "--groups has 1 lines but --ref has 3" in res.output


def test_score_corpus_rejects_groups_of_the_wrong_length():
    from gyara.metrics import score_corpus

    with pytest.raises(ValueError, match="groups has 1 entries"):
        score_corpus(["1", "2"], ["a", "b"], ["a", "b"], groups=["s"])


def test_score_ignores_a_bom_and_explains_non_utf8(tmp_path):
    """S3: Notepad's BOM is not a word error; UTF-16 gets a plain message."""
    ref = _w(tmp_path / "r.txt", "﻿ƙasa ɗaya\r\nna gode\r\n")
    hyp = _w(tmp_path / "h.txt", "ƙasa ɗaya\nna gode\n")
    out = tmp_path / "s.json"
    res = runner.invoke(cli.app, ["score", "--ref", ref, "--hyp", hyp, "--json", str(out)])
    assert res.exit_code == 0, res.output
    modes = json.loads(out.read_text(encoding="utf-8"))["modes"]
    assert all(modes[m]["wer"]["estimate"] == 0 for m in ("raw", "standard", "lenient"))
    bad = _w(tmp_path / "u16.txt", "ƙasa ɗaya\n", enc="utf-16")
    res = runner.invoke(cli.app, ["score", "--ref", bad, "--hyp", hyp])
    assert res.exit_code == 2 and "save it as UTF-8" in res.output


def test_manifest_read_ignores_a_bom(tmp_path):
    from gyara import manifest

    p = tmp_path / "m.jsonl"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"id": "u1", "audio": "a.wav", "text": "x"}).encode())
    assert manifest.read(p)[0].id == "u1"


def test_score_empty_and_mismatched_inputs_are_plain_errors(tmp_path):
    """M3, M5: no nan table; line counts are given; a trailing blank line is fine."""
    empty = _w(tmp_path / "e.txt", "\n\n")
    res = runner.invoke(cli.app, ["score", "--ref", empty, "--hyp", empty])
    assert res.exit_code == 2 and "no reference words" in res.output and "nan" not in res.output
    ref = _w(tmp_path / "r.txt", "a\nb\nc\n")
    hyp = _w(tmp_path / "h.txt", "a\nb\n")
    res = runner.invoke(cli.app, ["score", "--ref", ref, "--hyp", hyp])
    assert res.exit_code == 2 and "--ref has 3 lines but --hyp has 2" in res.output
    hyp2 = _w(tmp_path / "h2.txt", "a\nb\nc\n\n\n")
    assert runner.invoke(cli.app, ["score", "--ref", ref, "--hyp", hyp2]).exit_code == 0


def test_export_from_missing_or_empty_workspace_fails(tmp_path):
    """S4: no new DB, no empty files, no false success."""
    db = tmp_path / "ws" / "gyara.db"
    res = runner.invoke(cli.app, ["export", "srt", "--asset", "1", "--db", str(db),
                                  "--out", str(tmp_path / "a.srt")])
    assert res.exit_code == 2 and "No Gyara workspace" in res.output
    assert not db.exists() and not (tmp_path / "a.srt").exists()

    from gyara.store import Store

    Store(str(db)).close()  # an empty workspace
    res = runner.invoke(cli.app, ["export", "dataset", "--db", str(db), "--out", str(tmp_path / "ds")])
    assert res.exit_code == 1 and "Nothing to export yet" in res.output
    assert not (tmp_path / "ds").exists()
    res = runner.invoke(cli.app, ["export", "srt", "--asset", "7", "--db", str(db),
                                  "--out", str(tmp_path / "a.srt")])
    assert res.exit_code == 2 and "no recording with id 7" in res.output


def test_transcribe_non_audio_is_one_line(tmp_path):
    """S5: a text file renamed .mp3 gets a message, not a traceback."""
    pytest.importorskip("torch")
    fake = _w(tmp_path / "fake.mp3", "this is not audio\n")
    res = runner.invoke(cli.app, ["transcribe", fake])
    assert res.exit_code == 2 and "Could not read fake.mp3 as audio" in res.output
    assert "Traceback" not in res.output


def test_gated_model_gives_three_plain_lines(monkeypatch):
    """B3: fail fast, before any model download, with what to do next."""
    import huggingface_hub
    from huggingface_hub.utils import GatedRepoError

    def gated(model):
        raise GatedRepoError("401 Client Error. Cannot access gated repo", response=None)

    monkeypatch.setattr(huggingface_hub, "auth_check", gated)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(cli.typer.Exit) as e:
        cli._check_model_access("NCAIR1/Hausa-ASR")
    assert e.value.exit_code == 2

    res = runner.invoke(cli.app, ["eval", "--manifest", __file__, "--model", "NCAIR1/Hausa-ASR"])
    assert res.exit_code == 2
    assert "accept the licence" in res.output and "hf auth login" in res.output
    assert "Traceback" not in res.output


def test_transcribe_help_has_no_curly_braces():
    res = runner.invoke(cli.app, ["transcribe", "--help"])
    assert "{audio}" not in res.output and "AUDIO" in res.output
