"""Audio, VAD, transcriber logic and the eval pipeline, without downloading anything.

Real-model tests are marked ``slow`` (run with ``-m slow``).
"""

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pytest
import soundfile as sf

import gyara
from gyara import asr, audio, data, evaluate, manifest, normalize, vad

SR = 16000


def _tone(seconds, freq=220.0, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _speechy(seconds, seed=0, sr=SR):
    """Amplitude-modulated noise: crude but energy-wise speech-like."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * np.arange(n) / sr)  # ~4 syllables/s
    return (0.3 * env * rng.standard_normal(n)).astype(np.float32)


def _silence(seconds, sr=SR, seed=1):
    return (1e-4 * np.random.default_rng(seed).standard_normal(int(seconds * sr))).astype(np.float32)


# --- audio -------------------------------------------------------------------------


def test_audio_load_wav_resamples_and_downmixes(tmp_path):
    x = _tone(1.0, sr=44100)
    stereo = np.stack([x, x], axis=1)
    p = tmp_path / "s.wav"
    sf.write(p, stereo, 44100)
    y = audio.load(p)
    assert y.dtype == np.float32 and y.ndim == 1
    assert abs(len(y) - SR) <= 2
    assert abs(audio.duration(p) - 1.0) < 1e-3


def test_audio_save_and_load_bytes(tmp_path):
    p = audio.save_wav(tmp_path / "a.wav", _tone(0.5))
    y = audio.load(p.read_bytes())
    assert len(y) == SR // 2


# --- vad -----------------------------------------------------------------------------


def test_energy_vad_finds_bursts_and_skips_silence():
    sig = np.concatenate([_silence(1), _speechy(2), _silence(3), _speechy(1, seed=2), _silence(2)])
    regions = vad.energy_regions(sig, SR)
    assert len(regions) == 2
    (a0, b0), (a1, b1) = regions
    assert a0 == pytest.approx(1.0, abs=0.1) and b0 == pytest.approx(3.0, abs=0.1)
    assert a1 == pytest.approx(6.0, abs=0.1) and b1 == pytest.approx(7.0, abs=0.1)
    chunks = vad.chunk(sig, SR, backend="energy")
    assert len(chunks) == 1  # both fit into one ≤30 s chunk
    assert chunks[0][0] <= 1.0 and chunks[0][1] >= 7.0


def test_vad_hard_splits_long_speech():
    sig = _speechy(70)
    chunks = vad.chunk(sig, SR, max_s=30, backend="energy")
    assert all(b - a <= 30 + 1e-6 for a, b in chunks)
    assert chunks[0][0] == 0.0 and chunks[-1][1] == pytest.approx(70.0, abs=0.05)
    assert len(chunks) >= 3


def test_vad_no_speech_returns_fixed_windows_not_nothing():
    sig = np.zeros(int(65 * SR), dtype=np.float32)
    chunks = vad.chunk(sig, SR, backend="energy")
    assert chunks == [(0.0, 30.0), (30.0, 60.0), (60.0, 65.0)]


def test_plan_merges_greedily_and_widens_short():
    regions = [(1.0, 2.0), (3.0, 4.0), (20.0, 29.0), (40.0, 40.3)]
    out = vad.plan(regions, total_s=50, max_s=30, min_s=1.0, pad_s=0.2)
    assert out[0] == (0.8, 29.2)  # first three merged, padded
    a, b = out[1]
    assert b - a == pytest.approx(1.0) and a <= 39.8 and b >= 40.5


# --- asr (no model) -------------------------------------------------------------------


def test_max_new_tokens_bounded():
    assert asr.max_new_tokens_for(1) == 28
    assert asr.max_new_tokens_for(30) == 260
    assert asr.max_new_tokens_for(1000) == asr.MAX_NEW_TOKENS_CAP


def _fake_transcriber(responses):
    """Transcriber whose _generate returns scripted outputs (no model load)."""
    t = asr.Transcriber("fake/model", device="cpu")
    calls = []

    def gen(arrays, **kw):
        calls.append(kw)
        return [responses.pop(0) for _ in arrays]

    t._generate = gen
    return t, calls


def test_loop_is_retried_with_temperature_then_accepted():
    loop = " ".join(["na gode"] * 10)
    t, calls = _fake_transcriber([(loop, -0.2), (loop, -0.3), ("sannu da zuwa", -0.5)])
    (seg,) = t.transcribe_batch([_tone(3)])
    assert seg.text == "sannu da zuwa" and seg.flags == ["retried_t0.4"]
    assert calls[1]["do_sample"] and calls[1]["no_repeat_ngram_size"] == 4


def test_persistent_loop_is_trimmed_and_flagged_for_human():
    loop = "ina " + " ".join(["na gode"] * 10)
    t, _ = _fake_transcriber([(loop, -0.2)] * 4)
    (seg,) = t.transcribe_batch([_tone(3)])
    assert seg.flags == ["loop"]
    assert seg.text == "ina na gode na gode"


def test_transcribe_assigns_vad_times(monkeypatch):
    sig = np.concatenate([_silence(1), _speechy(2), _silence(40), _speechy(3, seed=3)])
    monkeypatch.setattr(vad, "detect", lambda a, sr, backend="auto": (vad.energy_regions(a, sr), "energy"))
    t, _ = _fake_transcriber([("sannu", -0.1), ("", -0.9)])
    segs = t.transcribe(sig)
    assert len(segs) == 2
    assert segs[0].start == pytest.approx(0.8, abs=0.1) and segs[1].end == pytest.approx(46, abs=0.1)
    assert segs[1].flags == ["empty"]


def test_transcribe_batch_rejects_long_clips():
    t = asr.Transcriber("fake/model", device="cpu")
    with pytest.raises(ValueError):
        t.transcribe_batch([np.zeros(31 * SR, np.float32)])


# --- data -----------------------------------------------------------------------------


def test_fix_mojibake_only_touches_real_mojibake():
    # Quote mojibake next to a real hooked letter: only the quote is repaired.
    fixed, changed = data.fix_mojibake("â€œSannu da ɗan")
    assert changed and fixed == "“Sannu da ɗan"
    assert data.fix_mojibake("café ƙasa") == ("café ƙasa", False)
    assert data.fix_mojibake("plain") == ("plain", False)


def test_build_manifest_from_folder(tmp_path):
    src = tmp_path / "own"
    src.mkdir()
    audio.save_wav(src / "a1.wav", _tone(1.5))
    (src / "a1.txt").write_text("Sannu da zuwa\n", encoding="utf-8")
    audio.save_wav(src / "a2.wav", _tone(1.0))  # no transcript -> skipped
    (src / "speakers.csv").write_text("file,speaker,dialect,licence\na1.wav,spk1,kano,consent-001\n",
                                      encoding="utf-8")
    out = data.build_manifest_from_folder(src, tmp_path / "m" / "own.jsonl")
    rows = manifest.read(out)
    assert [r.id for r in rows] == ["a1"]
    r = rows[0]
    assert (r.speaker, r.dialect, r.licence, r.text) == ("spk1", "kano", "consent-001", "Sannu da zuwa")
    assert r.duration == pytest.approx(1.5) and audio.duration(r.audio) == pytest.approx(1.5)


# --- evaluate ------------------------------------------------------------------------------


REFS = {
    "u1": ("Sannu da zuwa.", "g1"),
    "u2": ("Ina kwana?", "g1"),
    "u3": ("Ƙasar Najeriya tana da girma.", "g2"),
    "u4": ("Yaro ya tafi makaranta.", "g3"),
    "u5": ("Mun gode sosai.", "g4"),
    "u6": ("Ruwa yana da daɗi.", "g5"),
}


@pytest.fixture
def tiny_manifest(tmp_path):
    rows = []
    for i, (uid, (text, g)) in enumerate(REFS.items()):
        # distinct lengths let the fake transcriber identify each clip
        a = audio.save_wav(tmp_path / "wav" / f"{uid}.wav", _tone(1.0 + 0.25 * i, 200 + 50 * i))
        rows.append({"id": uid, "audio": f"wav/{a.name}", "text": text, "group": g,
                     "dialect": "kano" if i % 2 else "sokoto", "source": "own",
                     "speaker": f"s{i % 3}", "duration": 1.0 + 0.25 * i})
    # one >30 s clip exercises the VAD path
    a = audio.save_wav(tmp_path / "wav" / "long.wav", np.concatenate([_speechy(20), _speechy(15, 5)]))
    rows.append({"id": "long", "audio": "wav/long.wav", "text": "dogon jawabi ne wannan",
                 "group": "g6", "source": "own", "duration": 35.0})
    p = tmp_path / "test.jsonl"
    manifest.write(p, rows)
    return p


@dataclass
class _Seg:
    text: str
    avg_logprob: float | None
    flags: list = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0


class FakeTranscriber:
    """Maps clip length → scripted hypothesis. Mirrors the real interface."""

    def __init__(self, by_len, long_text="dogon jawabi"):
        self.by_len, self.long_text = by_len, long_text

    def transcribe_batch(self, arrays):
        return [_Seg(*self.by_len[len(a)]) for a in arrays]

    def transcribe(self, arr, sr=16000):
        return [_Seg(self.long_text, -0.4, ["loop"], 0, 20), _Seg("ne wannan", -0.2, [], 20, 35)]

    def info(self):
        return {"device": "cpu", "language_used": "hausa"}


def _fake(hyps):
    by_len = {}
    for i, uid in enumerate(REFS):
        n = int((1.0 + 0.25 * i) * SR)
        by_len[n] = hyps[uid]
    return FakeTranscriber(by_len)


GOOD = {
    "u1": ("sannu da zuwa", -0.1), "u2": ("ina kwana", -0.1),
    "u3": ("kasar najeriya tana da girma", -0.3),  # hook error only
    "u4": ("yaro ya tafi makaranta", -0.1), "u5": ("mun gode sosai", -0.1),
    "u6": ("ruwa yana da daɗi", -0.1),
}
BAD = {
    "u1": ("sanu zuwa", -1.5), "u2": ("ina kwana", -0.2), "u3": ("kasa na tana girma", -2.0),
    "u4": ("yaro tafi", -1.2), "u5": ("mun gode", -0.9), "u6": ("ruwa da dadi", -1.1),
}


class _Sugg:
    @dataclass
    class S:
        suggested: str

    def suggest(self, text):
        return self.S(text.replace("kasar", "ƙasar"))


def test_run_writes_predictions_metrics_and_report(tiny_manifest, tmp_path):
    rd = evaluate.run("fake/good", tiny_manifest, out_dir=tmp_path / "runs",
                      transcriber=_fake(GOOD), suggester=_Sugg(), b=500)
    preds = [json.loads(l) for l in (rd / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [p["id"] for p in preds] == list(REFS) + ["long"]
    assert {"id", "ref", "hyp", "avg_logprob", "flags", "duration", "group", "dialect",
            "speaker", "hyp_suggested"} <= set(preds[0])
    long = preds[-1]
    assert long["hyp"] == "dogon jawabi ne wannan" and "loop" in long["flags"]
    assert long["avg_logprob"] == pytest.approx((-0.4 * 20 + -0.2 * 15) / 35)

    m = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
    assert m["meta"]["rules_version"] == normalize.RULES_VERSION
    assert m["meta"]["manifest_sha256"] == manifest.sha256_file(tiny_manifest)
    assert m["meta"]["rtf"] is not None and m["meta"]["language"] == "hausa"
    std, len_ = m["modes"]["standard"], m["modes"]["lenient"]
    # one hook error (kasar vs ƙasar) in 25 standard ref words; lenient folds it
    assert std["ref_words"] == 25
    assert std["wer"]["estimate"] == pytest.approx(1 / 25)
    assert len_["wer"]["estimate"] == 0
    assert m["modes"]["raw"]["wer"]["estimate"] > std["wer"]["estimate"]
    assert std["wer"]["n_units"] == 6 and std["wer"]["unit"] == "cluster"
    assert set(std["breakdown"]["duration"]) == {"<5s", ">30s"}
    assert set(std["breakdown"]["dialect"]) == {"kano", "sokoto"}
    assert std["top_confusions"][0] == {"op": "S", "ref": "ƙasar", "hyp": "kasar", "count": 1}
    assert m["loop_flagged"] == 1
    se = std["suggestion_effect"]
    assert se["changed_utterances"] == 1 and se["wer"]["delta"] == pytest.approx(-1 / 25)

    report = (rd / "report.md").read_text(encoding="utf-8")
    assert gyara.ATTRIBUTION in report
    assert "| standard | 4.00 [" in report
    assert "Suggestion effect" in report


def test_score_run_is_rerunnable_without_inference(tiny_manifest, tmp_path):
    rd = evaluate.run("fake/bad", tiny_manifest, out_dir=tmp_path / "runs",
                      transcriber=_fake(BAD), b=300)
    before = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
    after = evaluate.score_run(rd, b=300)
    assert after["modes"]["standard"]["wer"] == before["modes"]["standard"]["wer"]
    c = after["confidence"]
    assert c["n"] == 7 and c["spearman_avg_logprob_vs_wer"] < 0  # low confidence ↔ errors
    assert "hyp_suggested" not in (rd / "predictions.jsonl").read_text(encoding="utf-8")


def test_compare_and_leaderboard(tiny_manifest, tmp_path):
    good = evaluate.run("fake/good", tiny_manifest, out_dir=tmp_path / "runs",
                        transcriber=_fake(GOOD), b=300)
    bad = evaluate.run("fake/bad", tiny_manifest, out_dir=tmp_path / "runs",
                       transcriber=_fake(BAD), b=300, limit=5)
    res = evaluate.compare(good, bad, b=500)
    assert res["n_common"] == 5 and res["dropped_a"] == 2 and res["dropped_b"] == 0
    assert res["wer"]["delta"] < 0
    md = (good / "compare.md").read_text(encoding="utf-8")
    assert "fake/good" in md and gyara.ATTRIBUTION in md
    # compare.json is the committed source for any published comparison number
    cj = json.loads((good / "compare.json").read_text(encoding="utf-8"))
    assert cj["n_common"] == 5 and cj["wer"]["delta"] == res["wer"]["delta"]
    assert cj["sentence_wer"] in md

    board = evaluate.leaderboard([bad, good])
    lines = board.splitlines()
    assert lines[2].startswith("| fake/good") and lines[3].startswith("| fake/bad")

    # a run scored under other rules must be refused
    mp = bad / "metrics.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["meta"]["rules_version"] = "0.0.1"
    mp.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="normalisation rules"):
        evaluate.compare(good, bad)


def test_metrics_json_is_strict_json(tiny_manifest, tmp_path):
    # Single cluster -> CI undefined (NaN); file must still be valid strict JSON.
    rd = evaluate.run("fake/good", tiny_manifest, out_dir=tmp_path / "runs",
                      transcriber=_fake(GOOD), b=100, limit=1)
    txt = (rd / "metrics.json").read_text(encoding="utf-8")
    json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert "CI n/a" in (rd / "report.md").read_text(encoding="utf-8")


# --- slow: real model + real data -------------------------------------------------------------


@pytest.mark.slow
def test_whisper_tiny_on_fleurs_smoke(tmp_path):
    man = data.fetch_fleurs("test", out_dir=tmp_path / "fleurs", limit=4)
    rd = evaluate.run("openai/whisper-tiny", man, out_dir=tmp_path / "runs", device="cpu", b=200)
    m = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
    assert m["n"]["utterances"] == 4
    assert not math.isnan(m["modes"]["standard"]["wer"]["estimate"])
