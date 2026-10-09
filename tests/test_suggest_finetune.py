import json

import numpy as np
import pytest
import soundfile as sf

from gyara import manifest as mf
from gyara import suggest
from gyara.finetune import train as ft
from gyara.stats import PairedResult

# --- Guard -------------------------------------------------------------------


def test_guard_accepts_hook_fix():
    s = suggest.guard("kasa ta", "ƙasa ta")
    assert s.accepted_by_guard and s.suggested == "ƙasa ta"
    assert s.edits == [{"op": "replace", "from": "kasa", "to": "ƙasa", "index": 0}]


def test_guard_accepts_hooks_apostrophe_case_punct_across_whole_sentence():
    s = suggest.guard("yan kasa sun dauki mataki", "'Yan ƙasa sun ɗauki mataki.")
    assert s.accepted_by_guard, s.reason


def test_guard_accepts_small_spelling_fix_and_word_merge():
    assert suggest.guard("sun tafi makarantaa", "sun tafi makaranta").accepted_by_guard
    assert suggest.guard("ya zo a kan hanya", "ya zo akan hanya").accepted_by_guard


@pytest.mark.parametrize("out, why", [
    ("kasa ta yi", "adds"),                           # added word
    ("duniya ta", "rewrites"),                        # different word
    ("ta kasa", None),                                # reorder = two rewrites
    ("Here is the corrected text: ƙasa ta", None),    # chatter
    ("ƙasa ta\nI fixed the hooked k.", None),         # explanation line
    ("", "empty"),
])
def test_guard_rejects(out, why):
    s = suggest.guard("kasa ta", out)
    assert not s.accepted_by_guard
    assert s.suggested == "kasa ta"  # rejection never changes text
    if why:
        assert why in s.reason


def test_guard_rejects_english_and_word_count_jump():
    assert not suggest.guard("kasa ta yi kyau", "the land is good").accepted_by_guard
    assert not suggest.guard("ina kwana", "ina kwana malam yaya aiki").accepted_by_guard


def test_clean_output_strips_wrapping_but_keeps_y_apostrophe():
    assert suggest.clean_output('"ƙasa ta"') == "ƙasa ta"
    assert suggest.clean_output("Corrected text: ƙasa ta") == "ƙasa ta"
    assert suggest.clean_output("```\nƙasa ta\n```") == "ƙasa ta"
    assert suggest.clean_output("'yan ƙasa'").startswith("'yan")
    assert suggest.guard("kasa ta", "Corrected: ƙasa ta").accepted_by_guard


# --- Prompt and backends ------------------------------------------------------


def test_build_messages_is_strict_few_shot_chat():
    msgs = suggest.build_messages("kasa ta")
    assert msgs[0]["role"] == "system"
    assert "Awarri" in msgs[0]["content"] and "ɓ ɗ ƙ" in msgs[0]["content"]
    assert "Never add, remove, reorder or translate" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:-1]] == ["user", "assistant"] * len(suggest.FEW_SHOT)
    assert msgs[-1] == {"role": "user", "content": "kasa ta"}
    # Every few-shot pair must itself pass the guard, or we teach the model to fail it.
    for src, dst in suggest.FEW_SHOT:
        assert suggest.guard(src, dst).accepted_by_guard, (src, dst)


def test_none_backend_returns_original():
    s = suggest.Suggester("none").suggest("kasa ta")
    assert s.suggested == "kasa ta" and not s.accepted_by_guard
    assert s.reason == "suggestions off"
    assert suggest.Suggester("none").translate("kasa ta") == ""


def test_unknown_backend_and_missing_url():
    with pytest.raises(ValueError):
        suggest.Suggester("magic")
    with pytest.raises(ValueError):
        suggest.Suggester("openai", base_url=None)


def _mock_client(reply: str, seen: list):
    httpx = pytest.importorskip("httpx")

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant",
                                                                  "content": reply}}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_backend_posts_chat_completion_and_caches():
    seen = []
    s = suggest.Suggester("openai", model_id="n-atlas-q4", base_url="http://llm:8080",
                          api_key="k", client=_mock_client("ƙasa ta", seen))
    out = s.suggest("kasa ta")
    assert out.accepted_by_guard and out.suggested == "ƙasa ta" and out.model == "n-atlas-q4"
    assert s.suggest("kasa ta") is out  # cached: no second request
    assert len(seen) == 1
    req = seen[0]
    assert str(req.url) == "http://llm:8080/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer k"
    body = json.loads(req.content)
    assert body["model"] == "n-atlas-q4" and body["temperature"] == 0
    assert body["messages"][-1]["content"] == "kasa ta"


def test_openai_backend_guard_applies_and_v1_base_url():
    seen = []
    s = suggest.Suggester("openai", base_url="http://llm/v1",
                          client=_mock_client("Sure! Here it is: duniya ta", seen))
    out = s.suggest("kasa ta")
    assert not out.accepted_by_guard and out.suggested == "kasa ta"
    assert str(seen[0].url) == "http://llm/v1/chat/completions"


def test_translate_uses_translation_prompt():
    seen = []
    s = suggest.Suggester("openai", base_url="http://llm", client=_mock_client('"Good morning"', seen))
    assert s.translate("Ina kwana") == "Good morning"
    assert "English" in json.loads(seen[0].content)["messages"][0]["content"]


# --- Fine-tune data preparation ----------------------------------------------


def _wav(path, seed, seconds=1.0):
    rng = np.random.default_rng(seed)
    sf.write(path, (rng.standard_normal(int(16000 * seconds)) * 0.05).astype("float32"), 16000)


def _export(root, name, speakers, per_speaker=2, seed0=0, texts=None, draft_last=False):
    d = root / name
    (d / "audio").mkdir(parents=True)
    rows = []
    k = seed0
    for s in speakers:
        for j in range(per_speaker):
            k += 1
            fn = f"audio/{s}_{j}.wav"
            _wav(d / fn, k)
            rows.append({"id": f"{name}-{s}-{j}", "audio": fn,
                         "text": (texts or ["sannu da zuwa", "ina kwana"])[j % 2],
                         "speaker": s, "duration": 1.0, "source": "own"})
    if draft_last:
        rows[-1]["draft"] = True
    mf.write(d / "manifest.jsonl", rows)
    return d


def _heldout(root, speakers, seed0=1000, copy_from=None):
    d = root / "held"
    d.mkdir(exist_ok=True)
    rows = []
    for i, s in enumerate(speakers):
        p = d / f"h{i}.wav"
        if copy_from is not None and i == 0:
            p.write_bytes(copy_from.read_bytes())
        else:
            _wav(p, seed0 + i)
        rows.append({"id": f"h{i}", "audio": p.name, "text": "wani abu", "speaker": s})
    mf.write(d / "heldout.jsonl", rows)
    return d / "heldout.jsonl"


def test_load_config_merges_defaults():
    cfg = ft.load_config({"training": {"max_steps": 2}, "data": {"exports": "x"}})
    assert cfg["training"]["max_steps"] == 2
    assert cfg["training"]["learning_rate"] == 1e-5  # default kept
    assert cfg["data"]["exports"] == ["x"]
    assert cfg["base_model"] == "NCAIR1/Hausa-ASR" and cfg["language"] == "hausa"
    with pytest.raises(ValueError):
        ft.load_config({"data": {"fleurs_fraction": 1.0}})


def test_load_config_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("attempt: 3\ntraining:\n  max_steps: 7\n", encoding="utf-8")
    cfg = ft.load_config(p)
    assert cfg["attempt"] == 3 and cfg["training"]["max_steps"] == 7


def test_prepare_data_aborts_on_shared_speaker(tmp_path):
    exp = _export(tmp_path, "exp", ["amina", "bala", "musa"])
    held = _heldout(tmp_path, ["bala"])
    cfg = ft.load_config({"data": {"exports": [str(exp)], "heldout": [str(held)]}})
    with pytest.raises(ft.LeakageError, match="speaker"):
        ft.prepare_data(cfg)


def test_prepare_data_aborts_on_shared_audio(tmp_path):
    exp = _export(tmp_path, "exp", ["amina", "bala"])
    held = _heldout(tmp_path, ["zainab"], copy_from=exp / "audio" / "amina_0.wav")
    cfg = ft.load_config({"data": {"exports": [str(exp)], "heldout": [str(held)]}})
    with pytest.raises(ft.LeakageError, match="identical"):
        ft.prepare_data(cfg)


def test_prepare_data_requires_heldout(tmp_path):
    exp = _export(tmp_path, "exp", ["amina", "bala"])
    with pytest.raises(ValueError, match="heldout"):
        ft.prepare_data(ft.load_config({"data": {"exports": [str(exp)]}}))


def test_prepare_data_clean_split_and_drafts(tmp_path):
    exp = _export(tmp_path, "exp", ["amina", "bala", "musa", "zainab"], draft_last=True)
    held = _heldout(tmp_path, ["other"])
    cfg = ft.load_config({"data": {"exports": [str(exp)], "heldout": [str(held)],
                                   "dev_fraction": 0.25}})
    with pytest.warns(UserWarning, match="draft"):
        d = ft.prepare_data(cfg)
    assert len(d["train"]) + len(d["dev"]) == 7  # 8 rows, 1 draft dropped
    train_spk = {r.speaker for r in d["train"]}
    dev_spk = {r.speaker for r in d["dev"]}
    assert dev_spk and train_spk and not (train_spk & dev_spk)


def test_speaker_dev_split_single_speaker_falls_back_with_warning(tmp_path):
    rows = [mf.Row(id=str(i), audio="x", text="a", speaker="solo") for i in range(10)]
    with pytest.warns(UserWarning, match="by utterance"):
        tr, dev = ft.speaker_dev_split(rows, 0.2, seed=1)
    assert len(dev) == 2 and len(tr) == 8


def test_speaker_dev_split_is_deterministic():
    rows = [mf.Row(id=f"{s}{i}", audio="x", speaker=s) for s in "abcdefgh" for i in range(3)]
    a = ft.speaker_dev_split(rows, 0.25, seed=7)
    b = ft.speaker_dev_split(rows, 0.25, seed=7)
    assert [r.id for r in a[1]] == [r.id for r in b[1]]
    assert not ({r.speaker for r in a[0]} & {r.speaker for r in a[1]})


def test_mix_fleurs_fraction():
    own = [mf.Row(id=f"o{i}", audio="x") for i in range(70)]
    fl = [mf.Row(id=f"f{i}", audio="x", source="fleurs") for i in range(500)]
    mixed = ft.mix_fleurs(own, fl, 0.3, seed=1)
    assert sum(r.source == "fleurs" for r in mixed) == 30
    assert ft.mix_fleurs(own, fl, 0.0) == own


def test_verdict_and_before_after(tmp_path):
    pr = PairedResult(0.30, 0.35, -0.05, -0.08, -0.02, 0.001, 0.99, 50, "cluster", 10000, 0.02)
    sentence = ft.verdict(pr.to_dict(), "tuned", "base")
    assert sentence.startswith("tuned makes fewer errors than base")
    assert ft.verdict({"paired": pr.to_dict()}) == sentence
    summary = {"base_model": "b", "model_dir": "m", "attempt": 2, "train_utterances": 10,
               "train_hours": 0.1, "own_hours": 0.1, "fleurs_utterances": 0, "speakers": 3,
               "rules_version": "1.0.0",
               "comparisons": [{"heldout": "h.jsonl", "sentence": sentence, "compare": {}}]}
    text = ft.write_before_after(tmp_path / "ba.md", summary).read_text(encoding="utf-8")
    assert "Configs tried for this model so far: **2**" in text and sentence in text
    assert "Awarri" in text


@pytest.mark.parametrize("language", ["hausa", None])
def test_evaluate_after_uses_config_language_for_base_and_tuned(tmp_path, monkeypatch, language):
    # Before/after must decode exactly as the config (and the published baseline)
    # says; it used to drop `language` and always force Hausa.
    from gyara import evaluate as ev

    calls = []
    monkeypatch.setattr(ev, "run", lambda model, man, name, **kw: calls.append((name, kw)) or name)
    pr = PairedResult(0.30, 0.35, -0.05, -0.08, -0.02, 0.001, 0.99, 50, "cluster", 10000, 0.02)
    monkeypatch.setattr(ev, "compare", lambda a, b, mode: {"wer": pr.to_dict()})
    cfg = ft.load_config({"language": language, "attempt": 3})
    res = ft._evaluate_after(cfg, tmp_path / "model", [tmp_path / "test.jsonl"], tmp_path)
    assert [n for n, _ in calls] == ["base-test", "hausa-asr-gyara-a3-test"]
    assert all(kw["language"] == language for _, kw in calls)
    assert res[0]["sentence"].startswith("tuned makes fewer errors than base")


# --- End to end (downloads openai/whisper-tiny) --------------------------------


@pytest.mark.slow
def test_train_smoke_whisper_tiny(tmp_path):
    pytest.importorskip("transformers")
    exp = _export(tmp_path, "exp", ["amina", "bala", "musa"], per_speaker=2)
    held = _heldout(tmp_path, ["other"])
    cfg = {
        "base_model": "openai/whisper-tiny",
        "processor_fallback": "openai/whisper-tiny",
        "output_dir": str(tmp_path / "out"),
        "data": {"exports": [str(exp)], "heldout": [str(held)], "dev_fraction": 0.34},
        "training": {"max_steps": 2, "per_device_train_batch_size": 2,
                     "per_device_eval_batch_size": 2, "gradient_accumulation_steps": 1,
                     "eval_steps": 1, "generation_max_length": 16, "logging_steps": 1},
        "eval": {"after": False},
    }
    s = ft.train(cfg)
    assert s["global_step"] == 2
    assert s["encoder_frozen"] is True  # a few seconds of audio << 5 h
    assert s["dev_wer"] is not None
    card = (tmp_path / "out" / "model" / "README.md").read_text(encoding="utf-8")
    assert "Awarri" in card and "openai/whisper-tiny" in card
    assert (tmp_path / "out" / "model" / "config.json").exists()
    assert (tmp_path / "out" / "model" / "preprocessor_config.json").exists()
