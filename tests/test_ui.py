from dataclasses import dataclass, field

import numpy as np
import pytest
import soundfile as sf

gr = pytest.importorskip("gradio")

from gyara.ui import app as ui  # noqa: E402

SR = 16_000


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    flags: list = field(default_factory=list)
    words: list | None = None


class FakeTranscriber:
    def __init__(self):
        self.calls = 0

    def transcribe(self, path_or_array, sr=SR):
        self.calls += 1
        return [
            FakeSegment(0.0, 1.5, "sannu da zuwa", -0.2),
            FakeSegment(1.5, 3.0, "ina kwana", -1.1),
            FakeSegment(3.0, 4.5, "na gode " * 6, -0.3, 3.1, ["loop"]),
        ]


class BrokenTranscriber:
    def transcribe(self, *a, **k):
        raise OSError("401 gated repo")


@dataclass
class FakeSuggestion:
    original: str
    suggested: str
    edits: list
    accepted_by_guard: bool
    reason: str = ""
    model: str = "fake-natlas"


class FakeSuggester:
    def suggest(self, text):
        return FakeSuggestion(text, text.replace("da", "ɗa"), [{"op": "replace"}], True)

    def translate(self, text):
        return "EN " + text


def _wav(tmp_path, name="rec.wav", seconds=4.5):
    t = np.arange(int(seconds * SR)) / SR
    p = tmp_path / name
    sf.write(str(p), (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), SR)
    return str(p)


@pytest.fixture
def demo(tmp_path):
    d = ui.build_app(tmp_path / "ws" / "g.db", "fake/model", "none",
                     transcriber=FakeTranscriber(), suggester=FakeSuggester(),
                     runs_dir=tmp_path / "runs")
    d.gyara_ctx.clip_dir = tmp_path / "clips"
    yield d
    d.gyara_ctx.store.close()


def test_build_app_returns_blocks(demo):
    assert isinstance(demo, gr.Blocks)


def test_full_correction_loop(demo, tmp_path):
    ctx = demo.gyara_ctx
    status, picker, tabs = ui.transcribe(ctx, _wav(tmp_path), "spk1", "kano", True, "", None)
    assert "First draft ready" in status and "3 segments" in status
    aid = picker["value"]
    assert tabs["selected"] == "correct"

    queue, pos, sug, *view = ui.open_asset(ctx, aid, ui.ORDER_CONF)
    assert len(queue) == 3 and pos == 0 and len(view) == 11
    first = ctx.store.segment(queue[0])
    assert first["loop_flagged"]  # loop first in confidence order
    assert view[3]["visible"] is True  # banner shown
    assert view[4] and view[4].endswith(".wav")  # segment clip written
    assert sf.info(view[4]).duration == pytest.approx(1.5, abs=0.01)

    # Save with a correction, moves on to next unchecked.
    pos2, sug, *view = ui.save_and_next(ctx, aid, queue, pos, "na gode", "amina", None)
    assert pos2 == 1
    assert ctx.store.segment(queue[0])["status"] == "verified"
    assert ctx.store.current_text(queue[0]) == "na gode"
    assert "1</b> of 3" in view[8]

    # Suggestion: shown as a diff, never applied until the person saves.
    current = ctx.store.current_text(queue[1])
    typed = "sannu da zuwa"
    html, controls, state = ui.get_suggestion(ctx, queue, pos2, typed)
    assert controls["visible"] is True and "<ins>ɗa</ins>" in html
    new_text, msg, controls, state = ui.accept_suggestion(ctx, state, typed)
    assert state["accepted"] and new_text["value"] == "sannu ɗa zuwa"
    assert ctx.store.current_text(queue[1]) == current  # still not saved
    pos3, *_ = ui.save_and_next(ctx, aid, queue, pos2, new_text["value"], "amina", state)
    assert ctx.store.revisions(queue[1])[-1]["source"] == "suggestion"
    assert ctx.store.suggestions(queue[1])[0]["status"] == "accepted"

    # Skip the last one -> done state.
    pos4, _, *view = ui.skip_and_next(ctx, aid, queue, pos3)
    assert pos4 == 3 and view[0]["visible"] is True and "Every segment" in view[0]["value"]
    pos5, _, *view = ui.go_previous(ctx, aid, queue, pos4)
    assert pos5 == 2 and view[1]["visible"] is True

    # Export: subtitles and dataset.
    f, msg = ui.make_subtitles(ctx, aid, "SRT (.srt)", "Corrected text", False)
    assert f["value"].endswith(".ha.srt") and "subtitles ready" in msg
    f, msg = ui.make_subtitles(ctx, aid, "WebVTT (.vtt)", "Corrected text", True)
    assert "EN " in open(f["value"], encoding="utf-8").read()
    f, msg = ui.build_dataset(ctx, "All files", aid)
    assert f["value"].endswith(".zip") and "Training dataset ready" in msg
    assert "segments checked" in ui.workspace_summary(ctx)


def test_reupload_reopens_existing_work(demo, tmp_path):
    ctx = demo.gyara_ctx
    p = _wav(tmp_path)
    ui.transcribe(ctx, p, "spk1", None, False, "", None)
    status, picker, tabs = ui.transcribe(ctx, p, "spk1", None, True, "form-7", None)
    assert "already in your workspace" in status
    assert ctx.transcriber.calls == 1
    assert ctx.store.asset(picker["value"])["consent_ref"] == "form-7"


def test_no_consent_still_transcribes_but_excluded_from_dataset(demo, tmp_path):
    ctx = demo.gyara_ctx
    status, picker, _ = ui.transcribe(ctx, _wav(tmp_path), "spk1", None, False, "", None)
    assert "not be used for training" in status
    aid = picker["value"]
    queue, pos, *_ = ui.open_asset(ctx, aid, ui.ORDER_TIME)
    ui.save_and_next(ctx, aid, queue, pos, "sannu da zuwa", "a", None)
    f, msg = ui.build_dataset(ctx, "This file only", aid)
    assert f["value"] is None and "Nothing to export" in msg and "consent" in msg


def test_model_failure_is_reported_not_raised(tmp_path):
    d = ui.build_app(tmp_path / "g.db", "NCAIR1/Hausa-ASR", "none",
                     transcriber=BrokenTranscriber(), runs_dir=tmp_path)
    ctx = d.gyara_ctx
    status, picker, tabs = ui.transcribe(ctx, _wav(tmp_path), "s", None, False, "", None)
    assert "could not load" in status and "HF_TOKEN" in status and "401" not in status
    assert ctx.store.assets()  # file kept so the user can retry

    class OtherFailure:
        def transcribe(self, *a, **k):
            raise RuntimeError("out of memory")

    ctx.transcriber = OtherFailure()
    status, *_ = ui.transcribe(ctx, None, "s", None, False, "", picker["value"])
    assert "could not run" in status and "out of memory" in status
    ctx.store.close()


def test_empty_states_and_no_file(demo):
    ctx = demo.gyara_ctx
    out = ui.render_segment(ctx, None, [], 0)
    assert out[0]["value"] == ui.EMPTY_NO_FILE and out[1]["visible"] is False
    status, *_ = ui.transcribe(ctx, None, "", None, False, "", None)
    assert "Choose a file first" in status


def test_suggestions_off_message(tmp_path):
    d = ui.build_app(tmp_path / "g.db", "m", "none", transcriber=FakeTranscriber(),
                     runs_dir=tmp_path)
    ctx = d.gyara_ctx
    html, controls, state = ui.get_suggestion(ctx, [1], 0, "x")
    assert "switched off" in html and state is None
    ctx.store.close()


def test_live_score_and_diff():
    s = ui.live_score("ina kwana", "Ina kwana lafiya.")
    assert "33.3%" in s  # standard: 1 deletion / 3 words
    assert "Type what you hear" in ui.live_score("x", "  ")
    d = ui.word_diff_html("da zuwa", "ɗa zuwa")
    assert "<del>da</del>" in d and "<ins>ɗa</ins>" in d


def test_benchmark_tab(demo, tmp_path):
    ctx = demo.gyara_ctx
    md, picker = ui.benchmark_overview(ctx)
    assert "No benchmark runs yet" in md
    run = tmp_path / "runs" / "base"
    run.mkdir(parents=True)
    (run / "report.md").write_text("# Report\nhello", encoding="utf-8")
    md, picker = ui.benchmark_overview(ctx)
    assert picker["choices"] == ["base"] or picker["choices"] == [("base", "base")]
    assert "hello" in ui.show_report(ctx, "base")


def test_upload_cap_refuses_long_files_before_saving(demo, tmp_path):
    ctx = demo.gyara_ctx
    ctx.max_minutes = 0.05  # 3 seconds
    status, picker, tabs = ui.transcribe(ctx, _wav(tmp_path, seconds=4.5), "s", None, True, "", None)
    assert "takes up to" in status and "gyara transcribe" in status
    assert ctx.store.assets() == [] and ctx.transcriber.calls == 0
    ctx.max_minutes = None  # no cap locally
    status, *_ = ui.transcribe(ctx, _wav(tmp_path, seconds=4.5), "s", None, True, "", None)
    assert "First draft ready" in status


def test_model_notice_only_when_gated_model_cannot_load(monkeypatch):
    monkeypatch.setattr(ui, "_hf_token", lambda: None)
    monkeypatch.setattr(ui, "_model_cached", lambda m: False)
    assert "HF_TOKEN" in ui.model_notice("NCAIR1/Hausa-ASR")
    assert ui.model_notice("openai/whisper-small") == ""
    monkeypatch.setattr(ui, "_hf_token", lambda: "hf_x")
    assert ui.model_notice("NCAIR1/Hausa-ASR") == ""


def _visible_texts(demo):
    out = []
    for block in demo.blocks.values():
        cfg = block.get_config() if hasattr(block, "get_config") else {}
        if cfg.get("visible", True) is not False:
            out.append(str(cfg.get("value", "")) + str(cfg.get("label", "")))
    return " ".join(out)


def test_suggest_button_hidden_when_backend_none(tmp_path):
    d = ui.build_app(tmp_path / "g.db", "fake/model", "none", transcriber=FakeTranscriber())
    try:
        texts = _visible_texts(d)
        assert "suggestions are off on this server" in texts
        buttons = [b for b in d.blocks.values() if isinstance(b, gr.Button)
                   and b.value == "Get suggestion from N-ATLaS"]
        assert buttons and buttons[0].visible is False
    finally:
        d.gyara_ctx.store.close()


def test_footer_carries_attribution_and_licence(demo):
    from gyara import ATTRIBUTION

    texts = _visible_texts(demo)
    assert ATTRIBUTION in texts and "1,000 active users" in texts
