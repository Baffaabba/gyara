"""The Gyara correction app: upload Hausa audio, fix the AI draft, export.

Four tabs:

``Transcribe``  upload a recording (audio or video), say who is speaking and
                whether we have their written consent, run the speech model.
``Correct``     the heart of Gyara: one segment at a time, listen, fix the
                text, see how far the AI was, save and move on. N-ATLaS can
                propose a fix; it is shown as a diff and only a person can
                accept it (and even then it is not saved until they save).
``Export``      subtitles (.srt/.vtt, optionally English) and the training
                dataset of verified, consented segments.
``Benchmark``   read-only view of ``runs/*`` reports.

Every event handler is a plain module-level function taking an ``AppContext``
first, bound with ``functools.partial``. That keeps them testable without a
browser, and lets tests inject a fake transcriber and suggester.
"""

from __future__ import annotations

import argparse
import difflib
import functools
import html
import os
import tempfile
import threading
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import numpy as np

from gyara import ATTRIBUTION, DEFAULT_ASR_MODEL
from gyara.export import export_dataset, export_subtitles, zip_export
from gyara.metrics import char_counts, word_counts
from gyara.normalize import get_normalizer
from gyara.store import Store

SR = 16_000
LICENCE_NOTE = "Free use limited to 1,000 active users under the N-ATLAS licence."
ORDER_TIME = "In order"
ORDER_CONF = "Least confident first"
DIALECTS = [
    ("Kano", "kano"), ("Katsina", "katsina"), ("Sokoto", "sokoto"), ("Zaria", "zaria"),
    ("Bauchi", "bauchi"), ("Hausa of Niger", "niger"), ("Other / not sure", "other"),
]
HOOKS = ["ɓ", "ɗ", "ƙ", "'y", "Ɓ", "Ɗ", "Ƙ"]

EMPTY_NO_FILE = "No file yet. Upload a Hausa recording to start."
EMPTY_NO_SEGMENTS = ("This file has no transcript yet. Open the **Transcribe** tab and "
                     "press **Transcribe**.")
SUGGEST_OFF = ("N-ATLaS spelling suggestions are off on this server. You can run them on "
               "your own computer with a GGUF build and llama.cpp. The README shows how.")
EMPTY_DONE = ("Every segment in this file has been checked. Well done. "
              "Make subtitles or a training dataset in the **Export** tab.")


# -- context ------------------------------------------------------------------


@dataclass
class AppContext:
    """Everything the handlers need. One per running app."""

    store: Store
    asr_model: str = DEFAULT_ASR_MODEL
    suggester_backend: str = "none"
    device: str | None = None
    transcriber: Any = None  # injected (tests) or loaded on first use
    suggester: Any = None
    runs_dir: Path = Path("runs")
    heldout_manifest: str | None = None
    max_minutes: float | None = None  # upload cap (set on the Space); None = no cap
    clip_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("GRADIO_TEMP_DIR") or Path(tempfile.gettempdir()) / "gradio") / "gyara")
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _audio: dict = field(default_factory=dict)

    def get_transcriber(self):
        with self._lock:
            if self.transcriber is None:
                from gyara.asr import get_transcriber  # heavy: torch + transformers

                self.transcriber = get_transcriber(self.asr_model, device=self.device)
            return self.transcriber

    def get_suggester(self):
        with self._lock:
            if self.suggester is None:
                from gyara.suggest import Suggester

                self.suggester = Suggester(self.suggester_backend, device=self.device)
            return self.suggester

    @property
    def suggestions_on(self) -> bool:
        return self.suggester is not None or self.suggester_backend != "none"

    def asset_audio(self, asset: dict) -> np.ndarray:
        """Decoded audio of an asset, cached (only the most recent two)."""
        key = asset["id"]
        if key not in self._audio:
            from gyara.export import _load_audio

            if len(self._audio) >= 2:
                self._audio.pop(next(iter(self._audio)))
            self._audio[key] = _load_audio(asset["path"])
        return self._audio[key]

    def clip(self, seg: dict) -> str | None:
        """Write just this segment to a small wav for the player."""
        asset = self.store.asset(seg["asset_id"])
        if asset is None:
            return None
        try:
            audio = self.asset_audio(asset)
        except Exception:
            return None
        a, b = int(round(seg["start"] * SR)), int(round(seg["end"] * SR))
        piece = audio[max(a, 0): max(b, a + 1)]
        if len(piece) == 0:
            return None
        import soundfile as sf

        self.clip_dir.mkdir(parents=True, exist_ok=True)
        path = self.clip_dir / f"seg{seg['id']}_{a}_{b}.wav"
        if not path.exists():
            sf.write(str(path), piece.astype(np.float32), SR, subtype="PCM_16")
        return str(path)

    def out_dir(self) -> Path:
        d = self.clip_dir / "downloads"
        d.mkdir(parents=True, exist_ok=True)
        return d


# -- small pure helpers ---------------------------------------------------------


def _esc(s: str | None) -> str:
    return html.escape(s or "")


def _pct(x: float) -> str:
    return "–" if x != x else f"{x * 100:.1f}%"  # NaN-safe


def _clock(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def word_diff_html(old: str, new: str) -> str:
    """Word-level diff: removed words struck through, added words highlighted."""
    a, b = (old or "").split(), (new or "").split()
    out: list[str] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(_esc(" ".join(a[i1:i2])))
            continue
        if i2 > i1:
            out.append(f"<del>{_esc(' '.join(a[i1:i2]))}</del>")
        if j2 > j1:
            out.append(f"<ins>{_esc(' '.join(b[j1:j2]))}</ins>")
    return f'<div class="gy-diff">{" ".join(out)}</div>'


def live_score(draft: str, text: str) -> str:
    """How far the AI draft was from the text the person typed, for one segment."""
    if not (text or "").strip():
        return ('<div class="gy-score gy-muted">Type what you hear. '
                "We will show how far the AI draft was from your text.</div>")
    cells = []
    for mode, label in (("standard", "Standard rules"), ("raw", "Exact, every comma")):
        n = get_normalizer(mode)
        ref, hyp = n(text), n(draft or "")
        w, _ = word_counts(ref, hyp)
        c = char_counts(ref, hyp)
        cells.append(
            f'<div class="gy-score-cell"><div class="gy-score-label">{label}</div>'
            f'<div class="gy-score-num">{_pct(w.rate)}</div>'
            f'<div class="gy-score-sub">WER · {w.errors} of {w.ref_len} words</div>'
            f'<div class="gy-score-sub">CER {_pct(c.rate)}</div></div>'
        )
    return ('<div class="gy-score"><div class="gy-score-title">How far the AI draft was from '
            'your text</div><div class="gy-score-row">' + "".join(cells) + "</div></div>")


def progress_html(ctx: AppContext, asset_id: int | None) -> str:
    if asset_id is None:
        return ""
    st = ctx.store.asset_stats(asset_id)
    frac = st["verified"] / st["total"] if st["total"] else 0.0
    if st["verified"]:
        wer = (f"AI word error rate so far: <b>{_pct(st['wer'])}</b> "
               f"on {st['ref_words']} words you checked (CER {_pct(st['cer'])}, "
               f"exact {_pct(st['raw']['wer'])}).")
    else:
        wer = "Save your first segment to see how accurate the AI was on this file."
    return (f'<div class="gy-progress"><div class="gy-progress-head"><b>{st["verified"]}</b> of '
            f'{st["total"]} segments checked'
            + (f' · {st["skipped"]} skipped' if st["skipped"] else "")
            + f'</div><div class="gy-bar"><span style="width:{frac * 100:.1f}%"></span></div>'
            f'<div class="gy-progress-sub">{wer}</div></div>')


def _card(kind: str, title: str, body: str = "") -> str:
    return (f'<div class="gy-card gy-{kind}"><div class="gy-card-title">{title}</div>'
            + (f'<div class="gy-card-body">{body}</div>' if body else "") + "</div>")


def _hf_token() -> str | None:
    """The Hugging Face token this server would use (HF_TOKEN env var or a saved login)."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _model_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:
        return False


def model_notice(model_id: str) -> str:
    """A warning shown at the top when the gated speech model clearly cannot load.

    No network call: we only check for a token and for a cached copy. The app
    still starts, so people can open, correct and export earlier work.
    """
    if not model_id.startswith("NCAIR1/") or Path(model_id).exists():
        return ""
    if _hf_token() or _model_cached(model_id):
        return ""
    return _card(
        "warn", "Transcription is not available on this server yet.",
        f"The speech model <code>{_esc(model_id)}</code> needs a Hugging Face token, and this "
        "server has none. You can still open earlier work, correct it and export it.<br>"
        "<b>To fix it:</b> accept the licence on the model page, then add your token as "
        "<code>HF_TOKEN</code> (on a Space: Settings, then Secrets) and restart.")


def asset_choices(ctx: AppContext) -> list[tuple[str, int]]:
    out = []
    for a in ctx.store.assets():
        label = a["name"]
        if a["speaker"]:
            label += f" · {a['speaker']}"
        if not a["consent_ref"]:
            label += " · not for training"
        out.append((label, a["id"]))
    return out


def _tick(progress, frac: float, desc: str) -> None:
    try:
        if progress is not None:
            progress(frac, desc=desc)
    except Exception:
        pass  # progress bars are cosmetic; never fail a transcription over one


# -- Transcribe ------------------------------------------------------------------


def transcribe(ctx: AppContext, file: Any, speaker: str, dialect: str, consent: bool,
               consent_ref: str, asset_id: int | None, progress=gr.Progress()):
    """Upload (or reuse) a recording and run the speech model on it.

    Returns ``(status_html, picker_update, tabs_update)``.
    """
    path = file if isinstance(file, (str, os.PathLike)) or file is None else getattr(file, "name", None)
    speaker = (speaker or "").strip() or None
    ref = None
    if consent:
        ref = (consent_ref or "").strip() or f"attested-in-app:{date.today().isoformat()}"
    stay = gr.update()

    if path:
        _tick(progress, 0.02, "Saving your file")
        try:
            dur = None
            try:
                from gyara.audio import duration as audio_duration

                dur = audio_duration(path)
            except Exception:
                pass
            if ctx.max_minutes and dur and dur > ctx.max_minutes * 60:
                return (_card(
                    "warn", f"This recording is {dur / 60:.0f} minutes long. This server takes "
                    f"up to {ctx.max_minutes:g} minutes.",
                    "Cut it into shorter files, or run Gyara on your own computer, which has "
                    "no limit: <code>pip install -e \".[all]\"</code>, then "
                    "<code>gyara transcribe your-file.mp3</code> or <code>gyara ui</code>."),
                    stay, stay)
            aid = ctx.store.import_file(path, name=Path(path).name, speaker=speaker,
                                        dialect=dialect or None, consent_ref=ref,
                                        duration=dur, model_id=ctx.asr_model)
        except Exception as e:  # unreadable upload
            return (_card("error", "We could not read that file.", _esc(str(e))), stay, stay)
    elif asset_id is not None:
        aid = int(asset_id)
        if ref and not ctx.store.asset(aid)["consent_ref"]:
            ctx.store.set_consent(aid, ref)
    else:
        return (_card("warn", "Choose a file first.",
                      "Upload a Hausa recording (audio or video), then press Transcribe."),
                stay, stay)

    picker = gr.update(choices=asset_choices(ctx), value=aid)
    if ctx.store.has_segments(aid):
        return (_card("ok", "This recording is already in your workspace.",
                      "We opened your earlier work on it. Nothing was lost."),
                picker, gr.update(selected="correct"))

    asset = ctx.store.asset(aid)
    _tick(progress, 0.1, "Loading the speech model")
    try:
        tr = ctx.get_transcriber()
        _tick(progress, 0.3, "Listening and writing a first draft")
        segments = tr.transcribe(asset["path"])
    except Exception as e:
        msg = str(e).lower()
        if "gated" in msg or "401" in msg or "403" in msg:
            body = (f"<code>{_esc(ctx.asr_model)}</code> is a gated model on Hugging Face, and "
                    "this server does not have access to it.<br>Your file is saved. To fix it: "
                    "accept the licence on the model page, add a token with access as "
                    "<code>HF_TOKEN</code> (on a Space: Settings, then Secrets), restart, and "
                    "press Transcribe again.")
            return (_card("error", "The speech model could not load.", body), picker, stay)
        body = (f"Model: <code>{_esc(ctx.asr_model)}</code><br>{_esc(type(e).__name__)}: "
                f"{_esc(str(e))[:400]}<br><br>Your file is saved. Press Transcribe again once "
                "the model is available. If the model is gated on Hugging Face, add "
                "<code>HF_TOKEN</code> as a secret.")
        return (_card("error", "The speech model could not run.", body), picker, stay)

    _tick(progress, 0.95, "Saving the draft")
    ctx.store.add_segments(aid, segments)
    n = len(segments)
    flagged = sum(1 for s in ctx.store.segments(aid) if s["loop_flagged"])
    body = f"{n} segment{'s' if n != 1 else ''} ready to check."
    if flagged:
        body += f" {flagged} may contain repeated words; they are marked."
    if not ref:
        body += (" No consent recorded, so this file can be subtitled but will not be "
                 "used for training.")
    return (_card("ok", "First draft ready.", body), picker, gr.update(selected="correct"))


# -- Correct ---------------------------------------------------------------------


def _hidden_editor(message: str) -> tuple:
    return (
        gr.update(value=message, visible=True),   # empty state
        gr.update(visible=False),                 # editor column
        "", gr.update(visible=False), None, "", gr.update(value=""), "",
        gr.update(), "", gr.update(visible=False),
    )


def render_segment(ctx: AppContext, asset_id: int | None, queue: list[int], pos: int) -> tuple:
    """Outputs for the Correct screen (see ``VIEW_OUTPUTS`` in ``build_app``)."""
    if asset_id is None:
        return _hidden_editor(EMPTY_NO_FILE)
    prog = progress_html(ctx, asset_id)
    if not queue:
        out = list(_hidden_editor(EMPTY_NO_SEGMENTS))
        out[8] = prog
        return tuple(out)
    if pos >= len(queue):
        out = list(_hidden_editor(EMPTY_DONE))
        out[8] = prog
        return tuple(out)
    seg = ctx.store.segment(queue[pos])
    status = {"verified": f" · checked by {_esc(seg['verified_by'] or 'someone')}",
              "skipped": " · skipped"}.get(seg["status"], "")
    where = (f'<div class="gy-where"><span class="gy-pill">Segment <b>{pos + 1}</b> of '
             f'{len(queue)}</span><span class="gy-time">{_clock(seg["start"])} – '
             f'{_clock(seg["end"])}</span><span class="gy-status">{status}</span></div>')
    banner = gr.update(visible=bool(seg["loop_flagged"]), value=(
        '<div class="gy-banner"><b>The AI may have repeated itself here — check carefully.</b> '
        "Listen to the whole clip and delete any words that were not said.</div>"))
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        where,
        banner,
        ctx.clip(seg),
        seg["asr_text"],
        gr.update(value=seg["text"]),
        live_score(seg["asr_text"], seg["text"]),
        prog,
        "",
        gr.update(visible=False),
    )


def open_asset(ctx: AppContext, asset_id: int | None, order: str):
    """Load the segment queue for a file. Starts at the first unchecked segment.

    Returns ``(queue, pos, sug_state, *view)``.
    """
    if asset_id is None:
        return ([], 0, None) + render_segment(ctx, None, [], 0)
    segs = ctx.store.segments(int(asset_id), order="confidence" if order == ORDER_CONF else "time")
    queue = [s["id"] for s in segs]
    pos = next((i for i, s in enumerate(segs) if s["status"] == "draft"), len(segs))
    if pos == len(segs) and not segs:
        pos = 0
    return (queue, pos, None) + render_segment(ctx, int(asset_id), queue, pos)


def _next_open(ctx: AppContext, queue: list[int], pos: int) -> int:
    for i in range(pos + 1, len(queue)):
        if ctx.store.segment(queue[i])["status"] == "draft":
            return i
    return len(queue)


def save_and_next(ctx: AppContext, asset_id: int | None, queue: list[int], pos: int, text: str,
                  author: str, sug_state: dict | None):
    """Save the text as checked by a person, then move to the next unchecked segment."""
    if asset_id is None or not queue or pos >= len(queue):
        return (pos, None) + render_segment(ctx, asset_id, queue, pos)
    seg_id = queue[pos]
    source = "human"
    if sug_state and sug_state.get("accepted") and \
            (text or "").strip() == (sug_state.get("suggested") or "").strip():
        source = "suggestion"
    ctx.store.verify(seg_id, author=(author or "").strip() or None, text=text or "",
                     source=source)
    nxt = _next_open(ctx, queue, pos)
    return (nxt, None) + render_segment(ctx, asset_id, queue, nxt)


def skip_and_next(ctx: AppContext, asset_id: int | None, queue: list[int], pos: int):
    """Mark as skipped (music, noise, not Hausa). Skipped segments never train a model."""
    if asset_id is None or not queue or pos >= len(queue):
        return (pos, None) + render_segment(ctx, asset_id, queue, pos)
    ctx.store.skip(queue[pos])
    nxt = _next_open(ctx, queue, pos)
    return (nxt, None) + render_segment(ctx, asset_id, queue, nxt)


def go_previous(ctx: AppContext, asset_id: int | None, queue: list[int], pos: int):
    pos = max(0, min(pos, len(queue)) - 1)
    return (pos, None) + render_segment(ctx, asset_id, queue, pos)


def get_suggestion(ctx: AppContext, queue: list[int], pos: int, text: str):
    """Ask N-ATLaS for a fix. Returns ``(html, controls_update, sug_state)``. Never applies it."""
    hide = gr.update(visible=False)
    if not queue or pos >= len(queue):
        return "", hide, None
    if not ctx.suggestions_on:
        return (_card("info", "Suggestions are switched off on this server.",
                      "An administrator can turn on N-ATLaS by setting "
                      "<code>GYARA_LLM_BACKEND</code>."), hide, None)
    try:
        sugg = ctx.get_suggester().suggest(text or "")
    except Exception as e:
        return (_card("error", "N-ATLaS could not answer.",
                      f"{_esc(type(e).__name__)}: {_esc(str(e))[:300]}"), hide, None)
    sid = ctx.store.add_suggestion(
        queue[pos], sugg.original, sugg.suggested, model=getattr(sugg, "model", None) or None,
        edits=list(sugg.edits or []), guard_ok=sugg.accepted_by_guard)
    if not sugg.accepted_by_guard:
        return (_card("info", "No safe suggestion this time.",
                      f"N-ATLaS proposed something our check did not trust: "
                      f"{_esc(sugg.reason)}. Your text is unchanged."), hide, None)
    if sugg.suggested.strip() == (text or "").strip():
        return _card("ok", "N-ATLaS has no changes to suggest."), hide, None
    body = word_diff_html(text, sugg.suggested)
    return (_card("suggest", "N-ATLaS suggests", body + '<div class="gy-hint">Nothing changes '
                  "until you accept. Listen again before you decide.</div>"),
            gr.update(visible=True), {"id": sid, "suggested": sugg.suggested, "accepted": False})


def accept_suggestion(ctx: AppContext, sug_state: dict | None, text: str):
    """Put the suggestion in the edit box. It is saved only when the person saves."""
    if not sug_state:
        return gr.update(), "", gr.update(visible=False), None
    ctx.store.decide_suggestion(sug_state["id"], accept=True)
    state = dict(sug_state, accepted=True)
    return (gr.update(value=sug_state["suggested"]),
            _card("ok", "Suggestion placed in your text.", "Check it, then save."),
            gr.update(visible=False), state)


def reject_suggestion(ctx: AppContext, sug_state: dict | None):
    if sug_state:
        ctx.store.decide_suggestion(sug_state["id"], accept=False)
    return _card("info", "Suggestion dismissed."), gr.update(visible=False), None


# -- Export ----------------------------------------------------------------------


def make_subtitles(ctx: AppContext, asset_id: int | None, fmt: str, source: str, english: bool):
    """Returns ``(file_update, message_html)``."""
    if asset_id is None:
        return gr.update(value=None), _card("warn", "Choose a file first.")
    fmt = "vtt" if "vtt" in (fmt or "").lower() else "srt"
    use = "asr" if (source or "").startswith("AI") else "current"
    translate: Callable[[str], str] | None = None
    if english:
        if not ctx.suggestions_on:
            return gr.update(value=None), _card(
                "info", "English subtitles need N-ATLaS.",
                "Ask an administrator to set <code>GYARA_LLM_BACKEND</code>.")
        translate = ctx.get_suggester().translate
    try:
        text = export_subtitles(ctx.store, int(asset_id), fmt, use=use, translate=translate)
    except Exception as e:
        return gr.update(value=None), _card("error", "Could not make subtitles.", _esc(str(e)))
    asset = ctx.store.asset(int(asset_id))
    stem = Path(asset["name"]).stem
    path = ctx.out_dir() / f"{stem}.{'en' if english else 'ha'}.{fmt}"
    path.write_text(text, encoding="utf-8")
    cues = text.count(" --> ")
    return gr.update(value=str(path)), _card("ok", f"{cues} subtitles ready.",
                                             "Download them below.")


def build_dataset(ctx: AppContext, scope: str, asset_id: int | None):
    """Export verified, consented segments; returns ``(file_update, summary_html)``."""
    ids = None
    if scope == "This file only":
        if asset_id is None:
            return gr.update(value=None), _card("warn", "Choose a file first.")
        ids = [int(asset_id)]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = ctx.store.root / "exports" / f"gyara-dataset-{stamp}"
    try:
        res = export_dataset(ctx.store, out, asset_ids=ids, heldout_manifest=ctx.heldout_manifest)
    except Exception as e:
        return gr.update(value=None), _card("error", "Could not build the dataset.", _esc(str(e)))
    if res.rows == 0:
        why = ("Only segments a person has checked, from recordings with recorded consent, "
               "go into training data.")
        return gr.update(value=None), _card("warn", "Nothing to export yet.", why + _excluded(res))
    z = zip_export(res)
    body = (f"<b>{res.rows}</b> clips · <b>{res.hours * 60:.1f}</b> minutes · "
            f"<b>{res.speakers}</b> speaker{'s' if res.speakers != 1 else ''}."
            + (f"<br>Held-out check: {_esc(res.leakage)}" if res.leakage else "")
            + _excluded(res))
    return gr.update(value=str(z)), _card("ok", "Training dataset ready.", body)


def _excluded(res) -> str:
    if not res.excluded:
        return ""
    reasons: dict[str, int] = {}
    for e in res.excluded:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    items = "".join(f"<li>{n} × {_esc(r)}</li>" for r, n in sorted(reasons.items()))
    return f"<br>Left out:<ul>{items}</ul>"


def workspace_summary(ctx: AppContext) -> str:
    g = ctx.store.global_stats()
    return (f'<div class="gy-stats">'
            f'<div><span>{g["assets"]}</span>files</div>'
            f'<div><span>{g["verified"]}</span>segments checked</div>'
            f'<div><span>{g["hours_verified"] * 60:.1f}</span>minutes checked</div>'
            f'<div><span>{g["speakers"]}</span>speakers</div>'
            f'<div><span>{_pct(g["wer"])}</span>AI word error rate ({g["ref_words"]} words)</div>'
            f"</div>")


# -- Benchmark -------------------------------------------------------------------


def _run_dirs(ctx: AppContext) -> list[Path]:
    return sorted(p.parent for p in Path(ctx.runs_dir).glob("*/metrics.json"))


def benchmark_overview(ctx: AppContext):
    """Returns ``(leaderboard_markdown, report_picker_update)``."""
    reports = sorted(p.parent.name for p in Path(ctx.runs_dir).glob("*/report.md"))
    picker = gr.update(choices=reports, value=reports[0] if reports else None)
    dirs = _run_dirs(ctx)
    if not dirs:
        return ("No benchmark runs yet. Run `gyara eval` to create one; results appear here.",
                picker)
    try:
        from gyara.evaluate import leaderboard

        table = leaderboard(dirs)
    except Exception as e:
        table = f"Could not read the runs: `{type(e).__name__}: {e}`"
    note = ("WER is pooled over all words, under the *standard* normalisation rules, with a "
            "95% bootstrap confidence interval. Lower is better. Compare models only on the "
            "same test set.")
    return f"{table}\n\n{note}", picker


def show_report(ctx: AppContext, name: str | None) -> str:
    if not name:
        return ""
    p = Path(ctx.runs_dir) / name / "report.md"
    return p.read_text(encoding="utf-8") if p.exists() else f"No report for `{name}`."


# -- Look and feel ---------------------------------------------------------------

GREEN = gr.themes.Color(
    c50="#eef7f1", c100="#d5ede0", c200="#acdabf", c300="#7cc29a", c400="#4fa676",
    c500="#2e8a5a", c600="#1f7048", c700="#175a3a", c800="#12472f", c900="#0e3a27",
    c950="#072117", name="gyara_green",
)


def make_theme() -> gr.themes.Base:
    font = [gr.themes.GoogleFont("Noto Sans"), "Noto Sans", "Segoe UI", "system-ui", "sans-serif"]
    mono = [gr.themes.GoogleFont("Noto Sans Mono"), "Consolas", "monospace"]
    return gr.themes.Base(
        primary_hue=GREEN, secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.stone, radius_size=gr.themes.sizes.radius_md,
        text_size=gr.themes.sizes.text_lg, font=font, font_mono=mono,
    ).set(
        body_background_fill="#faf7f0",
        body_background_fill_dark="#14120f",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#1d1a16",
        block_background_fill="#ffffff",
        block_background_fill_dark="#1d1a16",
        block_border_color="#e7e0d2",
        block_border_color_dark="#3a342b",
        block_shadow="0 1px 2px rgba(60, 45, 20, 0.06)",
        block_label_text_weight="600",
        button_primary_background_fill="*primary_700",
        button_primary_background_fill_hover="*primary_800",
        button_primary_background_fill_dark="*primary_500",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#f1ebdf",
        button_secondary_background_fill_hover="#e7dfcf",
        button_secondary_background_fill_dark="#2a251e",
        input_background_fill="#fffdf8",
        input_background_fill_dark="#221e19",
        checkbox_label_background_fill_selected="*primary_50",
        slider_color="*primary_600",
    )


CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
footer { display: none !important; }
.gy-header { display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap;
  padding: 18px 4px 6px; border-bottom: 2px solid #175a3a; margin-bottom: 4px; }
.gy-brand { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; color: #175a3a; }
.gy-tag { font-size: 1.05rem; color: #5b5346; }
.gy-steps { margin-left: auto; font-size: .9rem; color: #7a705f; }
.gy-steps b { color: #175a3a; }
.dark .gy-brand, .dark .gy-steps b { color: #7cc29a; }
.dark .gy-tag, .dark .gy-steps { color: #b8ad99; }
#edit-box textarea { font-size: 1.6rem !important; line-height: 1.55 !important;
  font-family: 'Noto Sans', 'Segoe UI', sans-serif !important; }
#draft-box textarea { font-size: 1.25rem !important; line-height: 1.5 !important;
  color: #5b5346 !important; background: #f6f1e6 !important; }
.dark #draft-box textarea { color: #cfc5b2 !important; background: #26211b !important; }
.gy-where { display: flex; gap: 14px; align-items: center; font-size: 1rem; }
.gy-pill { background: #175a3a; color: #fff; padding: 3px 12px; border-radius: 999px; }
.gy-time { color: #7a705f; font-variant-numeric: tabular-nums; }
.gy-status { color: #1f7048; font-weight: 600; }
.gy-banner { background: #fff4d6; border: 1px solid #e8b931; border-left: 6px solid #d99a00;
  color: #5a4300; padding: 12px 16px; border-radius: 8px; font-size: 1.05rem; }
.gy-score { border: 1px solid #e7e0d2; border-radius: 10px; padding: 10px 14px; }
.gy-score-title { font-size: .85rem; text-transform: uppercase; letter-spacing: .06em;
  color: #7a705f; margin-bottom: 6px; }
.gy-score-row { display: flex; gap: 22px; flex-wrap: wrap; }
.gy-score-label { font-size: .85rem; color: #7a705f; }
.gy-score-num { font-size: 1.9rem; font-weight: 700; color: #175a3a;
  font-variant-numeric: tabular-nums; }
.dark .gy-score-num { color: #7cc29a; }
.gy-score-sub { font-size: .85rem; color: #5b5346; }
.gy-muted { color: #7a705f; }
.gy-progress-head { font-size: 1.05rem; }
.gy-progress-sub { font-size: .9rem; color: #5b5346; margin-top: 6px; }
.gy-bar { height: 10px; background: #ece5d6; border-radius: 999px; overflow: hidden;
  margin-top: 6px; }
.gy-bar span { display: block; height: 100%; background: #2e8a5a; }
.gy-card { border-radius: 10px; padding: 12px 16px; border: 1px solid #e7e0d2; }
.gy-card-title { font-weight: 700; font-size: 1.05rem; }
.gy-card-body { margin-top: 4px; }
.gy-ok { background: #eef7f1; border-color: #acdabf; }
.gy-warn { background: #fff4d6; border-color: #e8b931; }
.gy-error { background: #fdecea; border-color: #f0a39b; color: #6b1c13; }
.gy-info { background: #f3f0e8; }
.gy-suggest { background: #f4f2fb; border-color: #c9c1ea; }
.dark .gy-card { background: #221e19; border-color: #3a342b; color: inherit; }
.gy-diff { font-size: 1.35rem; line-height: 1.7; margin: 6px 0; }
.gy-diff del { background: #fbd5d0; color: #8a1f12; text-decoration: line-through; padding: 0 3px;
  border-radius: 4px; }
.gy-diff ins { background: #cdeed8; color: #0e3a27; text-decoration: none; padding: 0 3px;
  border-radius: 4px; font-weight: 600; }
.gy-hint { font-size: .85rem; color: #7a705f; }
.gy-empty { font-size: 1.2rem; padding: 28px 8px; text-align: center; color: #5b5346; }
.gy-hooks button { font-size: 1.25rem !important; min-width: 48px !important; }
.gy-kbd { font-size: .85rem; color: #7a705f; }
.gy-kbd kbd { border: 1px solid #cfc5b2; border-bottom-width: 2px; border-radius: 4px;
  padding: 0 5px; font-family: inherit; background: #fffdf8; }
.gy-stats { display: flex; gap: 12px; flex-wrap: wrap; }
.gy-stats div { flex: 1 1 140px; border: 1px solid #e7e0d2; border-radius: 10px;
  padding: 10px 14px; color: #5b5346; font-size: .9rem; }
.gy-stats span { display: block; font-size: 1.6rem; font-weight: 700; color: #175a3a; }
.gy-footer { margin-top: 18px; padding-top: 12px; border-top: 1px solid #e7e0d2;
  font-size: .85rem; color: #7a705f; text-align: center; }
"""

# Ctrl+Enter (Cmd+Enter on a Mac) presses "Save & verify, next" when it is visible.
HEAD = """
<script>
document.addEventListener('keydown', function (e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    var b = document.getElementById('save-btn');
    if (b && b.offsetParent !== null) { e.preventDefault(); b.click(); }
  }
});
</script>
"""


def _insert_js(ch: str) -> str:
    """Browser-side: type a hooked letter at the cursor in the edit box."""
    lit = ch.replace("\\", "\\\\").replace("'", "\\'")
    return (
        "() => { const t = document.querySelector('#edit-box textarea'); if (!t) return;"
        " const s = t.selectionStart ?? t.value.length, e = t.selectionEnd ?? s;"
        f" t.setRangeText('{lit}', s, e, 'end');"
        " t.dispatchEvent(new Event('input', {bubbles: true})); t.focus(); }"
    )


FOOTER = (f'<div class="gy-footer">{html.escape(ATTRIBUTION)}<br>{LICENCE_NOTE}'
          "<br>Gyara keeps every human correction. Only segments a person checked, from "
          "speakers who gave written consent, become training data.</div>")


# -- Layout ----------------------------------------------------------------------


def build_app(
    store_path: str | Path = "workspace/gyara.db",
    asr_model: str = DEFAULT_ASR_MODEL,
    suggester_backend: str = "none",
    device: str | None = None,
    *,
    transcriber: Any = None,
    suggester: Any = None,
    runs_dir: str | Path = "runs",
    heldout_manifest: str | Path | None = None,
    max_minutes: float | None = None,
) -> gr.Blocks:
    """Build the Gradio app. ``transcriber``/``suggester`` can be injected (tests, demos)."""
    ctx = AppContext(
        store=Store(store_path), asr_model=asr_model, suggester_backend=suggester_backend,
        device=device, transcriber=transcriber, suggester=suggester, runs_dir=Path(runs_dir),
        heldout_manifest=str(heldout_manifest) if heldout_manifest else None,
        max_minutes=max_minutes or None,
    )
    P = functools.partial
    notice = "" if transcriber is not None else model_notice(asr_model)

    # Gradio 5.x warns that theme/css/head move to launch() in 6.0. Its launch()
    # does not accept them yet, and we pin gradio<6, so the warning is noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The '(theme|css|head)' parameter in the "
                                r"Blocks constructor", category=DeprecationWarning)
        blocks = gr.Blocks(theme=make_theme(), css=CSS, head=HEAD,
                           title="Gyara · Hausa transcripts", analytics_enabled=False)

    with blocks as demo:
        gr.HTML('<div class="gy-header"><div class="gy-brand">Gyara</div>'
                '<div class="gy-tag">Fix Hausa transcripts. Every fix you make teaches '
                'the model.</div><div class="gy-steps"><b>1</b> Upload · <b>2</b> Correct · '
                '<b>3</b> Export</div></div>')
        gr.HTML(notice, visible=bool(notice))
        with gr.Row(equal_height=True):
            picker = gr.Dropdown(label="Working on", choices=asset_choices(ctx), value=None,
                                 scale=3, interactive=True,
                                 info="Pick a file you uploaded before, or upload a new one.")
            order = gr.Radio([ORDER_TIME, ORDER_CONF], value=ORDER_TIME, scale=2,
                             label="Check segments", info="Least confident first finds "
                             "the AI's likely mistakes sooner.")
            author = gr.Textbox(label="Your name", placeholder="e.g. Amina", scale=1,
                                info="Saved with each check.")

        queue = gr.State([])
        pos = gr.State(0)
        sug_state = gr.State(None)

        with gr.Tabs(selected="transcribe") as tabs:
            # -- Transcribe ---------------------------------------------------
            with gr.Tab("1 · Transcribe", id="transcribe"):
                with gr.Row():
                    with gr.Column(scale=3):
                        upload = gr.File(label="Hausa recording (audio or video)",
                                         file_types=["audio", "video"], type="filepath")
                        with gr.Row():
                            speaker = gr.Textbox(label="Speaker",
                                                 placeholder="A name or code, e.g. spk-014",
                                                 info="Keeps test speakers out of training.")
                            dialect = gr.Dropdown(DIALECTS, label="Dialect", value=None,
                                                  allow_custom_value=True)
                        consent = gr.Checkbox(
                            label="I have the speaker's written consent",
                            info="Needed to use this recording for training. Without it you "
                                 "can still make subtitles.")
                        consent_ref = gr.Textbox(label="Consent form reference (optional)",
                                                 placeholder="e.g. form 2026-031")
                        go = gr.Button("Transcribe", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        status = gr.HTML(_card(
                            "info", "How it works",
                            "1. Upload a recording.<br>2. The AI writes a first draft.<br>"
                            "3. You listen and fix it, one short piece at a time.<br>"
                            "4. Download subtitles, or a dataset to train a better model."))

            # -- Correct ------------------------------------------------------
            with gr.Tab("2 · Correct", id="correct"):
                empty = gr.Markdown(EMPTY_NO_FILE, elem_classes="gy-empty")
                with gr.Column(visible=False) as editor:
                    with gr.Row(equal_height=True):
                        where = gr.HTML()
                        gr.HTML('<div class="gy-kbd" style="text-align:right">'
                                "<kbd>Ctrl</kbd> + <kbd>Enter</kbd> saves and moves on</div>")
                    banner = gr.HTML(visible=False)
                    clip = gr.Audio(label="Listen", type="filepath", interactive=False)
                    draft = gr.Textbox(label="AI draft (for reference)", interactive=False,
                                       lines=2, elem_id="draft-box")
                    text = gr.Textbox(label="Your text: write exactly what you hear",
                                      lines=3, elem_id="edit-box", interactive=True)
                    with gr.Row(elem_classes="gy-hooks"):
                        hook_btns = [gr.Button(h, size="sm", min_width=48) for h in HOOKS]
                    with gr.Row():
                        save = gr.Button("Save & verify, next", variant="primary", size="lg",
                                         elem_id="save-btn", scale=3)
                        skip = gr.Button("Skip (music, noise, not Hausa)", scale=2)
                        prev = gr.Button("Back", scale=1)
                    with gr.Row():
                        with gr.Column(scale=3):
                            score = gr.HTML()
                        with gr.Column(scale=2):
                            suggest_btn = gr.Button("Get suggestion from N-ATLaS",
                                                    visible=ctx.suggestions_on)
                            gr.Markdown(SUGGEST_OFF, visible=not ctx.suggestions_on,
                                        elem_classes="gy-hint")
                            sug_html = gr.HTML()
                            with gr.Row(visible=False) as sug_controls:
                                accept = gr.Button("Accept suggestion", variant="primary")
                                reject = gr.Button("Reject")
                file_prog = gr.HTML()

            # -- Export -------------------------------------------------------
            with gr.Tab("3 · Export", id="export"):
                ws_stats = gr.HTML(workspace_summary(ctx))
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Subtitles\nFor the file you are working on.")
                        sub_fmt = gr.Radio(["SRT (.srt)", "WebVTT (.vtt)"], value="SRT (.srt)",
                                           label="Format")
                        sub_src = gr.Radio(["Corrected text", "AI draft only"],
                                           value="Corrected text", label="Text")
                        sub_en = gr.Checkbox(label="Translate to English with N-ATLaS")
                        sub_btn = gr.Button("Make subtitles", variant="primary")
                        sub_msg = gr.HTML()
                        sub_file = gr.File(label="Download", interactive=False)
                    with gr.Column():
                        gr.Markdown("### Training dataset\nOnly checked segments from speakers "
                                    "who gave consent. Hugging Face audiofolder format.")
                        ds_scope = gr.Radio(["This file only", "All files"], value="All files",
                                            label="Include")
                        ds_btn = gr.Button("Build dataset (.zip)", variant="primary")
                        ds_msg = gr.HTML()
                        ds_file = gr.File(label="Download", interactive=False)

            # -- Benchmark ----------------------------------------------------
            with gr.Tab("Benchmark", id="benchmark"):
                gr.Markdown("### How the speech models score on held-out Hausa audio")
                board = gr.Markdown()
                report_pick = gr.Dropdown(label="Full report", choices=[])
                report = gr.Markdown()

        gr.HTML(FOOTER)

        # -- wiring -------------------------------------------------------------
        view = [empty, editor, where, banner, clip, draft, text, score, file_prog, sug_html,
                sug_controls]
        nav = [pos, sug_state] + view

        go.click(P(transcribe, ctx),
                 [upload, speaker, dialect, consent, consent_ref, picker],
                 [status, picker, tabs])
        picker.change(P(open_asset, ctx), [picker, order], [queue, pos, sug_state] + view)
        order.change(P(open_asset, ctx), [picker, order], [queue, pos, sug_state] + view)
        save.click(P(save_and_next, ctx), [picker, queue, pos, text, author, sug_state], nav)
        skip.click(P(skip_and_next, ctx), [picker, queue, pos], nav)
        prev.click(P(go_previous, ctx), [picker, queue, pos], nav)
        text.change(live_score, [draft, text], [score], show_progress="hidden")
        suggest_btn.click(P(get_suggestion, ctx), [queue, pos, text],
                          [sug_html, sug_controls, sug_state])
        accept.click(P(accept_suggestion, ctx), [sug_state, text],
                     [text, sug_html, sug_controls, sug_state])
        reject.click(P(reject_suggestion, ctx), [sug_state], [sug_html, sug_controls, sug_state])
        for b, h in zip(hook_btns, HOOKS):
            b.click(None, None, None, js=_insert_js(h))

        sub_btn.click(P(make_subtitles, ctx), [picker, sub_fmt, sub_src, sub_en],
                      [sub_file, sub_msg])
        ds_btn.click(P(build_dataset, ctx), [ds_scope, picker], [ds_file, ds_msg])
        report_pick.change(P(show_report, ctx), [report_pick], [report])
        demo.load(P(benchmark_overview, ctx), None, [board, report_pick])
        demo.load(P(workspace_summary, ctx), None, [ws_stats])
        demo.load(P(_refresh_picker, ctx), None, [picker])

    demo.gyara_ctx = ctx  # for tests and launchers
    demo.gyara_allowed_paths = [str(ctx.store.root.resolve()), str(ctx.clip_dir)]
    return demo


def _refresh_picker(ctx: AppContext):
    return gr.update(choices=asset_choices(ctx))


def launch(demo: gr.Blocks, **kwargs):
    """Launch with the workspace folder allowed for downloads (dataset zips)."""
    allowed = list(kwargs.pop("allowed_paths", []) or []) + getattr(demo, "gyara_allowed_paths", [])
    return demo.queue(default_concurrency_limit=2).launch(allowed_paths=allowed, **kwargs)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="gyara-ui", description="Gyara correction app")
    p.add_argument("--db", default=os.environ.get("GYARA_DB", "workspace/gyara.db"))
    p.add_argument("--model", default=os.environ.get("GYARA_ASR_MODEL", DEFAULT_ASR_MODEL))
    p.add_argument("--llm", default=os.environ.get("GYARA_LLM_BACKEND", "none"),
                   choices=["none", "transformers", "openai"])
    p.add_argument("--device", default=os.environ.get("GYARA_DEVICE") or None)
    p.add_argument("--heldout", default=os.environ.get("GYARA_HELDOUT") or None)
    p.add_argument("--max-minutes", type=float,
                   default=float(os.environ.get("GYARA_MAX_MINUTES") or 0) or None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--share", action="store_true")
    a = p.parse_args(argv)
    demo = build_app(a.db, a.model, a.llm, a.device, heldout_manifest=a.heldout,
                     max_minutes=a.max_minutes)
    launch(demo, server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
