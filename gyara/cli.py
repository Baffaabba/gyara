"""Command-line interface. Heavy imports stay inside commands so `--help` is instant."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import ATTRIBUTION, DEFAULT_ASR_MODEL, DEFAULT_LLM_MODEL, __version__, load_env

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Gyara: measure, correct and improve Hausa speech-to-text on N-ATLAS.",
)
console = Console()
MODES_HELP = "raw, standard or lenient."


def _utf8_output() -> None:
    """Print Hausa letters (ɓ ɗ ƙ ƴ) even when output is piped or redirected.

    On Windows a redirected stdout defaults to cp1252, which cannot encode them.
    """
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _fail(*lines: str, code: int = 2):
    """Print a short, plain error (no traceback) and exit."""
    for line in lines:
        console.print(line, highlight=False, markup=False, soft_wrap=True)
    raise typer.Exit(code)


def _check_mode(mode: str) -> str:
    from .normalize import MODES

    if mode not in MODES:
        _fail(f"Unknown --mode '{mode}'. Use {', '.join(MODES)}.")
    return mode


def _gated(model: str):
    _fail(f"{model} is a gated model on Hugging Face, and you do not have access to it yet.",
          f"1. Open https://huggingface.co/{model} and accept the licence.",
          "2. Run `hf auth login`, or set the HF_TOKEN environment variable, then try again.")


def _check_model_access(model: str) -> None:
    """Fail in seconds, with three plain lines, if a Hub model is gated or missing."""
    if Path(model).exists() or os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true"):
        return
    try:
        from huggingface_hub import auth_check
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return
    try:
        auth_check(model)
    except GatedRepoError:
        _gated(model)
    except RepositoryNotFoundError:
        _fail(f"Hugging Face has no model called {model}, or it is private.",
              "Check the name. If it is private, run `hf auth login` or set HF_TOKEN.")
    except Exception:
        return  # offline or the Hub is down: let the real load report it


def _is_access_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return "gated repo" in msg or "401 client error" in msg or "403 client error" in msg


def _version(value: bool):
    if value:
        console.print(f"gyara {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True,
                                 help="Show the version and exit."),
):
    _utf8_output()
    load_env()


# --- Transcribe -------------------------------------------------------------


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, metavar="AUDIO",
                                 help="Audio or video file."),
    model: str = typer.Option(DEFAULT_ASR_MODEL, help="Speech model id or local path."),
    out: Optional[Path] = typer.Option(None, help="Output file. Format from the extension: .srt .vtt .json .txt"),
    device: Optional[str] = typer.Option(None, help="cuda, cpu or mps. Default: best available."),
    language: Optional[str] = typer.Option("hausa", help="Whisper language. Use 'auto' to let the model decide."),
):
    """Transcribe a Hausa recording into timed segments, subtitles or text."""
    try:
        from .asr import SAMPLE_RATE, Transcriber
        from .audio import load as load_audio
    except ImportError as e:
        _fail(f"Transcription needs the speech extras ({e.name} is missing).",
              'Install them with: pip install -e ".[asr]"')
    from .subtitles import build_cues, to_srt, to_vtt

    try:
        wav = load_audio(str(audio), SAMPLE_RATE)
    except Exception:
        _fail(f"Could not read {audio.name} as audio or video. Try an mp3, wav, m4a or mp4 file.")
    _check_model_access(model)
    lang = None if language in (None, "auto") else language
    try:
        tr = Transcriber(model, device=device, language=lang)
        with console.status(f"Transcribing {audio.name} with {model}..."):
            segments = tr.transcribe(wav, SAMPLE_RATE)
    except OSError as e:
        if _is_access_error(e):
            _gated(model)
        raise
    if not segments:
        _fail(f"No speech found in {audio.name}. Nothing was written.", code=1)
    segs = [
        {"start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob,
         "flags": s.flags, "words": s.words}
        for s in segments
    ]
    flagged = sum(1 for s in segs if s["flags"])
    out = out or audio.with_suffix(".srt")
    ext = out.suffix.lower()
    if ext == ".srt":
        content = to_srt(build_cues(segs))
    elif ext == ".vtt":
        content = to_vtt(build_cues(segs))
    elif ext == ".json":
        content = json.dumps({"audio": str(audio), "model": model, "segments": segs},
                             ensure_ascii=False, indent=2)
    else:
        content = "\n".join(s["text"] for s in segs) + "\n"
    out.write_text(content, encoding="utf-8")
    console.print(f"[green]Wrote {out}[/green]: {len(segs)} segments"
                  + (f", [yellow]{flagged} flagged for review[/yellow]" if flagged else ""))


# --- Evaluate ---------------------------------------------------------------


@app.command("eval")
def eval_(
    manifest: Path = typer.Option(..., exists=True, help="JSONL manifest with audio + reference text."),
    model: str = typer.Option(DEFAULT_ASR_MODEL, help="Speech model id or local path."),
    name: Optional[str] = typer.Option(None, help="Run name. Default: model + manifest."),
    out_dir: Path = typer.Option(Path("runs"), help="Where runs are written."),
    device: Optional[str] = typer.Option(None),
    batch_size: int = typer.Option(8),
    limit: Optional[int] = typer.Option(None, help="Only the first N utterances (smoke tests)."),
    language: Optional[str] = typer.Option("hausa", help="Whisper language, or 'auto'."),
    suggest: str = typer.Option("none", help="Also measure N-ATLaS suggestions: none, transformers or openai."),
    llm_url: Optional[str] = typer.Option(None, help="Base URL for --suggest openai (llama.cpp, vLLM, Ollama)."),
):
    """Benchmark a speech model on a manifest: WER/CER three ways, with 95% CIs."""
    from . import evaluate

    suggester = None
    if suggest != "none":
        from .suggest import Suggester

        suggester = Suggester(backend=suggest, base_url=llm_url)
    _check_model_access(model)
    lang = None if language in (None, "auto") else language
    run_dir = evaluate.run(model, str(manifest), name=name, out_dir=str(out_dir), device=device,
                           batch_size=batch_size, limit=limit, suggester=suggester, language=lang)
    _print_metrics(Path(run_dir))


@app.command()
def rescore(run_dir: Path = typer.Argument(..., exists=True, file_okay=False)):
    """Re-score a run from its saved predictions (after a rules change, no model needed)."""
    from . import evaluate

    evaluate.score_run(str(run_dir))
    _print_metrics(run_dir)


@app.command()
def score(
    ref: Path = typer.Option(..., exists=True, help="References: .txt (one per line) or .jsonl with id,text."),
    hyp: Path = typer.Option(..., exists=True, help="Hypotheses, same format and order/ids as --ref."),
    groups: Optional[Path] = typer.Option(None, exists=True, help="Optional .txt of cluster ids (speaker), one per line."),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Also write the full result as JSON."),
):
    """Score transcripts you already have. No model, no audio: bring your own outputs."""
    from .metrics import score_corpus
    from .normalize import MODES, RULES_VERSION, get_normalizer
    from .stats import bootstrap_rate

    ids, refs = _read_texts(ref)
    hids, hyps = _read_texts(hyp)
    if ids != hids:
        if ref.suffix == ".jsonl" and hyp.suffix == ".jsonl":
            by_id = dict(zip(hids, hyps))
            missing = [i for i in ids if i not in by_id]
            if missing:
                console.print(f"[yellow]{len(missing)} reference ids have no hypothesis; "
                              f"scored as empty output.[/yellow]")
            hyps = [by_id.get(i, "") for i in ids]
        elif len(refs) != len(hyps):
            _fail(f"--ref has {len(refs)} lines but --hyp has {len(hyps)}. "
                  "Give one line per utterance, in the same order.")
    grp = None
    if groups:
        _, grp = _read_texts(groups)
        if len(grp) != len(refs):
            _fail(f"--groups has {len(grp)} lines but --ref has {len(refs)}. "
                  "Give one group (e.g. speaker) per utterance.")
    if not refs or not any(r.strip() for r in refs):
        _fail("There are no reference words to score. Check that --ref is not empty.")
    empty = sum(1 for r in refs if not r.strip())
    if empty:
        console.print(f"[yellow]{empty} reference line{'s are' if empty != 1 else ' is'} empty. "
                      "Any words the model wrote there count as insertions.[/yellow]")
    table = Table(title=f"Gyara score  (n={len(refs)}, rules v{RULES_VERSION})")
    for col in ("mode", "WER % [95% CI]", "CER % [95% CI]", "S / D / I"):
        table.add_column(col)
    result = {"rules_version": RULES_VERSION, "n": len(refs), "modes": {}}
    for mode in MODES:
        sc = score_corpus(ids, refs, hyps, get_normalizer(mode), grp)
        w = bootstrap_rate([u.words.errors for u in sc.utterances],
                           [u.words.ref_len for u in sc.utterances], grp)
        c = bootstrap_rate([u.chars.errors for u in sc.utterances],
                           [u.chars.ref_len for u in sc.utterances], grp)
        t = sc.words
        table.add_row(mode, w.fmt(), c.fmt(), f"{t.substitutions} / {t.deletions} / {t.insertions}")
        # summary() also has plain "wer"/"cer" floats; they must not replace the CIs.
        result["modes"][mode] = {**sc.summary(), "wer": w.to_dict(), "cer": c.to_dict(),
                                 "top_confusions": sc.top_confusions(15)}
    console.print(table)
    if json_out:
        json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"Wrote {json_out}")


def _read_texts(path: Path) -> tuple[list[str], list[str]]:
    """UTF-8 text, one utterance per line (or .jsonl with id, text).

    A byte-order mark (Notepad, Excel) is ignored so it is not scored as a
    word. Trailing blank lines are dropped so one stray newline does not make
    two files look different lengths.
    """
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        _fail(f"{path.name} is not UTF-8 text. Open it in a text editor and save it as UTF-8.")
    while lines and not lines[-1].strip():
        lines.pop()
    if path.suffix == ".jsonl":
        try:
            rows = [json.loads(x) for x in lines if x.strip()]
            return [str(r["id"]) for r in rows], [r.get("text", "") for r in rows]
        except (ValueError, KeyError, TypeError):
            _fail(f"{path.name}: every line must be JSON with an \"id\" and a \"text\".")
    return [str(i) for i in range(len(lines))], lines


@app.command()
def compare(
    run_a: Path = typer.Argument(..., exists=True, file_okay=False, help="Usually the new model."),
    run_b: Path = typer.Argument(..., exists=True, file_okay=False, help="Usually the baseline."),
    mode: str = typer.Option("standard", help="raw, standard or lenient."),
):
    """Is A really better than B? Paired bootstrap on the same utterances."""
    from . import evaluate

    res = evaluate.compare(str(run_a), str(run_b), mode=_check_mode(mode))
    console.print(f"[bold]WER:[/bold] {res['sentence_wer']}", highlight=False)
    console.print(f"[bold]CER:[/bold] {res['sentence_cer']}", highlight=False)
    if res.get("path"):
        console.print(f"Full table: {res['path']} (numbers in {Path(res['path']).with_suffix('.json')})")


@app.command()
def leaderboard(
    runs: list[Path] = typer.Argument(None, help="Run dirs. Default: every run under runs/."),
    out: Optional[Path] = typer.Option(None, help="Write the markdown table here."),
):
    """One table across runs: model, test set, n, WER with CI, CER, speed."""
    from . import evaluate

    dirs = [str(p) for p in runs] if runs else sorted(
        str(p.parent) for p in Path("runs").glob("*/metrics.json"))
    md = evaluate.leaderboard(dirs)
    if out:
        out.write_text(md, encoding="utf-8")
    console.print(md)


def _print_metrics(run_dir: Path):
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    console.print(f"[green]Run written to {run_dir}[/green]  (report: {run_dir / 'report.md'})")
    modes = m.get("modes") or {}
    table = Table(title=f"{m.get('meta', {}).get('model_id', '')}  n={m.get('meta', {}).get('n', '?')}")
    for col in ("mode", "WER %", "95% CI", "CER %"):
        table.add_column(col)
    for mode, v in modes.items():
        w, c = v.get("wer", {}), v.get("cer", {})
        table.add_row(mode, _pct(w.get("estimate")), f"{_pct(w.get('low'))} – {_pct(w.get('high'))}",
                      _pct(c.get("estimate")))
    console.print(table)


def _pct(x) -> str:
    return "n/a" if x is None or x != x else f"{x * 100:.2f}"


# --- Data -------------------------------------------------------------------


@app.command("fetch-fleurs")
def fetch_fleurs(
    split: str = typer.Option("test", help="test, validation or train."),
    out_dir: Path = typer.Option(Path("data/fleurs")),
    limit: Optional[int] = typer.Option(None),
):
    """Download FLEURS Hausa (CC-BY-4.0) and write a Gyara manifest."""
    from .data import fetch_fleurs as _fetch

    path = _fetch(split=split, out_dir=str(out_dir), limit=limit)
    console.print(f"[green]Manifest: {path}[/green]")


@app.command("make-manifest")
def make_manifest(
    folder: Path = typer.Argument(..., exists=True, file_okay=False,
                                  help="Folder of audio files, each with a same-named .txt transcript."),
    out: Path = typer.Option(..., help="Manifest to write (.jsonl)."),
):
    """Build a manifest from your own recordings (x.wav + x.txt, optional speakers.csv)."""
    from .data import build_manifest_from_folder

    path = build_manifest_from_folder(str(folder), str(out))
    console.print(f"[green]Manifest: {path}[/green]")


@app.command("check-leakage")
def check_leakage(
    train: Path = typer.Argument(..., exists=True, help="Training manifest."),
    heldout: list[Path] = typer.Argument(..., help="One or more held-out manifests."),
):
    """Confirm no audio clip or speaker in training also appears in a held-out set."""
    from .manifest import leakage_check, read

    tr = read(train)
    bad = False
    for h in heldout:
        rep = leakage_check(tr, read(h))
        colour = "green" if rep.clean else "red"
        console.print(f"[{colour}]{h}: {rep.summary()}[/{colour}]")
        bad |= not rep.clean
    raise typer.Exit(1 if bad else 0)


@app.command()
def normalize(
    text: str = typer.Argument(..., help="Hausa text."),
    mode: str = typer.Option("standard", help="raw, standard or lenient."),
):
    """Show what the scoring rules do to a piece of Hausa text."""
    from .normalize import RULES_VERSION, get_normalizer

    console.print(get_normalizer(_check_mode(mode))(text), highlight=False, markup=False)
    console.print(f"[dim]rules v{RULES_VERSION}, mode {mode}[/dim]")


# --- Export, fine-tune, UI --------------------------------------------------


@app.command()
def export(
    what: str = typer.Argument(..., help="srt, vtt or dataset."),
    db: Path = typer.Option(Path("workspace/gyara.db"), help="Gyara workspace database."),
    asset: Optional[int] = typer.Option(None, help="Asset id (subtitles: required)."),
    out: Path = typer.Option(..., help="Output file (subtitles) or folder (dataset)."),
    heldout: list[Path] = typer.Option(None, help="Held-out manifests to exclude from a dataset."),
    english: bool = typer.Option(False, help="Subtitles in English via N-ATLaS."),
    llm_backend: str = typer.Option("transformers", help="For --english: transformers or openai."),
    llm_url: Optional[str] = typer.Option(None),
):
    """Export subtitles for one recording, or a training dataset of verified segments."""
    from . import export as ex
    from .store import Store

    if what not in ("srt", "vtt", "dataset"):
        _fail(f"Unknown export '{what}'. Use srt, vtt or dataset.")
    if what in ("srt", "vtt") and asset is None:
        _fail("Subtitles are for one recording: add --asset <id>.")
    if not db.is_file():  # never create an empty workspace just to export from it
        _fail(f"No Gyara workspace at {db}. Nothing to export.",
              "Upload and correct a recording with `gyara ui` first, or point --db at your workspace.")
    store = Store(str(db))
    if what in ("srt", "vtt"):
        if store.asset(asset) is None:
            _fail(f"There is no recording with id {asset} in {db}.")
        if not store.has_segments(asset):
            _fail(f"Recording {asset} has no transcript yet. Nothing to export.", code=1)
        translate = None
        if english:
            from .suggest import Suggester

            translate = Suggester(backend=llm_backend, base_url=llm_url).translate
        out.write_text(ex.export_subtitles(store, asset, fmt=what, translate=translate), encoding="utf-8")
        console.print(f"[green]Wrote {out}[/green]")
    else:
        import shutil

        held = [str(h) for h in heldout] if heldout else None
        existed = out.exists()
        result = ex.export_dataset(store, str(out), heldout_manifest=held)
        if result.rows == 0:
            if not existed:
                shutil.rmtree(out, ignore_errors=True)
            _fail("Nothing to export yet. Only segments a person has checked, from recordings "
                  "with recorded consent, go into a training dataset.", code=1)
        console.print(result)


@app.command()
def finetune(config: Path = typer.Option(..., exists=True, help="YAML config (see gyara/finetune/config.yaml).")):
    """Fine-tune NCAIR1/Hausa-ASR on exported corrections, then measure before vs after."""
    from .finetune.train import train

    result = train(str(config))
    console.print(result)


@app.command()
def ui(
    db: Path = typer.Option(Path("workspace/gyara.db")),
    model: str = typer.Option(DEFAULT_ASR_MODEL),
    llm_backend: str = typer.Option("none", help="none, transformers or openai."),
    device: Optional[str] = typer.Option(None),
    llm_url: Optional[str] = typer.Option(None, help="Server URL for the openai backend, e.g. http://localhost:8080."),
    heldout: Optional[Path] = typer.Option(None, help="Held-out manifest to check at dataset export."),
    share: bool = typer.Option(False, help="Create a public Gradio link (for testers)."),
    host: Optional[str] = typer.Option(None, help="Address to listen on. Use 0.0.0.0 to serve your network."),
    port: int = typer.Option(7860),
):
    """Open the correction app in your browser."""
    import os

    from .ui.app import build_app, launch

    if llm_url:
        os.environ["GYARA_LLM_BASE_URL"] = llm_url
    demo = build_app(str(db), model, llm_backend, device,
                     heldout_manifest=str(heldout) if heldout else None)
    launch(demo, share=share, server_name=host, server_port=port)


@app.command()
def about():
    """Licence and attribution."""
    console.print(ATTRIBUTION)
    console.print(f"Speech model: {DEFAULT_ASR_MODEL}. Language model: {DEFAULT_LLM_MODEL}.")
    console.print("Free use is limited to 1,000 active users under the N-ATLAS licence.")


if __name__ == "__main__":
    app()
