"""Benchmark a speech model on a manifest, and compare runs honestly.

A run is a directory::

    runs/<name>/
        predictions.jsonl   one line per utterance: id, ref, hyp, avg_logprob, flags, ...
        meta.json           model, manifest hash, decoding settings, timing
        metrics.json        every number in the report, machine-readable
        report.md           the human-readable report

Inference and scoring are separate on purpose. ``run`` is slow (it decodes
audio); ``score_run`` re-scores ``predictions.jsonl`` in seconds, so a
normalisation-rule change or a new breakdown never needs a GPU, and anyone can
audit a published number from the predictions file alone.

The rules in the ``eval-integrity`` skill are enforced here: pooled WER, three
normalisation modes, cluster bootstrap CIs, paired tests for every comparison,
``RULES_VERSION`` stamped on every number and checked before comparing.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import gyara
from gyara import manifest, metrics, normalize, stats
from gyara import audio as audio_mod

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
TOP_CONFUSIONS = 25
DURATION_BUCKETS = ((0.0, 5.0, "<5s"), (5.0, 15.0, "5-15s"), (15.0, 30.0, "15-30s"))


# --- Small helpers --------------------------------------------------------------


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def _clean(x: Any) -> Any:
    """JSON-safe: NaN/inf → None (strict JSON has no NaN), numpy → python."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def _pct(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{nd}f}"


def _fmt_ci(iv: stats.Interval) -> str:
    if math.isnan(iv.estimate):
        return "n/a"
    if math.isnan(iv.low):
        return f"{iv.estimate * 100:.2f} [CI n/a]"
    return iv.fmt()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _bucket(d: float | None) -> str:
    if d is None:
        return "unknown"
    for lo, hi, name in DURATION_BUCKETS:
        if lo <= d < hi:
            return name
    return ">30s"


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return out


# --- Inference --------------------------------------------------------------------


def _transcribe_rows(transcriber, rows: list[manifest.Row], batch_size: int):
    """Yield (row, text, avg_logprob, flags, duration) in manifest order.

    Clips ≤30 s go through ``transcribe_batch`` (one Whisper window each, no
    VAD, so the reference and the audio stay aligned). Longer clips go through
    the VAD-chunked ``transcribe`` and their segments are joined.
    """
    limit = 30.0 * SAMPLE_RATE + 1
    pending: list[tuple[manifest.Row, np.ndarray]] = []

    def flush():
        segs = transcriber.transcribe_batch([a for _, a in pending])
        for (r, a), s in zip(pending, segs):
            yield r, s.text, s.avg_logprob, list(getattr(s, "flags", []) or []), len(a) / SAMPLE_RATE
        pending.clear()

    for row in rows:
        arr = audio_mod.load(row.audio, SAMPLE_RATE)
        if len(arr) > limit:
            yield from flush()
            segs = transcriber.transcribe(arr, sr=SAMPLE_RATE)
            text = " ".join(s.text for s in segs if s.text).strip()
            w = [(s.avg_logprob, s.end - s.start) for s in segs if s.avg_logprob is not None]
            lp = sum(l * d for l, d in w) / sum(d for _, d in w) if w and sum(d for _, d in w) else None
            flags = sorted({f for s in segs for f in (s.flags or [])} | {"long_clip_vad"})
            yield row, text, lp, flags, len(arr) / SAMPLE_RATE
            continue
        pending.append((row, arr))
        if len(pending) >= batch_size:
            yield from flush()
    yield from flush()


def run(
    model_id: str,
    manifest_path: str | Path,
    name: str | None = None,
    out_dir: str | Path = "runs",
    device: str | None = None,
    batch_size: int = 8,
    limit: int | None = None,
    suggester=None,
    language: str | None = "hausa",
    transcriber=None,
    b: int = stats.DEFAULT_B,
) -> Path:
    """Transcribe every manifest row with ``model_id``, then score. Returns the run dir.

    ``suggester`` (duck-typed: ``.suggest(text).suggested``) adds a
    ``hyp_suggested`` column so the report can measure the effect of
    accepting every N-ATLaS suggestion. ``transcriber`` overrides model
    loading (tests, or a pre-loaded instance in the UI).
    """
    manifest_path = Path(manifest_path)
    rows = manifest.read(manifest_path)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"{manifest_path} has no rows")
    name = name or f"{_slug(model_id)}__{_slug(manifest_path.stem)}"
    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    if transcriber is None:
        from gyara.asr import Transcriber

        transcriber = Transcriber(model_id, device=device, language=language, batch_size=batch_size)

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    audio_s = 0.0
    n_suggest_err = 0
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row, hyp, lp, flags, dur in _transcribe_rows(transcriber, rows, batch_size):
            audio_s += dur
            rec = {
                "id": row.id, "ref": row.text, "hyp": hyp, "avg_logprob": lp, "flags": flags,
                "duration": round(dur, 3), "group": row.cluster, "dialect": row.dialect,
                "speaker": row.speaker, "source": row.source,
            }
            if suggester is not None:
                try:
                    rec["hyp_suggested"] = suggester.suggest(hyp).suggested
                except Exception as e:  # noqa: BLE001 - one bad call must not sink the run
                    log.warning("suggester failed on %s: %s", row.id, e)
                    rec["hyp_suggested"] = hyp  # = "suggestion rejected"; counted below
                    rec["flags"] = flags + ["suggest_error"]
                    n_suggest_err += 1
            f.write(json.dumps(_clean(rec), ensure_ascii=False) + "\n")
    wall = time.perf_counter() - t0

    info = transcriber.info() if hasattr(transcriber, "info") else {}
    meta = {
        "model_id": model_id,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.sha256_file(manifest_path),
        "test_set": manifest_path.stem,
        "n": len(rows),
        "limit": limit,
        "device": info.get("device", device),
        "language": language,
        "decoding": info,
        "started_utc": started.isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 2),
        "audio_seconds": round(audio_s, 2),
        # Includes audio loading and, if present, suggestion time.
        "rtf": round(wall / audio_s, 4) if audio_s else None,
        "gyara_version": gyara.__version__,
        "rules_version": normalize.RULES_VERSION,
        "suggester": type(suggester).__name__ if suggester is not None else None,
        "suggest_errors": n_suggest_err if suggester is not None else None,
        "versions": _versions(),
    }
    (run_dir / "meta.json").write_text(json.dumps(_clean(meta), indent=2), encoding="utf-8")
    score_run(run_dir, b=b)
    return run_dir


# --- Scoring --------------------------------------------------------------------------


def _rate_block(cs: metrics.CorpusScore, groups: Sequence, b: int) -> dict:
    w_err = [u.words.errors for u in cs.utterances]
    w_len = [u.words.ref_len for u in cs.utterances]
    c_err = [u.chars.errors for u in cs.utterances]
    c_len = [u.chars.ref_len for u in cs.utterances]
    wer = stats.bootstrap_rate(w_err, w_len, groups, b=b)
    cer = stats.bootstrap_rate(c_err, c_len, groups, b=b)
    s = cs.summary()
    return {
        "n": s["utterances"], "n_scored": s["utterances_scored"], "ref_words": s["ref_words"],
        "ref_chars": int(sum(c_len)),
        "wer": wer.to_dict(), "cer": cer.to_dict(),
        "substitutions": s["substitutions"], "deletions": s["deletions"],
        "insertions": s["insertions"],
        "_wer": wer, "_cer": cer,
    }


def _strip_private(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [_strip_private(v) for v in d]
    return d


def _breakdown(cs: metrics.CorpusScore, preds: list[dict], key, groups: Sequence, b: int) -> dict:
    idx: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(preds):
        v = key(p)
        if v is not None and v != "":
            idx[str(v)].append(i)
    out = {}
    for val, ii in sorted(idx.items()):
        sub = metrics.CorpusScore([cs.utterances[i] for i in ii])
        out[val] = _rate_block(sub, [groups[i] for i in ii], b)
    return out


def _clusters(preds: list[dict]) -> list[str | None]:
    return [p.get("group") or p.get("speaker") for p in preds]


def score_run(run_dir: str | Path, b: int = stats.DEFAULT_B) -> dict:
    """Score ``predictions.jsonl`` (no inference). Writes metrics.json + report.md."""
    run_dir = Path(run_dir)
    preds = _read_jsonl(run_dir / "predictions.jsonl")
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("rules_version") and meta["rules_version"] != normalize.RULES_VERSION:
        log.warning("predictions made under rules %s, re-scored under %s",
                    meta["rules_version"], normalize.RULES_VERSION)
    meta["scored_rules_version"] = normalize.RULES_VERSION
    meta["rules_version"] = normalize.RULES_VERSION  # scoring, not decoding, applies the rules

    ids = [p["id"] for p in preds]
    refs = [p.get("ref") or "" for p in preds]
    hyps = [p.get("hyp") or "" for p in preds]
    groups = _clusters(preds)
    has_sugg = any("hyp_suggested" in p for p in preds)

    modes: dict[str, dict] = {}
    scores: dict[str, metrics.CorpusScore] = {}
    for mode in normalize.MODES:
        norm = normalize.get_normalizer(mode)
        cs = metrics.score_corpus(ids, refs, hyps, norm, groups)
        scores[mode] = cs
        block = _rate_block(cs, groups, b)
        block["breakdown"] = {
            "dialect": _breakdown(cs, preds, lambda p: p.get("dialect"), groups, b),
            "source": _breakdown(cs, preds, lambda p: p.get("source"), groups, b),
            "speaker": _breakdown(cs, preds, lambda p: p.get("speaker"), groups, b),
            "duration": _breakdown(cs, preds, lambda p: _bucket(p.get("duration")), groups, b),
        }
        if mode == "standard":
            block["top_confusions"] = cs.top_confusions(TOP_CONFUSIONS)
        if has_sugg:
            sugg = [p.get("hyp_suggested", p.get("hyp")) or "" for p in preds]
            cs_s = metrics.score_corpus(ids, refs, sugg, norm, groups)
            lens = [u.words.ref_len for u in cs.utterances]
            clens = [u.chars.ref_len for u in cs.utterances]
            pw = stats.paired_bootstrap(
                [u.words.errors for u in cs_s.utterances], [u.words.errors for u in cs.utterances],
                lens, groups, b=b)
            pc = stats.paired_bootstrap(
                [u.chars.errors for u in cs_s.utterances], [u.chars.errors for u in cs.utterances],
                clens, groups, b=b)
            block["suggestion_effect"] = {
                "wer": pw.to_dict(), "cer": pc.to_dict(),
                "changed_utterances": sum(1 for p in preds if p.get("hyp_suggested", p["hyp"]) != p["hyp"]),
                "_pw": pw, "_pc": pc,
            }
        modes[mode] = block

    # Confidence check: does a lower avg_logprob actually mean more errors?
    std = scores["standard"]
    lp, uw = [], []
    for p, u in zip(preds, std.utterances):
        if p.get("avg_logprob") is not None and u.words.ref_len > 0:
            lp.append(float(p["avg_logprob"]))
            uw.append(u.words.rate)
    rho = stats.spearman(lp, uw)
    flag_counts: dict[str, int] = defaultdict(int)
    for p in preds:
        for fl in p.get("flags") or []:
            flag_counts[fl] += 1

    n_clusters = len({g if g is not None else f"__solo_{i}" for i, g in enumerate(groups)})
    result = {
        "meta": meta,
        "n": {
            "utterances": len(preds),
            "clusters": n_clusters,
            "cluster_unit": "group/speaker" if any(g is not None for g in groups) else "utterance",
            "hours": round(sum(p.get("duration") or 0 for p in preds) / 3600, 4),
            "ref_words_standard": modes["standard"]["ref_words"],
            "bootstrap_resamples": b,
        },
        "modes": modes,
        "hook_error_share_points": (
            (modes["standard"]["_wer"].estimate - modes["lenient"]["_wer"].estimate)
        ),
        "confidence": {
            "spearman_avg_logprob_vs_wer": rho, "n": len(lp),
            "note": "negative rho means lower confidence goes with more errors (what we want)",
        },
        "flags": dict(flag_counts),
        "loop_flagged": flag_counts.get("loop", 0),
    }
    (run_dir / "report.md").write_text(_report_md(result, modes), encoding="utf-8")
    clean = _clean(_strip_private(result))
    (run_dir / "metrics.json").write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    return clean


# --- Report -----------------------------------------------------------------------


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(lines)


def _breakdown_table(bd: dict, title: str) -> str:
    if not bd:
        return ""
    rows = [
        (k, v["n"], v["ref_words"], _fmt_ci(v["_wer"]), _fmt_ci(v["_cer"]))
        for k, v in bd.items()
    ]
    return f"### By {title}\n\n" + _table(
        [title, "utterances", "ref words", "WER % [95% CI]", "CER % [95% CI]"], rows) + "\n"


def _report_md(res: dict, modes: dict) -> str:
    meta, n = res["meta"], res["n"]
    std = modes["standard"]
    out = [f"# Gyara evaluation: `{meta.get('model_id', '?')}` on `{meta.get('test_set', '?')}`", ""]
    out.append(
        f"- **n** = {n['utterances']} utterances, {n['clusters']} clusters "
        f"({n['cluster_unit']}), {std['ref_words']} reference words (standard), "
        f"{n['hours']:.2f} h of audio"
    )
    if meta:
        out.append(f"- Manifest: `{meta.get('manifest', '?')}` (sha256 `{str(meta.get('manifest_sha256', ''))[:12]}`)")
        out.append(
            f"- Device: {meta.get('device')}, language forcing: {meta.get('language')!r} "
            f"(used: {meta.get('decoding', {}).get('language_used')!r}), "
            f"real-time factor: {meta.get('rtf')}"
        )
        out.append(
            f"- Normalisation rules {meta.get('rules_version')}, gyara {meta.get('gyara_version')}, "
            f"run started {meta.get('started_utc')}"
        )
    out.append(
        f"- 95% CIs: cluster bootstrap, {n['bootstrap_resamples']} resamples "
        f"(clusters = sentence id / speaker, so correlated readings are not counted as independent)."
    )
    out += ["", "## Headline", "",
            "`standard` is the headline. `raw` only collapses whitespace; `lenient` also folds hooked letters.", ""]
    out.append(_table(
        ["mode", "WER % [95% CI]", "CER % [95% CI]", "S", "D", "I", "ref words"],
        [(m, _fmt_ci(modes[m]["_wer"]), _fmt_ci(modes[m]["_cer"]), modes[m]["substitutions"],
          modes[m]["deletions"], modes[m]["insertions"], modes[m]["ref_words"]) for m in modes],
    ))
    hook = res["hook_error_share_points"]
    if hook is not None and not math.isnan(hook):
        out += ["", f"Standard minus lenient WER = **{hook * 100:.2f} points**: the part of the "
                    "error rate that is hooked-letter (ɓ ɗ ƙ ƴ) mistakes only."]

    out += ["", "## Where it fails (standard)", ""]
    for key, title in (("duration", "clip length"), ("source", "source"),
                       ("dialect", "dialect"), ("speaker", "speaker")):
        t = _breakdown_table(std["breakdown"][key], title)
        if t:
            out += [t]
    conf = std.get("top_confusions") or []
    if conf:
        out += [f"### Top {len(conf)} word confusions", "",
                _table(["op", "reference", "hypothesis", "count"],
                       [(c["op"], c["ref"] or "∅", c["hyp"] or "∅", c["count"]) for c in conf]), ""]

    c = res["confidence"]
    rho = c["spearman_avg_logprob_vs_wer"]
    out += ["## Is the confidence score useful?", ""]
    if rho is None or math.isnan(rho):
        out.append(f"Not measurable here (n = {c['n']} utterances with a score).")
    else:
        strength = "weak" if abs(rho) < 0.3 else "moderate" if abs(rho) < 0.5 else "strong"
        direction = "as hoped, lower confidence goes with more errors" if rho < 0 else \
            "the wrong direction: confidence does not flag the bad utterances"
        out.append(
            f"Spearman correlation between `avg_logprob` and utterance WER: **ρ = {rho:.3f}** "
            f"(n = {c['n']}), {strength}; {direction}."
        )
        if c["n"] < 30:
            out.append(f"With only {c['n']} utterances this correlation is not reliable either way.")
        elif abs(rho) < 0.3 or rho >= 0:
            out.append("Do not market \"least confident first\" on this evidence.")
    out += ["", "## Loops and flags", ""]
    if res["flags"]:
        out.append(_table(["flag", "utterances"], sorted(res["flags"].items())))
    else:
        out.append("No utterance was flagged.")
    out.append("")
    out.append(f"`loop` = still repetitive after temperature fallback; trimmed and left for a human "
               f"({res['loop_flagged']} utterance(s)).")

    if "suggestion_effect" in std:
        out += ["", "## Suggestion effect (accepting every suggestion vs ASR alone)", ""]
        rows = []
        for m in modes:
            se = modes[m].get("suggestion_effect")
            if not se:
                continue
            pw: stats.PairedResult = se["_pw"]
            rows.append((m, _pct(pw.rate_b), _pct(pw.rate_a), f"{pw.delta * 100:+.2f}",
                         f"[{pw.low * 100:+.2f}, {pw.high * 100:+.2f}]", f"{pw.p_value:.4f}",
                         f"{pw.mde * 100:.2f}"))
        out.append(_table(["mode", "ASR WER %", "with suggestions WER %", "Δ points",
                           "95% CI", "p", "MDE"], rows))
        se = std["suggestion_effect"]
        out += ["", "Standard WER: " + se["_pw"].sentence("With suggestions", "ASR alone"),
                "", "Standard CER: " + se["_pc"].sentence("With suggestions", "ASR alone"),
                "", f"Suggestions changed {se['changed_utterances']} of {n['utterances']} utterances. "
                "In the product, a human accepts or rejects each one; this measures the upper "
                "bound where every suggestion is accepted."]
    out += ["", "---", "", f"*{gyara.ATTRIBUTION}*", ""]
    return "\n".join(out)


# --- Comparison -----------------------------------------------------------------------


def _run_rules(run_dir: Path) -> str | None:
    for fn in ("metrics.json", "meta.json"):
        p = run_dir / fn
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            v = d.get("meta", d).get("rules_version")
            if v:
                return v
    return None


def _run_label(run_dir: Path) -> str:
    p = run_dir / "meta.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")).get("model_id") or run_dir.name
    return run_dir.name


def compare(run_a: str | Path, run_b: str | Path, mode: str = "standard",
            out: str | Path | None = None, b: int = stats.DEFAULT_B) -> dict:
    """Paired bootstrap of run A vs run B on the utterances both contain.

    Negative delta means A makes fewer errors. Refuses runs scored under
    different normalisation rules, or whose references disagree, because
    then the comparison is not like for like.
    """
    run_a, run_b = Path(run_a), Path(run_b)
    ra, rb = _run_rules(run_a), _run_rules(run_b)
    if ra != rb:
        raise ValueError(f"runs use different normalisation rules ({ra} vs {rb}); re-score both "
                         "with `score_run` under the same RULES_VERSION first")
    pa = {p["id"]: p for p in _read_jsonl(run_a / "predictions.jsonl")}
    pb = {p["id"]: p for p in _read_jsonl(run_b / "predictions.jsonl")}
    common = [i for i in pa if i in pb]
    if not common:
        raise ValueError("the two runs share no utterance ids")
    norm = normalize.get_normalizer(mode)
    bad = [i for i in common if norm(pa[i].get("ref")) != norm(pb[i].get("ref"))]
    if bad:
        raise ValueError(f"{len(bad)} shared id(s) have different references, e.g. {bad[:3]}")
    refs = [pa[i].get("ref") or "" for i in common]
    groups = [pa[i].get("group") or pa[i].get("speaker") for i in common]
    sa = metrics.score_corpus(common, refs, [pa[i].get("hyp") or "" for i in common], norm, groups)
    sb = metrics.score_corpus(common, refs, [pb[i].get("hyp") or "" for i in common], norm, groups)
    pw = stats.paired_bootstrap([u.words.errors for u in sa.utterances],
                                [u.words.errors for u in sb.utterances],
                                [u.words.ref_len for u in sa.utterances], groups, b=b)
    pc = stats.paired_bootstrap([u.chars.errors for u in sa.utterances],
                                [u.chars.errors for u in sb.utterances],
                                [u.chars.ref_len for u in sa.utterances], groups, b=b)
    la, lb = _run_label(run_a), _run_label(run_b)
    if la == lb:
        la, lb = run_a.name, run_b.name
    res = {
        "run_a": str(run_a), "run_b": str(run_b), "label_a": la, "label_b": lb, "mode": mode,
        "rules_version": ra, "n_common": len(common),
        "dropped_a": len(pa) - len(common), "dropped_b": len(pb) - len(common),
        "wer": pw.to_dict(), "cer": pc.to_dict(),
        "sentence_wer": pw.sentence(la, lb), "sentence_cer": pc.sentence(la, lb),
    }
    md = [f"# Compare: `{la}` (A) vs `{lb}` (B), {mode} normalisation", "",
          f"- Same {len(common)} utterances, same references, rules {ra}. "
          f"Dropped (not in both runs): {res['dropped_a']} from A, {res['dropped_b']} from B.",
          f"- Paired cluster bootstrap, {pw.resamples} resamples over {pw.n_units} {pw.unit}s. "
          "Δ = A − B; negative means A makes fewer errors.", "",
          _table(["metric", "A %", "B %", "Δ points", "95% CI", "p", "MDE (points)"],
                 [(name, _pct(r.rate_a), _pct(r.rate_b), f"{r.delta * 100:+.2f}",
                   f"[{r.low * 100:+.2f}, {r.high * 100:+.2f}]", f"{r.p_value:.4f}",
                   f"{r.mde * 100:.2f}") for name, r in (("WER", pw), ("CER", pc))]),
          "", f"**WER:** {res['sentence_wer']}", "", f"**CER:** {res['sentence_cer']}",
          "", "---", "", f"*{gyara.ATTRIBUTION}*", ""]
    out_path = Path(out) if out else run_a / "compare.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    # The machine-readable twin: every published comparison number must come
    # from a committed file, not from the markdown or a console print.
    json_path = out_path.with_suffix(".json")
    res["path"] = str(out_path)
    res["json_path"] = str(json_path)
    res = _clean(res)
    json_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


def leaderboard(run_dirs: Iterable[str | Path]) -> str:
    """Markdown table across runs, sorted by standard WER."""
    entries = []
    for d in run_dirs:
        p = Path(d) / "metrics.json"
        if not p.exists():
            log.warning("no metrics.json in %s; skipped", d)
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        meta, std, len_ = m.get("meta", {}), m["modes"]["standard"], m["modes"]["lenient"]
        iv = stats.Interval(**{k: (float("nan") if v is None else v) for k, v in std["wer"].items()})
        entries.append((
            std["wer"]["estimate"] if std["wer"]["estimate"] is not None else float("inf"),
            (meta.get("model_id", Path(d).name), meta.get("test_set", "?"), m["n"]["utterances"],
             _fmt_ci(iv), _pct(std["cer"]["estimate"]), _pct(len_["wer"]["estimate"]),
             meta.get("rtf", "n/a"), meta.get("rules_version", "?")),
        ))
    entries.sort(key=lambda e: e[0])
    return _table(["model", "test set", "n", "WER std % [95% CI]", "CER std %", "WER lenient %",
                   "RTF", "rules"], [e[1] for e in entries])
