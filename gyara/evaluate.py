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


NOT_RECORDED = "not recorded"
# Run files are written with "\n" on every OS, so a re-score on Windows does
# not turn every line of a committed report into a diff.
LF = "\n"
_FLEURS_ID = re.compile(r"^fleurs_([a-z]{2,3}_[a-z]{2})_(train|validation|test)_")


def _test_set_label(manifest_path: str | Path | None, ids: Sequence[str] = ()) -> str:
    """A human name for the test set, e.g. ``fleurs ha_ng test``.

    Order of evidence: the ``<stem>.summary.json`` that ``fetch-fleurs`` writes
    next to the manifest; the FLEURS id pattern (``fleurs_ha_ng_test_…``) when
    every id follows it; otherwise the manifest's folder and file name.
    """
    p = Path(manifest_path) if manifest_path else None
    if p is not None:
        summ = p.with_name(p.stem + ".summary.json")
        if summ.exists():
            try:
                s = json.loads(summ.read_text(encoding="utf-8"))
                repo = str(s.get("repo", "")).split("/")[-1]
                parts = [repo, s.get("config"), s.get("split")]
                # A summary from a different fetch (other --limit) would mislabel the run.
                if all(parts) and (not ids or s.get("rows") == len(ids)):
                    label = " ".join(str(x) for x in parts)
                    return label + (f" (first {s['limit']})" if s.get("limit") else "")
            except (ValueError, OSError):
                pass
    ms = [_FLEURS_ID.match(str(i)) for i in ids]
    if ms and all(ms) and len({m.groups() for m in ms}) == 1:
        return "fleurs " + " ".join(ms[0].groups())
    if p is not None:
        parent = p.parent.name
        return f"{parent} {p.stem}" if parent and parent not in (".", "") else p.stem
    return "?"


def _suggester_meta(sg) -> dict:
    """What produced the suggestions, so a suggestion effect can be reproduced.

    Duck-typed: anything we cannot read off the object is "not recorded"
    rather than guessed. The prompt and guard are identified by a hash of their
    text and thresholds, so a changed prompt shows up as a changed id.
    """
    out: dict[str, Any] = {"class": type(sg).__name__}
    backend = getattr(sg, "backend", None)
    out["backend"] = backend or NOT_RECORDED
    out["model_id"] = getattr(sg, "model_id", None) or NOT_RECORDED
    out["quantisation"] = _quantisation(sg)
    out["prompt_guard_id"] = _prompt_guard_id() if backend else NOT_RECORDED
    if backend == "openai":
        out["base_url"] = getattr(sg, "base_url", None) or NOT_RECORDED
    for k in ("max_new_tokens", "repetition_penalty"):
        if hasattr(sg, k):
            out[k] = getattr(sg, k)
    return out


def _quantisation(sg) -> str:
    backend = getattr(sg, "backend", None)
    if backend == "openai":
        return "set by the server; not visible to Gyara"
    if backend == "none":
        return "n/a (suggestions off)"
    model = getattr(sg, "_model", None)
    if model is None:
        want = getattr(sg, "load_in_4bit", None)
        return (f"{NOT_RECORDED} (model not loaded; 4-bit requested: {want})"
                if want is not None else NOT_RECORDED)
    if getattr(model, "is_loaded_in_4bit", False):
        qc = getattr(getattr(model, "config", None), "quantization_config", None)
        qt = (qc.get("bnb_4bit_quant_type") if isinstance(qc, dict)
              else getattr(qc, "bnb_4bit_quant_type", None))
        return f"4-bit bitsandbytes{f' {qt}' if qt else ''}"
    if getattr(model, "is_loaded_in_8bit", False):
        return "8-bit bitsandbytes"
    dtype = getattr(model, "dtype", None)
    return f"none ({str(dtype).replace('torch.', '')})" if dtype is not None else NOT_RECORDED


def _prompt_guard_id() -> str:
    """sha256 (first 12 hex) of the proofreading prompt, few-shot examples and guard thresholds."""
    import hashlib

    from gyara import suggest as sug

    blob = json.dumps({
        "identity": sug.NATLAS_IDENTITY, "instructions": sug.PROOFREAD_INSTRUCTIONS,
        "few_shot": sug.FEW_SHOT, "max_word_edit": sug.MAX_WORD_EDIT,
        "max_changed_word_share": sug.MAX_CHANGED_WORD_SHARE,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


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
    suggest_s = 0.0
    guard_rejections: dict[str, int] = defaultdict(int)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row, hyp, lp, flags, dur in _transcribe_rows(transcriber, rows, batch_size):
            audio_s += dur
            rec = {
                "id": row.id, "ref": row.text, "hyp": hyp, "avg_logprob": lp, "flags": flags,
                "duration": round(dur, 3), "group": row.cluster, "dialect": row.dialect,
                "speaker": row.speaker, "source": row.source,
            }
            if suggester is not None:
                ts = time.perf_counter()
                try:
                    sg = suggester.suggest(hyp)
                    rec["hyp_suggested"] = sg.suggested
                    accepted = getattr(sg, "accepted_by_guard", None)
                    reason = getattr(sg, "reason", None)
                    rec["suggest_accepted"] = accepted
                    rec["suggest_reason"] = reason
                    if accepted is False:
                        guard_rejections[str(reason or "unspecified")] += 1
                except Exception as e:  # noqa: BLE001 - one bad call must not sink the run
                    log.warning("suggester failed on %s: %s", row.id, e)
                    rec["hyp_suggested"] = hyp  # = "suggestion rejected"; counted below
                    rec["suggest_accepted"] = False
                    rec["suggest_reason"] = f"error: {type(e).__name__}"
                    rec["flags"] = flags + ["suggest_error"]
                    n_suggest_err += 1
                suggest_s += time.perf_counter() - ts
            f.write(json.dumps(_clean(rec), ensure_ascii=False) + "\n")
    wall = time.perf_counter() - t0

    info = transcriber.info() if hasattr(transcriber, "info") else {}
    meta = {
        "model_id": model_id,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.sha256_file(manifest_path),
        "test_set": manifest_path.stem,
        "test_set_label": _test_set_label(manifest_path, [r.id for r in rows]),
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
        "rtf_without_suggest": (round((wall - suggest_s) / audio_s, 4)
                                if audio_s and suggester is not None else None),
        "gyara_version": gyara.__version__,
        "rules_version": normalize.RULES_VERSION,
        "suggester": _suggester_meta(suggester) if suggester is not None else None,
        "suggest_errors": n_suggest_err if suggester is not None else None,
        "suggest_seconds": round(suggest_s, 2) if suggester is not None else None,
        "suggest_guard_rejections": (
            {"total": sum(guard_rejections.values()), "by_reason": dict(guard_rejections)}
            if suggester is not None else None),
        "versions": _versions(),
    }
    (run_dir / "meta.json").write_text(json.dumps(_clean(meta), indent=2), encoding="utf-8", newline=LF)
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


def _cluster_keys(preds: list[dict]) -> dict:
    """Which field each utterance's bootstrap cluster actually came from."""
    by_group = sum(1 for p in preds if p.get("group"))
    by_speaker = sum(1 for p in preds if not p.get("group") and p.get("speaker"))
    solo = len(preds) - by_group - by_speaker
    used = [k for k, v in (("group", by_group), ("speaker", by_speaker)) if v]
    return {"group": by_group, "speaker": by_speaker, "own_cluster": solo,
            "unit": "+".join(used) if used else "utterance"}


def _cluster_text(ck: dict, preds: list[dict]) -> str:
    """Plain-English description of the cluster key, naming only what was used."""
    fleurs = all((p.get("source") or "") == "fleurs" for p in preds)
    names = {"group": "`group` (FLEURS sentence id)" if fleurs else "`group`",
             "speaker": "`speaker`"}
    parts = [f"{names[k]} for {ck[k]} utterances" for k in ("group", "speaker") if ck[k]]
    if ck["own_cluster"]:
        parts.append(f"{ck['own_cluster']} utterances with neither, each its own cluster")
    if not ck["group"] and not ck["speaker"]:
        return "utterances (no `group` or `speaker` in the manifest), resampled independently"
    return "; ".join(parts)


# Hooked letters as they appear after ``standard`` normalisation (ƴ is canonicalised to 'y).
_HOOKS = ("ɓ", "ɗ", "ƙ", "'y")


def _has_hook(w: str) -> bool:
    return any(h in w for h in _HOOKS)


def _hook_split(cs_standard: metrics.CorpusScore) -> dict:
    """Word substitutions (standard) that differ only in hooked letters, by direction.

    ``hyp_hooked_ref_plain``: the transcript has a hook where the reference has
    none (e.g. ref *daya*, hyp *ɗaya*). ``ref_hooked_hyp_plain``: the reverse.
    ``both_hooked``: both carry hooks, in different places. Who is "right" is
    not decided here: references can be missing hooks too.
    """
    lenient = normalize.get_normalizer("lenient")
    out = {"hook_only_substitutions": 0, "hyp_hooked_ref_plain": 0,
           "ref_hooked_hyp_plain": 0, "both_hooked": 0}
    for (op, r, h), k in cs_standard.confusions.items():
        if op != "S" or r == h or lenient(r) != lenient(h):
            continue
        out["hook_only_substitutions"] += k
        hr, hh = _has_hook(r), _has_hook(h)
        key = ("hyp_hooked_ref_plain" if hh and not hr else
               "ref_hooked_hyp_plain" if hr and not hh else "both_hooked")
        out[key] += k
    ref_w = [w for u in cs_standard.utterances for w in u.ref.split()]
    hyp_w = [w for u in cs_standard.utterances for w in u.hyp.split()]
    out["ref_words_with_hook_share"] = (sum(map(_has_hook, ref_w)) / len(ref_w)) if ref_w else None
    out["hyp_words_with_hook_share"] = (sum(map(_has_hook, hyp_w)) / len(hyp_w)) if hyp_w else None
    return out


_PUNCT_KEEP = set("'’ʼ‘")  # apostrophes are part of Hausa spelling ('yan, ƙur'ani)


def _case_punct_fold(s: str) -> str:
    """Casefold and drop punctuation (apostrophes kept), for "changed beyond case/punctuation"."""
    import unicodedata

    s = unicodedata.normalize("NFC", s or "").casefold()
    s = "".join(" " if unicodedata.category(c).startswith("P") and c not in _PUNCT_KEEP else c
                for c in s)
    return " ".join(s.split())


def _suggestion_counts(preds: list[dict]) -> dict:
    """How many utterances the suggestions touched, and how deeply."""
    std = normalize.get_normalizer("standard")
    n_any = n_beyond = n_std = 0
    for p in preds:
        h, s = p.get("hyp") or "", p.get("hyp_suggested", p.get("hyp")) or ""
        if s == h:
            continue
        n_any += 1
        if _case_punct_fold(s) != _case_punct_fold(h):
            n_beyond += 1
        if std(s) != std(h):
            n_std += 1
    out: dict[str, Any] = {
        "utterances": len(preds),
        "changed_any": n_any,
        "changed_beyond_case_punctuation": n_beyond,
        "changed_after_standard_normalisation": n_std,
    }
    if any("suggest_accepted" in p for p in preds):
        rej: dict[str, int] = defaultdict(int)
        for p in preds:
            if p.get("suggest_accepted") is False:
                rej[str(p.get("suggest_reason") or "unspecified")] += 1
        out["guard_rejections"] = {"total": sum(rej.values()), "by_reason": dict(rej)}
    else:
        out["guard_rejections"] = NOT_RECORDED
    return out


def _upgrade_meta(meta: dict, preds: list[dict]) -> dict:
    """Bring meta.json from older runs up to the current fields, without guessing.

    Anything the old run did not write down becomes "not recorded".
    """
    if "test_set_label" not in meta:
        meta["test_set_label"] = _test_set_label(meta.get("manifest"), [p["id"] for p in preds])
    if any("hyp_suggested" in p for p in preds):
        sg = meta.get("suggester")
        if not isinstance(sg, dict):
            meta["suggester"] = {
                "class": sg if isinstance(sg, str) else NOT_RECORDED,
                "backend": NOT_RECORDED, "model_id": NOT_RECORDED,
                "quantisation": NOT_RECORDED, "prompt_guard_id": NOT_RECORDED,
            }
        for k in ("suggest_seconds", "suggest_guard_rejections", "rtf_without_suggest"):
            meta.setdefault(k, NOT_RECORDED)
    return meta


def score_run(run_dir: str | Path, b: int = stats.DEFAULT_B) -> dict:
    """Score ``predictions.jsonl`` (no inference). Writes metrics.json + report.md."""
    run_dir = Path(run_dir)
    preds = _read_jsonl(run_dir / "predictions.jsonl")
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("rules_version") and meta["rules_version"] != normalize.RULES_VERSION:
        log.warning("predictions made under rules %s, re-scored under %s",
                    meta["rules_version"], normalize.RULES_VERSION)
    if meta_path.exists():
        # Old runs lack newer fields. Add them as "not recorded" (never
        # overwriting what the run did record) so meta.json says so itself.
        before = json.dumps(meta, sort_keys=True)
        meta = _upgrade_meta(meta, preds)
        if json.dumps(meta, sort_keys=True) != before:
            meta_path.write_text(json.dumps(_clean(meta), indent=2, ensure_ascii=False),
                                 encoding="utf-8", newline=LF)
    meta["scored_rules_version"] = normalize.RULES_VERSION
    meta["rules_version"] = normalize.RULES_VERSION  # scoring, not decoding, applies the rules
    meta = _upgrade_meta(meta, preds)

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
            # Per-utterance direction under this mode's WER: the pooled delta
            # hides that accepting everything makes some clips worse.
            per = [(s.words.errors, a.words.errors) for s, a in zip(cs_s.utterances, cs.utterances)]
            block["suggestion_effect"] = {
                "wer": pw.to_dict(), "cer": pc.to_dict(),
                "changed_utterances": sum(1 for p in preds if p.get("hyp_suggested", p["hyp"]) != p["hyp"]),
                "wer_improved_utterances": sum(1 for s, a in per if s < a),
                "wer_worsened_utterances": sum(1 for s, a in per if s > a),
                "wer_unchanged_utterances": sum(1 for s, a in per if s == a),
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
    ck = _cluster_keys(preds)
    gap = modes["standard"]["_wer"].estimate - modes["lenient"]["_wer"].estimate
    result = {
        "meta": meta,
        "n": {
            "utterances": len(preds),
            "clusters": n_clusters,
            "cluster_unit": ck["unit"],
            "cluster_keys": ck,
            "cluster_description": _cluster_text(ck, preds),
            "hours": round(sum(p.get("duration") or 0 for p in preds) / 3600, 4),
            "ref_words_standard": modes["standard"]["ref_words"],
            "bootstrap_resamples": b,
        },
        "modes": modes,
        # Standard minus lenient WER. Not "the model's hook errors": the
        # reference can be the side missing the hook. See hook_substitutions.
        "standard_minus_lenient_wer": gap,  # a fraction, like every rate here
        "hook_substitutions": _hook_split(scores["standard"]),
        "suggestions": _suggestion_counts(preds) if has_sugg else None,
        "confidence": {
            "spearman_avg_logprob_vs_wer": rho, "n": len(lp),
            "note": "negative rho means lower confidence goes with more errors (what we want)",
        },
        "flags": dict(flag_counts),
        "loop_flagged": flag_counts.get("loop", 0),
    }
    (run_dir / "report.md").write_text(_report_md(result, modes), encoding="utf-8", newline=LF)
    clean = _clean(_strip_private(result))
    (run_dir / "metrics.json").write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8", newline=LF)
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


def _p_cell(r: stats.PairedResult) -> str:
    """p for a table column headed "p": ``0.0016`` or ``< 0.001``."""
    s = r.p_str()
    return s[4:] if s.startswith("p = ") else s[2:] if s.startswith("p ") else s


def _hook_lines(res: dict) -> list[str]:
    gap = res.get("standard_minus_lenient_wer")
    if gap is None or math.isnan(gap):
        return []
    hs = res.get("hook_substitutions") or {}
    out = ["", f"Standard minus lenient WER = **{gap * 100:.2f} points**. This is how much of the "
               "error rate disappears when hooked letters (ɓ ɗ ƙ 'y) are folded. It measures "
               "disagreement between transcript and reference; it does not say which side is wrong."]
    if hs.get("hook_only_substitutions"):
        share = lambda x: "n/a" if x is None else f"{x * 100:.1f}%"  # noqa: E731
        out += ["", f"Of the word substitutions under `standard`, {hs['hook_only_substitutions']} "
                    "differ only in hooked letters:", "",
                _table(["direction", "substitutions"], [
                    ("hook in the transcript, not in the reference", hs["hyp_hooked_ref_plain"]),
                    ("hook in the reference, not in the transcript", hs["ref_hooked_hyp_plain"]),
                    ("hooks on both sides, in different places", hs["both_hooked"]),
                ]), "",
                f"Words carrying a hook: {share(hs.get('ref_words_with_hook_share'))} of reference "
                f"words, {share(hs.get('hyp_words_with_hook_share'))} of transcript words."]
    return out


def _suggestion_lines(res: dict, modes: dict) -> list[str]:
    std, n, meta = modes["standard"], res["n"], res["meta"]
    out = ["", "## Suggestion effect: every suggestion accepted without review, vs ASR alone", ""]
    sg = meta.get("suggester")
    if isinstance(sg, dict):
        secs = meta.get("suggest_seconds")
        secs = f"{secs} s" if isinstance(secs, (int, float)) else str(secs)
        mid = sg.get("model_id")
        mid = mid if mid == NOT_RECORDED else f"`{mid}`"
        out += [f"- Suggester: {mid}, backend {sg.get('backend')}, "
                f"quantisation {sg.get('quantisation')}, prompt/guard id {sg.get('prompt_guard_id')}.",
                f"- Time spent on suggestions: {secs}. Real-time factor without them: "
                f"{meta.get('rtf_without_suggest')}.", ""]
    rows = []
    for m in modes:
        se = modes[m].get("suggestion_effect")
        if not se:
            continue
        pw: stats.PairedResult = se["_pw"]
        rows.append((m, _pct(pw.rate_b), _pct(pw.rate_a), f"{pw.delta * 100:+.2f}",
                     f"[{pw.low * 100:+.2f}, {pw.high * 100:+.2f}]", _p_cell(pw),
                     f"{pw.mde * 100:.2f}", se["wer_improved_utterances"],
                     se["wer_worsened_utterances"], se["wer_unchanged_utterances"]))
    out.append(_table(["mode", "ASR WER %", "with suggestions WER %", "Δ points", "95% CI", "p",
                       "MDE", "clips better", "clips worse", "clips same"], rows))
    se = std["suggestion_effect"]
    out += ["", "Standard WER: " + se["_pw"].sentence("With suggestions", "ASR alone"),
            "", "Standard CER: " + se["_pc"].sentence("With suggestions", "ASR alone")]
    sc = res.get("suggestions") or {}
    if sc:
        out += ["", f"The suggestions changed the text of {sc['changed_any']} of {sc['utterances']} "
                    f"utterances. {sc['changed_beyond_case_punctuation']} of those changed more than "
                    "capitals and punctuation, and "
                    f"{sc['changed_after_standard_normalisation']} still differ after `standard` "
                    "normalisation."]
        gr = sc.get("guard_rejections")
        if isinstance(gr, dict):
            reasons = ", ".join(f"{k}: {v}" for k, v in sorted(gr["by_reason"].items(),
                                                              key=lambda kv: -kv[1]))
            out += ["", f"The guard rejected {gr['total']} of {sc['utterances']} suggestions"
                        + (f" ({reasons})." if reasons else ".")]
        else:
            out += ["", f"Guard rejections: {gr} for this run."]
    out += ["", f"Under standard WER, accepting every suggestion made "
                f"{se['wer_improved_utterances']} utterance{'' if se['wer_improved_utterances'] == 1 else 's'} better, "
                f"{se['wer_worsened_utterances']} worse and {se['wer_unchanged_utterances']} no "
                "different. This is not an upper bound: a reviewer who rejects the harmful "
                "suggestions does better than accepting them all. In the product, a human accepts "
                "or rejects each one."]
    return out


def _report_md(res: dict, modes: dict) -> str:
    meta, n = res["meta"], res["n"]
    std = modes["standard"]
    label = meta.get("test_set_label") or meta.get("test_set", "?")
    out = [f"# Gyara evaluation: `{meta.get('model_id', '?')}` on {label}", ""]
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
        f"- 95% CIs: cluster bootstrap, {n['bootstrap_resamples']} resamples. Clusters: "
        f"{n['cluster_description']}."
    )
    out += ["", "## Headline", "",
            "`standard` is the headline. `raw` only collapses whitespace; `lenient` also folds hooked letters.", ""]
    out.append(_table(
        ["mode", "WER % [95% CI]", "CER % [95% CI]", "S", "D", "I", "ref words"],
        [(m, _fmt_ci(modes[m]["_wer"]), _fmt_ci(modes[m]["_cer"]), modes[m]["substitutions"],
          modes[m]["deletions"], modes[m]["insertions"], modes[m]["ref_words"]) for m in modes],
    ))
    out += _hook_lines(res)

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
        out += _suggestion_lines(res, modes)
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


def _paired(pa: dict, pb: dict, common: list[str], mode: str, groups: list, b: int):
    norm = normalize.get_normalizer(mode)
    refs = [pa[i].get("ref") or "" for i in common]
    sa = metrics.score_corpus(common, refs, [pa[i].get("hyp") or "" for i in common], norm, groups)
    sb = metrics.score_corpus(common, refs, [pb[i].get("hyp") or "" for i in common], norm, groups)
    pw = stats.paired_bootstrap([u.words.errors for u in sa.utterances],
                                [u.words.errors for u in sb.utterances],
                                [u.words.ref_len for u in sa.utterances], groups, b=b)
    pc = stats.paired_bootstrap([u.chars.errors for u in sa.utterances],
                                [u.chars.errors for u in sb.utterances],
                                [u.chars.ref_len for u in sa.utterances], groups, b=b)
    return pw, pc


def compare(run_a: str | Path, run_b: str | Path, mode: str = "standard",
            out: str | Path | None = None, b: int = stats.DEFAULT_B) -> dict:
    """Paired bootstrap of run A vs run B on the utterances both contain.

    Negative delta means A makes fewer errors. Refuses runs scored under
    different normalisation rules, or whose references disagree, because
    then the comparison is not like for like.

    All three modes × WER/CER are tested (six tests) and reported; ``mode`` is
    the headline, fixed before looking, and fills the top-level ``wer``/``cer``
    and sentences.
    """
    run_a, run_b = Path(run_a), Path(run_b)
    ra, rb = _run_rules(run_a), _run_rules(run_b)
    if ra != rb:
        raise ValueError(f"runs use different normalisation rules ({ra} vs {rb}); re-score both "
                         "with `score_run` under the same RULES_VERSION first")
    if mode not in normalize.MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {normalize.MODES}")
    pa = {p["id"]: p for p in _read_jsonl(run_a / "predictions.jsonl")}
    pb = {p["id"]: p for p in _read_jsonl(run_b / "predictions.jsonl")}
    common = [i for i in pa if i in pb]
    if not common:
        raise ValueError("the two runs share no utterance ids")
    norm = normalize.get_normalizer(mode)
    bad = [i for i in common if norm(pa[i].get("ref")) != norm(pb[i].get("ref"))]
    if bad:
        raise ValueError(f"{len(bad)} shared id(s) have different references, e.g. {bad[:3]}")
    groups = [pa[i].get("group") or pa[i].get("speaker") for i in common]
    ck = _cluster_keys([pa[i] for i in common])
    by_mode = {m: _paired(pa, pb, common, m, groups, b) for m in normalize.MODES}
    pw, pc = by_mode[mode]
    la, lb = _run_label(run_a), _run_label(run_b)
    if la == lb:
        la, lb = run_a.name, run_b.name
    res = {
        "run_a": run_a.as_posix(), "run_b": run_b.as_posix(), "label_a": la, "label_b": lb, "mode": mode,
        "headline": f"{mode} WER", "n_tests": 2 * len(by_mode),
        "rules_version": ra, "n_common": len(common),
        "dropped_a": len(pa) - len(common), "dropped_b": len(pb) - len(common),
        "cluster_keys": ck,
        "wer": pw.to_dict(), "cer": pc.to_dict(),
        "sentence_wer": pw.sentence(la, lb), "sentence_cer": pc.sentence(la, lb),
        "modes": {m: {"wer": w.to_dict(), "cer": c.to_dict()} for m, (w, c) in by_mode.items()},
    }
    rows = []
    for m, (w, c) in by_mode.items():
        for name, r in (("WER", w), ("CER", c)):
            head = m == mode and name == "WER"
            cell = f"**{m}**" if head else m
            rows.append((cell, name, _pct(r.rate_a), _pct(r.rate_b), f"{r.delta * 100:+.2f}",
                         f"[{r.low * 100:+.2f}, {r.high * 100:+.2f}]", _p_cell(r),
                         f"{r.mde * 100:.2f}"))
    md = [f"# Compare: `{la}` (A) vs `{lb}` (B)", "",
          f"- Same {len(common)} utterances, same references, rules {ra}. "
          f"Dropped (not in both runs): {res['dropped_a']} from A, {res['dropped_b']} from B.",
          f"- Paired cluster bootstrap, {pw.resamples} resamples over {pw.n_units} {pw.unit}s "
          f"({_cluster_text(ck, [pa[i] for i in common])}). "
          "Δ = A − B; negative means A makes fewer errors.",
          f"- Headline, fixed in advance: **{mode} WER**. The table shows all "
          f"{res['n_tests']} tests (3 normalisation modes × WER/CER); read the others as "
          "supporting evidence, not as extra chances to find a difference.", "",
          _table(["mode", "metric", "A %", "B %", "Δ points", "95% CI", "p", "MDE (points)"], rows),
          "", f"**WER ({mode}):** {res['sentence_wer']}", "",
          f"**CER ({mode}):** {res['sentence_cer']}",
          "", "---", "", f"*{gyara.ATTRIBUTION}*", ""]
    out_path = Path(out) if out else run_a / "compare.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8", newline=LF)
    # The machine-readable twin: every published comparison number must come
    # from a committed file, not from the markdown or a console print.
    json_path = out_path.with_suffix(".json")
    res["path"] = out_path.as_posix()
    res["json_path"] = json_path.as_posix()
    res = _clean(res)
    json_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8", newline=LF)
    return res


def _iv(d: dict) -> stats.Interval:
    return stats.Interval(**{k: (float("nan") if v is None else v) for k, v in d.items()})


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
        model = meta.get("model_id", Path(d).name)
        if meta.get("suggester"):
            # WER is the speech model alone; RTF includes suggestion time.
            model += " (run with suggestions; WER is ASR alone)"
        label = meta.get("test_set_label") or meta.get("test_set", "?")
        entries.append((
            std["wer"]["estimate"] if std["wer"]["estimate"] is not None else float("inf"),
            (model, label, m["n"]["utterances"], _fmt_ci(_iv(std["wer"])), _fmt_ci(_iv(std["cer"])),
             _pct(len_["wer"]["estimate"]), meta.get("rtf", "n/a"), meta.get("rules_version", "?")),
        ))
    entries.sort(key=lambda e: e[0])
    return _table(["model", "test set", "n", "WER std % [95% CI]", "CER std % [95% CI]",
                   "WER lenient %", "RTF", "rules"], [e[1] for e in entries])
