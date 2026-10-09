"""Fine-tune a Whisper Hausa model on human-verified Gyara exports.

This is the "improve" end of the loop: corrected segments exported by
``gyara export`` become training data for ``NCAIR1/Hausa-ASR``, and the result
is re-measured on held-out data it never saw.

The order of operations is the point:

1. Read the export manifests (draft rows dropped) and, optionally, mix in
   FLEURS ha_ng *train* utterances so a few hours of own data do not make the
   model forget general Hausa.
2. **Leakage check** against every held-out manifest. Any shared audio or
   shared speaker aborts training (eval-integrity rule 4); shared sentence
   text is only a warning.
3. Split a **speaker-disjoint dev set** off the training pool. Early stopping
   and checkpoint choice use dev WER only, never the test set.
4. Train with conservative small-data defaults (low LR, warmup, frozen encoder
   under 5 h, light SpecAugment).
5. Save model + processor + a model card carrying the N-ATLAS licence and
   attribution, then evaluate base and tuned **once** on each held-out
   manifest and write a paired comparison, ``before_after.md``, whatever it
   shows.

Run: ``python -m gyara.finetune.train --config my.yaml`` (or ``gyara finetune``).
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import math
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .. import ATTRIBUTION
from .. import manifest as mf
from ..manifest import Row
from ..metrics import score_corpus
from ..normalize import RULES_VERSION, get_normalizer
from ..stats import PairedResult

log = logging.getLogger("gyara.finetune")

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
SAMPLE_RATE = 16_000
MAX_LABEL_TOKENS = 448  # Whisper decoder context


class LeakageError(RuntimeError):
    """Training data overlaps a held-out manifest (audio or speaker)."""


# --- Config -------------------------------------------------------------------


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config: str | Path | dict | None = None) -> dict:
    """Packaged defaults, deep-merged with a YAML path or a dict."""
    defaults = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if config is None:
        return defaults
    if isinstance(config, (str, Path)):
        config = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    cfg = _deep_merge(defaults, config)
    for key in ("exports", "heldout"):
        v = cfg["data"].get(key)
        cfg["data"][key] = [] if v is None else ([v] if isinstance(v, (str, Path)) else list(v))
    f = float(cfg["data"].get("fleurs_fraction") or 0.0)
    if not 0.0 <= f < 1.0:
        raise ValueError("data.fleurs_fraction must be in [0, 1)")
    return cfg


# --- Data ---------------------------------------------------------------------


def read_export(export_dir: str | Path) -> list[Row]:
    """Rows of one Gyara export dir. Draft (unverified) rows are dropped.

    Prefers ``manifest.jsonl``; falls back to the HF ``metadata.jsonl``.
    """
    d = Path(export_dir)
    m = d / "manifest.jsonl"
    if m.exists():
        rows = mf.read(m)
    elif (d / "metadata.jsonl").exists():
        rows = []
        for line in (d / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(Row(
                id=f"{d.name}/{Path(r['file_name']).stem}",
                audio=str((d / r["file_name"]).resolve()),
                text=r.get("transcription", ""),
                speaker=r.get("speaker"),
                dialect=r.get("dialect"),
                duration=r.get("duration"),
                source=r.get("source", "own"),
                # asset_sha256 hashes the source recording, not this clip: not used here.
                extra={"draft": r.get("draft", False)},
            ))
    else:
        raise FileNotFoundError(f"{d}: no manifest.jsonl or metadata.jsonl (is this a gyara export?)")
    kept = [r for r in rows if not r.extra.get("draft")]
    if len(kept) < len(rows):
        warnings.warn(f"{d}: dropped {len(rows) - len(kept)} draft (unverified) row(s)", stacklevel=2)
    return [r for r in kept if (r.text or "").strip()]


def _fleurs_rows(cfg: dict) -> list[Row]:
    path = cfg["data"].get("fleurs_manifest")
    if not path:
        from .. import data as gdata  # lazy: pulls in datasets

        res = gdata.fetch_fleurs("train")
        path = res[0] if isinstance(res, (tuple, list)) else res
    p = Path(path)
    if p.is_dir():
        p = p / "manifest.jsonl"
    rows = mf.read(p)
    for r in rows:
        r.source = r.source or "fleurs"
    return rows


def _duration(r: Row) -> float:
    if r.duration:
        return float(r.duration)
    import soundfile as sf

    try:
        r.duration = float(sf.info(r.audio).duration)
    except Exception:  # noqa: BLE001 - non-wav; decoded later anyway
        r.duration = float(len(load_audio(r.audio)) / SAMPLE_RATE)
    return r.duration


def check_leakage(train_rows: list[Row], heldout_paths: list[str | Path]) -> list[str]:
    """Raise ``LeakageError`` on any audio/speaker overlap. Returns warnings."""
    notes = []
    for hp in heldout_paths:
        held = mf.read(hp)
        rep = mf.leakage_check(train_rows, held, normalizer=get_normalizer("standard"))
        if not rep.clean:
            raise LeakageError(
                f"Refusing to train: training data overlaps held-out manifest {hp}: "
                f"{rep.summary()}. Remove those clips/speakers from the training "
                f"exports (held-out is sacred: eval-integrity rule 4)."
            )
        if rep.shared_text:
            notes.append(f"{hp}: {rep.summary()}")
            warnings.warn(f"{hp}: {rep.summary()}", stacklevel=2)
    return notes


def speaker_dev_split(
    rows: list[Row], dev_fraction: float, seed: int = 42
) -> tuple[list[Row], list[Row]]:
    """Split rows into (train, dev) with no speaker in both.

    Whole speakers are moved to dev until it holds about ``dev_fraction`` of the
    utterances. With a single speaker (or no speaker labels at all) that is
    impossible, so it falls back to an utterance split and warns: dev WER is
    then optimistic for unseen voices.
    """
    if dev_fraction <= 0 or len(rows) < 2:
        return list(rows), []
    by_spk: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r.speaker:
            by_spk[r.speaker].append(r)
    rng = random.Random(seed)
    target = max(1, round(dev_fraction * len(rows)))
    if len(by_spk) >= 2:
        unlabelled = [r for r in rows if not r.speaker]
        speakers = sorted(by_spk)
        rng.shuffle(speakers)
        dev_spk: list[str] = []
        n = 0
        # Keep at least one speaker for training.
        for s in speakers[:-1]:
            if n >= target:
                break
            dev_spk.append(s)
            n += len(by_spk[s])
        dev = [r for s in dev_spk for r in by_spk[s]]
        train = [r for s in speakers if s not in dev_spk for r in by_spk[s]] + unlabelled
        return train, dev
    warnings.warn(
        "Fewer than two speakers in the training pool: dev split is by utterance, "
        "not by speaker, so dev WER will be optimistic for new voices.",
        stacklevel=2,
    )
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    dev_idx = set(idx[:min(target, len(rows) - 1)])
    return ([r for i, r in enumerate(rows) if i not in dev_idx],
            [r for i, r in enumerate(rows) if i in dev_idx])


def mix_fleurs(own: list[Row], fleurs: list[Row], fraction: float, seed: int = 42) -> list[Row]:
    """Add FLEURS rows so they make up ``fraction`` of the result."""
    if fraction <= 0 or not fleurs:
        return list(own)
    if not own:
        return list(fleurs)
    n = min(len(fleurs), round(fraction / (1 - fraction) * len(own)))
    rng = random.Random(seed)
    return list(own) + rng.sample(fleurs, n)


def load_audio(path: str) -> np.ndarray:
    try:
        from ..audio import load
    except ImportError:  # pragma: no cover - audio.py is part of the package
        load = None
    if load is not None:
        return load(path, sr=SAMPLE_RATE)
    import soundfile as sf

    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: {sr} Hz; resample to 16 kHz first")
    return a


class WhisperRows:
    """Lazy torch-style dataset: features are computed per item, not up front."""

    def __init__(self, rows: list[Row], processor):
        self.rows, self.processor = rows, processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        audio = load_audio(r.audio)
        feats = self.processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE).input_features[0]
        labels = self.processor.tokenizer(r.text).input_ids[:MAX_LABEL_TOKENS]
        return {"input_features": feats, "labels": labels}


@dataclass
class SpeechCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # The model prepends the start token itself when shifting labels right.
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# --- Model --------------------------------------------------------------------


def _load_model_and_processor(cfg: dict):
    from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

    base, fallback = cfg["base_model"], cfg.get("processor_fallback") or "openai/whisper-small"
    model = WhisperForConditionalGeneration.from_pretrained(base)
    notes = []
    try:
        processor = WhisperProcessor.from_pretrained(base)
    except Exception as e:  # noqa: BLE001 - missing/partial processor files
        processor = WhisperProcessor.from_pretrained(fallback)
        notes.append(f"processor loaded from {fallback} ({base} has none: {type(e).__name__})")

    language, task = cfg.get("language"), cfg.get("task") or "transcribe"
    if language:
        gc = model.generation_config
        if not getattr(gc, "lang_to_id", None):
            # Older fine-tunes save a generation config without the language
            # table; borrow the multilingual one from the matching base Whisper.
            model.generation_config = GenerationConfig.from_pretrained(fallback)
            notes.append(f"generation_config taken from {fallback} (base had no lang_to_id)")
        model.generation_config.language = language
        model.generation_config.task = task
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None
        processor.tokenizer.set_prefix_tokens(language=language, task=task)
        notes.append(f"language/task tokens: <|{language}|> <|{task}|>")
    else:
        notes.append("language: null - base generation_config and tokenizer prefix kept")
    return model, processor, notes


def _training_args(cfg: dict, out: Path, has_dev: bool, cuda: bool):
    from transformers import Seq2SeqTrainingArguments

    t = cfg["training"]
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kw: dict[str, Any] = dict(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        warmup_ratio=float(t["warmup_ratio"]),
        max_steps=int(t["max_steps"]),
        gradient_checkpointing=bool(t["gradient_checkpointing"]),
        fp16=cuda,
        predict_with_generate=True,
        generation_max_length=int(t["generation_max_length"]),
        logging_steps=int(t["logging_steps"]),
        save_total_limit=int(t["save_total_limit"]),
        report_to="none",
        seed=int(cfg["seed"]),
        data_seed=int(cfg["seed"]),
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=int(t.get("dataloader_num_workers", 0)),
    )
    if t["gradient_checkpointing"] and "gradient_checkpointing_kwargs" in params:
        # Non-reentrant checkpointing works with a frozen encoder (no input grads).
        kw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if has_dev:
        steps = int(t["eval_steps"])
        kw.update({
            strategy_key: "steps",
            "eval_steps": steps,
            "save_strategy": "steps",
            "save_steps": steps,
            "load_best_model_at_end": True,
            "metric_for_best_model": "wer",
            "greater_is_better": False,
        })
    else:
        kw.update({strategy_key: "no", "save_strategy": "no"})
    return Seq2SeqTrainingArguments(**kw)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        import torch

        torch.manual_seed(seed)


# --- Train --------------------------------------------------------------------


def prepare_data(cfg: dict) -> dict:
    """Everything before the model: rows, leakage check, dev split, FLEURS mix.

    Kept separate from ``train`` so it can be tested (and run as a dry run)
    without downloading a model.
    """
    d = cfg["data"]
    own: list[Row] = []
    for e in d["exports"]:
        own.extend(read_export(e))
    fleurs = _fleurs_rows(cfg) if float(d.get("fleurs_fraction") or 0) > 0 else []
    if not own and not fleurs:
        raise ValueError("No training data: set data.exports (and/or data.fleurs_fraction).")

    max_s = float(d.get("max_duration_s") or 30.0)
    too_long = [r for r in own if _duration(r) > max_s]
    if too_long:
        warnings.warn(f"dropped {len(too_long)} clip(s) longer than {max_s}s", stacklevel=2)
        own = [r for r in own if r not in too_long]

    heldout = [Path(h) for h in d["heldout"]]
    if not heldout:
        if d.get("require_heldout", True):
            raise ValueError("data.heldout is empty: give at least one held-out manifest "
                             "(or set data.require_heldout: false for a throwaway run).")
        warnings.warn("No held-out manifest: nothing guards against leakage and there "
                      "will be no before/after numbers.", stacklevel=2)
    # Check the whole pool (own + FLEURS candidates) before anything is split.
    text_notes = check_leakage(own + fleurs, heldout)

    seed = int(cfg["seed"])
    if own:
        train_own, dev = speaker_dev_split(own, float(d["dev_fraction"]), seed)
        train_rows = mix_fleurs(train_own, fleurs, float(d.get("fleurs_fraction") or 0), seed)
    else:
        train_rows, dev = speaker_dev_split(fleurs, float(d["dev_fraction"]), seed)
    hours = sum(_duration(r) for r in train_rows) / 3600
    return {
        "train": train_rows,
        "dev": dev,
        "heldout": heldout,
        "hours": hours,
        "own_hours": sum(_duration(r) for r in train_rows if r.source != "fleurs") / 3600,
        "speakers": len({r.speaker for r in train_rows if r.speaker}),
        "dev_speakers": len({r.speaker for r in dev if r.speaker}),
        "n_fleurs": sum(1 for r in train_rows if r.source == "fleurs"),
        "text_overlap_warnings": text_notes,
    }


def train(config: str | Path | dict | None = None) -> dict:
    """Fine-tune per config and return a summary dict (paths and numbers)."""
    import torch
    from transformers import EarlyStoppingCallback, Seq2SeqTrainer

    cfg = load_config(config)
    seed = int(cfg["seed"])
    _seed_everything(seed)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log.info("attempt %s: config #%s tried for this model", cfg["attempt"], cfg["attempt"])

    data = prepare_data(cfg)
    model, processor, notes = _load_model_and_processor(cfg)

    t = cfg["training"]
    freeze = t.get("freeze_encoder", "auto")
    if freeze == "auto":
        freeze = data["hours"] < float(t.get("freeze_encoder_below_hours", 5.0))
    if freeze:
        model.freeze_encoder()
    if t.get("spec_augment"):
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = float(t.get("mask_time_prob", 0.05))
    model.config.use_cache = False  # incompatible with gradient checkpointing

    norm = get_normalizer("standard")
    tok = processor.tokenizer

    def compute_metrics(pred) -> dict:
        pred_ids, label_ids = pred.predictions, pred.label_ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids = np.where(pred_ids < 0, tok.pad_token_id, pred_ids)
        label_ids = np.where(label_ids == -100, tok.pad_token_id, label_ids)
        hyps = tok.batch_decode(pred_ids, skip_special_tokens=True)
        refs = tok.batch_decode(label_ids, skip_special_tokens=True)
        s = score_corpus(range(len(refs)), refs, hyps, normalizer=norm)
        wer = s.wer if not math.isnan(s.wer) else 1.0
        return {"wer": wer, "cer": s.cer}

    cuda = torch.cuda.is_available()
    has_dev = bool(data["dev"])
    args = _training_args(cfg, out, has_dev, cuda)
    callbacks = []
    if has_dev and int(t.get("early_stopping_patience") or 0) > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(t["early_stopping_patience"])))
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=WhisperRows(data["train"], processor),
        eval_dataset=WhisperRows(data["dev"], processor) if has_dev else None,
        data_collator=SpeechCollator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics if has_dev else None,
        callbacks=callbacks,
    )
    train_out = trainer.train()
    dev_metrics = trainer.evaluate() if has_dev else {}

    model_dir = out / "model"
    model.config.use_cache = True
    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))

    summary = {
        "attempt": cfg["attempt"],
        "seed": seed,
        "base_model": cfg["base_model"],
        "language": cfg.get("language"),
        "output_dir": str(out),
        "model_dir": str(model_dir),
        "train_utterances": len(data["train"]),
        "dev_utterances": len(data["dev"]),
        "train_hours": data["hours"],
        "own_hours": data["own_hours"],
        "fleurs_utterances": data["n_fleurs"],
        "speakers": data["speakers"],
        "dev_speakers": data["dev_speakers"],
        "encoder_frozen": bool(freeze),
        "global_step": int(train_out.global_step),
        "train_loss": float(train_out.training_loss),
        "dev_wer": dev_metrics.get("eval_wer"),
        "dev_cer": dev_metrics.get("eval_cer"),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "heldout": [str(h) for h in data["heldout"]],
        "rules_version": RULES_VERSION,
        "notes": notes + data["text_overlap_warnings"],
        "config": cfg,
        "comparisons": [],
        "before_after": None,
    }
    if cfg["eval"].get("after") and data["heldout"]:
        summary["comparisons"] = _evaluate_after(cfg, model_dir, data["heldout"], out)
        summary["before_after"] = str(write_before_after(out / "before_after.md", summary))
    write_model_card(model_dir / "README.md", summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


# --- After training -----------------------------------------------------------


def _evaluate_after(cfg: dict, model_dir: Path, heldout: list[Path], out: Path) -> list[dict]:
    from .. import evaluate as ev  # lazy: owns model loading for eval

    e = cfg["eval"]
    tag = f"{cfg.get('model_name') or 'tuned'}-a{cfg['attempt']}"
    results = []
    for h in heldout:
        stem = h.stem
        # Same language forcing as training and as the published baseline run:
        # without it, a config with language: null would still be decoded with
        # forced Hausa here, and before/after would not match the baseline.
        kw = dict(out_dir=str(out / "eval"), batch_size=int(e.get("batch_size") or 8),
                  limit=e.get("limit"), language=cfg.get("language"))
        base_run = ev.run(cfg["base_model"], str(h), name=f"base-{stem}", **kw)
        tuned_run = ev.run(str(model_dir), str(h), name=f"{tag}-{stem}", **kw)
        cmp = ev.compare(tuned_run, base_run, mode="standard")
        results.append({
            "heldout": str(h),
            "base_run": str(base_run),
            "tuned_run": str(tuned_run),
            "compare": cmp,
            "sentence": verdict(cmp, "tuned", "base"),
        })
    return results


def verdict(cmp: dict, a: str = "tuned", b: str = "base") -> str:
    """``PairedResult.sentence`` from whatever ``evaluate.compare`` returned."""
    fields = set(PairedResult.__dataclass_fields__)
    for src in (cmp, cmp.get("paired") if isinstance(cmp, dict) else None,
                cmp.get("standard") if isinstance(cmp, dict) else None,
                (cmp.get("wer") if isinstance(cmp, dict) else None)):
        if isinstance(src, dict) and fields <= src.keys():
            return PairedResult(**{k: src[k] for k in fields}).sentence(a, b)
    if isinstance(cmp, dict) and isinstance(cmp.get("sentence"), str):
        return cmp["sentence"]
    return "Comparison result not understood; see the JSON below."


def write_before_after(path: Path, summary: dict) -> Path:
    lines = [
        "# Before / after fine-tuning",
        "",
        f"Base model: `{summary['base_model']}`. Tuned model: `{summary['model_dir']}`.",
        f"Configs tried for this model so far: **{summary['attempt']}** (`attempt`). "
        "Hyperparameters were chosen on the speaker-disjoint dev split; each held-out "
        "set below is evaluated once.",
        f"Training data: {summary['train_utterances']} utterances, "
        f"{summary['train_hours']:.2f} h ({summary['own_hours']:.2f} h own audio, "
        f"{summary['fleurs_utterances']} FLEURS train utterances), "
        f"{summary['speakers']} speakers. Normalisation rules v{summary['rules_version']} "
        "(`standard` mode).",
        "",
    ]
    for c in summary["comparisons"]:
        lines += [f"## {Path(c['heldout']).name}", "", c["sentence"], "",
                  "<details><summary>Full comparison</summary>", "", "```json",
                  json.dumps(c["compare"], indent=2, default=str), "```", "", "</details>", ""]
    lines += ["", ATTRIBUTION, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_model_card(path: Path, s: dict) -> Path:
    cfg = s["config"]
    name = cfg.get("model_name") or "hausa-asr-gyara"
    natlas = str(s["base_model"]).startswith("NCAIR1/")
    title = f"{name} (Powered by Awarri)" if natlas else name
    licence = (
        "This model is a fine-tune of an N-ATLAS model and is released under the "
        "**N-ATLAS licence** of its base model. Free use is capped at 1,000 active "
        "end-users (rolling 30 days); derivatives keep the same licence; renamed "
        "derivatives must carry the suffix \"Powered by Awarri\"."
        if natlas else
        f"This model inherits the licence of its base model `{s['base_model']}`."
    )
    results = "\n".join(f"- **{Path(c['heldout']).name}**: {c['sentence']}"
                        for c in s["comparisons"]) or "- Not evaluated on held-out data yet."
    dev = (f"{s['dev_wer'] * 100:.2f}% WER (standard)" if s.get("dev_wer") is not None
           else "no dev split")
    card = f"""---
language: [ha]
base_model: {s['base_model']}
tags: [automatic-speech-recognition, whisper, hausa, gyara]
---

# {title}

Fine-tuned from [`{s['base_model']}`](https://huggingface.co/{s['base_model']}) with
[Gyara](https://github.com/) on human-verified Hausa transcripts.

## Training data

- {s['train_utterances']} utterances, {s['train_hours']:.2f} h
  ({s['own_hours']:.2f} h own audio, {s['fleurs_utterances']} FLEURS ha_ng train utterances)
- {s['speakers']} training speakers; dev split: {s['dev_utterances']} utterances,
  {s['dev_speakers']} speakers, disjoint from training speakers
- Leakage check against every held-out manifest passed (no shared audio, no shared speakers)

## Training

- Language/task tokens: `{s['language']}` / `{cfg.get('task')}`
- Steps: {s['global_step']}, LR {cfg['training']['learning_rate']}, encoder frozen: {s['encoder_frozen']}
- Seed {s['seed']}; configs tried so far (`attempt`): {s['attempt']}
- Best dev result: {dev}
- Notes: {'; '.join(s['notes']) or 'none'}

## Held-out results (paired bootstrap vs base, `standard` normalisation v{s['rules_version']})

{results}

## Licence

{licence}

{ATTRIBUTION}
"""
    path.write_text(card, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="YAML config (merged over the defaults)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    s = train(a.config)
    print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=2, default=str))


if __name__ == "__main__":
    main()
