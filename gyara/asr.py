"""Whisper-family transcription with confidence scores and loop protection.

We call ``WhisperForConditionalGeneration.generate`` directly instead of the
``pipeline`` helper for three reasons:

* **Confidence.** ``avg_logprob`` per segment (mean log-probability of the
  generated tokens) drives "least confident first" in the correction UI, and
  ``evaluate`` checks that it actually correlates with WER before anyone
  markets the feature. The pipeline throws the scores away.
* **Loop control.** Each decoded chunk is checked by ``gyara.hallucination``.
  A flagged chunk is re-decoded with temperature fallback (as OpenAI's Whisper
  does) plus ``no_repeat_ngram_size``; if it is still flagged it is trimmed and
  marked ``loop`` for a human. Nothing flagged is accepted silently.
* **Language forcing is explicit.** ``language="hausa"`` forces the Hausa
  token. Some fine-tunes (``NCAIR1/Hausa-ASR``'s card calls ``generate()``
  with no language) may have been trained without it, so ``language=None``
  lets the model use its own generation config. The setting is recorded in
  every eval run, because it can move WER a lot.

Long audio is cut by ``gyara.vad`` into ≤30 s speech chunks, decoded in
batches.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gyara import DEFAULT_ASR_MODEL, hallucination, vad
from gyara import audio as audio_mod

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MAX_CHUNK_S = 30.0
FALLBACK_TEMPERATURES = (0.2, 0.4, 0.6)
NO_REPEAT_NGRAM = 4
TOKENS_PER_SECOND = 8  # Hausa is ~2-3.5 words/s; Whisper BPE ~2-3 tokens/word
MAX_NEW_TOKENS_CAP = 440  # decoder has 448 positions, minus the 4 prompt tokens
FALLBACK_PROCESSOR = "openai/whisper-small"


def max_new_tokens_for(duration_s: float) -> int:
    """Token budget for a chunk: generous for real speech, fatal for loops."""
    return int(min(MAX_NEW_TOKENS_CAP, TOKENS_PER_SECOND * max(duration_s, 0.0) + 20))


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    flags: list[str] = field(default_factory=list)
    # [{text, start, end}] when the model gives real word times; None means
    # "not available" and subtitles.py estimates them from character lengths.
    words: list[dict] | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


# --- Model loading (lazy, shared) ---------------------------------------------

_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_processor(model_id: str):
    from transformers import WhisperProcessor

    try:
        return WhisperProcessor.from_pretrained(model_id)
    except (OSError, ValueError, TypeError) as e:
        # Some fine-tunes ship weights only. Whisper-small's tokenizer and
        # feature extractor are identical for every whisper-small fine-tune.
        log.warning("No processor in %s (%s); using %s", model_id, e, FALLBACK_PROCESSOR)
        return WhisperProcessor.from_pretrained(FALLBACK_PROCESSOR)


def load_model(model_id: str, device: str, dtype_name: str):
    """Load (processor, model) once per (model_id, device, dtype)."""
    key = (model_id, device, dtype_name)
    with _CACHE_LOCK:
        if key not in _CACHE:
            import torch
            from transformers import WhisperForConditionalGeneration

            processor = _load_processor(model_id)
            # use_safetensors left at its default (None) so repos that ship only
            # pytorch_model.bin (NCAIR1/Hausa-ASR) load too.
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=getattr(torch, dtype_name)
            )
            model.to(device).eval()
            _CACHE[key] = (processor, model)
        return _CACHE[key]


_INSTANCES: dict[tuple, "Transcriber"] = {}


def get_transcriber(model_id: str = DEFAULT_ASR_MODEL, device: str | None = None, **kw) -> "Transcriber":
    """Shared Transcriber per (model_id, device, options) for the UI and CLI."""
    key = (model_id, _resolve_device(device), tuple(sorted(kw.items())))
    if key not in _INSTANCES:
        _INSTANCES[key] = Transcriber(model_id, device=device, **kw)
    return _INSTANCES[key]


# --- Transcriber ---------------------------------------------------------------


class Transcriber:
    def __init__(
        self,
        model_id: str = DEFAULT_ASR_MODEL,
        device: str | None = None,
        language: str | None = "hausa",
        task: str = "transcribe",
        batch_size: int = 8,
        dtype: str | None = None,
        num_beams: int = 1,
    ):
        self.model_id = model_id
        self.device = _resolve_device(device)
        self.language = language
        self.task = task
        self.batch_size = max(1, int(batch_size))
        # fp16 halves memory and doubles speed on GPU; CPU fp16 is slow/unsupported.
        self.dtype = dtype or ("float16" if self.device.startswith("cuda") else "float32")
        self.num_beams = num_beams
        self.language_used: str | None = None  # what was actually forced, set on load
        self._processor = None
        self._model = None

    # -- loading

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._processor, self._model = load_model(self.model_id, self.device, self.dtype)
        self.language_used = self._setup_language()

    def _setup_language(self) -> str | None:
        """Make ``generate(language=...)`` work, or degrade to model default.

        Fine-tunes saved with an old transformers often lack ``lang_to_id`` in
        their generation config, and then ``generate(language=...)`` raises. We
        rebuild the maps from the tokenizer's special tokens when we can.
        """
        if not self.language:
            return None
        gc = self._model.generation_config
        if getattr(gc, "lang_to_id", None) and getattr(gc, "task_to_id", None):
            return self.language
        try:
            from transformers.models.whisper.tokenization_whisper import LANGUAGES

            tok = self._processor.tokenizer
            unk = tok.unk_token_id
            lang_to_id = {}
            for code in LANGUAGES:
                tid = tok.convert_tokens_to_ids(f"<|{code}|>")
                if tid is not None and tid != unk:
                    lang_to_id[f"<|{code}|>"] = tid
            task_to_id = {t: tok.convert_tokens_to_ids(f"<|{t}|>") for t in ("transcribe", "translate")}
            if "<|ha|>" not in lang_to_id or unk in task_to_id.values():
                raise ValueError("tokenizer has no Hausa/task tokens")
            gc.lang_to_id, gc.task_to_id, gc.is_multilingual = lang_to_id, task_to_id, True
            if getattr(gc, "no_timestamps_token_id", None) is None:
                gc.no_timestamps_token_id = tok.convert_tokens_to_ids("<|notimestamps|>")
            log.info("Rebuilt lang_to_id/task_to_id for %s from its tokenizer", self.model_id)
            return self.language
        except Exception as e:  # noqa: BLE001 - any failure means "cannot force"
            log.warning(
                "%s cannot force language=%r (%s); decoding with the model's own settings",
                self.model_id, self.language, e,
            )
            return None

    def info(self) -> dict:
        """Decoding settings, for run metadata."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "language": self.language,
            "language_used": self.language_used,
            "task": self.task,
            "num_beams": self.num_beams,
            "batch_size": self.batch_size,
            "fallback_temperatures": list(FALLBACK_TEMPERATURES),
            "no_repeat_ngram_size_on_retry": NO_REPEAT_NGRAM,
        }

    # -- decoding

    def _generate(self, arrays: Sequence[np.ndarray], **extra) -> list[tuple[str, float | None]]:
        """One ``generate`` call over ≤30 s arrays → [(text, avg_logprob)]."""
        import torch

        self._ensure_loaded()
        proc, model = self._processor, self._model
        feats = proc.feature_extractor(
            [np.asarray(a, dtype=np.float32) for a in arrays],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        kwargs: dict[str, Any] = {
            "input_features": feats.input_features.to(self.device, dtype=model.dtype),
            "max_new_tokens": max_new_tokens_for(max(len(a) for a in arrays) / SAMPLE_RATE),
            "output_scores": True,
            "return_dict_in_generate": True,
            "num_beams": self.num_beams,
        }
        if "attention_mask" in feats:
            kwargs["attention_mask"] = feats.attention_mask.to(self.device)
        if self.language_used:
            kwargs["language"] = self.language_used
            kwargs["task"] = self.task
        kwargs.update(extra)
        with torch.inference_mode():
            out = model.generate(**kwargs)
        texts = proc.batch_decode(out.sequences, skip_special_tokens=True)
        logprobs = self._avg_logprobs(out)
        return [(t.strip(), lp) for t, lp in zip(texts, logprobs)]

    def _avg_logprobs(self, out) -> list[float | None]:
        """Mean token log-prob of each generated sequence, up to and incl. EOS."""
        n = out.sequences.shape[0]
        try:
            scores = self._model.compute_transition_scores(
                out.sequences, out.scores, getattr(out, "beam_indices", None), normalize_logits=True
            ).float().cpu().numpy()
        except Exception as e:  # noqa: BLE001 - confidence is optional, text is not
            log.debug("compute_transition_scores failed: %s", e)
            return [None] * n
        gen = out.sequences[:, -scores.shape[1]:].cpu().numpy()
        eos = self._model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        res: list[float | None] = []
        for toks, sc in zip(gen, scores):
            keep = []
            for t, s in zip(toks, sc):
                if np.isfinite(s):
                    keep.append(s)
                if int(t) in eos_ids:
                    break
            res.append(float(np.mean(keep)) if keep else None)
        return res

    def _decode_checked(self, arrays: Sequence[np.ndarray]) -> list[Segment]:
        """Batched greedy decode, then per-chunk loop check and fallback."""
        out: list[Segment] = []
        for i in range(0, len(arrays), self.batch_size):
            batch = arrays[i : i + self.batch_size]
            for arr, (text, lp) in zip(batch, self._generate(batch)):
                out.append(self._check_and_retry(arr, text, lp))
        return out

    def _check_and_retry(self, arr: np.ndarray, text: str, lp: float | None) -> Segment:
        dur = len(arr) / SAMPLE_RATE
        seg = Segment(0.0, dur, text, lp, hallucination.compression_ratio(text), [])
        first = hallucination.check(text, dur)
        if not first.flagged:
            return seg
        import torch

        for temp in FALLBACK_TEMPERATURES:
            torch.manual_seed(int(temp * 1000))  # reproducible sampling
            (t2, lp2), = self._generate(
                [arr], do_sample=True, temperature=temp, no_repeat_ngram_size=NO_REPEAT_NGRAM,
                num_beams=1,
            )
            if not hallucination.check(t2, dur).flagged:
                return Segment(0.0, dur, t2, lp2, hallucination.compression_ratio(t2),
                               [f"retried_t{temp}"])
        trimmed = hallucination.trim_loop(text)
        return Segment(0.0, dur, trimmed, lp, hallucination.compression_ratio(trimmed),
                       ["loop"])

    # -- public API

    def transcribe_batch(self, arrays: Sequence[np.ndarray]) -> list[Segment]:
        """Decode pre-cut clips (each ≤30 s, 16 kHz). Used by ``evaluate``."""
        for a in arrays:
            if len(a) > MAX_CHUNK_S * SAMPLE_RATE + 1:
                raise ValueError("transcribe_batch takes clips of at most 30 s; use transcribe()")
        return self._decode_checked(list(arrays))

    def transcribe(self, path_or_array: str | Path | bytes | np.ndarray, sr: int = SAMPLE_RATE) -> list[Segment]:
        """Transcribe a file or array of any length into timed segments."""
        if isinstance(path_or_array, np.ndarray):
            wav = path_or_array.astype(np.float32, copy=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1 if wav.shape[-1] <= 8 else 0)
            if sr != SAMPLE_RATE:
                wav = audio_mod.resample(wav, sr, SAMPLE_RATE)
        else:
            wav = audio_mod.load(path_or_array, SAMPLE_RATE)
        regions = vad.chunk(wav, SAMPLE_RATE, max_s=MAX_CHUNK_S)
        pieces = [wav[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)] for a, b in regions]
        segs = self._decode_checked(pieces)
        for (a, b), s in zip(regions, segs):
            s.start, s.end = a, b
            if not s.text:
                s.flags.append("empty")  # VAD heard speech, model wrote nothing
        return segs
