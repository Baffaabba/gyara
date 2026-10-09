"""N-ATLaS proofreading suggestions for Hausa transcripts, behind a guard.

N-ATLaS (``NCAIR1/N-ATLaS``, a Llama-3 8B fine-tune by Awarri) reads *text*,
not audio. It can fix what a speech model gets wrong in writing (missing hooked
letters, apostrophes, capitals, punctuation), but it cannot know what was said.
Left alone, an LLM will also "improve" a correct but unusual word into a more
common one, or answer in English, or chat. So every suggestion passes a guard
before anyone sees it, and nothing is ever applied automatically: the UI shows
the suggestion, a human accepts or rejects it (hard rule 3).

The guard rejects a suggestion when:

* the output is chatter rather than a corrected sentence ("Here is...",
  explanation lines, English prose);
* the number of words changes, except a pure split/merge (``a kan`` ->
  ``akan``) of at most one word;
* an aligned word changes by more than ``MAX_WORD_EDIT`` normalised character
  edit distance, unless the change is only hooks, apostrophes, case or
  punctuation (those are exactly what we want it to fix);
* too many words change at all.

Backends:

``none``          suggestions off; returns the text unchanged.
``transformers``  loads N-ATLaS locally. 4-bit (bitsandbytes) on CUDA when
                  available, fp16 on CUDA otherwise. On CPU an 8B model is
                  impractical (minutes per sentence); it is allowed, with a
                  warning.
``openai``        posts to ``{base_url}/v1/chat/completions``. This is how
                  N-ATLaS runs without a GPU: community GGUF builds
                  (``tosinamuda/N-ATLaS-GGUF``, ``Abu-Dju/N-ATLaS-Q8_0-GGUF``)
                  served by llama.cpp on a laptop CPU::

                      llama-server -m N-ATLaS-Q4_K_M.gguf --port 8080
                      Suggester("openai", base_url="http://localhost:8080")

                  vLLM and Ollama expose the same endpoint.

N-ATLAS is an initiative of the Federal Ministry of Communications, Innovation
and Digital Economy, and powered by Awarri Technologies.
"""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from rapidfuzz.distance import Levenshtein

from . import DEFAULT_LLM_MODEL
from .normalize import get_normalizer

BACKENDS = ("none", "transformers", "openai")

# A word may change by at most this share of its characters (normalised
# Levenshtein on the ``standard`` form) unless the change is purely cosmetic.
# 0.4 lets "kasa"->"kassa" or a one-letter typo through and stops "kasa"->"duniya".
MAX_WORD_EDIT = 0.4
# Even if each change is small, an output that touches most of the words is a
# rewrite, not a proofread.
MAX_CHANGED_WORD_SHARE = 0.5

# The identity line from the N-ATLaS model card's own system prompt. The model
# was tuned with it, so we keep it and add our task after it.
NATLAS_IDENTITY = (
    "You are a large language model trained by Awarri AI technologies. "
    "You are a friendly assistant and you are here to help."
)

PROOFREAD_INSTRUCTIONS = (
    "You are a strict Hausa proofreader for speech-to-text transcripts. "
    "Fix ONLY: spelling, hooked letters (ɓ ɗ ƙ 'y), apostrophes, capitalisation "
    "and punctuation. Never add, remove, reorder or translate words. Never "
    "change a word into a different word. If the text is already correct, "
    "return it unchanged. Reply with the corrected Hausa text only: no "
    "explanation, no quotes, no English."
)

# Tiny, uncontroversial examples. Each shows one kind of fix and the last shows
# "already correct -> unchanged", which matters most.
FEW_SHOT: list[tuple[str, str]] = [
    ("yan kasa sun dauki mataki", "'Yan ƙasa sun ɗauki mataki."),
    ("ina kwana malam ya aiki", "Ina kwana malam, ya aiki?"),
    ("Mun gode da zuwanku.", "Mun gode da zuwanku."),
]

TRANSLATE_INSTRUCTIONS = (
    "Translate the following Hausa text into clear, natural English for "
    "subtitles. Reply with the English translation only."
)

# Prefixes models put in front of the answer. Stripped, not rejected.
_PREFIX_RE = re.compile(
    r"^\s*(corrected(\s+text)?|correction|output|answer|gyara|gyararre|rubutu)\s*[:：]\s*",
    re.IGNORECASE,
)
_QUOTES = "\"'`“”‘’«»"
# Openers that mean the model is talking to us, not returning the sentence.
_CHATTER_START = re.compile(
    r"^\s*(here|sure|certainly|okay|ok,|note|the corrected|the text|i |i'm|as an)\b",
    re.IGNORECASE,
)
# English function words that do not double as common Hausa words. ("to",
# "a", "in", "da" are Hausa words and are deliberately absent.)
_ENGLISH = {
    "the", "is", "are", "was", "this", "that", "here", "corrected", "correct",
    "text", "sentence", "of", "and", "with", "your", "you", "i", "translation",
    "means", "word", "words", "please", "it", "has", "have", "been",
}

_standard = get_normalizer("standard")
_lenient = get_normalizer("lenient")


@dataclass
class Suggestion:
    original: str
    suggested: str
    edits: list[dict] = field(default_factory=list)  # {op, from, to, index}
    accepted_by_guard: bool = False
    reason: str = ""
    model: str = ""

    @property
    def changed(self) -> bool:
        return self.accepted_by_guard and self.suggested != self.original

    def to_dict(self) -> dict:
        return asdict(self)


# --- Prompt -------------------------------------------------------------------


def build_messages(text: str, few_shot: bool = True) -> list[dict]:
    """Chat messages for one proofreading request (system, few-shot, user)."""
    msgs = [{"role": "system", "content": f"{NATLAS_IDENTITY}\n\n{PROOFREAD_INSTRUCTIONS}"}]
    if few_shot:
        for src, dst in FEW_SHOT:
            msgs.append({"role": "user", "content": src})
            msgs.append({"role": "assistant", "content": dst})
    msgs.append({"role": "user", "content": text})
    return msgs


def build_translate_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": f"{NATLAS_IDENTITY}\n\n{TRANSLATE_INSTRUCTIONS}"},
        {"role": "user", "content": text},
    ]


# --- Guard --------------------------------------------------------------------


def clean_output(raw: str) -> str:
    """Strip the wrapping a model adds around an answer (prefix, quotes, fences)."""
    t = (raw or "").strip()
    t = re.sub(r"^```\w*\s*|\s*```$", "", t).strip()
    t = _PREFIX_RE.sub("", t).strip()
    # Matching quotes around the whole answer; never strip a leading 'y apostrophe.
    while len(t) >= 2 and t[0] in _QUOTES and t[-1] in _QUOTES and t[:2].lower() != "'y":
        t = t[1:-1].strip()
    return t


def word_edits(original: str, suggested: str) -> list[dict]:
    """Word-level edit list between two texts (whitespace tokens)."""
    a, b = original.split(), suggested.split()
    out = []
    for op in Levenshtein.editops(a, b):
        out.append({
            "op": op.tag,
            "from": a[op.src_pos] if op.tag != "insert" else "",
            "to": b[op.dest_pos] if op.tag != "delete" else "",
            "index": op.src_pos,
        })
    return out


def _cosmetic(a: str, b: str) -> bool:
    """True when two words differ only by hooks, apostrophes, case or punctuation."""
    return _lenient(a) == _lenient(b)


def _chatter(original: str, out: str) -> str | None:
    if not out:
        return "empty output"
    if "\n" in out.strip() and "\n" not in original.strip():
        return "output has extra lines (explanation?)"
    if _CHATTER_START.match(out) and not _CHATTER_START.match(original):
        return "output is chatter, not a corrected sentence"
    orig_words = {w.lower() for w in _standard(original).split()}
    new_english = [w for w in _standard(out).split() if w in _ENGLISH and w not in orig_words]
    if len(new_english) >= 2:
        return "output looks like English, not Hausa"
    return None


def guard(original: str, raw_output: str, model: str = "") -> Suggestion:
    """Decide whether a model output may be shown as a suggestion.

    On rejection the suggestion equals the original, so a careless caller that
    applies ``.suggested`` still changes nothing.
    """
    out = clean_output(raw_output)

    def reject(reason: str) -> Suggestion:
        return Suggestion(original, original, [], False, reason, model)

    why = _chatter(original, out)
    if why:
        return reject(why)

    a, b = _standard(original).split(), _standard(out).split()
    if abs(len(a) - len(b)) > 1:
        return reject(f"word count changed {len(a)} -> {len(b)}")
    if len(a) != len(b):
        # Only a split/merge is allowed: the letters must be the same.
        if "".join(_lenient(original).split()) != "".join(_lenient(out).split()):
            return reject("adds or removes a word")
    else:
        changed = 0
        for wa, wb in zip(a, b):
            if wa == wb:
                continue
            if _cosmetic(wa, wb):
                continue
            changed += 1
            d = Levenshtein.normalized_distance(wa, wb)
            if d > MAX_WORD_EDIT:
                return reject(f"rewrites a word: {wa!r} -> {wb!r}")
        # Only real (non-cosmetic) spelling changes count: fixing every hook in
        # a sentence is the point, respelling half its words is not.
        if a and changed / len(a) > MAX_CHANGED_WORD_SHARE and changed > 1:
            return reject(f"changes too many words ({changed}/{len(a)})")

    edits = word_edits(original, out)
    reason = "no change" if out == original else "ok"
    return Suggestion(original, out, edits, True, reason, model)


# --- Suggester ----------------------------------------------------------------


class Suggester:
    """Proofreading suggestions from N-ATLaS (or any chat LLM) behind ``guard``.

    ``client`` (an ``httpx.Client``) can be passed for the ``openai`` backend;
    tests use it with ``httpx.MockTransport``.
    """

    def __init__(
        self,
        backend: str = "none",
        model_id: str = DEFAULT_LLM_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        load_in_4bit: bool = True,
        device: str | None = None,
        max_new_tokens: int = 256,
        # Greedy decoding copies the input; a repetition penalty > 1 would
        # penalise exactly the tokens a proofreader must repeat, so it is off
        # (1.0) by default even though the card suggests 1.12 for chat.
        repetition_penalty: float = 1.0,
        timeout: float = 120.0,
        client: Any = None,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
        self.backend = backend
        self.model_id = model_id
        self.base_url = base_url or os.environ.get("GYARA_LLM_BASE_URL")
        self.api_key = api_key or os.environ.get("GYARA_LLM_API_KEY")
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.timeout = timeout
        self._client = client
        self._model = None
        self._tok = None
        self._cache: dict[str, Any] = {}
        if backend == "openai" and not self.base_url:
            raise ValueError("the openai backend needs base_url (e.g. http://localhost:8080)")

    # -- public API --

    def suggest(self, text: str) -> Suggestion:
        if self.backend == "none":
            return Suggestion(text, text, [], False, "suggestions off", "none")
        if not text or not text.strip():
            return Suggestion(text, text, [], False, "empty text", self.model_id)
        key = self._key("suggest", text)
        if key not in self._cache:
            raw = self._chat(build_messages(text))
            self._cache[key] = guard(text, raw, self.model_id)
        return self._cache[key]

    def suggest_many(self, texts: Iterable[str]) -> list[Suggestion]:
        return [self.suggest(t) for t in texts]

    def translate(self, text: str) -> str:
        """Hausa -> English, for subtitles. Unguarded: the output is a new language.

        Returns "" when suggestions are off, so callers can skip the track.
        """
        if self.backend == "none" or not text or not text.strip():
            return ""
        key = self._key("translate", text)
        if key not in self._cache:
            self._cache[key] = clean_output(self._chat(build_translate_messages(text)))
        return self._cache[key]

    # -- backends --

    def _key(self, kind: str, text: str) -> str:
        return kind + ":" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _chat(self, messages: list[dict]) -> str:
        if self.backend == "openai":
            return self._chat_openai(messages)
        return self._chat_transformers(messages)

    def _chat_openai(self, messages: list[dict]) -> str:
        import httpx

        base = self.base_url.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
            "stream": False,
        }
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        finally:
            if self._client is None:
                client.close()

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available() and self.device != "cpu"
        kwargs: dict[str, Any] = {}
        if cuda:
            bnb = False
            if self.load_in_4bit:
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                    bnb = True
                except ImportError:
                    warnings.warn("bitsandbytes not installed; loading N-ATLaS in fp16 "
                                  "(needs ~16 GB GPU memory).", stacklevel=3)
            if not bnb:
                kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = self.device or "auto"
        else:
            warnings.warn(
                "Loading an 8B LLM on CPU: expect minutes per sentence and ~32 GB RAM. "
                "Use the 'openai' backend with a GGUF build under llama.cpp instead.",
                stacklevel=3,
            )
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()

    def _chat_transformers(self, messages: list[dict]) -> str:
        import torch
        from datetime import datetime

        if self._model is None:
            self._load_transformers()
        tok, model = self._tok, self._model
        # The N-ATLaS card renders its Llama-3 template with a date_string;
        # templates that do not use it ignore the extra variable.
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            date_string=datetime.now().strftime("%d %b %Y"),
        )
        # The template already contains <|begin_of_text|>; do not add it twice.
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        eos = [tok.eos_token_id]
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot != tok.unk_token_id:
            eos.append(eot)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=eos,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos[0],
            )
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
