"""Cut long audio into Whisper-sized speech chunks.

Whisper sees at most 30 s at a time. Cutting blindly every 30 s splits words
in half and feeds the model long silences, which is exactly where it
hallucinates ("na gode na gode ..."). So we find speech with a voice activity
detector, pad each region a little (VADs clip word onsets), merge neighbouring
regions greedily up to ``max_s`` so the model gets context, and hard-split any
single region that is still too long.

Silero VAD is used when installed; a simple energy detector is the fallback so
the toolkit still works in a minimal install. Neither may lose audio silently:
if no speech is found at all (very quiet recording, music bed, a VAD failure)
we return fixed 30 s windows over the whole file and let the transcriber and a
human decide.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)

PAD_S = 0.2

Region = tuple[float, float]


# --- Detectors ----------------------------------------------------------------


@lru_cache(maxsize=1)
def _silero_model():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def silero_regions(audio: np.ndarray, sr: int = 16000) -> list[Region]:
    """Speech regions in seconds from Silero VAD. Raises ImportError if absent."""
    import torch
    from silero_vad import get_speech_timestamps

    if sr not in (8000, 16000):
        from gyara.audio import resample

        audio, sr = resample(audio, sr, 16000), 16000
    ts = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
        _silero_model(),
        sampling_rate=sr,
    )
    return [(t["start"] / sr, t["end"] / sr) for t in ts]


def energy_regions(
    audio: np.ndarray,
    sr: int = 16000,
    frame_s: float = 0.03,
    min_speech_s: float = 0.25,
    min_silence_s: float = 0.3,
) -> list[Region]:
    """Frame-energy VAD: frames well above the noise floor are speech.

    The threshold adapts to the recording: 10 dB above the 10th-percentile
    frame energy, but never below -50 dBFS (so near-digital silence with a tiny
    hiss is not called speech). Gaps shorter than ``min_silence_s`` are bridged
    and blips shorter than ``min_speech_s`` dropped.
    """
    hop = max(int(sr * frame_s), 1)
    n = len(audio) // hop
    if n == 0:
        return []
    frames = np.asarray(audio[: n * hop], dtype=np.float64).reshape(n, hop)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    floor = np.percentile(db, 10)
    # If the clip is nearly all speech the 10th percentile *is* speech; keep the
    # threshold at least 15 dB under the loud frames so we do not miss it all.
    thr = max(min(floor + 10.0, np.percentile(db, 95) - 15.0), -50.0)
    active = db > thr
    regions: list[list[float]] = []
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            regions.append([i * frame_s, j * frame_s])
            i = j
        else:
            i += 1
    merged: list[list[float]] = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < min_silence_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_speech_s]


def detect(audio: np.ndarray, sr: int = 16000, backend: str = "auto") -> tuple[list[Region], str]:
    """Return (regions, backend_used). ``backend`` is auto, silero or energy."""
    if backend in ("auto", "silero"):
        try:
            return silero_regions(audio, sr), "silero"
        except ImportError:
            if backend == "silero":
                raise
            log.info("silero-vad not installed; using energy VAD")
    return energy_regions(audio, sr), "energy"


# --- Chunking -------------------------------------------------------------------


def fixed_windows(total_s: float, max_s: float = 30.0) -> list[Region]:
    out, t = [], 0.0
    while t < total_s - 1e-6:
        out.append((t, min(t + max_s, total_s)))
        t += max_s
    return out


def _split_long(a: float, b: float, max_s: float) -> list[Region]:
    """Split [a, b] into equal pieces of at most ``max_s`` seconds."""
    k = int(np.ceil((b - a) / max_s - 1e-9))
    if k <= 1:
        return [(a, b)]
    step = (b - a) / k
    return [(a + i * step, a + (i + 1) * step) for i in range(k)]


def plan(
    regions: list[Region], total_s: float, max_s: float = 30.0, min_s: float = 1.0,
    pad_s: float = PAD_S,
) -> list[Region]:
    """Pad, merge greedily up to ``max_s``, hard-split, and widen tiny chunks.

    Pure function on region lists, so it is testable without any detector.
    """
    if total_s <= 0:
        return []
    if not regions:
        return fixed_windows(total_s, max_s)
    # 1. pad and clip, then fuse overlaps the padding created
    padded: list[list[float]] = []
    for a, b in sorted(regions):
        a, b = max(0.0, a - pad_s), min(total_s, b + pad_s)
        if b <= a:
            continue
        if padded and a <= padded[-1][1]:
            padded[-1][1] = max(padded[-1][1], b)
        else:
            padded.append([a, b])
    # 2. hard-split anything longer than max_s
    pieces: list[Region] = []
    for a, b in padded:
        pieces.extend(_split_long(a, b, max_s))
    # 3. greedy merge: extend the current chunk while it still fits in max_s
    chunks: list[list[float]] = []
    for a, b in pieces:
        if chunks and b - chunks[-1][0] <= max_s + 1e-9:
            chunks[-1][1] = b
        else:
            chunks.append([a, b])
    # 4. widen chunks shorter than min_s, without crossing neighbours
    for i, c in enumerate(chunks):
        short = min_s - (c[1] - c[0])
        if short <= 0:
            continue
        lo = chunks[i - 1][1] if i > 0 else 0.0
        hi = chunks[i + 1][0] if i + 1 < len(chunks) else total_s
        c[0] = max(lo, c[0] - short / 2)
        c[1] = min(hi, c[0] + min_s)
    return [(round(a, 3), round(b, 3)) for a, b in chunks]


def chunk(
    audio: np.ndarray, sr: int = 16000, max_s: float = 30.0, min_s: float = 1.0,
    backend: str = "auto",
) -> list[Region]:
    """Speech chunks ``[(start_s, end_s), ...]``, each at most ``max_s`` long."""
    total_s = len(audio) / sr
    if total_s <= 0:
        return []
    regions, used = detect(audio, sr, backend)
    if not regions:
        log.warning("VAD (%s) found no speech; falling back to fixed %.0f s windows", used, max_s)
    return plan(regions, total_s, max_s=max_s, min_s=min_s)
