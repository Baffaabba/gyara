"""Load any audio file as 16 kHz mono float32, without a system ffmpeg.

Whisper models expect 16 kHz mono. Users upload whatever their phone or editor
produced (wav, mp3, m4a, even mp4 video), and the dev laptops and HF Spaces do
not reliably have ffmpeg installed. So:

* wav / flac / ogg are read with ``soundfile`` (libsndfile, ships in the wheel);
* everything else is decoded by the ffmpeg binary that ``imageio-ffmpeg``
  bundles, piped back as raw float32, so no temp files and no system install.

Resampling happens here, once, so every downstream module can assume 16 kHz.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000

# Formats libsndfile reads natively. Anything else goes through ffmpeg.
_SOUNDFILE_EXT = {".wav", ".flac", ".ogg", ".oga", ".aiff", ".aif"}


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as e:  # pragma: no cover - depends on install
        raise RuntimeError(
            "Decoding this format needs ffmpeg. Install it with `pip install imageio-ffmpeg` "
            "(or `pip install gyara[asr]`)."
        ) from e
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffmpeg_decode(src: str | bytes, sr: int) -> np.ndarray:
    """Decode with ffmpeg to mono float32 at ``sr``. ``src`` is a path or raw bytes."""
    is_bytes = isinstance(src, (bytes, bytearray))
    cmd = [
        _ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0" if is_bytes else str(src),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, input=bytes(src) if is_bytes else None, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg could not decode the audio: {msg[-500:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def resample(audio: np.ndarray, orig_sr: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Band-limited resampling (soxr via librosa when present, else polyphase)."""
    if orig_sr == sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    try:
        import librosa

        out = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    except ImportError:
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(orig_sr, sr)
            out = resample_poly(audio, sr // g, orig_sr // g)
        except ImportError:
            # Last resort, no anti-aliasing: fine for speech going *down* to 16 kHz
            # from 22-48 kHz only in a pinch. librosa is in gyara[asr].
            n_out = int(round(len(audio) * sr / orig_sr))
            t_out = np.arange(n_out) * (orig_sr / sr)
            out = np.interp(t_out, np.arange(len(audio)), audio)
    return np.asarray(out, dtype=np.float32)


def load(path_or_bytes: str | Path | bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return the audio as a 1-D float32 array at ``sr`` Hz, channels averaged."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        try:
            data, file_sr = sf.read(io.BytesIO(path_or_bytes), dtype="float32", always_2d=True)
        except Exception:
            return _ffmpeg_decode(path_or_bytes, sr)
    else:
        path = Path(path_or_bytes)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in _SOUNDFILE_EXT:
            return _ffmpeg_decode(str(path), sr)
        try:
            data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception:
            # e.g. a .wav that is really ADPCM or a mislabelled mp3.
            return _ffmpeg_decode(str(path), sr)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return resample(mono, file_sr, sr)


def save_wav(path: str | Path, array: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    """Write 16-bit PCM wav (small, universally readable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(array, dtype=np.float32), sr, subtype="PCM_16")
    return path


def duration(path: str | Path) -> float:
    """Duration in seconds. Header-only for soundfile formats; decodes otherwise."""
    path = Path(path)
    if path.suffix.lower() in _SOUNDFILE_EXT:
        try:
            info = sf.info(str(path))
            return info.frames / info.samplerate
        except Exception:
            pass
    return len(load(path)) / SAMPLE_RATE
