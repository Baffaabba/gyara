"""Gyara: measure, correct and improve Hausa speech-to-text on N-ATLAS.

N-ATLAS is an initiative of the Federal Ministry of Communications, Innovation
and Digital Economy, and powered by Awarri Technologies.
"""

__version__ = "0.1.0"

ATTRIBUTION = (
    "N-ATLAS is an initiative of the Federal Ministry of Communications, "
    "Innovation and Digital Economy, and powered by Awarri Technologies."
)

DEFAULT_ASR_MODEL = "NCAIR1/Hausa-ASR"
DEFAULT_LLM_MODEL = "NCAIR1/N-ATLaS"


def load_env(path=".env") -> None:
    """Read KEY=value lines from a .env file into the environment.

    Variables already set win, and blank values are skipped, so an empty
    HF_TOKEN= line never hides a token saved by `hf auth login`.
    """
    import os
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
