"""Hugging Face Space entry point for the Gyara correction app.

Configure with Space variables (Settings -> Variables and secrets):

    GYARA_ASR_MODEL     speech model id         (default NCAIR1/Hausa-ASR)
    GYARA_LLM_BACKEND   none | transformers | openai   (default none)
    GYARA_LLM_BASE_URL  for the openai backend, e.g. a llama.cpp server
    GYARA_DB            SQLite workspace file   (default workspace/gyara.db)
    GYARA_HELDOUT       optional held-out manifest checked at dataset export
    GYARA_MAX_MINUTES   longest upload in minutes (default 10 on a Space,
                        no limit locally; 0 means no limit)
    HF_TOKEN            a *secret*: NCAIR1 models are gated on Hugging Face

Without persistent storage a Space's disk is wiped on restart, so export your
work (Export tab) before the Space sleeps, or point GYARA_DB at /data.

N-ATLAS is an initiative of the Federal Ministry of Communications, Innovation
and Digital Economy, and powered by Awarri Technologies.
"""

import os

from gyara import DEFAULT_ASR_MODEL, load_env
from gyara.ui.app import build_app, launch

load_env()  # local runs only; a Space has no .env and uses its secrets

ON_SPACE = bool(os.environ.get("SPACE_ID"))
_cap = os.environ.get("GYARA_MAX_MINUTES")

demo = build_app(
    store_path=os.environ.get("GYARA_DB", "workspace/gyara.db"),
    asr_model=os.environ.get("GYARA_ASR_MODEL", DEFAULT_ASR_MODEL),
    suggester_backend=os.environ.get("GYARA_LLM_BACKEND", "none"),
    device=os.environ.get("GYARA_DEVICE") or None,
    heldout_manifest=os.environ.get("GYARA_HELDOUT") or None,
    max_minutes=float(_cap) if _cap else (10.0 if ON_SPACE else None),
)

if __name__ == "__main__":
    # On a Space, bind all interfaces and refuse very large uploads before they
    # fill the disk; locally, Gradio's defaults (127.0.0.1, no size limit).
    if ON_SPACE:
        launch(demo, server_name="0.0.0.0", max_file_size="300mb")
    else:
        launch(demo)
