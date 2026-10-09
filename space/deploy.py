"""Assemble the Hugging Face Space and, optionally, upload it.

    python space/deploy.py                       # build only, into build/space/
    python space/deploy.py --repo YOU/gyara      # build, then create/update the Space

The Space gets: app.py, the gyara/ package, space/README.md (with the Space
front matter) as README.md, space/requirements.txt, and any committed
benchmark summaries (runs/*/metrics.json, runs/*/report.md, except runs
named smoke-*) for the Benchmark tab. Nothing else: no audio, no database, no tokens.

Uploading needs a Hugging Face *write* token (`hf auth login` or HF_TOKEN).
The Space itself needs a *read* token with access to NCAIR1/Hausa-ASR, added
as the secret HF_TOKEN in the Space settings.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "space"
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.db", "*.wav", "*.mp3", "*.m4a",
                              "*.mp4", "*.flac", "*.ogg", ".env")


def build(out: Path = OUT) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(ROOT / "app.py", out / "app.py")
    shutil.copytree(ROOT / "gyara", out / "gyara", ignore=SKIP)
    shutil.copy2(ROOT / "space" / "README.md", out / "README.md")
    shutil.copy2(ROOT / "space" / "requirements.txt", out / "requirements.txt")
    for f in list((ROOT / "runs").glob("*/metrics.json")) + list((ROOT / "runs").glob("*/report.md")):
        if f.parent.name.startswith("smoke"):
            continue  # smoke tests check the pipeline runs; they are not results
        dest = out / "runs" / f.parent.name / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
    return out


def upload(out: Path, repo: str, private: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="space",
                      commit_message="Deploy Gyara", delete_patterns=["gyara/**", "runs/**"])
    return f"https://huggingface.co/spaces/{repo}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", help="Space id, e.g. your-name/gyara. Omit to build only.")
    p.add_argument("--private", action="store_true", help="Create the Space as private.")
    a = p.parse_args(argv)

    out = build()
    files = sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file())
    print(f"Built {out} ({len(files)} files).")
    if not a.repo:
        print("Build only. To deploy: python space/deploy.py --repo YOUR-NAME/gyara")
        return 0
    url = upload(out, a.repo, a.private)
    print(f"Uploaded. The Space builds in a few minutes: {url}")
    print(f"Next: add the secret HF_TOKEN at {url}/settings (a read token that has "
          "access to NCAIR1/Hausa-ASR), then restart the Space.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
