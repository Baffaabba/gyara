"""Real export -> load_dataset -> fine-tune smoke run (CPU). Text evidence only.

Run from the repo root (needs data/fleurs/probe-dev and data/fleurs/test.jsonl
from `gyara fetch-fleurs`, and access to NCAIR1/Hausa-ASR):

    .venv/Scripts/python runs/finetune-export-smoke/export_smoke.py

It goes through the correction app's own handlers (gyara.ui.app), the same
functions the buttons call:

1. "Transcribe" on 6 FLEURS ha_ng validation clips with NCAIR1/Hausa-ASR, consent
   ticked, plus 1 clip with consent NOT ticked (must be left out of the export).
2. "Save and next" on every segment, with the FLEURS reference as the person's
   correction (the stand-in for a human editor).
3. "Build dataset" (all files), with FLEURS test as the held-out manifest.
4. datasets.load_dataset("audiofolder") on the export.
5. `gyara finetune` on the export: NCAIR1/Hausa-ASR, 2 steps, eval.after false.

The workspace (database, audio, export, weights) goes to runs/smoke-finetune-export/,
which git ignores. Logs go next to this script.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")  # Hausa hooked letters on a Windows console
HERE = Path("runs/finetune-export-smoke")
WS = Path("runs/smoke-finetune-export")
HELDOUT = "data/fleurs/test.jsonl"
CONSENT = "FLEURS ha_ng validation, CC-BY-4.0 (public dataset, used as a stand-in)"
log_lines: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_lines.append(s)


from gyara import manifest as mf  # noqa: E402
from gyara.store import Store  # noqa: E402
from gyara.ui import app as ui  # noqa: E402

t0 = time.time()
if WS.exists():
    import shutil
    shutil.rmtree(WS)
WS.mkdir(parents=True)
store = Store(WS / "workspace" / "gyara.db")
ctx = ui.AppContext(store=store, asr_model="NCAIR1/Hausa-ASR", device="cpu",
                    heldout_manifest=HELDOUT)

dev = mf.read("data/fleurs/probe-dev/validation.jsonl")
short = [r for r in dev if r.duration and r.duration <= 25.0]
picked, no_consent = short[:6], short[6]
refs: dict[int, str] = {}
log("## 1. Transcribe (ui.transcribe)")
for r in picked + [no_consent]:
    consent = r is not no_consent
    status, _, _ = ui.transcribe(ctx, r.audio, speaker="", dialect="", consent=consent,
                                 consent_ref=CONSENT if consent else "", asset_id=None,
                                 progress=None)
    aid = max(a["id"] for a in store.assets())
    refs[aid] = r.text
    segs = store.segments(aid)
    log(f"- {r.id} ({r.duration:.1f} s, consent={consent}): asset {aid}, {len(segs)} segment(s); "
        f"status card says: {'First draft ready' in status}")

log("\n## 2. Correct and verify (ui.save_and_next)")
for aid, ref in refs.items():
    segs = store.segments(aid)
    queue = [s["id"] for s in segs]
    if len(segs) != 1:
        log(f"- asset {aid}: {len(segs)} segments, cannot map one reference onto them; "
            "verifying the drafts unchanged")
    for pos, s in enumerate(segs):
        text = ref if len(segs) == 1 else s["text"]
        ui.save_and_next(ctx, aid, queue, pos, text, "export-smoke", None)
    after = store.segments(aid)
    log(f"- asset {aid}: draft '{segs[0]['asr_text'][:60]}...' -> verified "
        f"'{after[0]['text'][:60]}...' status={after[0]['status']}")

log("\n## 3. Build dataset (ui.build_dataset, 'All files', held-out = FLEURS test)")
file_update, card = ui.build_dataset(ctx, "All files", None)
zip_path = Path(file_update["value"])
export_dir = zip_path.with_suffix("") if zip_path.suffix == ".zip" else zip_path
if not (export_dir / "metadata.jsonl").exists():
    export_dir = next(p.parent for p in (store.root / "exports").rglob("metadata.jsonl"))
log(f"- zip: {zip_path.as_posix()} ({zip_path.stat().st_size} bytes)")
log(f"- export dir: {export_dir.as_posix()}")
log(f"- files: {sorted(p.relative_to(export_dir).as_posix() for p in export_dir.rglob('*') if p.is_file())}")
import re  # noqa: E402
log(f"- app card: {re.sub('<[^>]+>', ' ', card).split()!r}"[:600])
meta = [json.loads(x) for x in (export_dir / "metadata.jsonl").read_text("utf-8").splitlines() if x]
log(f"- metadata.jsonl rows: {len(meta)}; first: "
    f"{json.dumps({k: meta[0][k] for k in ('file_name', 'transcription', 'duration', 'source')}, ensure_ascii=False)}")

log("\n## 4. datasets.load_dataset('audiofolder')")
from datasets import load_dataset  # noqa: E402

ds = load_dataset("audiofolder", data_dir=str(export_dir))
log(f"- {ds}")
ex = ds["train"][0]
arr = ex["audio"]["array"] if isinstance(ex["audio"], dict) else ex["audio"].get_all_samples().data
sr = ex["audio"]["sampling_rate"] if isinstance(ex["audio"], dict) else ex["audio"].get_all_samples().sample_rate
log(f"- row 0: transcription={ex['transcription'][:80]!r}, sampling_rate={sr}, "
    f"samples={len(arr) if hasattr(arr, '__len__') else arr.shape}")

log("\n## 5. gyara finetune on the export (NCAIR1/Hausa-ASR, 2 steps, CPU, eval.after false)")
cfg = {
    "attempt": 1,
    "base_model": "NCAIR1/Hausa-ASR",
    "language": "hausa",
    "output_dir": str(WS / "out"),
    "model_name": "hausa-asr-gyara-export-smoke",
    "data": {"exports": [str(export_dir)], "fleurs_fraction": 0.0,
             "heldout": [HELDOUT], "dev_fraction": 0.34},
    "training": {"max_steps": 2, "eval_steps": 1, "per_device_train_batch_size": 2,
                 "per_device_eval_batch_size": 2, "gradient_accumulation_steps": 1,
                 "generation_max_length": 64, "logging_steps": 1},
    "eval": {"after": False},
}
(WS / "finetune.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
(HERE / "finetune.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8",
                                    newline="\n")
gyara_exe = Path(sys.executable).with_name("gyara")
t1 = time.time()
r = subprocess.run([str(gyara_exe), "finetune", "--config", str(WS / "finetune.yaml")],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
(HERE / "finetune-output.txt").write_text(
    f"$ gyara finetune --config {WS.as_posix()}/finetune.yaml\n[exit {r.returncode}, "
    f"{time.time() - t1:.0f} s]\n\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}",
    encoding="utf-8", newline="\n")
log(f"- exit {r.returncode} in {time.time() - t1:.0f} s (full output: finetune-output.txt)")
summ = WS / "out" / "summary.json"
if summ.exists():
    s = json.loads(summ.read_text(encoding="utf-8"))
    keep = {k: s[k] for k in ("train_utterances", "dev_utterances", "train_hours", "own_hours",
                              "global_step", "train_loss", "dev_wer", "encoder_frozen",
                              "dev_split", "leakage_note", "notes", "before_after")}
    log(f"- summary.json: {json.dumps(keep, indent=2, ensure_ascii=False)}")
    (HERE / "summary.json").write_text(json.dumps({k: v for k, v in s.items()},
                                                  indent=2, default=str, ensure_ascii=False),
                                       encoding="utf-8", newline="\n")
    card_path = WS / "out" / "model" / "README.md"
    (HERE / "model-card.md").write_text(card_path.read_text(encoding="utf-8"), encoding="utf-8",
                                        newline="\n")
    log(f"- weights written: {sorted(p.name for p in (WS / 'out' / 'model').iterdir())}")
log(f"\nTotal {time.time() - t0:.0f} s")
(HERE / "run-log.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8", newline="\n")
