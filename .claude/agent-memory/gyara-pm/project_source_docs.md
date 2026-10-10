---
name: project-source-docs
description: The source PDFs (Concept v1.0, Execution Plan draft) match docs/source/*.md in version, but the Markdown drops each task's "done when" line and the "(or team server)" demo option
metadata:
  type: project
---

Checked 2026-10-09: Baffa's PDFs `Gyara Concept Document.pdf` (v1.0, 6 Oct) and `Naic Execution Plan.pdf` (draft, "nothing built until Go") are the same versions as `docs/source/CONCEPT.md` / `EXECUTION-PLAN.md`. No conflicting numbers or dates.

The Markdown copies are abridged. They drop, from the Execution Plan: every task's "done when" criterion (e.g. 2.2 `.srt` opens in a video player/YouTube, 2.3 a tester corrects a full file, 3.3 English `.srt` works, 3.6 testers confirm receipt, 4.5 script approved by team, 5.4 names/roles/affiliations, 5.5 confirmation before midnight Sunday), "English loanwords" in task 1.1, and "Deploy demo to HF Space **(or team server)**" in 3.5.

**Why:** the "done when" lines are the real acceptance tests; the Markdown alone under-specifies the DoD.
**How to apply:** trace against the PDF wording (now copied into `docs/STATUS.md` traceability tables), not the Markdown summary. See [[project-demo-hosting]].
