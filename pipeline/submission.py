"""
submission.py — write ranked retrieve.py results to the on-disk format the
competition expects.

The exact format (file naming, delimiter, header, whether frame_id or a
timestamp is expected, one file per query vs one combined file, VQA answer
column format, etc.) is set by the organizers per round and hasn't arrived
in any material available while writing this module. Rather than guess and
bake a wrong format into retrieve.py's core logic, every writer here is a
narrow, isolated function fed a plain ranked-result list — so confirming
the real spec later is a one-place edit, not a refactor.

Current best-guess format (adjust when the organizer spec is confirmed):
one CSV per query, no header, columns as shown per task type below — this
matches the shape described in AICDraft/AIC_context.md (video_id + frame_id
for KIS, + answer for VQA, + one frame_id per sub-event for TRAKE).
"""

import csv
from pathlib import Path


def write_kis(results: list, out_path: Path) -> None:
    """results: list of dicts with 'video_id', 'frame_idx' (already ranked,
    best first). Writes up to 100 rows -- the competition's submission cap."""
    _write_csv(out_path, [(r["video_id"], r["frame_idx"]) for r in results[:100]])


def write_vqa(results: list, out_path: Path) -> None:
    """results: list of dicts with 'video_id', 'frame_idx', 'answer'."""
    _write_csv(out_path, [(r["video_id"], r["frame_idx"], r["answer"]) for r in results[:100]])


def write_trake(results: list, out_path: Path) -> None:
    """results: list of dicts with 'video_id', 'frame_ids' (list[int], one
    per sub-event, in order)."""
    _write_csv(out_path, [(r["video_id"], *r["frame_ids"]) for r in results[:100]])


def _write_csv(out_path: Path, rows: list) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
