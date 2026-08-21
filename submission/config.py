"""submission/config.py — batch configuration: input/output roots, Mixed-mode
weights & legs, row caps, CSV formatting rules.

Everything here is a plain dataclass default. Override via the CLI flags or
by editing the @dataclass, matching the repo's "constants live in config.py /
no env-file" convention (see backend/config.py).

Signal weight keys follow the UI: Keyframe / ASR / Caption / OCR. `legs` are
the per-signal sub-leg toggles that Mixed consumes (same keys the backend's
`routes/search.py` Mixed endpoint and `search/mixed.py` expect).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Sensible Mixed-mode defaults: lean on the keyframe embeddings, then caption,
# then the fuzzy full-text legs. Tune per round if you want.
DEFAULT_WEIGHTS = {"Keyframe": 3, "Caption": 2, "ASR": 1, "OCR": 1}
DEFAULT_LEGS = {
    # Keyframe sub-legs
    "kf_siglip2": True,
    "kf_clip": True,
    # ASR sub-legs
    "asr_siglip": True,
    "asr_fuzzy": True,
    # Caption sub-legs
    "cap_siglip": True,
    "cap_fuzzy": True,
}


@dataclass
class SubmissionConfig:
    queries_dir: Path = REPO_ROOT / "queries"
    output_dir: Path = REPO_ROOT / "submissions"

    # Mixed-mode fusion weights/legs used for KIS + Q&A (and each TRAKE event).
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    legs: dict = field(default_factory=lambda: dict(DEFAULT_LEGS))

    # Per-query row cap — the CodaBench hard limit is 100 rows / query.
    max_rows: int = 100
    # Candidate pool the pipeline fetches per leg before RRF fusion.
    top_k: int = 100

    # CSV "Frame Idx" output source. "n" = 1-based keyframe number (the value
    # map-keyframes/{video}.csv keys, and the value thumbnails / frame ids in
    # this pipeline already expose). "frame_id" = 0-based row index.
    frame_index: str = "n"

    # TRAKE events: which signal each event runs through. "Mixed" is the
    # user's requested mode and lets every event fuse Keyframe/ASR/Caption/OCR.
    trake_event_signal: str = "Mixed"
    # How many top TRAKE videos to emit rows for (<= max_rows anyway).
    trake_top_videos: int = 100

    # Q&A answer hook mode — see answer.py. "none" (default) emits an empty
    # answer field for a format/validation run. Wire a VQA model into
    # answer.py's generate_answer() for a scored run.
    answer_mode: str = "none"
    answer_max_len: int = 100  # CodaBench hard limit on the Answer field

    # CSV formatting rules (see csvout.py).
    quote_all_answers: bool = False  # when True, every Q&A answer gets quoted
    line_ending: str = "\r\n"        # CRLF is the documented default (LF allowed)
    encoding: str = "utf-8"
    write_header: bool = False       # CodaBench wants NO header row