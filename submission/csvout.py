"""submission/csvout.py — CodaBench CSV writers.

Rules enforced here (from the competition brief):
  * .csv is a plain-text file — UTF-8, comma-delimited, no header row.
  * One row per attempted result (<= max_rows / query).
  * Frame Idx is numeric; never quoted.
  * Q&A <Answer>: quote only when it contains a comma, double-quote, CR/LF,
    or leading/trailing whitespace; escape `"` as `""`; cap at answer_max_len
    chars. `quote_all_answers` forces quoting of every answer (also valid).
  * TRAKE: <video>, <Frame ID_1>, ..., <Frame ID_N> in chronological order.

Note on spaces: the brief's examples sometimes show "<video>, <idx>" with a
space after the comma. Both are comma-delimited CSV; we emit the compact
`<video>,<idx>` form (always parseable).
"""

from __future__ import annotations

from pathlib import Path


def _needs_quoting(s: str) -> bool:
    return (s != s.strip()) or any(c in s for c in [",", '"', "\r", "\n"])


def _quote(s: str, always: bool) -> str:
    s = "" if s is None else str(s)
    if always or _needs_quoting(s):
        return '"' + s.replace('"', '""') + '"'
    return s


def _frame_id(cfg, n: int) -> str:
    """Render the CSV Frame Idx for a byte frame number.

    Default `cfg.frame_index == "n"` -> the 1-based keyframe number the
    pipeline already resolves. `"frame_id"` -> the 0-based row index.
    """
    if cfg.frame_index == "frame_id":
        return str(max(0, int(n) - 1))
    return str(int(n))


def _encode_line(line: str, cfg) -> str:
    return line + cfg.line_ending


def _write(path: Path, lines, cfg) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Hard cap per query (CodaBench allows at most 100 rows).
    lines = list(lines)[: cfg.max_rows]
    with open(path, "w", encoding=cfg.encoding, newline="") as fh:
        fh.writelines(_encode_line(ln, cfg) for ln in lines)


# ---------------------------------------------------------------------------
# Per-type writers
# ---------------------------------------------------------------------------

def write_kis(path: Path, results, cfg) -> None:
    """results: iterable of {"video_id","n"} -> `<video>,<Frame Idx>` rows."""
    lines = [f"{r['video_id']},{_frame_id(cfg, r['n'])}" for r in results]
    _write(path, lines, cfg)


def write_qa(path: Path, results, cfg) -> None:
    """results: iterable of {"video_id","n","answer"} ->
    `<video>,<Frame Idx>,<Answer>` rows with CodaBench-safe answer quoting.

    The answer is NOT trimmed (per the brief, leading/trailing whitespace is
    preserved and made round-trip safe by quoting); only capped at
    answer_max_len characters.
    """
    lines = []
    for r in results:
        answer = r.get("answer") or ""
        answer = str(answer)[: cfg.answer_max_len]
        lines.append(f"{r['video_id']},{_frame_id(cfg, r['n'])},{_quote(answer, cfg.quote_all_answers)}")
    _write(path, lines, cfg)


def write_trake(path: Path, results, cfg) -> None:
    """results: iterable of {"video_id","frame_ids":[int,...]} ->
    `<video>,<id1>,<id2>,...,<idN>` (ids already in chronological order)."""
    lines = []
    for r in results:
        ids = ",".join(_frame_id(cfg, x) for x in r["frame_ids"])
        lines.append(f"{r['video_id']},{ids}")
    _write(path, lines, cfg)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def write_for(name: str, results, cfg) -> None:
    from .query import QueryType, Query

    suffix = QueryType.from_filename(name)  # re-infer from the output stem
    out_path = Path(cfg.output_dir) / f"{name}.csv"
    if suffix is QueryType.KIS:
        write_kis(out_path, results, cfg)
    elif suffix is QueryType.QA:
        write_qa(out_path, results, cfg)
    elif suffix is QueryType.TRAKE:
        write_trake(out_path, results, cfg)
    return out_path