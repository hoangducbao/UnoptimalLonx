"""
backend/export.py -- AIC submission CSV export: ranking/dedup logic that
turns a signal's ranked candidate list into <=100 rows for one query, per
the AIC scoring model (Final Score = average of R@k for k in
{1, 5, 20, 50, 100}, where R@k = max score among the first k rows -- so
only the best-scoring row within each threshold band matters, and a
duplicate of an already-placed row can never raise that max, only waste a
slot). Pure logic, no FastAPI/file I/O -- mirrors the backend/search/*.py
vs backend/routes/*.py split already used for TRAKE; backend/routes/export.py
is the thin endpoint on top of this module.

Deliberately reuses the app's *existing* result-dict shapes instead of a
parallel Candidate model:
  - KIS/VQA candidates: {video_id, n, rank, score_label, score_val, text}
    (backend/common.py::df_to_results -- what every non-TRAKE signal
    already returns).
  - TRAKE candidates: {video_id, video_score, order_valid, coverage,
    events: [...]} (backend/search/trake.py::trake_rank_videos), where
    each events[i] is {event_index, label, matched, n, score_val, ...}
    when matched.

Rows built here stay n-space (the app's internal 1-indexed keyframe
ordinal); rows_to_csv_text() below does the n -> frame_idx translation AIC
submissions actually expect (frame_idx_for_n in backend/common.py) plus
final text formatting -- kept separate so the ranking/dedup logic is
testable without touching map-keyframes files.
"""

from typing import Optional

from .common import frame_idx_for_n, valid_ns_for_video

# Confirmed-mode rows 2-5: same video/event, n offset by these (in order),
# skipping any offset that doesn't resolve to a real keyframe.
HEDGE_OFFSETS = [1, -1, 2, -2]
HEDGE_ROWS = 4


def _flat_key(video_id, n) -> tuple:
    return (video_id, int(n))


def _confidence(c: dict) -> float:
    return c.get("video_score", c.get("score_val", 0.0))


def _hedge_rows_flat(confirmed: dict) -> list:
    """KIS/VQA hedge rows: same video, n +/- HEDGE_OFFSETS."""
    video_id, n = confirmed["video_id"], int(confirmed["n"])
    valid_ns = valid_ns_for_video(video_id)
    rows = []
    for off in HEDGE_OFFSETS:
        cand_n = n + off
        if cand_n in valid_ns:
            rows.append({"video_id": video_id, "n": cand_n})
    return rows


def _hedge_rows_trake(confirmed: dict) -> list:
    """TRAKE hedge rows: perturb one event's n at a time, holding the rest
    at their confirmed value. Cycles events lowest-score-first within each
    offset so the least-confident alignment gets hedged first."""
    events = confirmed["events"]
    video_id = confirmed["video_id"]
    valid_ns = valid_ns_for_video(video_id)
    base_ns = [int(e["n"]) for e in events]
    order = sorted(range(len(events)), key=lambda i: events[i].get("score_val", 0.0))
    rows = []
    for off in HEDGE_OFFSETS:
        for ev_i in order:
            cand_n = base_ns[ev_i] + off
            if cand_n not in valid_ns:
                continue
            new_ns = list(base_ns)
            new_ns[ev_i] = cand_n
            rows.append({"video_id": video_id, "ns": new_ns})
    return rows


def generate_export(query_type: str, candidates: list, mode: str,
                     confirmed: Optional[dict] = None, max_rows: int = 100) -> list:
    """Rank/dedup `candidates` into <=max_rows rows for one query's
    submission CSV. `answer` (VQA) isn't handled here -- it's the same
    string on every row, applied later by rows_to_csv_text()."""
    is_trake = query_type == "TRAKE"

    def row_and_key(c: dict):
        if is_trake:
            row = {"video_id": c["video_id"], "ns": [int(e["n"]) for e in c["events"]]}
            return row, (row["video_id"], tuple(row["ns"]))
        row = {"video_id": c["video_id"], "n": int(c["n"])}
        return row, _flat_key(row["video_id"], row["n"])

    ranked = sorted(candidates, key=_confidence, reverse=True)

    if mode == "confirmed":
        if confirmed is None:
            raise ValueError("mode='confirmed' requires a confirmed candidate")
        seen, rows = set(), []

        def add(row, key):
            if key in seen or len(rows) >= max_rows:
                return
            seen.add(key)
            rows.append(row)

        conf_row, conf_key = row_and_key(confirmed)
        add(conf_row, conf_key)
        hedges = _hedge_rows_trake(confirmed) if is_trake else _hedge_rows_flat(confirmed)
        for row in hedges[:HEDGE_ROWS]:
            key = (row["video_id"], tuple(row["ns"])) if is_trake else _flat_key(row["video_id"], row["n"])
            add(row, key)

        # Rows 6-100: remaining candidates by confidence, exact-repeat
        # dedup only against what's already placed above (spec: "no exact
        # repeats") -- not the window-overlap dedup unconfirmed mode uses.
        for c in ranked:
            if is_trake and not all(e.get("matched") for e in c["events"]):
                continue
            row, key = row_and_key(c)
            add(row, key)
        return rows

    # mode == "unconfirmed": one greedy pass, sorted by confidence, skip
    # any candidate whose hypothesis key was already placed. A duplicate
    # hypothesis can never outscore what's already at that key (R@k only
    # rewards the best row within a band), so it's never worth a slot --
    # this single rule reproduces every row-band behavior the spec
    # describes (1 / 2-5 / 6-20 / 21-50 / 51-100) without branching on row
    # index; those bands are R@k eval checkpoints, not distinct code paths.
    seen, rows = set(), []
    for c in ranked:
        if len(rows) >= max_rows:
            break
        if is_trake:
            if not all(e.get("matched") for e in c["events"]):
                continue
            key = c["video_id"]  # video-level hypothesis: wrong video scores 0 regardless of frame accuracy
        else:
            key = _flat_key(c["video_id"], c["n"])
        if key in seen:
            continue
        seen.add(key)
        row, _ = row_and_key(c)
        rows.append(row)
    return rows


def rows_to_csv_text(query_type: str, rows: list, answer: str = "") -> str:
    """n -> frame_idx translation + final CSV text: no header row,
    video_id/frame_idx columns unquoted, VQA answer column always quoted
    (confirmed against the real submission/*.csv samples already in the
    repo -- e.g. `L30_V072,676,"Giang Ly"`, quoted even with no comma, so
    plain csv.QUOTE_MINIMAL wouldn't reproduce it)."""
    lines = []
    for row in rows:
        video_id = row["video_id"]
        if query_type == "TRAKE":
            frame_idxs = [frame_idx_for_n(video_id, n) for n in row["ns"]]
            if any(f is None for f in frame_idxs):
                continue  # unresolvable n -- drop rather than emit a malformed row
            fields = [video_id, *(str(f) for f in frame_idxs)]
        else:
            frame_idx = frame_idx_for_n(video_id, row["n"])
            if frame_idx is None:
                continue
            fields = [video_id, str(frame_idx)]
            if query_type == "VQA":
                fields.append('"' + answer.replace('"', '""') + '"')
        lines.append(",".join(fields))
    return "\n".join(lines) + ("\n" if lines else "")
