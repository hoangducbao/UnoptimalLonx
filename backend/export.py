"""
backend/export.py -- AIC submission CSV export: ranking/dedup logic that
turns a query's answer (a confirmed frame, or a user-curated ordered list)
plus its search candidates into <=100 rows for one query's submission CSV,
per the AIC scoring model (Final Score = average of R@k for k in
{1, 5, 20, 50, 100}, where R@k = max score among the first k rows -- so
only the best-scoring row within each threshold band matters, and a
duplicate of an already-placed row can never raise that max, only waste a
slot). Pure logic, no FastAPI/file I/O -- mirrors the backend/search/*.py
vs backend/routes/*.py split already used for TRAKE; backend/routes/export.py
is the thin endpoint (+ two small preview-data routes) on top of this
module.

KIS/VQA row structure (the export popup's "answer/similar/neighbours-of-
similar" tiers):
  confirmed mode:  [confirmed frame] + [its nearest keyframes by time] +
                    [similar-semantic candidates] + [nearest keyframes by
                    time of each similar, as filler] -- up to max_rows.
  unconfirmed mode: [user-ordered answer frames, kept in order] + [similar-
                    semantic candidates] + [nearest keyframes by time of
                    each similar, as filler] -- up to max_rows.
"Nearest keyframes by time" is deliberately keyframe-only for now (picks
from the video's own existing extracted n's, ordered by |n - center| --
equivalent to time-gap order since map-keyframes rows are chronological) --
arbitrary non-keyframe frame_idx neighbours are a later phase.

TRAKE keeps the same n-offset hedge/rank logic as before (unchanged) --
its export-popup UI is a later phase too, still served through the
existing confirm+export-bar flow.

Deliberately reuses the app's *existing* result-dict shapes instead of a
parallel Candidate model:
  - KIS/VQA candidates ("similars"): {video_id, n, rank, score_label,
    score_val, text} (backend/common.py::df_to_results -- what every
    non-TRAKE signal already returns).
  - KIS/VQA confirmed/answers: {video_id, n} (n = the app's internal
    1-indexed keyframe ordinal).
  - TRAKE candidates: {video_id, video_score, order_valid, coverage,
    events: [...]} (backend/search/trake.py::trake_rank_videos), where
    each events[i] is {event_index, label, matched, n, score_val, ...}
    when matched.

Rows built here stay n-space; rows_to_csv_text() below does the n ->
frame_idx translation AIC submissions actually expect (frame_idx_for_n in
backend/common.py) plus final text formatting -- kept separate so the
ranking/dedup logic is testable without touching map-keyframes files.
"""

from typing import Optional

from .common import frame_idx_for_n, valid_ns_for_video

DEFAULT_NEIGHBOUR_COUNT = 10

# TRAKE confirmed-mode hedge rows: same event, n offset by these (in
# order), skipping any offset that doesn't resolve to a real keyframe.
HEDGE_OFFSETS = [1, -1, 2, -2]
HEDGE_ROWS = 4


def _flat_key(video_id, n) -> tuple:
    return (video_id, int(n))


def _confidence(c: dict) -> float:
    return c.get("video_score", c.get("score_val", 0.0))


def nearest_keyframes_by_time(video_id: str, center_n: int, count: int) -> list:
    """The `count` keyframes closest to center_n by |n - center_n| (n is
    chronological, so this is time-gap order), excluding center_n itself.
    Keyframe-only for now -- see module docstring; this is what backs both
    the export popup's "Neighbours" preview section and the "neighbours of
    similars" filler tier."""
    center_n = int(center_n)
    valid_ns = [n for n in valid_ns_for_video(video_id) if n != center_n]
    valid_ns.sort(key=lambda n: abs(n - center_n))
    return valid_ns[:count]


def generate_export(query_type: str, candidates: list, mode: str,
                     confirmed: Optional[dict] = None, answers: Optional[list] = None,
                     neighbour_count: int = DEFAULT_NEIGHBOUR_COUNT, max_rows: int = 100) -> list:
    """Rank/dedup into <=max_rows rows for one query's submission CSV.
    `answer` (VQA text) isn't handled here -- it's the same string on
    every row, applied later by rows_to_csv_text()."""
    if query_type == "TRAKE":
        return _generate_export_trake(candidates, mode, confirmed, max_rows)
    return _generate_export_flat(candidates, mode, confirmed, answers, neighbour_count, max_rows)


def _generate_export_flat(candidates: list, mode: str, confirmed: Optional[dict],
                           answers: Optional[list], neighbour_count: int, max_rows: int) -> list:
    seen, rows = set(), []

    def add(video_id, n) -> bool:
        key = _flat_key(video_id, n)
        if key in seen or len(rows) >= max_rows:
            return False
        seen.add(key)
        rows.append({"video_id": video_id, "n": int(n)})
        return True

    if mode == "confirmed":
        if confirmed is None:
            raise ValueError("mode='confirmed' requires a confirmed frame")
        add(confirmed["video_id"], confirmed["n"])
        for n in nearest_keyframes_by_time(confirmed["video_id"], confirmed["n"], neighbour_count):
            add(confirmed["video_id"], n)
    else:
        if not answers:
            raise ValueError("mode='unconfirmed' requires at least one answer frame")
        for a in answers:
            add(a["video_id"], a["n"])

    # Similar-semantic tier, ranked order, exact-repeat dedup only.
    ranked = sorted(candidates, key=_confidence, reverse=True)
    for c in ranked:
        add(c["video_id"], c["n"])

    # Filler tier: nearest-by-time keyframes of each similar, in the same
    # ranked order, until the row budget is exhausted.
    for c in ranked:
        if len(rows) >= max_rows:
            break
        for n in nearest_keyframes_by_time(c["video_id"], c["n"], neighbour_count):
            if len(rows) >= max_rows:
                break
            add(c["video_id"], n)

    return rows


def _generate_export_trake(candidates: list, mode: str, confirmed: Optional[dict], max_rows: int) -> list:
    def row_and_key(c: dict):
        row = {"video_id": c["video_id"], "ns": [int(e["n"]) for e in c["events"]]}
        return row, (row["video_id"], tuple(row["ns"]))

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
        for row in _hedge_rows_trake(confirmed)[:HEDGE_ROWS]:
            add(row, (row["video_id"], tuple(row["ns"])))

        # Rows 6-100: remaining candidates by confidence, exact-repeat
        # dedup only against what's already placed above (spec: "no exact
        # repeats") -- not the window-overlap dedup unconfirmed mode uses.
        for c in ranked:
            if not all(e.get("matched") for e in c["events"]):
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
        if not all(e.get("matched") for e in c["events"]):
            continue
        key = c["video_id"]  # video-level hypothesis: wrong video scores 0 regardless of frame accuracy
        if key in seen:
            continue
        seen.add(key)
        row, _ = row_and_key(c)
        rows.append(row)
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


def rows_to_csv_text(query_type: str, rows: list, mode: str = "unconfirmed", answer: str = "") -> str:
    """n -> frame_idx translation + final CSV text: no header row,
    video_id/frame_idx columns unquoted, VQA answer column always quoted
    (confirmed against the real submission/*.csv samples already in the
    repo -- e.g. `L30_V072,676,"Giang Ly"`, quoted even with no comma, so
    plain csv.QUOTE_MINIMAL wouldn't reproduce it). VQA + unconfirmed has
    no real answer yet (LLM answering is a later phase) -- every row gets
    the literal placeholder regardless of what's passed."""
    if query_type == "VQA" and mode != "confirmed":
        answer = "LLM needed"

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
