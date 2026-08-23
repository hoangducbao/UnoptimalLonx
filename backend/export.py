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

TRAKE row structure is different in kind, not just in row shape: a TRAKE
answer is a human's exact (video_id, frame_idx_1..N) pick from watching the
video directly, in *native* frame_idx space -- there's no keyframe `n` to
translate, unlike KIS/VQA (its export-popup UI is a later phase, still
just a placeholder):
  confirmed mode:   [human-picked (video_id, frame_idxs)] + [shell hedges:
                     every event shifted together by the same +/-15,
                     +/-30, ... native-frame offset, widening outward] --
                     up to max_rows.
  unconfirmed mode: [one row per shortlisted candidate video, its own
                     proposed frame_idx sequence, confidence order, no
                     hedging] + [shell hedges of each candidate's sequence,
                     same widening pattern, video-by-video in confidence
                     order, as filler] -- up to max_rows.
A shell hedge shifts *every* event by the same signed offset in one row,
not one event at a time: it's a bet that the whole pick is off by a
consistent timing drift (the human's playback-position sense was off, or a
candidate's whole alignment is shifted), which is both a more plausible
failure mode than one event drifting independently and a far more
row-efficient hedge than a cross-product across events would be, since
only the single best row per R@k band ever counts.

Deliberately reuses the app's *existing* result-dict shapes instead of a
parallel Candidate model:
  - KIS/VQA candidates ("similars"): {video_id, n, rank, score_label,
    score_val, text} (backend/common.py::df_to_results -- what every
    non-TRAKE signal already returns).
  - KIS/VQA confirmed/answers: {video_id, n} (n = the app's internal
    1-indexed keyframe ordinal).
  - TRAKE candidates (unconfirmed mode): {video_id, video_score,
    order_valid, coverage, events: [...]} (backend/search/trake.py::
    trake_rank_videos), where each events[i] is {event_index, label,
    matched, n, score_val, ...} when matched -- n gets translated to
    frame_idx once, up front, in _generate_export_trake().
  - TRAKE confirmed: {video_id, frame_idxs: [f1, ..., fN]} -- already
    native frame_idx, straight from a human's manual pick, no lookup.

KIS/VQA rows built here stay n-space; rows_to_csv_text() below does the n
-> frame_idx translation AIC submissions actually expect (frame_idx_for_n
in backend/common.py) plus final text formatting -- kept separate so the
ranking/dedup logic is testable without touching map-keyframes files. TRAKE
rows are already resolved to frame_idx by the time they leave
_generate_export_trake(), so rows_to_csv_text() does no lookup for them.
"""

from typing import Optional

from .common import frame_idx_for_n, valid_ns_for_video

DEFAULT_NEIGHBOUR_COUNT = 10

# TRAKE hedge shells: every event shifted together by this many native
# frames, widening outward (+15, -15, +30, -30, ...). ~10 frames is the
# scoring model's typical GT window width, so 15 clears one window's worth
# per shell without being so small that early shells barely move.
TRAKE_SHELL_STEP = 15
# Upper bound on shells generated -- 60 shells * 2 signs = up to 120 hedge
# sequences, comfortably more than a 100-row budget (minus the 1 top row)
# could ever consume, even after dedup/negative-frame skips.
TRAKE_MAX_SHELLS = 60


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


def _shell_hedges(base_frame_idxs: list) -> list:
    """Every uniform-shift hedge sequence around base_frame_idxs, widening
    outward in shells of TRAKE_SHELL_STEP native frames: all events +15,
    all events -15, then +30, -30, ... (see module docstring for why a
    shift is applied to every event together rather than one at a time).
    A sequence is skipped if any event would go negative; shells stop
    after TRAKE_MAX_SHELLS regardless (see its comment)."""
    hedges = []
    for shell in range(1, TRAKE_MAX_SHELLS + 1):
        radius = TRAKE_SHELL_STEP * shell
        for offset in (radius, -radius):
            seq = [f + offset for f in base_frame_idxs]
            if all(f >= 0 for f in seq):
                hedges.append(seq)
    return hedges


def _generate_export_trake(candidates: list, mode: str, confirmed: Optional[dict], max_rows: int) -> list:
    seen, rows = set(), []

    def add(video_id: str, frame_idxs: list) -> bool:
        key = (video_id, tuple(frame_idxs))
        if key in seen or len(rows) >= max_rows:
            return False
        seen.add(key)
        rows.append({"video_id": video_id, "frame_idxs": list(frame_idxs)})
        return True

    if mode == "confirmed":
        if confirmed is None:
            raise ValueError("mode='confirmed' requires a confirmed candidate")
        video_id = confirmed["video_id"]
        base = [int(f) for f in confirmed["frame_idxs"]]
        add(video_id, base)
        for seq in _shell_hedges(base):
            if len(rows) >= max_rows:
                break
            add(video_id, seq)
        return rows

    # mode == "unconfirmed": resolve each shortlisted candidate's n's to
    # frame_idx once, up front (skipping any candidate with an unmatched or
    # otherwise unresolvable event -- no frame_idx to put in that column).
    ranked = sorted(candidates, key=_confidence, reverse=True)
    resolved = []
    for c in ranked:
        if not all(e.get("matched") for e in c["events"]):
            continue
        frame_idxs = [frame_idx_for_n(c["video_id"], e["n"]) for e in c["events"]]
        if any(f is None for f in frame_idxs):
            continue
        resolved.append((c["video_id"], frame_idxs))

    # Top tier: one row per candidate video, its own proposed sequence, no
    # hedging -- wrong video scores 0 regardless of frame precision, so
    # this tier is pure distinct-hypothesis diversification.
    for video_id, frame_idxs in resolved:
        add(video_id, frame_idxs)

    # Filler tier: shell-hedge each candidate's own sequence, video-by-
    # video in the same confidence order, until the row budget is spent.
    for video_id, frame_idxs in resolved:
        if len(rows) >= max_rows:
            break
        for seq in _shell_hedges(frame_idxs):
            if len(rows) >= max_rows:
                break
            add(video_id, seq)

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
            # Already native frame_idx by construction (_generate_export_trake
            # resolves n -> frame_idx, or takes a human's frame_idx pick
            # directly) -- no lookup needed here.
            fields = [video_id, *(str(f) for f in row["frame_idxs"])]
        else:
            frame_idx = frame_idx_for_n(video_id, row["n"])
            if frame_idx is None:
                continue
            fields = [video_id, str(frame_idx)]
            if query_type == "VQA":
                fields.append('"' + answer.replace('"', '""') + '"')
        lines.append(",".join(fields))
    return "\n".join(lines) + ("\n" if lines else "")
