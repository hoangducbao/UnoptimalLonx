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

TRAKE row structure is different in kind, not just in row shape, and lives
entirely outside generate_export()/rows_to_csv_text() below -- see
generate_trake_rows() and its own docstring. A TRAKE answer is a human's
exact (video_id, frame_idx_1..N) pick from watching the video directly, in
*native* frame_idx space -- there's no keyframe `n` to translate, unlike
KIS/VQA. There's no confirmed/unconfirmed distinction for TRAKE any more:
curation happens per video (an ordered event list, each a native
frame_idx, built by watching that video in the Export tab), row generation
happens per video into an in-memory cache (frontend/js/export-ui.js), and
a human merges however many cached videos they curated into one final
<=100-row CSV at export time -- see generate_trake_rows()'s docstring for
the row-generation half of that, and CLAUDE.md's "Export architecture"
section for the curate/cache/merge flow end to end.

Deliberately reuses the app's *existing* result-dict shapes instead of a
parallel Candidate model:
  - KIS/VQA candidates ("similars"): {video_id, n, rank, score_label,
    score_val, text} (backend/common.py::df_to_results -- what every
    non-TRAKE signal already returns).
  - KIS/VQA confirmed/answers: {video_id, n} (n = the app's internal
    1-indexed keyframe ordinal).

KIS/VQA rows built here stay n-space; rows_to_csv_text() below does the n
-> frame_idx translation AIC submissions actually expect (frame_idx_for_n
in backend/common.py) plus final text formatting -- kept separate so the
ranking/dedup logic is testable without touching map-keyframes files. TRAKE
rows (video_id + a list of already-native frame_idxs, straight from
generate_trake_rows() or a merge of several cached calls to it) are always
already resolved to frame_idx, so rows_to_csv_text() does no lookup for
them.
"""

from random import Random
from typing import Optional

from .common import frame_idx_for_n, n_for_frame_idx, native_frame_range_for_video, valid_ns_for_video

DEFAULT_NEIGHBOUR_COUNT = 10


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
    every row, applied later by rows_to_csv_text(). KIS/VQA only -- TRAKE
    has its own generate_trake_rows() below, called through the dedicated
    /api/export/trake-rows + /api/export/trake-write routes instead of
    this one, since its curate/cache/merge flow doesn't fit this
    single-shot candidates-in-rows-out shape."""
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


TRAKE_INTERP_SEED = 20260828  # arbitrary fixed seed -- see generate_trake_rows()'s docstring


def _event_neighbour_stream(video_id: str, frame_idx: int, count: int) -> list:
    """Up to `count` ranked (nearest-first) neighbour frame_idxs for one
    TRAKE event's pick. A keyframe-backed pick (frame_idx matches some
    keyframe's own native frame exactly) ranks by keyframe-index (n)
    distance, same metric/pool as the KIS/VQA "Neighbours" tier -- reuses
    nearest_keyframes_by_time() and translates its n's back to frame_idx.
    A pick with no keyframe behind it has no embedding and thus no
    "similar" pool to search at all -- see module docstring -- so it ranks
    by plain frame-number distance instead: frame_idx+1, frame_idx-1,
    frame_idx+2, frame_idx-2, ..., clipped to the video's real frame
    range."""
    n = n_for_frame_idx(video_id, frame_idx)
    if n is not None:
        return [f for f in (frame_idx_for_n(video_id, nb) for nb in nearest_keyframes_by_time(video_id, n, count))
                if f is not None]
    lo, hi = native_frame_range_for_video(video_id)
    out = []
    for shell in range(1, count + 1):
        for cand in (frame_idx + shell, frame_idx - shell):
            if lo <= cand <= hi:
                out.append(cand)
        if len(out) >= count:
            break
    return out[:count]


def _fill_trake_row(raw: list, lo: int, hi: int, rng: Random) -> list:
    """One row's worth of per-event frame numbers, left to right, given
    each event's raw k-th-neighbour candidate (None where that event's
    stream ran dry, or never had one -- see generate_trake_rows()). Walks
    events in sequence order, tracking the previous event's already-chosen
    frame number as a running lower bound: a raw candidate is used as-is
    if it clears that bound (i.e. it's already in temporal order with
    what came before), otherwise -- whether because it's None or because
    it would tie/violate the i<j frame-number ordering -- a frame is
    interpolated instead, picked uniformly at random strictly between the
    running lower bound and the nearest usable upper bound (the next
    event's own raw candidate if it clears the running bound too, else the
    video's own end). This is the one rule the spec leaves open ("no fixed
    rule") -- it satisfies temporal ordering per-row by construction,
    without needing a second corrective pass."""
    n = len(raw)
    chosen = []
    prev = lo
    for i in range(n):
        val = raw[i]
        if val is not None and val > prev:
            chosen.append(val)
            prev = val
            continue
        upper = hi
        for j in range(i + 1, n):
            if raw[j] is not None and raw[j] > prev:
                upper = raw[j]
                break
        upper = max(upper, prev + 1)
        pick = rng.randint(prev + 1, upper - 1) if upper - 1 > prev else prev + 1
        pick = min(pick, hi)
        chosen.append(pick)
        prev = pick
    return chosen


def generate_trake_rows(video_id: str, frame_idxs: list, max_rows: int = 100) -> list:
    """<=max_rows candidate sequences for one video's curated TRAKE event
    list, per the spec's row-generation rules:
      row 1            = the curated picks, in event order, as given.
      rows 2..max_rows = row k's event i = event i's k-th nearest
                          neighbour (see _event_neighbour_stream), taken
                          independently per event and zipped by rank.
    Temporal ordering (event i's frame number < event j's for i<j) is
    enforced within each row only, via _fill_trake_row's interpolation
    fallback -- used whenever an event's own k-th neighbour is missing
    (its stream ran dry, or it had none to begin with) or would break that
    row's ordering. One row is attempted per rank 1..max_rows-1; an exact
    duplicate of an already-emitted row (which interpolation's randomness
    makes rare but not impossible, especially in a short/crowded video) is
    dropped rather than retried, so the result can come back shorter than
    max_rows -- consistent with every other export tier in this module,
    where a spent dedup slot is never backfilled.
    A fixed RNG seed (TRAKE_INTERP_SEED) makes an unchanged event list's
    output byte-identical across repeated "Generate rows" clicks, which
    matters once a video's cache entry has been included in a merge and
    then regenerated -- otherwise every regenerate would silently reshuffle
    already-reviewed filler rows for no reason."""
    if not frame_idxs:
        raise ValueError("need at least one curated event")
    frame_idxs = [int(f) for f in frame_idxs]
    lo, hi = native_frame_range_for_video(video_id)
    rng = Random(TRAKE_INTERP_SEED)

    streams = [_event_neighbour_stream(video_id, f, max_rows) for f in frame_idxs]

    seen = {tuple(frame_idxs)}
    rows = [list(frame_idxs)]
    for k in range(max_rows - 1):
        raw = [streams[i][k] if k < len(streams[i]) else None for i in range(len(frame_idxs))]
        row = _fill_trake_row(raw, lo, hi, rng)
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

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
            # Already native frame_idx by construction (generate_trake_rows,
            # or a merged list of several of its calls) -- no lookup needed
            # here.
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
