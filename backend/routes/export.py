"""
backend/routes/export.py -- export popup endpoints, on top of backend/
export.py's pure ranking/dedup logic:
  POST /api/export             -- KIS/VQA only: generate + return the
                                   finished CSV text.
  GET  /api/export/frame       -- single frame's {n, frame_idx, thumbnail_url},
                                   for the popup's answer-area card(s).
  GET  /api/export/neighbors   -- a frame's N nearest keyframes by time, for
                                   the popup's "Neighbours" preview section.
  GET  /api/export/similar     -- confirmed mode's "Similars" preview, a
                                   fresh visual search seeded by the
                                   confirmed frame itself.
  GET  /api/export/nearest-keyframe -- snaps a raw native frame_idx to its
                                   nearest indexed keyframe n, for the
                                   Export tab's "Keyframes" checkbox
                                   re-check (Keyframes-unchecked ->
                                   checked).
  POST /api/export/trake-rows  -- TRAKE: one video's curated event list ->
                                   <=99 candidate row sequences, for the
                                   Export tab's client-side per-video cache.
  POST /api/export/trake-write -- TRAKE: a human-merged row set (built
                                   client-side from that cache, several
                                   videos' rows interleaved) -> the
                                   finished CSV text. No ranking/dedup
                                   logic of its own -- just the same
                                   rows_to_csv_text() formatting step
                                   /api/export uses, on rows that already
                                   arrive fully resolved.

TRAKE has no single-shot candidates-in-CSV-out endpoint any more (unlike
KIS/VQA's /api/export) -- see backend/export.py's module docstring for why
its curate/cache/merge flow needs two separate calls instead.

CSV endpoints return CSV text directly rather than JSON (there's no
existing StreamingResponse/FileResponse use in this repo to match instead
-- PlainTextResponse is the simplest fit for small in-memory generated
content like this).
"""

import re
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .. import export as export_mod
from ..common import frame_idx_for_n, nearest_keyframe_n_for_frame_idx, thumbnail_url

router = APIRouter()


class ExportRequest(BaseModel):
    query_type: Literal["KIS", "VQA"]
    mode: Literal["confirmed", "unconfirmed"]
    candidates: list = []
    confirmed: Optional[dict] = None
    answers: list = []
    answer: str = ""
    neighbour_count: int = export_mod.DEFAULT_NEIGHBOUR_COUNT
    max_rows: int = 99
    filename: str = "export"
    # Export tab's "Keyframes" checkbox: True (default, old behavior) means
    # `confirmed`/`answers` carry {video_id, n} (an indexed keyframe).
    # False means they carry {video_id, frame_idx} instead -- a raw native
    # frame, e.g. picked straight from video playback rather than a
    # result card -- see backend/export.py's generate_export() docstring.
    keyframes: bool = True


class TrakeRowsRequest(BaseModel):
    video_id: str
    frame_idxs: list[int]
    max_rows: int = 99


class TrakeRow(BaseModel):
    video_id: str
    frame_idxs: list[int]


class TrakeWriteRequest(BaseModel):
    rows: list[TrakeRow]
    filename: str = "export"


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("_.")
    return (name or "export") + ".csv"


@router.post("/api/export")
def export_csv(body: ExportRequest):
    # Confirmed mode's "similar" tier is a fresh visual search seeded by
    # the confirmed frame itself, not the opener tab's original query
    # candidates -- see backend/export.py's module docstring and
    # similar_candidates_for_frame(). Unconfirmed mode has no single
    # confirmed frame to re-query from, so it keeps using whatever
    # candidates the frontend sent (the opener tab's last search results).
    candidates = body.candidates
    try:
        if body.mode == "confirmed" and body.confirmed:
            candidates = (
                export_mod.similar_candidates_for_frame(body.confirmed["video_id"], body.confirmed["n"])
                if body.keyframes else
                export_mod.similar_candidates_for_native_frame(body.confirmed["video_id"], body.confirmed["frame_idx"])
            )
        rows = export_mod.generate_export(
            candidates, body.mode,
            confirmed=body.confirmed, answers=body.answers,
            neighbour_count=body.neighbour_count, max_rows=body.max_rows,
            keyframes=body.keyframes,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    csv_text = export_mod.rows_to_csv_text(body.query_type, rows, answer=body.answer)
    filename = _safe_filename(body.filename)
    return PlainTextResponse(
        csv_text, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/api/export/trake-rows")
def get_trake_rows(body: TrakeRowsRequest):
    if not body.frame_idxs:
        raise HTTPException(400, "need at least one curated event")
    try:
        rows = export_mod.generate_trake_rows(body.video_id, body.frame_idxs, max_rows=body.max_rows)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"video_id": body.video_id, "rows": rows}


@router.post("/api/export/trake-write")
def write_trake_csv(body: TrakeWriteRequest):
    if not body.rows:
        raise HTTPException(400, "no rows to export -- select at least one cached video")
    rows = [{"video_id": r.video_id, "frame_idxs": r.frame_idxs} for r in body.rows]
    csv_text = export_mod.rows_to_csv_text("TRAKE", rows)
    filename = _safe_filename(body.filename)
    return PlainTextResponse(
        csv_text, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/api/export/frame")
def get_frame_info(video_id: str, n: int):
    frame_idx = frame_idx_for_n(video_id, n)
    if frame_idx is None:
        raise HTTPException(404, f"n={n} not found for {video_id}")
    return {"video_id": video_id, "n": n, "frame_idx": frame_idx, "thumbnail_url": thumbnail_url(video_id, n)}


@router.get("/api/export/neighbors")
def get_export_neighbors(video_id: str, n: int, count: int = export_mod.DEFAULT_NEIGHBOUR_COUNT):
    ns = export_mod.nearest_keyframes_by_time(video_id, n, count)
    return {
        "video_id": video_id, "center_n": n,
        "frames": [
            {"n": nb, "frame_idx": frame_idx_for_n(video_id, nb), "thumbnail_url": thumbnail_url(video_id, nb)}
            for nb in ns
        ],
    }


@router.get("/api/export/nearest-keyframe")
def get_nearest_keyframe(video_id: str, frame_idx: int):
    """Backs the Export tab's "Keyframes" checkbox: re-checking it while a
    raw native frame (frame_idx, no n) is curated snaps that frame to its
    nearest indexed keyframe, so the popup's keyframe-mode UI has an n to
    show/work with again -- see backend/common.py's
    nearest_keyframe_n_for_frame_idx()."""
    n = nearest_keyframe_n_for_frame_idx(video_id, frame_idx)
    if n is None:
        raise HTTPException(404, f"no keyframes indexed for {video_id}")
    return {"video_id": video_id, "n": n, "frame_idx": frame_idx_for_n(video_id, n), "thumbnail_url": thumbnail_url(video_id, n)}


@router.get("/api/export/similar")
def get_export_similar(video_id: str, n: int, count: int = export_mod.DEFAULT_SIMILAR_COUNT):
    """Confirmed mode's "Similars" preview -- same visual-search-by-frame
    pool /api/export itself uses to build the CSV in confirmed mode (see
    export_csv above), exposed so the popup can show it before exporting."""
    try:
        results = export_mod.similar_candidates_for_frame(video_id, n, k=count)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"video_id": video_id, "n": n, "results": results}
