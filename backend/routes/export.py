"""
backend/routes/export.py -- export popup endpoints, on top of backend/
export.py's pure ranking/dedup logic:
  POST /api/export           -- generate + return the finished CSV text.
  GET  /api/export/frame     -- single frame's {n, frame_idx, thumbnail_url},
                                 for the popup's answer-area card(s).
  GET  /api/export/neighbors -- a frame's N nearest keyframes by time, for
                                 the popup's "Neighbours" preview section.

/api/export returns CSV text directly rather than JSON (there's no
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
from ..common import frame_idx_for_n, thumbnail_url

router = APIRouter()


class ExportRequest(BaseModel):
    query_type: Literal["KIS", "VQA", "TRAKE"]
    mode: Literal["confirmed", "unconfirmed"]
    candidates: list = []
    confirmed: Optional[dict] = None
    answers: list = []
    answer: str = ""
    neighbour_count: int = export_mod.DEFAULT_NEIGHBOUR_COUNT
    max_rows: int = 100
    filename: str = "export"


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("_.")
    return (name or "export") + ".csv"


@router.post("/api/export")
def export_csv(body: ExportRequest):
    try:
        rows = export_mod.generate_export(
            body.query_type, body.candidates, body.mode,
            confirmed=body.confirmed, answers=body.answers,
            neighbour_count=body.neighbour_count, max_rows=body.max_rows,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    csv_text = export_mod.rows_to_csv_text(body.query_type, rows, mode=body.mode, answer=body.answer)
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
