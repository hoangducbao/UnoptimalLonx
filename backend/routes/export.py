"""
backend/routes/export.py -- POST /api/export: thin endpoint on top of
backend/export.py's pure ranking/dedup logic, returning the finished CSV
text directly rather than JSON (there's no existing StreamingResponse/
FileResponse use in this repo to match instead -- PlainTextResponse is the
simplest fit for small in-memory generated content like this).
"""

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .. import export as export_mod

router = APIRouter()


class ExportRequest(BaseModel):
    query_type: Literal["KIS", "VQA", "TRAKE"]
    mode: Literal["confirmed", "unconfirmed"]
    candidates: list = []
    confirmed: Optional[dict] = None
    answer: str = ""
    max_rows: int = 100


@router.post("/api/export")
def export_csv(body: ExportRequest):
    try:
        rows = export_mod.generate_export(
            body.query_type, body.candidates, body.mode,
            confirmed=body.confirmed, max_rows=body.max_rows,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    csv_text = export_mod.rows_to_csv_text(body.query_type, rows, answer=body.answer)
    return PlainTextResponse(
        csv_text, media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="export.csv"'},
    )
