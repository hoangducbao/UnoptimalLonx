"""
backend/routes/search.py -- per-signal search endpoints. Each mirrors the
matching `if mode == "..."` block in ui/app.py 1:1 (same fetch_k
computation, same apply_filters-before-RRF ordering, same skip messages for
picture queries). Phase 1 of the rewrite: Keyframe only -- ASR/Caption/OCR/
Summary/Mixed/TRAKE/Hierarchy land in later phases, same module.
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from .. import config
from ..common import apply_filters, df_to_results, parse_lot_range
from ..models import is_image_query
from ..search import keyframe as kf
from .query_image import resolve_query

router = APIRouter()


class KeyframeLegs(BaseModel):
    siglip2: bool = True
    clip: bool = True
    rrf: bool = True


class KeyframeSearchRequest(BaseModel):
    query: Optional[str] = None
    image_id: Optional[str] = None
    top_k: int = config.DISPLAY_N
    video_filter: str = ""
    lot_filter: str = ""
    legs: KeyframeLegs = KeyframeLegs()


class LegResult(BaseModel):
    skipped: Optional[str] = None
    results: list = []


class KeyframeSearchResponse(BaseModel):
    siglip2: Optional[LegResult] = None
    clip: Optional[LegResult] = None
    rrf: Optional[LegResult] = None


@router.post("/api/search/keyframe", response_model=KeyframeSearchResponse)
def search_keyframe(body: KeyframeSearchRequest):
    query = resolve_query(body.query, body.image_id)
    top_k = body.top_k
    fetch_k = max(config.FETCH_K, top_k)
    lot_filter = parse_lot_range(body.lot_filter)
    image_query = is_image_query(query)

    siglip2_df = clip_df = None
    if body.legs.siglip2 or body.legs.rrf:
        siglip2_df = apply_filters(kf.search_siglip2_frame(query, k=fetch_k), body.video_filter, lot_filter)
    if body.legs.clip or body.legs.rrf:
        clip_df = apply_filters(kf.search_clip_frame(query, k=fetch_k), body.video_filter, lot_filter)

    resp = KeyframeSearchResponse()
    if body.legs.siglip2:
        resp.siglip2 = LegResult(results=df_to_results(siglip2_df.head(top_k), "score"))
    if body.legs.clip:
        if image_query:
            resp.clip = LegResult(skipped="Skipped — picture queries are SigLIP2-only.")
        else:
            resp.clip = LegResult(results=df_to_results(clip_df.head(top_k), "score"))
    if body.legs.rrf:
        if image_query:
            resp.rrf = LegResult(skipped="Skipped — picture queries only ever have one active leg (SigLIP2), nothing to fuse.")
        else:
            fused = kf.rrf_fuse_frame([siglip2_df, clip_df], top_n=top_k)
            resp.rrf = LegResult(results=df_to_results(fused, "rrf_score"))
    return resp
