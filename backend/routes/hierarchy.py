"""
backend/routes/hierarchy.py -- Hierarchy Search endpoints. Ports
ui/app.py:2112-2179. Two endpoints because step 3's drill-down is
per-video and re-triggerable independently (seed-frame change, "Expand"
button) without re-running step 1 -- the frontend caches each group's
step-1 `frames` list from the initial search response and resends it
verbatim on a later expand call, mirroring ui/app.py's own behavior
(`hierarchy_expand_group` always drills down from the ORIGINAL step-1
`groups[vid]` list, never a previously-drilled/compounded one -- see
ui/app.py:2162, which reads `groups[vid]` fresh, not the prior render's
drilled result).
"""

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from .. import config
from ..common import parse_lot_range
from ..search import hierarchy as hier_mod
from .query_image import resolve_query

router = APIRouter()


class HierarchySearchRequest(BaseModel):
    query: Optional[str] = None
    image_id: Optional[str] = None
    top_k: int = config.DISPLAY_N
    top_g: int = config.TOP_G_DEFAULT
    video_filter: str = ""
    lot_filter: str = ""


class HierarchyGroup(BaseModel):
    video_id: str
    best_rank: int
    score_label: str
    best_score_val: float
    step1_frames: list
    seed_options: List[int]
    top1_n: int
    drilled_frames: list


class HierarchySearchResponse(BaseModel):
    groups: List[HierarchyGroup] = []


@router.post("/api/search/hierarchy", response_model=HierarchySearchResponse)
def search_hierarchy(body: HierarchySearchRequest):
    query = resolve_query(body.query, body.image_id)
    fetch_k = max(config.FETCH_K, body.top_k)
    lot_filter = parse_lot_range(body.lot_filter)

    raw_groups = hier_mod.base_search_grouped(query, fetch_k, body.video_filter, lot_filter, body.top_k)

    groups = []
    for g in raw_groups:
        frames = g["frames"]
        best = frames[0]
        seed_options = [f["n"] for f in frames]
        top1_n = seed_options[0]
        drilled = hier_mod.hierarchy_expand_group(g["video_id"], frames, body.top_g, fetch_k, seed_n=top1_n)
        groups.append(HierarchyGroup(
            video_id=g["video_id"], best_rank=best["rank"], score_label=best["score_label"],
            best_score_val=best["score_val"], step1_frames=frames, seed_options=seed_options,
            top1_n=top1_n, drilled_frames=drilled,
        ))
    return HierarchySearchResponse(groups=groups)


class HierarchyExpandRequest(BaseModel):
    video_id: str
    step1_frames: list
    top_g: int = config.TOP_G_DEFAULT
    seed_n: Optional[int] = None
    top_k: int = config.DISPLAY_N


class HierarchyExpandResponse(BaseModel):
    frames: list = []


@router.post("/api/hierarchy/expand", response_model=HierarchyExpandResponse)
def expand_hierarchy(body: HierarchyExpandRequest):
    fetch_k = max(config.FETCH_K, body.top_k)
    frames = hier_mod.hierarchy_expand_group(body.video_id, body.step1_frames, body.top_g, fetch_k, seed_n=body.seed_n)
    return HierarchyExpandResponse(frames=frames)
