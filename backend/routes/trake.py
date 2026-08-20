"""
backend/routes/trake.py -- TRAKE endpoint. Ports ui/app.py:2181-2262's
render block on top of backend/search/trake.py's ported logic. Each
candidate's matched events carry a thumbnail_url (frontend renders them
directly, same shape convention as everywhere else) and their own
`timestamp` (seconds) so the frontend can drive TRAKE's marker-bar
playback dialog without a second round-trip -- it only needs GET
/api/playback (already built) to resolve fps for the live timer.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from .. import config
from ..common import parse_lot_range, thumbnail_url
from ..search import trake as trake_mod

router = APIRouter()


class TrakeEvent(BaseModel):
    text: str
    signal: str


class TrakeSearchRequest(BaseModel):
    context: Optional[TrakeEvent] = None
    events: List[TrakeEvent]
    top_k: int = config.DISPLAY_N
    top_v: int = 10
    video_filter: str = ""
    lot_filter: str = ""
    mixed_weights: Dict[str, int] = {}
    mixed_legs: Dict[str, bool] = {}


class TrakeSearchResponse(BaseModel):
    message: Optional[str] = None
    candidates: list = []


@router.post("/api/search/trake", response_model=TrakeSearchResponse)
def search_trake(body: TrakeSearchRequest):
    texts = [ev.text.strip() for ev in body.events]
    if len(body.events) < 1 or not all(texts):
        return TrakeSearchResponse(message="Fill in every event's query text to search (minimum 1 event).")

    fetch_k = max(config.FETCH_K, body.top_k)
    lot_filter = parse_lot_range(body.lot_filter)
    ctx_text = (body.context.text.strip() if body.context else "")
    ctx_signal = body.context.signal if body.context else "Summary"

    def run_event(text, signal):
        return trake_mod.trake_search_event(
            text, signal, fetch_k, body.video_filter, lot_filter,
            mixed_weights=body.mixed_weights, mixed_legs=body.mixed_legs,
        )

    all_dfs, labels = [], []
    if ctx_text:
        all_dfs.append(run_event(ctx_text, ctx_signal))
        labels.append("E0")
    for i, ev in enumerate(body.events):
        all_dfs.append(run_event(ev.text, ev.signal))
        labels.append(f"E{i + 1}")

    # Context bonus: any video in the context query's own top-(Top-K/2)
    # candidates gets a flat score bump, independent of whether that video
    # also satisfies the ordered-chain match above.
    bonus_video_ids = None
    if ctx_text:
        ctx_df = all_dfs[0]
        if ctx_df is not None and not ctx_df.empty:
            half = max(1, body.top_k // 2)
            bonus_video_ids = set(ctx_df.sort_values("rank").head(half)["video_id"])

    candidates = trake_mod.trake_rank_videos(all_dfs, labels, body.top_v, bonus_video_ids=bonus_video_ids)

    for c in candidates:
        for e in c["events"]:
            if e["matched"]:
                e["thumbnail_url"] = thumbnail_url(e["video_id"], e["n"])

    message = None
    if not candidates:
        message = ("No video matches every event in the required order. Try broader event text, "
                   "fewer events, or a different signal per event.")
    return TrakeSearchResponse(message=message, candidates=candidates)
