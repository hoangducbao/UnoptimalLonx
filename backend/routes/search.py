"""
backend/routes/search.py -- per-signal search endpoints. Each mirrors the
matching `if mode == "..."` block in ui/app.py 1:1 (same fetch_k
computation, same apply_filters-before-RRF ordering, same skip messages for
picture queries, same graceful-degrade-on-ES-down behavior). Phase 2 of the
rewrite adds ASR/Caption/OCR/Summary alongside Phase 1's Keyframe --
Mixed/TRAKE/Hierarchy land in later phases, same module.
"""

from typing import Dict, List, Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from .. import config
from .. import metadata_filter as md
from .. import od_filter as od
from ..common import apply_filters, df_to_results, parse_lot_range
from ..models import is_image_query
from ..search import asr as asr_mod
from ..search import caption as cap_mod
from ..search import keyframe as kf
from ..search import mixed as mixed_mod
from ..search import ocr as ocr_mod
from ..search import summary as sum_mod
from ..search import trake as trake_mod
from .query_image import resolve_query

router = APIRouter()

# Shared skip-message text -- identical wording to ui/app.py's st.caption()
# calls (ui/app.py:1986, 1992, 2016-2017, 2042-2043, 2079-2080) so the API
# response reads the same regardless of which signal it came from.
_SKIP_SIGLIP2_ONLY = "Skipped — picture queries are SigLIP2-only."
_SKIP_NOTHING_TO_FUSE = "Skipped — picture queries only ever have one active leg (SigLIP2), nothing to fuse."


class LegResult(BaseModel):
    skipped: Optional[str] = None
    warning: Optional[str] = None
    results: list = []


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
    exclude_lot: bool = False
    od_filter: str = ""
    facet_field: str = ""
    facet_value: str = ""
    legs: KeyframeLegs = KeyframeLegs()


class KeyframeSearchResponse(BaseModel):
    siglip2: Optional[LegResult] = None
    clip: Optional[LegResult] = None
    rrf: Optional[LegResult] = None


@router.post("/api/search/keyframe", response_model=KeyframeSearchResponse)
def search_keyframe(body: KeyframeSearchRequest):
    query = resolve_query(body.query, body.image_id)
    top_k = body.top_k
    fetch_k = max(config.FETCH_K, top_k)
    lot_filter = parse_lot_range(body.lot_filter, body.exclude_lot)
    image_query = is_image_query(query)
    od_matched, od_unmatched = od.match_classes(body.od_filter)
    od_warning = od.unmatched_warning(od_unmatched)

    siglip2_df = clip_df = None
    if body.legs.siglip2 or body.legs.rrf:
        siglip2_df = apply_filters(kf.search_siglip2_frame(query, k=fetch_k), body.video_filter, lot_filter)
        siglip2_df = md.apply_facet_filter(siglip2_df, body.facet_field, body.facet_value)
    if body.legs.clip or body.legs.rrf:
        clip_df = apply_filters(kf.search_clip_frame(query, k=fetch_k), body.video_filter, lot_filter)
        clip_df = md.apply_facet_filter(clip_df, body.facet_field, body.facet_value)

    resp = KeyframeSearchResponse()
    if body.legs.siglip2:
        filtered = od.apply_od_filter(siglip2_df, od_matched)
        resp.siglip2 = LegResult(warning=od_warning, results=df_to_results(filtered.head(top_k), "score"))
    if body.legs.clip:
        if image_query:
            resp.clip = LegResult(skipped="Skipped — picture queries are SigLIP2-only.")
        else:
            filtered = od.apply_od_filter(clip_df, od_matched)
            resp.clip = LegResult(warning=od_warning, results=df_to_results(filtered.head(top_k), "score"))
    if body.legs.rrf:
        if image_query:
            resp.rrf = LegResult(skipped=_SKIP_NOTHING_TO_FUSE)
        else:
            fused = kf.rrf_fuse_frame([siglip2_df, clip_df], top_n=fetch_k)
            fused = od.apply_od_filter(fused, od_matched)
            resp.rrf = LegResult(warning=od_warning, results=df_to_results(fused.head(top_k), "rrf_score"))
    return resp


# ---------------------------------------------------------------------------
# ASR / Caption / Summary share one shape: a SigLIP2 leg + a fuzzy leg + RRF,
# all resolved to a keyframe `n` via that signal's attach_keyframe_* before
# df_to_results. OCR is the one-leg exception (its own endpoint, no RRF).
# ---------------------------------------------------------------------------

class TextSignalLegs(BaseModel):
    siglip: bool = True
    fuzzy: bool = True
    rrf: bool = True


class TextSignalSearchRequest(BaseModel):
    query: Optional[str] = None
    image_id: Optional[str] = None
    top_k: int = config.DISPLAY_N
    video_filter: str = ""
    lot_filter: str = ""
    exclude_lot: bool = False
    od_filter: str = ""  # applied by ASR/Caption; ignored by Summary (video-level, no per-frame OD)
    facet_field: str = ""
    facet_value: str = ""
    legs: TextSignalLegs = TextSignalLegs()


class TextSignalSearchResponse(BaseModel):
    siglip: Optional[LegResult] = None
    fuzzy: Optional[LegResult] = None
    rrf: Optional[LegResult] = None


@router.post("/api/search/asr", response_model=TextSignalSearchResponse)
def search_asr(body: TextSignalSearchRequest):
    query = resolve_query(body.query, body.image_id)
    top_k = body.top_k
    fetch_k = max(config.FETCH_K, top_k)
    lot_filter = parse_lot_range(body.lot_filter, body.exclude_lot)
    image_query = is_image_query(query)
    od_matched, od_unmatched = od.match_classes(body.od_filter)
    od_warning = od.unmatched_warning(od_unmatched)

    siglip_df = fuzzy_df = None
    fuzzy_warning = None
    if body.legs.siglip or body.legs.rrf:
        siglip_df = apply_filters(asr_mod.search_siglip_asr(query, k=fetch_k), body.video_filter, lot_filter)
        siglip_df = md.apply_facet_filter(siglip_df, body.facet_field, body.facet_value)
    if body.legs.fuzzy or body.legs.rrf:
        fuzzy_df, fuzzy_warning = asr_mod.search_asr_fuzzy(query, k=fetch_k)
        fuzzy_df = apply_filters(fuzzy_df, body.video_filter, lot_filter)
        fuzzy_df = md.apply_facet_filter(fuzzy_df, body.facet_field, body.facet_value)

    resp = TextSignalSearchResponse()
    if body.legs.siglip:
        filtered = od.apply_od_filter(asr_mod.attach_keyframe_asr(siglip_df), od_matched)
        resp.siglip = LegResult(warning=od_warning, results=df_to_results(filtered.head(top_k), "score", "text"))
    if body.legs.fuzzy:
        filtered = od.apply_od_filter(asr_mod.attach_keyframe_asr(fuzzy_df), od_matched)
        resp.fuzzy = LegResult(warning=fuzzy_warning or od_warning, results=df_to_results(filtered.head(top_k), "score", "text"))
    if body.legs.rrf:
        if image_query:
            resp.rrf = LegResult(skipped=_SKIP_NOTHING_TO_FUSE)
        else:
            fused = asr_mod.attach_keyframe_asr(asr_mod.rrf_fuse_asr({"siglip_asr": siglip_df, "fuzzy": fuzzy_df}, top_n=fetch_k))
            fused = od.apply_od_filter(fused, od_matched)
            resp.rrf = LegResult(warning=od_warning, results=df_to_results(fused.head(top_k), "rrf_score", "text"))
    return resp


@router.post("/api/search/caption", response_model=TextSignalSearchResponse)
def search_caption(body: TextSignalSearchRequest):
    query = resolve_query(body.query, body.image_id)
    top_k = body.top_k
    fetch_k = max(config.FETCH_K, top_k)
    lot_filter = parse_lot_range(body.lot_filter, body.exclude_lot)
    image_query = is_image_query(query)
    od_matched, od_unmatched = od.match_classes(body.od_filter)
    od_warning = od.unmatched_warning(od_unmatched)

    siglip_df = fuzzy_df = None
    fuzzy_warning = None
    if body.legs.siglip or body.legs.rrf:
        siglip_df = apply_filters(cap_mod.search_siglip_caption(query, k=fetch_k), body.video_filter, lot_filter)
        siglip_df = md.apply_facet_filter(siglip_df, body.facet_field, body.facet_value)
    if body.legs.fuzzy or body.legs.rrf:
        fuzzy_df, fuzzy_warning = cap_mod.search_caption_fuzzy(query, k=fetch_k)
        fuzzy_df = apply_filters(fuzzy_df, body.video_filter, lot_filter)
        fuzzy_df = md.apply_facet_filter(fuzzy_df, body.facet_field, body.facet_value)

    resp = TextSignalSearchResponse()
    if body.legs.siglip:
        filtered = od.apply_od_filter(cap_mod.attach_keyframe_caption(siglip_df), od_matched)
        resp.siglip = LegResult(warning=od_warning, results=df_to_results(filtered.head(top_k), "score", "text"))
    if body.legs.fuzzy:
        filtered = od.apply_od_filter(cap_mod.attach_keyframe_caption(fuzzy_df), od_matched)
        resp.fuzzy = LegResult(warning=fuzzy_warning or od_warning, results=df_to_results(filtered.head(top_k), "score", "text"))
    if body.legs.rrf:
        if image_query:
            resp.rrf = LegResult(skipped=_SKIP_NOTHING_TO_FUSE)
        else:
            fused = cap_mod.attach_keyframe_caption(cap_mod.rrf_fuse_caption({"siglip_caption": siglip_df, "fuzzy": fuzzy_df}, top_n=fetch_k))
            fused = od.apply_od_filter(fused, od_matched)
            resp.rrf = LegResult(warning=od_warning, results=df_to_results(fused.head(top_k), "rrf_score", "text"))
    return resp


@router.post("/api/search/summary", response_model=TextSignalSearchResponse)
def search_summary(body: TextSignalSearchRequest):
    query = resolve_query(body.query, body.image_id)
    top_k = body.top_k
    fetch_k = max(config.FETCH_K, top_k)
    lot_filter = parse_lot_range(body.lot_filter, body.exclude_lot)
    image_query = is_image_query(query)

    siglip_df = fuzzy_df = None
    fuzzy_warning = None
    if body.legs.siglip or body.legs.rrf:
        siglip_df = apply_filters(sum_mod.search_siglip_summary(query, k=fetch_k), body.video_filter, lot_filter)
        siglip_df = md.apply_facet_filter(siglip_df, body.facet_field, body.facet_value)
    if body.legs.fuzzy or body.legs.rrf:
        fuzzy_df, fuzzy_warning = sum_mod.search_summary_fuzzy(query, k=fetch_k)
        fuzzy_df = apply_filters(fuzzy_df, body.video_filter, lot_filter)
        fuzzy_df = md.apply_facet_filter(fuzzy_df, body.facet_field, body.facet_value)

    resp = TextSignalSearchResponse()
    if body.legs.siglip:
        resp.siglip = LegResult(results=df_to_results(sum_mod.attach_keyframe_summary(siglip_df).head(top_k), "score", "text"))
    if body.legs.fuzzy:
        resp.fuzzy = LegResult(warning=fuzzy_warning, results=df_to_results(sum_mod.attach_keyframe_summary(fuzzy_df).head(top_k), "score", "text"))
    if body.legs.rrf:
        if image_query:
            resp.rrf = LegResult(skipped=_SKIP_NOTHING_TO_FUSE)
        else:
            fused = sum_mod.attach_keyframe_summary(sum_mod.rrf_fuse_summary({"siglip_summary": siglip_df, "fuzzy": fuzzy_df}, top_n=top_k))
            resp.rrf = LegResult(results=df_to_results(fused, "rrf_score", "text"))
    return resp


# ---------------------------------------------------------------------------
# OCR: single leg by design, no embedding leg, no RRF (ui/app.py:684-757).
# ---------------------------------------------------------------------------

class OcrSearchRequest(BaseModel):
    query: Optional[str] = None
    image_id: Optional[str] = None
    top_k: int = config.DISPLAY_N
    video_filter: str = ""
    lot_filter: str = ""
    exclude_lot: bool = False
    od_filter: str = ""
    facet_field: str = ""
    facet_value: str = ""


class OcrSearchResponse(BaseModel):
    fuzzy: Optional[LegResult] = None
    image_query_unavailable: bool = False


@router.post("/api/search/ocr", response_model=OcrSearchResponse)
def search_ocr(body: OcrSearchRequest):
    query = resolve_query(body.query, body.image_id)
    if is_image_query(query):
        return OcrSearchResponse(image_query_unavailable=True)

    top_k = body.top_k
    fetch_k = max(config.FETCH_K, top_k)
    lot_filter = parse_lot_range(body.lot_filter, body.exclude_lot)
    od_matched, od_unmatched = od.match_classes(body.od_filter)
    od_warning = od.unmatched_warning(od_unmatched)

    df, warning = ocr_mod.search_ocr_fuzzy(query, k=fetch_k)
    df = apply_filters(df, body.video_filter, lot_filter)
    df = md.apply_facet_filter(df, body.facet_field, body.facet_value)
    filtered = od.apply_od_filter(ocr_mod.attach_keyframe_ocr(df), od_matched)
    results = df_to_results(filtered.head(top_k), "score", "text")
    return OcrSearchResponse(fuzzy=LegResult(warning=warning or od_warning, results=results))


# ---------------------------------------------------------------------------
# Mixed: many independent sub-queries, each with its own text and its own
# single signal (Keyframe/ASR/Caption/OCR -- no nested "Mixed", no Summary
# [video-level, always resolves to frame 1 -- reserved for TRAKE's context
# row instead, see backend/routes/trake.py], no reverse-image-search per
# sub-query), combined with a user-weighted RRF (0-3 per sub-query). Each
# sub-query is resolved via trake_search_event() -- the exact same
# per-signal search + internal RRF every standalone signal route and TRAKE
# event already use -- then rrf_fuse_weighted() combines them keyed by
# sub-query index (not signal name), so two sub-queries can share a signal
# without colliding. TRAKE's own per-event "Mixed" signal option (backed by
# the shared mixedConfig weights/legs) is unrelated and unchanged.
# ---------------------------------------------------------------------------

class MixedSubQuery(BaseModel):
    text: str
    signal: Literal["Keyframe", "ASR", "Caption", "OCR"]
    weight: int = 1


class MixedSearchRequest(BaseModel):
    queries: List[MixedSubQuery] = []
    top_k: int = config.DISPLAY_N
    video_filter: str = ""
    lot_filter: str = ""
    exclude_lot: bool = False
    od_filter: str = ""
    facet_field: str = ""
    facet_value: str = ""
    show_transcript: bool = False


class MixedSearchResponse(BaseModel):
    empty: bool = False
    warning: Optional[str] = None
    results: list = []


@router.post("/api/search/mixed", response_model=MixedSearchResponse)
def search_mixed(body: MixedSearchRequest):
    top_k = body.top_k
    fetch_k = max(config.FETCH_K, top_k)
    lot_filter = parse_lot_range(body.lot_filter, body.exclude_lot)
    od_matched, od_unmatched = od.match_classes(body.od_filter)
    od_warning = od.unmatched_warning(od_unmatched)

    sub_dfs, sub_weights = {}, {}
    for i, q in enumerate(body.queries):
        text = q.text.strip()
        if not text or q.weight <= 0:
            continue
        df = trake_mod.trake_search_event(
            text, q.signal, fetch_k, body.video_filter, lot_filter,
            facet_field=body.facet_field, facet_value=body.facet_value,
        )
        if df is not None and not df.empty:
            sub_dfs[i] = df[["video_id", "n", "rank"]]
            sub_weights[i] = q.weight

    if not sub_dfs:
        return MixedSearchResponse(empty=True)

    # trake_search_event() already applies the facet filter per sub-query,
    # so only OD filtering (which it doesn't apply) needs to happen here,
    # post-fusion, same as every per-signal route above.
    fused = mixed_mod.rrf_fuse_weighted(sub_dfs, sub_weights, top_n=fetch_k)
    fused = od.apply_od_filter(fused, od_matched)
    results = df_to_results(fused.head(top_k), "rrf_score")
    if body.show_transcript:
        # Attaches ASR transcript text under every result regardless of
        # which sub-query actually ranked it (unlike the per-signal routes'
        # `text_col`, which only ever carries text a signal's own search
        # produced) -- opt-in since it's an extra per-row lookup.
        for r in results:
            r["text"] = asr_mod.transcript_for_frame(r["video_id"], r["n"])
    return MixedSearchResponse(warning=od_warning, results=results)
