"""
backend/routes/search.py -- per-signal search endpoints. Each mirrors the
matching `if mode == "..."` block in ui/app.py 1:1 (same fetch_k
computation, same apply_filters-before-RRF ordering, same skip messages for
picture queries, same graceful-degrade-on-ES-down behavior). Phase 2 of the
rewrite adds ASR/Caption/OCR/Summary alongside Phase 1's Keyframe --
Mixed/TRAKE/Hierarchy land in later phases, same module.
"""

from typing import Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from .. import config
from ..common import apply_filters, df_to_results, parse_lot_range
from ..models import is_image_query
from ..search import asr as asr_mod
from ..search import caption as cap_mod
from ..search import keyframe as kf
from ..search import mixed as mixed_mod
from ..search import ocr as ocr_mod
from ..search import summary as sum_mod
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
            resp.rrf = LegResult(skipped=_SKIP_NOTHING_TO_FUSE)
        else:
            fused = kf.rrf_fuse_frame([siglip2_df, clip_df], top_n=top_k)
            resp.rrf = LegResult(results=df_to_results(fused, "rrf_score"))
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

    siglip_df = fuzzy_df = None
    fuzzy_warning = None
    if body.legs.siglip or body.legs.rrf:
        siglip_df = apply_filters(asr_mod.search_siglip_asr(query, k=fetch_k), body.video_filter, lot_filter)
    if body.legs.fuzzy or body.legs.rrf:
        fuzzy_df, fuzzy_warning = asr_mod.search_asr_fuzzy(query, k=fetch_k)
        fuzzy_df = apply_filters(fuzzy_df, body.video_filter, lot_filter)

    resp = TextSignalSearchResponse()
    if body.legs.siglip:
        resp.siglip = LegResult(results=df_to_results(asr_mod.attach_keyframe_asr(siglip_df).head(top_k), "score", "text"))
    if body.legs.fuzzy:
        resp.fuzzy = LegResult(warning=fuzzy_warning, results=df_to_results(asr_mod.attach_keyframe_asr(fuzzy_df).head(top_k), "score", "text"))
    if body.legs.rrf:
        if image_query:
            resp.rrf = LegResult(skipped=_SKIP_NOTHING_TO_FUSE)
        else:
            fused = asr_mod.attach_keyframe_asr(asr_mod.rrf_fuse_asr({"siglip_asr": siglip_df, "fuzzy": fuzzy_df}, top_n=top_k))
            resp.rrf = LegResult(results=df_to_results(fused, "rrf_score", "text"))
    return resp


@router.post("/api/search/caption", response_model=TextSignalSearchResponse)
def search_caption(body: TextSignalSearchRequest):
    query = resolve_query(body.query, body.image_id)
    top_k = body.top_k
    fetch_k = max(config.FETCH_K, top_k)
    lot_filter = parse_lot_range(body.lot_filter, body.exclude_lot)
    image_query = is_image_query(query)

    siglip_df = fuzzy_df = None
    fuzzy_warning = None
    if body.legs.siglip or body.legs.rrf:
        siglip_df = apply_filters(cap_mod.search_siglip_caption(query, k=fetch_k), body.video_filter, lot_filter)
    if body.legs.fuzzy or body.legs.rrf:
        fuzzy_df, fuzzy_warning = cap_mod.search_caption_fuzzy(query, k=fetch_k)
        fuzzy_df = apply_filters(fuzzy_df, body.video_filter, lot_filter)

    resp = TextSignalSearchResponse()
    if body.legs.siglip:
        resp.siglip = LegResult(results=df_to_results(cap_mod.attach_keyframe_caption(siglip_df).head(top_k), "score", "text"))
    if body.legs.fuzzy:
        resp.fuzzy = LegResult(warning=fuzzy_warning, results=df_to_results(cap_mod.attach_keyframe_caption(fuzzy_df).head(top_k), "score", "text"))
    if body.legs.rrf:
        if image_query:
            resp.rrf = LegResult(skipped=_SKIP_NOTHING_TO_FUSE)
        else:
            fused = cap_mod.attach_keyframe_caption(cap_mod.rrf_fuse_caption({"siglip_caption": siglip_df, "fuzzy": fuzzy_df}, top_n=top_k))
            resp.rrf = LegResult(results=df_to_results(fused, "rrf_score", "text"))
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
    if body.legs.fuzzy or body.legs.rrf:
        fuzzy_df, fuzzy_warning = sum_mod.search_summary_fuzzy(query, k=fetch_k)
        fuzzy_df = apply_filters(fuzzy_df, body.video_filter, lot_filter)

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

    df, warning = ocr_mod.search_ocr_fuzzy(query, k=fetch_k)
    df = apply_filters(df, body.video_filter, lot_filter)
    results = df_to_results(ocr_mod.attach_keyframe_ocr(df).head(top_k), "score", "text")
    return OcrSearchResponse(fuzzy=LegResult(warning=warning, results=results))


# ---------------------------------------------------------------------------
# Mixed: a user-weighted RRF across Keyframe/ASR/Caption/OCR (ui/app.py:
# 896-978, 2089-2110). `weights`/`legs` come from the frontend's one shared
# mixedConfig (state.js) -- the backend stays stateless per request, see
# the rewrite plan's Decisions section 2.
# ---------------------------------------------------------------------------

class MixedSearchRequest(BaseModel):
    query: Optional[str] = None
    image_id: Optional[str] = None
    top_k: int = config.DISPLAY_N
    video_filter: str = ""
    lot_filter: str = ""
    exclude_lot: bool = False
    weights: Dict[str, int] = {}
    legs: Dict[str, bool] = {}


class MixedSearchResponse(BaseModel):
    empty: bool = False
    results: list = []


@router.post("/api/search/mixed", response_model=MixedSearchResponse)
def search_mixed(body: MixedSearchRequest):
    query = resolve_query(body.query, body.image_id)
    top_k = body.top_k
    fetch_k = max(config.FETCH_K, top_k)
    lot_filter = parse_lot_range(body.lot_filter, body.exclude_lot)

    signal_dfs = {}
    if body.weights.get("Keyframe", 0):
        signal_dfs["Keyframe"] = mixed_mod._mixed_keyframe_df(query, fetch_k, body.video_filter, lot_filter, body.legs)
    if body.weights.get("ASR", 0):
        signal_dfs["ASR"] = mixed_mod._mixed_asr_df(query, fetch_k, body.video_filter, lot_filter, body.legs)
    if body.weights.get("Caption", 0):
        signal_dfs["Caption"] = mixed_mod._mixed_caption_df(query, fetch_k, body.video_filter, lot_filter, body.legs)
    if body.weights.get("OCR", 0):
        signal_dfs["OCR"] = mixed_mod._mixed_ocr_df(query, fetch_k, body.video_filter, lot_filter)

    if not signal_dfs:
        return MixedSearchResponse(empty=True)

    fused = mixed_mod.rrf_fuse_weighted(signal_dfs, body.weights, top_n=top_k)
    return MixedSearchResponse(results=df_to_results(fused, "rrf_score"))
