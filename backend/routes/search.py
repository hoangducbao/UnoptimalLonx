"""
backend/routes/search.py -- per-signal search endpoints. Each mirrors the
matching `if mode == "..."` block in ui/app.py 1:1 (same fetch_k
computation, same apply_filters-before-RRF ordering, same skip messages for
picture queries, same graceful-degrade-on-ES-down behavior). Phase 2 of the
rewrite adds ASR/Caption/OCR/Summary alongside Phase 1's Keyframe --
Mixed/TRAKE/Hierarchy land in later phases, same module.
"""

from typing import List, Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from .. import config
from .. import metadata_filter as md
from .. import od_filter as od
from ..common import apply_filters, df_to_results, parse_lot_range
from ..models import is_image_query, siglip2_truncation_warning
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
# calls (ui/app.py:2016-2017, 2042-2043, 2079-2080) so the API response
# reads the same regardless of which signal it came from.
_SKIP_NOTHING_TO_FUSE = "Skipped — picture queries only ever have one active leg (SigLIP2), nothing to fuse."


class LegResult(BaseModel):
    skipped: Optional[str] = None
    warning: Optional[str] = None
    results: list = []


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


class KeyframeSearchResponse(BaseModel):
    warning: Optional[str] = None
    results: list = []


# Keyframe is SigLIP2-only -- its CLIP ViT-B/32 leg (and the Multilingual-CLIP
# query-time text encoder behind it) was removed entirely, see
# backend/search/keyframe.py's docstring. No legs to choose, no RRF to fuse
# -- same single-leg shape as OCR below.
@router.post("/api/search/keyframe", response_model=KeyframeSearchResponse)
def search_keyframe(body: KeyframeSearchRequest):
    query = resolve_query(body.query, body.image_id)
    top_k = body.top_k
    fetch_k = max(config.FETCH_K, top_k)
    lot_filter = parse_lot_range(body.lot_filter, body.exclude_lot)
    od_matched, od_unmatched = od.match_classes(body.od_filter)
    od_warning = od.unmatched_warning(od_unmatched)
    # Keyframe is SigLIP2-only (see docstring above), so a too-long query
    # takes priority over the OD warning -- it affects every result, not
    # just the ones an unmatched OD class would have narrowed.
    trunc_warning = None if is_image_query(query) else siglip2_truncation_warning(query)

    df = apply_filters(kf.search_siglip2_frame(query, k=fetch_k), body.video_filter, lot_filter)
    df = md.apply_facet_filter(df, body.facet_field, body.facet_value)
    filtered = od.apply_od_filter(df, od_matched)
    return KeyframeSearchResponse(warning=trunc_warning or od_warning, results=df_to_results(filtered.head(top_k) if filtered is not None else None, "score"))


# ---------------------------------------------------------------------------
# ASR / Caption / Summary share one shape: a SigLIP2 leg + a fuzzy leg + RRF,
# all resolved to a keyframe `n` via that signal's attach_keyframe_* before
# df_to_results. OCR is the one-leg exception (its own endpoint, no RRF).
# ASR alone carries a fourth leg, `exact` (Elasticsearch match_phrase) --
# hence the default-off field on the shared request/response models below,
# which Caption/Summary simply never set or read.
# ---------------------------------------------------------------------------

class TextSignalLegs(BaseModel):
    siglip: bool = True
    fuzzy: bool = True
    exact: bool = False  # ASR only -- Caption/Summary never send or read it
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
    exact: Optional[LegResult] = None  # ASR only, None for Caption/Summary
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
    # Only the siglip leg (below) is subject to the 64-token cap -- the
    # fuzzy leg's own warning, further down, is untouched by this.
    trunc_warning = None if image_query else siglip2_truncation_warning(query)

    siglip_df = fuzzy_df = exact_df = None
    fuzzy_warning = exact_warning = None
    if body.legs.siglip or body.legs.rrf:
        siglip_df = apply_filters(asr_mod.search_siglip_asr(query, k=fetch_k), body.video_filter, lot_filter)
        siglip_df = md.apply_facet_filter(siglip_df, body.facet_field, body.facet_value)
    if body.legs.fuzzy or body.legs.rrf:
        fuzzy_df, fuzzy_warning = asr_mod.search_asr_fuzzy(query, k=fetch_k)
        fuzzy_df = apply_filters(fuzzy_df, body.video_filter, lot_filter)
        fuzzy_df = md.apply_facet_filter(fuzzy_df, body.facet_field, body.facet_value)
    if body.legs.exact or body.legs.rrf:
        exact_df, exact_warning = asr_mod.search_asr_exact(query, k=fetch_k)
        exact_df = apply_filters(exact_df, body.video_filter, lot_filter)
        exact_df = md.apply_facet_filter(exact_df, body.facet_field, body.facet_value)

    resp = TextSignalSearchResponse()
    if body.legs.siglip:
        filtered = od.apply_od_filter(asr_mod.attach_keyframe_asr(siglip_df), od_matched)
        resp.siglip = LegResult(warning=trunc_warning or od_warning, results=df_to_results(filtered.head(top_k), "score", "text"))
    if body.legs.fuzzy:
        filtered = od.apply_od_filter(asr_mod.attach_keyframe_asr(fuzzy_df), od_matched)
        resp.fuzzy = LegResult(warning=fuzzy_warning or od_warning, results=df_to_results(filtered.head(top_k), "score", "text"))
    if body.legs.exact:
        filtered = od.apply_od_filter(asr_mod.attach_keyframe_asr(exact_df), od_matched)
        resp.exact = LegResult(warning=exact_warning or od_warning, results=df_to_results(filtered.head(top_k), "score", "text"))
    if body.legs.rrf:
        if image_query:
            resp.rrf = LegResult(skipped=_SKIP_NOTHING_TO_FUSE)
        else:
            fused = asr_mod.attach_keyframe_asr(asr_mod.rrf_fuse_asr({"siglip_asr": siglip_df, "fuzzy": fuzzy_df, "exact": exact_df}, top_n=fetch_k))
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
    # Only the siglip leg (below) is subject to the 64-token cap -- the
    # fuzzy leg's own warning, further down, is untouched by this.
    trunc_warning = None if image_query else siglip2_truncation_warning(query)

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
        resp.siglip = LegResult(warning=trunc_warning or od_warning, results=df_to_results(filtered.head(top_k), "score", "text"))
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
    # Only the siglip leg (below) is subject to the 64-token cap -- the
    # fuzzy leg's own warning, further down, is untouched by this.
    trunc_warning = None if image_query else siglip2_truncation_warning(query)

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
        resp.siglip = LegResult(warning=trunc_warning, results=df_to_results(sum_mod.attach_keyframe_summary(siglip_df).head(top_k), "score", "text"))
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
    # OCR sub-queries have no SigLIP2 leg at all (search/ocr.py is
    # fuzzy-only), so they're exempt from the token-limit check below --
    # every other signal option here (Keyframe/ASR/Caption) always runs a
    # SigLIP2 leg internally, via trake_search_event().
    truncated_labels = []
    for i, q in enumerate(body.queries):
        text = q.text.strip()
        if not text or q.weight <= 0:
            continue
        if q.signal != "OCR" and siglip2_truncation_warning(text):
            truncated_labels.append(f"#{i + 1}")
        df = trake_mod.trake_search_event(
            text, q.signal, fetch_k, body.video_filter, lot_filter,
            facet_field=body.facet_field, facet_value=body.facet_value,
        )
        if df is not None and not df.empty:
            sub_dfs[i] = df[["video_id", "n", "rank"]]
            sub_weights[i] = q.weight

    if not sub_dfs:
        return MixedSearchResponse(empty=True)

    trunc_warning = None
    if truncated_labels:
        trunc_warning = (
            f"Sub-quer{'y' if len(truncated_labels) == 1 else 'ies'} "
            f"{', '.join(truncated_labels)} too long for SigLIP2's embedding legs -- "
            f"only the first ~64 tokens of each were used (fuzzy legs unaffected)."
        )

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
    return MixedSearchResponse(warning=trunc_warning or od_warning, results=results)
