"""
backend/search/trake.py -- TRAKE: an ordered list of event sub-queries
(each with its own text + one of Keyframe/ASR/Caption/OCR/Mixed), find
videos where every event's best-matching frame occurs in the declared
order, plus an optional context (E0) query always matched via Summary --
reserved for that role since Summary is video-level (always resolves to
frame 1, see attach_keyframe_summary), not one of the events' own signal
choices. Ported from ui/app.py:980-1113 (trake_search_event,
trake_rank_videos). Reuses each signal's existing search + RRF pipeline
UNSCOPED by default (no video/lot filter -- TRAKE searches the whole
corpus per event, on purpose, though the caller may still pass a filter to
narrow every event the same way the UI lets a user do for other signals),
then adds a video-level coverage/order/score layer on top. No new
embedding models, no new per-signal fusion logic.
"""

import pandas as pd

from .. import config
from .. import metadata_filter as md
from ..common import apply_filters, keyframe_timestamp
from . import asr as asr_mod
from . import caption as cap_mod
from . import keyframe as kf
from . import mixed as mixed_mod
from . import ocr as ocr_mod
from . import summary as sum_mod

_EMPTY = pd.DataFrame(columns=["video_id", "n", "rank", "score", "text"])


def trake_search_event(query: str, signal: str, fetch_k: int, video_filter: str = "", lot_filter=None,
                        mixed_weights: dict = None, mixed_legs: dict = None,
                        facet_field: str = "", facet_value: str = "") -> pd.DataFrame:
    """One TRAKE event's candidate frames for `signal`, normalized to
    [video_id, n, rank, score] (+text where the signal carries one).
    Dispatches to the exact same search + RRF calls the standalone signal
    endpoints already make."""
    if not query:
        return _EMPTY

    # video/lot scope + the metadata facet filter together, same video-level
    # tier -- every apply_filters(...) call below goes through this instead
    # of being paired with its own separate apply_facet_filter call.
    def _scoped(df):
        return md.apply_facet_filter(apply_filters(df, video_filter, lot_filter), facet_field, facet_value)

    if signal == "Keyframe":
        df = _scoped(kf.search_siglip2_frame(query, k=fetch_k))
        return _EMPTY if df is None or df.empty else df[["video_id", "n", "rank", "score"]]

    if signal == "ASR":
        # Deliberately siglip + fuzzy only, no `exact` leg: this fusion backs
        # every Mixed sub-query and TRAKE event, so adding a third input here
        # would silently reshuffle existing Mixed/TRAKE results. The
        # standalone /api/search/asr RRF does fuse exact -- the two diverge
        # on purpose.
        siglip_df = _scoped(asr_mod.search_siglip_asr(query, k=fetch_k))
        fuzzy_raw, _w = asr_mod.search_asr_fuzzy(query, k=fetch_k)
        fuzzy_df = _scoped(fuzzy_raw)
        fused = asr_mod.attach_keyframe_asr(asr_mod.rrf_fuse_asr({"siglip_asr": siglip_df, "fuzzy": fuzzy_df}, top_n=fetch_k))
        return _EMPTY if fused is None or fused.empty else fused.rename(columns={"rrf_score": "score"})[["video_id", "n", "rank", "score", "text"]]

    if signal == "Caption":
        siglip_df = _scoped(cap_mod.search_siglip_caption(query, k=fetch_k))
        fuzzy_raw, _w = cap_mod.search_caption_fuzzy(query, k=fetch_k)
        fuzzy_df = _scoped(fuzzy_raw)
        fused = cap_mod.attach_keyframe_caption(cap_mod.rrf_fuse_caption({"siglip_caption": siglip_df, "fuzzy": fuzzy_df}, top_n=fetch_k))
        return _EMPTY if fused is None or fused.empty else fused.rename(columns={"rrf_score": "score"})[["video_id", "n", "rank", "score", "text"]]

    if signal == "OCR":
        fuzzy_raw, _w = ocr_mod.search_ocr_fuzzy(query, k=fetch_k)
        df = ocr_mod.attach_keyframe_ocr(_scoped(fuzzy_raw))
        return _EMPTY if df is None or df.empty else df[["video_id", "n", "rank", "score", "text"]]

    if signal == "Summary":
        siglip_df = _scoped(sum_mod.search_siglip_summary(query, k=fetch_k))
        fuzzy_raw, _w = sum_mod.search_summary_fuzzy(query, k=fetch_k)
        fuzzy_df = _scoped(fuzzy_raw)
        fused = sum_mod.attach_keyframe_summary(sum_mod.rrf_fuse_summary({"siglip_summary": siglip_df, "fuzzy": fuzzy_df}, top_n=fetch_k))
        return _EMPTY if fused is None or fused.empty else fused.rename(columns={"rrf_score": "score"})[["video_id", "n", "rank", "score", "text"]]

    if signal == "Mixed":
        # rrf_fuse_weighted only ever reads "rank" off its inputs (never a
        # score column), so the existing _mixed_*_df helpers -- already
        # trimmed to [video_id, n, rank] -- are safe to reuse unchanged;
        # the per-event "score" below is the weighted-RRF score, same as
        # standalone Mixed mode's own rrf_score. Facet filtering happens
        # once, post-fusion, same as standalone Mixed's own route.
        weights = mixed_weights or {}
        legs = mixed_legs or {}
        signal_dfs = {}
        if weights.get("Keyframe", 0):
            signal_dfs["Keyframe"] = mixed_mod._mixed_keyframe_df(query, fetch_k, video_filter, lot_filter)
        if weights.get("ASR", 0):
            signal_dfs["ASR"] = mixed_mod._mixed_asr_df(query, fetch_k, video_filter, lot_filter, legs)
        if weights.get("Caption", 0):
            signal_dfs["Caption"] = mixed_mod._mixed_caption_df(query, fetch_k, video_filter, lot_filter, legs)
        if weights.get("OCR", 0):
            signal_dfs["OCR"] = mixed_mod._mixed_ocr_df(query, fetch_k, video_filter, lot_filter)
        if not signal_dfs:
            return _EMPTY
        fused = mixed_mod.rrf_fuse_weighted(signal_dfs, weights, top_n=fetch_k)
        fused = md.apply_facet_filter(fused, facet_field, facet_value)
        return _EMPTY if fused.empty else fused.rename(columns={"rrf_score": "score"})[["video_id", "n", "rank", "score"]]

    raise ValueError(f"unknown TRAKE event signal: {signal!r}")


def trake_rank_videos(event_dfs: list, labels: list, top_n: int,
                       bonus_video_ids: set = None, bonus_multiplier: float = 1.10) -> list:
    """Group each event's candidates by video (keeping only the
    best-ranked hit per video per event), hard-drop any video whose
    matched-event timestamps aren't strictly increasing in the declared
    order, then rank survivors by coverage * mean(score across matched
    events). `labels` is the display label per event_dfs slot (e.g. "E0"
    for an included context query, "E1"/"E2"/... for the required events).
    `bonus_video_ids` (from the optional context query's own top-K/2
    results) gets a flat score multiplier -- applied before the final
    sort/top_n cut so it can actually affect which videos make the cut,
    not just their displayed order within it."""
    n_events = len(event_dfs)
    best_by_video: dict = {}
    for i, df in enumerate(event_dfs):
        if df is None or df.empty:
            continue
        best = df.sort_values("rank").groupby("video_id", as_index=False).first()
        for _, row in best.iterrows():
            best_by_video.setdefault(row["video_id"], {})[i] = row

    candidates = []
    for video_id, matched in best_by_video.items():
        events, timestamps = [], []
        for i in range(n_events):
            row = matched.get(i)
            if row is None:
                events.append({"event_index": i, "label": labels[i], "matched": False, "timestamp": None})
                continue
            ts, _fps = keyframe_timestamp(video_id, row["n"])
            text = row.get("text")
            events.append({
                "event_index": i, "label": labels[i], "matched": True, "video_id": video_id,
                "n": int(row["n"]), "rank": int(row["rank"]), "score_label": "score",
                "score_val": float(row["score"]), "text": text if isinstance(text, str) else None,
                "timestamp": ts,
            })
            if ts is not None:
                timestamps.append(ts)

        # Order check runs over resolved timestamps in declared query order
        # (events list above is already built in that order) -- hard rule:
        # any inversion drops the video outright.
        if any(timestamps[j] >= timestamps[j + 1] for j in range(len(timestamps) - 1)):
            continue

        matched_events = [e for e in events if e["matched"]]
        video_score = (len(matched_events) / n_events) * (sum(e["score_val"] for e in matched_events) / len(matched_events))
        if bonus_video_ids and video_id in bonus_video_ids:
            video_score *= bonus_multiplier
        candidates.append({
            "video_id": video_id, "video_score": video_score, "order_valid": True,
            "coverage": len(matched_events) / n_events, "events": events,
        })

    candidates.sort(key=lambda c: c["video_score"], reverse=True)
    return candidates[:top_n]
