"""
backend/search/mixed.py -- the legs+weights composite signal: a
user-weighted RRF across Keyframe/ASR/Caption/OCR (not Summary --
video-level, kept out of this signal). Ported from ui/app.py:896-978. For
each signal, only its *checked* legs contribute -- 2 checked legs are
fused with that signal's own existing rrf_fuse_* first (so this composite
reuses the exact same per-signal RRF already used standalone), 1 checked
leg is used at its own raw rank, 0 checked legs (or a 0 weight) drop the
signal entirely. The resulting per-signal rank lists are then combined
with a weighted RRF, keyed on (video_id, n) since every signal is
normalized to that shape already.

The standalone "Mixed" tab (`backend/routes/search.py::search_mixed`) no
longer uses `_mixed_*_df` below -- it moved to many independent
single-signal sub-queries (see that route's own docstring) -- but does
reuse `rrf_fuse_weighted()`, keyed by sub-query index instead of signal
name. `_mixed_*_df` themselves now only back TRAKE's per-event "Mixed"
signal option (`backend/search/trake.py::trake_search_event`), which still
works exactly as this module describes.
"""

import numpy as np
import pandas as pd

from .. import config
from ..common import apply_filters
from . import asr as asr_mod
from . import caption as cap_mod
from . import keyframe as kf
from . import ocr as ocr_mod


def _mixed_keyframe_df(query, fetch_k, video_filter, lot_filter, legs) -> pd.DataFrame:
    named = {}
    if legs.get("kf_siglip2"):
        named["siglip2"] = apply_filters(kf.search_siglip2_frame(query, k=fetch_k), video_filter, lot_filter)
    if legs.get("kf_clip"):
        named["clip"] = apply_filters(kf.search_clip_frame(query, k=fetch_k), video_filter, lot_filter)
    if not named:
        return None
    if len(named) == 1:
        df = next(iter(named.values())).copy()
        df["n"] = df["frame_id"] + 1
    else:
        df = kf.rrf_fuse_frame(list(named.values()), top_n=fetch_k)
    return df[["video_id", "n", "rank"]] if df is not None and not df.empty else None


def _mixed_asr_df(query, fetch_k, video_filter, lot_filter, legs) -> pd.DataFrame:
    named = {}
    if legs.get("asr_siglip"):
        named["siglip_asr"] = apply_filters(asr_mod.search_siglip_asr(query, k=fetch_k), video_filter, lot_filter)
    if legs.get("asr_fuzzy"):
        fuzzy_df, _warning = asr_mod.search_asr_fuzzy(query, k=fetch_k)
        named["fuzzy"] = apply_filters(fuzzy_df, video_filter, lot_filter)
    if not named:
        return None
    if len(named) == 1:
        df = asr_mod.attach_keyframe_asr(next(iter(named.values())))
    else:
        df = asr_mod.attach_keyframe_asr(asr_mod.rrf_fuse_asr(named, top_n=fetch_k))
    return df[["video_id", "n", "rank"]] if df is not None and not df.empty else None


def _mixed_caption_df(query, fetch_k, video_filter, lot_filter, legs) -> pd.DataFrame:
    named = {}
    if legs.get("cap_siglip"):
        named["siglip_caption"] = apply_filters(cap_mod.search_siglip_caption(query, k=fetch_k), video_filter, lot_filter)
    if legs.get("cap_fuzzy"):
        fuzzy_df, _warning = cap_mod.search_caption_fuzzy(query, k=fetch_k)
        named["fuzzy"] = apply_filters(fuzzy_df, video_filter, lot_filter)
    if not named:
        return None
    if len(named) == 1:
        df = cap_mod.attach_keyframe_caption(next(iter(named.values())))
    else:
        df = cap_mod.attach_keyframe_caption(cap_mod.rrf_fuse_caption(named, top_n=fetch_k))
    return df[["video_id", "n", "rank"]] if df is not None and not df.empty else None


def _mixed_ocr_df(query, fetch_k, video_filter, lot_filter) -> pd.DataFrame:
    """OCR has no leg choice -- its single Fuzzy OCR leg is used whenever
    its weight is > 0."""
    fuzzy_df, _warning = ocr_mod.search_ocr_fuzzy(query, k=fetch_k)
    df = ocr_mod.attach_keyframe_ocr(apply_filters(fuzzy_df, video_filter, lot_filter))
    return df[["video_id", "n", "rank"]] if df is not None and not df.empty else None


def rrf_fuse_weighted(signal_dfs: dict, weights: dict, k: int = config.RRF_K, top_n: int = config.DISPLAY_N) -> pd.DataFrame:
    """Weighted RRF across already-per-signal-ranked dfs, keyed on (video_id, n)."""
    scores: dict = {}
    for name, df in signal_dfs.items():
        w = weights.get(name, 0)
        if not w or df is None or df.empty:
            continue
        for _, row in df.iterrows():
            key = (row["video_id"], int(row["n"]))
            scores[key] = scores.get(key, 0.0) + w * (1.0 / (k + row["rank"]))
    rows = [{"video_id": vid, "n": n, "rrf_score": s} for (vid, n), s in scores.items()]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("rrf_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.head(top_n)
