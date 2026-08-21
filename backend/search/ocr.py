"""
backend/search/ocr.py -- OCR signal: Elasticsearch fuzzy over per-frame OCR
text only. Ported from ui/app.py:684-757. Single leg by design (no
embedding leg, no RRF).
"""

import pandas as pd
from cachetools import TTLCache

from .. import config
from ..common import query_hash
from ..es_client import get_es_client
from ..es_indexing import ensure_ocr_fuzzy_index
from ..models import is_image_query

_fuzzy_cache = TTLCache(maxsize=256, ttl=300)
_EMPTY_FUZZY = pd.DataFrame(columns=["rank", "score", "video_id", "frame_id", "text"])


def search_ocr_fuzzy(query, k: int = config.FETCH_K):
    if is_image_query(query):
        return _EMPTY_FUZZY, None
    cache_key = (query_hash(query), k)
    if cache_key in _fuzzy_cache:
        return _fuzzy_cache[cache_key], None
    try:
        ensure_ocr_fuzzy_index()
        es = get_es_client()
        resp = es.search(index=config.ES_INDEX_OCR, size=k, query={
            "match": {"text": {"query": query, "fuzziness": "AUTO"}}
        })
    except Exception as e:
        return _EMPTY_FUZZY, f"[OCR fuzzy] Elasticsearch not reachable at {config.ES_HOST} ({e}) — showing no results."

    rows = []
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        rows.append({"rank": rank, "score": float(hit["_score"]), "video_id": src["video_id"],
                      "frame_id": src["frame_id"], "text": src["text"]})
    result = pd.DataFrame(rows)
    _fuzzy_cache[cache_key] = result
    return result, None


def attach_keyframe_ocr(df: pd.DataFrame) -> pd.DataFrame:
    """OCR is frame-level: frame_id == map-keyframes.n directly, no lookup needed."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["n"] = out["frame_id"].astype(int)
    return out
