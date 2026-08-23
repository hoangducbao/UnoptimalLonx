"""
backend/search/summary.py -- Summary signal: SigLIP2-summary embeddings +
Elasticsearch fuzzy, RRF. Ported from ui/app.py:760-894. Video-level, not
frame-level: one row per video (its whole summary), so RRF keys on
video_id alone, and the displayed thumbnail is always that video's frame 1
(no per-summary frame to point at) -- see attach_keyframe_summary.
"""

import faiss
import numpy as np
import pandas as pd
from cachetools import TTLCache

from .. import config
from ..common import l2_normalize, query_hash, video_id_from_filename
from ..es_client import get_es_client
from ..es_indexing import ensure_summary_fuzzy_index
from ..models import encode_text_siglip2, is_image_query, siglip2_query_vec

_index = None
_meta: pd.DataFrame = None


def ensure_summary_embeddings():
    """One-time SigLIP2 text-tower embed of every summaries/*.txt, saved to
    SUMMARY_EMBED_DIR so later runs don't re-embed (resumable: skips any
    video that already has a .npy on disk)."""
    for txt_path in sorted(config.SUMMARY_DIR.glob("*.txt")):
        video_id = txt_path.stem
        out_path = config.SUMMARY_EMBED_DIR / f"{video_id}.npy"
        if out_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        vec = encode_text_siglip2([text])
        np.save(out_path, vec.astype("float32"))
    return True


def build_siglip_summary_index():
    global _index, _meta
    ensure_summary_embeddings()
    if not (config.SIGLIP_SUMMARY_FAISS.exists() and config.SIGLIP_SUMMARY_META.exists()):
        npy_paths = sorted(config.SUMMARY_EMBED_DIR.glob("*.npy"))  # was *_summary_siglip768.npy before the rename
        index = faiss.IndexFlatIP(768)
        rows = []
        gid = 0
        for npy_path in npy_paths:
            video_id = video_id_from_filename(str(npy_path), ("_summary_siglip768",))
            txt_path = config.SUMMARY_DIR / f"{video_id}.txt"
            if not txt_path.exists():
                continue
            vec = l2_normalize(np.load(npy_path))
            index.add(vec)
            rows.append((gid, video_id, txt_path.read_text(encoding="utf-8").strip()))
            gid += 1
        faiss.write_index(index, str(config.SIGLIP_SUMMARY_FAISS))
        pd.DataFrame(rows, columns=["global_id", "video_id", "text"]).to_csv(config.SIGLIP_SUMMARY_META, index=False)

    _index = faiss.read_index(str(config.SIGLIP_SUMMARY_FAISS))
    _meta = pd.read_csv(config.SIGLIP_SUMMARY_META)
    return _index, _meta


def _get_index():
    if _index is None:
        build_siglip_summary_index()
    return _index, _meta


_siglip_cache = TTLCache(maxsize=256, ttl=300)


def search_siglip_summary(query, k: int = config.FETCH_K) -> pd.DataFrame:
    cache_key = (query_hash(query), k)
    if cache_key in _siglip_cache:
        return _siglip_cache[cache_key]
    index, meta = _get_index()
    qvec = l2_normalize(siglip2_query_vec(query).reshape(1, -1))
    n = min(k, index.ntotal)
    scores, ids = index.search(qvec, n)
    rows = []
    for rank, (gid, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if gid == -1:
            continue
        row = meta.iloc[int(gid)]
        rows.append({"rank": rank, "score": float(score), "video_id": row["video_id"], "text": row["text"]})
    result = pd.DataFrame(rows)
    _siglip_cache[cache_key] = result
    return result


_fuzzy_cache = TTLCache(maxsize=256, ttl=300)
_EMPTY_FUZZY = pd.DataFrame(columns=["rank", "score", "video_id", "text"])


def search_summary_fuzzy(query, k: int = config.FETCH_K):
    if is_image_query(query):
        return _EMPTY_FUZZY, None
    cache_key = (query_hash(query), k)
    if cache_key in _fuzzy_cache:
        return _fuzzy_cache[cache_key], None
    try:
        ensure_summary_fuzzy_index()
        es = get_es_client()
        resp = es.search(index=config.ES_INDEX_SUMMARY, size=k, query={
            "match": {"text": {"query": query, "fuzziness": "AUTO"}}
        })
    except Exception as e:
        return _EMPTY_FUZZY, f"[Summary fuzzy] Elasticsearch not reachable at {config.ES_HOST} ({e}) — showing other legs only."

    rows = []
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        rows.append({"rank": rank, "score": float(hit["_score"]), "video_id": src["video_id"], "text": src["text"]})
    result = pd.DataFrame(rows)
    _fuzzy_cache[cache_key] = result
    return result, None


def rrf_fuse_summary(named_dfs: dict, k: int = config.RRF_K, top_n: int = config.DISPLAY_N) -> pd.DataFrame:
    scores, extra = {}, {}
    for df in named_dfs.values():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            key = row["video_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + row["rank"])
            extra.setdefault(key, {"text": row.get("text")})
    rows = [{"video_id": vid, "rrf_score": s, **extra[vid]} for vid, s in scores.items()]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("rrf_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.head(top_n)


def attach_keyframe_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Video-level: always point at frame 1 as the representative thumbnail."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["n"] = 1
    return out
