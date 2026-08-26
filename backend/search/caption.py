"""
backend/search/caption.py -- Caption signal: SigLIP2-caption embeddings +
Elasticsearch fuzzy, RRF. Ported from ui/app.py:558-683. Frame-level on
both legs (frame_id == map-keyframes.n directly), so RRF keys on frame_id
and `n` needs no time-based lookup at all.
"""

import faiss
import numpy as np
import pandas as pd
from cachetools import TTLCache

from .. import config
from ..common import l2_normalize, query_hash, video_id_from_filename
from ..es_client import get_es_client
from ..es_indexing import ensure_caption_fuzzy_index
from ..models import is_image_query, siglip2_query_vec

_index = None
_meta: pd.DataFrame = None


def build_siglip_caption_index():
    global _index, _meta
    if not (config.SIGLIP_CAPTION_FAISS.exists() and config.SIGLIP_CAPTION_META.exists()):
        # caption_embed/{video_id}.npy + {video_id}.csv (was siglip_caption/
        # {video_id}_caption_siglip768.npy + _frames.csv before the rename).
        npy_paths = sorted(config.SIGLIP_CAPTION_DIR.glob("*.npy"))
        index = faiss.IndexFlatIP(768)
        rows = []
        gid = 0
        for npy_path in npy_paths:
            video_id = video_id_from_filename(str(npy_path), ("_caption_siglip768", "_siglip768"))
            frames_path = config.SIGLIP_CAPTION_DIR / f"{video_id}.csv"
            if not frames_path.exists():
                candidates = [
                    config.SIGLIP_CAPTION_DIR / f"{video_id}_caption_siglip768_frames.csv",
                    config.SIGLIP_CAPTION_DIR / f"{video_id}_frames.csv",
                ] + list(config.SIGLIP_CAPTION_DIR.glob(f"{video_id}*.csv"))
                found = [c for c in candidates if c.exists()]
                if not found:
                    continue
                frames_path = found[0]
            vecs = l2_normalize(np.load(npy_path))
            frames = pd.read_csv(frames_path)
            if len(frames) != vecs.shape[0]:
                continue
            index.add(vecs)
            for _, r in frames.iterrows():
                rows.append((gid, video_id, int(r["frame_id"]), r.get("text", "")))
                gid += 1
        faiss.write_index(index, str(config.SIGLIP_CAPTION_FAISS))
        pd.DataFrame(rows, columns=["global_id", "video_id", "frame_id", "text"]).to_csv(config.SIGLIP_CAPTION_META, index=False)

    _index = faiss.read_index(str(config.SIGLIP_CAPTION_FAISS))
    _meta = pd.read_csv(config.SIGLIP_CAPTION_META)
    return _index, _meta


def _get_index():
    if _index is None:
        build_siglip_caption_index()
    return _index, _meta


_siglip_cache = TTLCache(maxsize=256, ttl=300)


def search_siglip_caption(query, k: int = config.FETCH_K) -> pd.DataFrame:
    cache_key = (query_hash(query), k)
    if cache_key in _siglip_cache:
        return _siglip_cache[cache_key]
    index, meta = _get_index()
    if index.ntotal == 0 or meta.empty:
        return pd.DataFrame(columns=["rank", "score", "video_id", "frame_id", "text"])
    qvec = l2_normalize(siglip2_query_vec(query).reshape(1, -1))
    n = min(k, index.ntotal)
    scores, ids = index.search(qvec, n)
    rows = []
    for rank, (gid, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if gid == -1 or gid >= len(meta):
            continue
        row = meta.iloc[int(gid)]
        rows.append({"rank": rank, "score": float(score), "video_id": row["video_id"],
                      "frame_id": int(row["frame_id"]), "text": row["text"]})
    result = pd.DataFrame(rows)
    _siglip_cache[cache_key] = result
    return result


_fuzzy_cache = TTLCache(maxsize=256, ttl=300)
_EMPTY_FUZZY = pd.DataFrame(columns=["rank", "score", "video_id", "frame_id", "text"])


def search_caption_fuzzy(query, k: int = config.FETCH_K):
    if is_image_query(query):
        return _EMPTY_FUZZY, None
    cache_key = (query_hash(query), k)
    if cache_key in _fuzzy_cache:
        return _fuzzy_cache[cache_key], None
    try:
        ensure_caption_fuzzy_index()
        es = get_es_client()
        resp = es.search(index=config.ES_INDEX_CAPTION, size=k, query={
            "match": {"text": {"query": query, "fuzziness": "AUTO"}}
        })
    except Exception as e:
        return _EMPTY_FUZZY, f"[Caption fuzzy] Elasticsearch not reachable at {config.ES_HOST} ({e}) — showing other legs only."

    rows = []
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        rows.append({"rank": rank, "score": float(hit["_score"]), "video_id": src["video_id"],
                      "frame_id": src["frame_id"], "text": src["text"]})
    result = pd.DataFrame(rows)
    _fuzzy_cache[cache_key] = result
    return result, None


def rrf_fuse_caption(named_dfs: dict, k: int = config.RRF_K, top_n: int = config.DISPLAY_N) -> pd.DataFrame:
    scores, extra = {}, {}
    for df in named_dfs.values():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            key = (row["video_id"], int(row["frame_id"]))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + row["rank"])
            extra.setdefault(key, {"text": row.get("text")})
    rows = [{"video_id": vid, "frame_id": fid, "rrf_score": s, **extra[(vid, fid)]}
            for (vid, fid), s in scores.items()]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("rrf_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.head(top_n)


def attach_keyframe_caption(df: pd.DataFrame) -> pd.DataFrame:
    """Captions are frame-level on both legs: frame_id == map-keyframes.n
    directly, no lookup needed."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["n"] = out["frame_id"].astype(int)
    return out
