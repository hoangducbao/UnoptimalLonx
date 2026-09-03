"""
backend/search/keyframe.py -- Keyframe signal: SigLIP2 frame embeddings.
Ported from ui/app.py:340-412 (build_frame_index etc). CLIP ViT-B/32 +
its Multilingual-CLIP query-time text encoder (XLM-RoBERTa-large) were
removed from this signal entirely -- that text tower alone cost ~4.6GB RAM
once lazily loaded, dwarfing every other model/index in the system
combined, for a second frame-embedding leg that mostly duplicated
SigLIP2's own ranking. @st.cache_resource -> build_frame_index() is called
once at startup (backend/main.py's lifespan) and the result held in
_FRAME_INDICES; @st.cache_data -> a small TTLCache per search function
keyed on (query_hash, k).
"""

import glob as glob_mod

import faiss
import numpy as np
import pandas as pd
from cachetools import TTLCache

from .. import config
from ..common import l2_normalize, query_hash, video_id_from_filename
from ..models import siglip2_query_vec

# glob_pattern -> (faiss.IndexFlatIP, lookup_df) -- built once by build_frame_index()
_FRAME_INDICES: dict = {}


def build_frame_index(glob_pattern: str):
    npy_paths = sorted(glob_mod.glob(glob_pattern))
    if not npy_paths:
        print(f"[Keyframe Warning] No .npy files matched: {glob_pattern}. Initializing empty index (dim={config.EMBED_DIM}).")
        index = faiss.IndexFlatIP(config.EMBED_DIM)
        result = (index, pd.DataFrame(columns=["video_id", "frame_id"]))
        _FRAME_INDICES[glob_pattern] = result
        return result

    all_vecs = []
    lookup_rows = []
    for npy_path in npy_paths:
        video_id = video_id_from_filename(npy_path, ("_siglip768", "_siglip2"))
        vecs = np.load(npy_path).astype("float32")
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[1] != config.EMBED_DIM:
            continue
        for row_idx in range(len(vecs)):
            lookup_rows.append({"video_id": video_id, "frame_id": row_idx})
        all_vecs.append(vecs)

    if not all_vecs:
        print(f"[Keyframe Warning] No .npy files matched dimension {config.EMBED_DIM} for pattern: {glob_pattern}. Initializing empty index.")
        index = faiss.IndexFlatIP(config.EMBED_DIM)
        result = (index, pd.DataFrame(columns=["video_id", "frame_id"]))
        _FRAME_INDICES[glob_pattern] = result
        return result

    matrix = l2_normalize(np.vstack(all_vecs).astype("float32"))
    index = faiss.IndexFlatIP(config.EMBED_DIM)
    index.add(matrix)
    result = (index, pd.DataFrame(lookup_rows))
    _FRAME_INDICES[glob_pattern] = result
    return result


def _get_frame_index(glob_pattern: str):
    if glob_pattern not in _FRAME_INDICES:
        build_frame_index(glob_pattern)
    return _FRAME_INDICES[glob_pattern]


def _search_frame(index, lookup_df, qvec: np.ndarray, k: int) -> pd.DataFrame:
    if index.ntotal == 0 or lookup_df.empty:
        return pd.DataFrame(columns=["rank", "score", "video_id", "frame_id", "n"])
    q = l2_normalize(qvec.reshape(1, -1))
    n = min(k, index.ntotal)
    scores, ids = index.search(q, n)
    valid_mask = (ids[0] >= 0) & (ids[0] < len(lookup_df))
    valid_ids = ids[0][valid_mask]
    if len(valid_ids) == 0:
        return pd.DataFrame(columns=["rank", "score", "video_id", "frame_id", "n"])
    results = lookup_df.iloc[valid_ids].copy().reset_index(drop=True)
    results["score"] = scores[0][valid_mask]
    results["rank"] = np.arange(1, len(results) + 1)
    results["n"] = results["frame_id"] + 1
    return results[["rank", "score", "video_id", "frame_id", "n"]]


_siglip2_cache = TTLCache(maxsize=256, ttl=300)


def search_siglip2_frame(query, k: int = config.FETCH_K) -> pd.DataFrame:
    cache_key = (query_hash(query), k)
    if cache_key in _siglip2_cache:
        return _siglip2_cache[cache_key]
    index, lookup_df = _get_frame_index(config.FRAME_SIGLIP2_GLOB)
    qvec = siglip2_query_vec(query)
    result = _search_frame(index, lookup_df, qvec, k)
    _siglip2_cache[cache_key] = result
    return result


def search_siglip2_by_frame(video_id: str, n: int, k: int = config.FETCH_K) -> pd.DataFrame:
    """Same ranking as search_siglip2_frame(), but the query is an existing
    keyframe already in the index (by video_id + 1-indexed n) rather than
    typed text or a pasted image -- reuses that keyframe's own stored
    embedding (index.reconstruct) instead of re-encoding a thumbnail
    through SigLIP2, so it's exact and needs no image bytes at all. Used
    by the export popup's confirmed-mode "Similars" tier: a fresh visual
    search seeded by the confirmed frame itself, rather than reusing
    whatever the opener tab's last query happened to be. The queried frame
    itself always comes back as the top hit (score 1.0, cosine similarity
    with itself) -- callers that don't want it in the results filter it
    out themselves (see backend/export.py)."""
    index, lookup_df = _get_frame_index(config.FRAME_SIGLIP2_GLOB)
    frame_id = int(n) - 1
    matches = lookup_df.index[(lookup_df["video_id"] == video_id) & (lookup_df["frame_id"] == frame_id)]
    if len(matches) == 0:
        raise ValueError(f"no indexed frame for {video_id} n={n}")
    qvec = index.reconstruct(int(matches[0]))
    return _search_frame(index, lookup_df, qvec, k + 1)
