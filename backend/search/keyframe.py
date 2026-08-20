"""
backend/search/keyframe.py -- Keyframe signal: SigLIP2 + CLIP ViT-B/32 frame
embeddings, RRF. Ported from ui/app.py:340-412 (build_frame_index through
rrf_fuse_frame). @st.cache_resource -> build_frame_index() is called once
per glob at startup (backend/main.py's lifespan) and the result held in
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
from ..models import is_image_query, siglip2_query_vec

import clip_encoder  # pipeline/clip_encoder.py -- Multilingual-CLIP text tower

# glob_pattern -> (faiss.IndexFlatIP, lookup_df) -- built once by build_frame_index()
_FRAME_INDICES: dict = {}


def build_frame_index(glob_pattern: str):
    npy_paths = sorted(glob_mod.glob(glob_pattern))
    if not npy_paths:
        raise FileNotFoundError(f"no .npy files matched: {glob_pattern}")

    all_vecs = []
    lookup_rows = []
    for npy_path in npy_paths:
        video_id = video_id_from_filename(npy_path, ("_viclip768", "_clip32", "_siglip768", "_siglip2"))
        vecs = np.load(npy_path).astype("float32")
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        for row_idx in range(len(vecs)):
            lookup_rows.append({"video_id": video_id, "frame_id": row_idx})
        all_vecs.append(vecs)

    matrix = l2_normalize(np.vstack(all_vecs).astype("float32"))
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    result = (index, pd.DataFrame(lookup_rows))
    _FRAME_INDICES[glob_pattern] = result
    return result


def _get_frame_index(glob_pattern: str):
    if glob_pattern not in _FRAME_INDICES:
        build_frame_index(glob_pattern)
    return _FRAME_INDICES[glob_pattern]


def _search_frame(index, lookup_df, qvec: np.ndarray, k: int) -> pd.DataFrame:
    q = l2_normalize(qvec.reshape(1, -1))
    n = min(k, index.ntotal)
    scores, ids = index.search(q, n)
    results = lookup_df.iloc[ids[0]].copy().reset_index(drop=True)
    results["score"] = scores[0]
    results["rank"] = np.arange(1, len(results) + 1)
    results["n"] = results["frame_id"] + 1
    return results[["rank", "score", "video_id", "frame_id", "n"]]


_siglip2_cache = TTLCache(maxsize=256, ttl=300)
_clip_cache = TTLCache(maxsize=256, ttl=300)


def search_siglip2_frame(query, k: int = config.FETCH_K) -> pd.DataFrame:
    cache_key = (query_hash(query), k)
    if cache_key in _siglip2_cache:
        return _siglip2_cache[cache_key]
    index, lookup_df = _get_frame_index(config.FRAME_SIGLIP2_GLOB)
    qvec = siglip2_query_vec(query)
    result = _search_frame(index, lookup_df, qvec, k)
    _siglip2_cache[cache_key] = result
    return result


def search_clip_frame(query, k: int = config.FETCH_K) -> pd.DataFrame:
    if is_image_query(query):
        # Picture queries are SigLIP2-only -- CLIP ViT-B/32 here is the
        # Multilingual-CLIP *text* tower's paired space, no image tower
        # wired up for it, so this leg contributes nothing to an image query.
        return pd.DataFrame(columns=["rank", "score", "video_id", "frame_id", "n"])
    cache_key = (query_hash(query), k)
    if cache_key in _clip_cache:
        return _clip_cache[cache_key]
    index, lookup_df = _get_frame_index(config.FRAME_CLIP_GLOB)
    qvec = clip_encoder.encode_text([query])[0]
    result = _search_frame(index, lookup_df, qvec, k)
    _clip_cache[cache_key] = result
    return result


def rrf_fuse_frame(dfs: list, k: int = config.RRF_K, top_n: int = config.DISPLAY_N) -> pd.DataFrame:
    scores, extra = {}, {}
    for df in dfs:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            key = (row["video_id"], int(row["frame_id"]))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + row["rank"])
            extra.setdefault(key, {"n": int(row["frame_id"]) + 1})
    rows = [{"video_id": vid, "frame_id": fid, "rrf_score": s, **extra[(vid, fid)]}
            for (vid, fid), s in scores.items()]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("rrf_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.head(top_n)
