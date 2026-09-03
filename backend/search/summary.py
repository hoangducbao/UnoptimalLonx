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
from ..common import faiss_search_pooled, l2_normalize, query_hash, video_id_from_filename
from ..es_client import get_es_client
from ..es_indexing import ensure_summary_fuzzy_index
from ..models import encode_text_siglip2, is_image_query, siglip2_query_mat

_index = None
_meta: pd.DataFrame = None


def ensure_summary_embeddings():
    """One-time SigLIP2 text-tower embed of every summaries/*.txt, saved to
    SUMMARY_EMBED_DIR so later runs don't re-embed (resumable: skips any
    video that already has a .npy on disk).

    Only for a profile whose summary vectors this app produces itself. A
    chunked profile's (config.SUMMARY_CHUNKED) come from an upstream
    pipeline that splits each summary to fit SigLIP2's 64-token text window
    *before* embedding; embedding a whole summary here would drop a single
    silently-truncated vector into an index where every other row is a
    chunk."""
    if config.SUMMARY_CHUNKED:
        return False
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
        index = faiss.IndexFlatIP(config.EMBED_DIM)
        rows = []
        gid = 0
        for npy_path in npy_paths:
            if npy_path.name == "summaries_all.npy":
                # A chunked profile ships one concatenation of every video's
                # chunks next to the per-video files; indexing it as well
                # would duplicate the whole corpus.
                continue
            video_id = video_id_from_filename(str(npy_path), ("_summary_siglip768",))
            txt_path = config.SUMMARY_DIR / f"{video_id}.txt"
            if not txt_path.exists():
                continue
            vecs = l2_normalize(np.load(npy_path))
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            if config.SUMMARY_CHUNKED:
                # One vector per chunk of the summary, each with its own text
                # in the sibling {video_id}.csv -- so a hit can show the
                # sentence that actually scored, not the whole paragraph.
                chunks_path = config.SUMMARY_EMBED_DIR / f"{video_id}.csv"
                if not chunks_path.exists():
                    continue
                chunks = pd.read_csv(chunks_path)
                if len(chunks) != vecs.shape[0]:
                    continue
                texts = [(int(r["chunk_index"]), r["text"]) for _, r in chunks.iterrows()]
            else:
                # One vector for the whole summary, text straight off disk.
                if vecs.shape[0] != 1:
                    continue
                texts = [(0, txt_path.read_text(encoding="utf-8").strip())]
            index.add(vecs)
            for chunk_index, text in texts:
                rows.append((gid, video_id, chunk_index, text))
                gid += 1
        faiss.write_index(index, str(config.SIGLIP_SUMMARY_FAISS))
        pd.DataFrame(rows, columns=["global_id", "video_id", "chunk_index", "text"]).to_csv(config.SIGLIP_SUMMARY_META, index=False)

    _index = faiss.read_index(str(config.SIGLIP_SUMMARY_FAISS))
    _meta = pd.read_csv(config.SIGLIP_SUMMARY_META)
    return _index, _meta


def _get_index():
    if _index is None:
        build_siglip_summary_index()
    return _index, _meta


_siglip_cache = TTLCache(maxsize=256, ttl=300)

# A chunked profile holds several rows per video (up to 6, mean 3.2 across
# the 1152 corpus's 2501 chunks over 785 videos), so a plain top-k FAISS
# search can spend most of its slots on one video. Overfetch by this factor
# and keep each video's best-scoring chunk -- a max-pool over chunks -- so
# the leg still hands exactly one row per video to rrf_fuse_summary, which
# keys on video_id alone. 4x covers the worst case comfortably.
_CHUNK_OVERFETCH = 4

_EMPTY_SIGLIP = pd.DataFrame(columns=["rank", "score", "video_id", "text"])


def search_siglip_summary(query, k: int = config.FETCH_K) -> pd.DataFrame:
    cache_key = (query_hash(query), k)
    if cache_key in _siglip_cache:
        return _siglip_cache[cache_key]
    index, meta = _get_index()
    # Two independent chunkings meet here and must not be confused: the
    # corpus rows are chunks of a *summary* (config.SUMMARY_CHUNKED, built
    # upstream), while the query matrix may hold chunks of a long *query*
    # (models.py). faiss_search_pooled RRF-fuses the second; the
    # drop_duplicates below max-pools the first. Both leave the rows in
    # descending score order, which is what keep="first" relies on.
    n = k * _CHUNK_OVERFETCH if config.SUMMARY_CHUNKED else k
    ids, scores = faiss_search_pooled(index, siglip2_query_mat(query), n, per_vec_k=n)
    rows = []
    for gid, score in zip(ids, scores):
        if gid == -1:
            continue
        row = meta.iloc[int(gid)]
        rows.append({"score": float(score), "video_id": row["video_id"], "text": row["text"]})
    if not rows:
        result = _EMPTY_SIGLIP
    else:
        # FAISS returns these in descending score order, so a video's first
        # row is already its best chunk -- keep="first" is the max-pool. On
        # an unchunked profile there is only ever one row per video, so this
        # is a no-op and the ranking is bit-for-bit what it always was.
        result = (pd.DataFrame(rows)
                  .drop_duplicates("video_id", keep="first")
                  .head(k)
                  .reset_index(drop=True))
        result["rank"] = np.arange(1, len(result) + 1)
        result = result[["rank", "score", "video_id", "text"]]
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
