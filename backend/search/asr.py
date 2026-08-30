"""
backend/search/asr.py -- ASR signal: SigLIP2-ASR embeddings + Elasticsearch
fuzzy, RRF. Ported from ui/app.py:415-555. Segment-level (the ES leg has no
direct frame_id), so RRF keys on segment_id; `n` is resolved per-leg
(direct for SigLIP2-ASR, nearest-time for fuzzy) via attach_keyframe_asr.
"""

import faiss
import numpy as np
import pandas as pd
from cachetools import TTLCache

from .. import config
from ..common import keyframe_timestamp, l2_normalize, nearest_keyframe_n_by_time, query_hash, video_id_from_filename
from ..es_client import get_es_client
from ..es_indexing import ensure_asr_fuzzy_index
from ..models import is_image_query, siglip2_query_vec

_index = None
_meta: pd.DataFrame = None


def build_siglip_asr_index():
    global _index, _meta
    if not (config.SIGLIP_ASR_FAISS.exists() and config.SIGLIP_ASR_META.exists()):
        # transcript_embed/{video_id}.npy + {video_id}.csv (was asr_embed/
        # {video_id}_asr_siglip768.npy + _frames.csv before the rename).
        npy_paths = sorted(config.ASR_EMBED_DIR.glob("*.npy"))
        index = faiss.IndexFlatIP(768)
        rows = []
        gid = 0
        for npy_path in npy_paths:
            video_id = video_id_from_filename(str(npy_path), ("_asr_siglip768",))
            frames_path = config.ASR_EMBED_DIR / f"{video_id}.csv"
            if not frames_path.exists():
                continue
            vecs = l2_normalize(np.load(npy_path))
            frames = pd.read_csv(frames_path)
            if len(frames) != vecs.shape[0]:
                continue
            index.add(vecs)
            for _, r in frames.iterrows():
                rows.append((gid, video_id, int(r["frame_id"]), int(r["segment_id"]),
                             float(r["start_sec"]), r["text"]))
                gid += 1
        faiss.write_index(index, str(config.SIGLIP_ASR_FAISS))
        pd.DataFrame(rows, columns=["global_id", "video_id", "frame_id", "segment_id", "start_sec", "text"]).to_csv(config.SIGLIP_ASR_META, index=False)

    _index = faiss.read_index(str(config.SIGLIP_ASR_FAISS))
    _meta = pd.read_csv(config.SIGLIP_ASR_META)
    return _index, _meta


def _get_index():
    if _index is None:
        build_siglip_asr_index()
    return _index, _meta


_siglip_cache = TTLCache(maxsize=256, ttl=300)


def search_siglip_asr(query, k: int = config.FETCH_K) -> pd.DataFrame:
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
        rows.append({"rank": rank, "score": float(score), "video_id": row["video_id"],
                      "segment_id": int(row["segment_id"]), "frame_id": int(row["frame_id"]),
                      "start_sec": row["start_sec"], "text": row["text"]})
    result = pd.DataFrame(rows)
    _siglip_cache[cache_key] = result
    return result


_fuzzy_cache = TTLCache(maxsize=256, ttl=300)
_EMPTY_FUZZY = pd.DataFrame(columns=["rank", "score", "video_id", "segment_id", "start_sec", "text"])


def search_asr_fuzzy(query, k: int = config.FETCH_K):
    """Returns (df, warning). warning is a str (mirrors ui/app.py's
    st.warning) if ES was unreachable, else None -- fuzzy legs need text,
    so an image query short-circuits to an empty df with no warning."""
    if is_image_query(query):
        return _EMPTY_FUZZY, None
    cache_key = (query_hash(query), k)
    if cache_key in _fuzzy_cache:
        return _fuzzy_cache[cache_key], None
    try:
        ensure_asr_fuzzy_index()
        es = get_es_client()
        resp = es.search(index=config.ES_INDEX_ASR, size=k, query={
            "match": {"text": {"query": query, "fuzziness": "AUTO"}}
        })
    except Exception as e:
        return _EMPTY_FUZZY, f"[ASR fuzzy] Elasticsearch not reachable at {config.ES_HOST} ({e}) — showing other legs only."

    rows = []
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        rows.append({"rank": rank, "score": float(hit["_score"]), "video_id": src["video_id"],
                      "segment_id": src["segment_id"], "start_sec": src["start_sec"], "text": src["text"]})
    result = pd.DataFrame(rows)
    _fuzzy_cache[cache_key] = result
    return result, None


def rrf_fuse_asr(named_dfs: dict, k: int = config.RRF_K, top_n: int = config.DISPLAY_N) -> pd.DataFrame:
    scores, extra = {}, {}
    for df in named_dfs.values():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            key = (row["video_id"], int(row["segment_id"]))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + row["rank"])
            e = extra.setdefault(key, {"text": row.get("text"), "start_sec": row.get("start_sec"), "frame_id": row.get("frame_id")})
            if pd.isna(e.get("frame_id")) and not pd.isna(row.get("frame_id", np.nan)):
                e["frame_id"] = row["frame_id"]
    rows = [{"video_id": vid, "segment_id": sid, "rrf_score": s, **extra[(vid, sid)]}
            for (vid, sid), s in scores.items()]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("rrf_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.head(top_n)


def attach_keyframe_asr(df: pd.DataFrame) -> pd.DataFrame:
    """ASR is segment-level: resolve `n` directly when frame_id is carried
    (the SigLIP2-ASR leg), else fall back to nearest-time (the fuzzy leg)."""
    if df is None or df.empty:
        return df
    ns = []
    for _, row in df.iterrows():
        fid = row.get("frame_id", np.nan)
        if pd.notna(fid):
            ns.append(int(fid))
        else:
            ns.append(nearest_keyframe_n_by_time(row["video_id"], row.get("start_sec")))
    out = df.copy()
    out["n"] = ns
    return out


_transcript_cache: dict = {}


def _load_transcript(video_id: str):
    """Per-video transcript segments (segment_id, start_sec, text), sorted
    by start_sec -- same source file ensure_asr_fuzzy_index() bulk-indexes
    from, read directly here instead of round-tripping through ES."""
    if video_id not in _transcript_cache:
        path = config.TRANSCRIPTS_DIR / f"{video_id}.csv"
        _transcript_cache[video_id] = pd.read_csv(path).sort_values("start_sec").reset_index(drop=True) if path.exists() else None
    return _transcript_cache[video_id]


def transcript_for_frame(video_id: str, n) -> str:
    """Reverse of the search-driven ASR legs above: given an arbitrary
    keyframe (any signal's hit, not necessarily one ASR itself ranked),
    find the transcript segment nearest that frame's timestamp. Backs
    Mixed's "Show transcript" toggle, which needs ASR text under a frame
    regardless of which sub-query actually ranked it."""
    segs = _load_transcript(video_id)
    if segs is None or segs.empty:
        return None
    ts, _fps = keyframe_timestamp(video_id, n)
    if ts is None:
        return None
    idx = (segs["start_sec"] - ts).abs().idxmin()
    text = segs.loc[idx, "text"]
    return text if isinstance(text, str) else None
