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
from ..common import faiss_search_pooled, keyframe_timestamp, l2_normalize, nearest_keyframe_n_by_time, query_hash, video_id_from_filename
from ..es_client import get_es_client
from ..es_indexing import ensure_asr_fuzzy_index
from ..models import is_image_query, siglip2_query_mat

_index = None
_meta: pd.DataFrame = None


def build_siglip_asr_index():
    global _index, _meta
    if not (config.SIGLIP_ASR_FAISS.exists() and config.SIGLIP_ASR_META.exists()):
        # transcript_embed/{video_id}.npy + {video_id}.csv (was asr_embed/
        # {video_id}_asr_siglip768.npy + _frames.csv before the rename).
        npy_paths = sorted(config.ASR_EMBED_DIR.glob("*.npy"))
        index = faiss.IndexFlatIP(config.EMBED_DIM)
        rows = []
        gid = 0
        for npy_path in npy_paths:
            video_id = video_id_from_filename(str(npy_path), ("_asr_siglip768", "_siglip768"))
            frames_path = config.ASR_EMBED_DIR / f"{video_id}.csv"
            if not frames_path.exists():
                candidates = [
                    config.ASR_EMBED_DIR / f"{video_id}_asr_siglip768_frames.csv",
                    config.ASR_EMBED_DIR / f"{video_id}_frames.csv",
                ] + list(config.ASR_EMBED_DIR.glob(f"{video_id}*.csv"))
                found = [c for c in candidates if c.exists()]
                if not found:
                    continue
                frames_path = found[0]
            vecs = l2_normalize(np.load(npy_path))
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            if vecs.shape[1] != config.EMBED_DIM:
                continue
            frames = pd.read_csv(frames_path)
            if len(frames) != vecs.shape[0]:
                continue
            # frame_id is only carried by the 768 profile's transcript CSVs.
            # The 1152 ones are segment-only (row_index, segment_id,
            # start_sec, end_sec, chunk_index, text), so resolve the keyframe
            # by timestamp instead -- the same fallback attach_keyframe_asr()
            # already applies to the ES fuzzy leg, just done once here at
            # build time so the meta CSV carries a real n on either profile.
            has_frame_id = "frame_id" in frames.columns
            keep_positions, keep_rows = [], []
            for pos, (_, r) in enumerate(frames.iterrows()):
                if has_frame_id and pd.notna(r["frame_id"]):
                    frame_id = int(r["frame_id"])
                else:
                    frame_id = nearest_keyframe_n_by_time(video_id, r["start_sec"])
                if frame_id is None:  # no map-keyframes row to hang this segment on
                    continue
                keep_positions.append(pos)
                keep_rows.append((video_id, frame_id, int(r["segment_id"]),
                                  float(r["start_sec"]), r["text"]))
            if not keep_rows:
                continue
            # global_id IS the FAISS row position, so vectors and meta rows
            # have to be dropped together -- adding all of vecs while skipping
            # a meta row would shift every later segment's lookup by one.
            index.add(vecs[keep_positions])
            for row in keep_rows:
                rows.append((gid, *row))
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
    if index.ntotal == 0 or meta.empty:
        return pd.DataFrame(columns=["rank", "score", "video_id", "segment_id", "frame_id", "start_sec", "text"])
    ids, scores = faiss_search_pooled(index, siglip2_query_mat(query), k)
    rows = []
    for rank, (gid, score) in enumerate(zip(ids, scores), start=1):
        if gid == -1 or gid >= len(meta):
            continue
        row = meta.iloc[int(gid)]
        rows.append({"rank": rank, "score": float(score), "video_id": row["video_id"],
                      "segment_id": int(row["segment_id"]), "frame_id": int(row["frame_id"]),
                      "start_sec": row["start_sec"], "text": row["text"]})
    result = pd.DataFrame(rows)
    _siglip_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Two Elasticsearch legs over the same asr_segments index, differing only in
# the query body: `fuzzy` casts wide (fuzziness AUTO + the implicit OR
# operator -- some of the query words, approximately matched), `exact` casts
# narrow (match_phrase -- every word, contiguous and in order). Everything
# wrapped around that query is identical, so it lives once in _es_leg() and
# each leg is a thin wrapper with its own cache.
# ---------------------------------------------------------------------------

_EMPTY_ES = pd.DataFrame(columns=["rank", "score", "video_id", "segment_id", "start_sec", "text"])
_fuzzy_cache = TTLCache(maxsize=256, ttl=300)
_exact_cache = TTLCache(maxsize=256, ttl=300)


def _es_leg(query, k: int, es_query: dict, cache: TTLCache, label: str):
    """Returns (df, warning). warning is a str (mirrors ui/app.py's
    st.warning) if ES was unreachable, else None -- these legs need text,
    so an image query short-circuits to an empty df with no warning."""
    if is_image_query(query):
        return _EMPTY_ES, None
    cache_key = (query_hash(query), k)
    if cache_key in cache:
        return cache[cache_key], None
    try:
        ensure_asr_fuzzy_index()
        es = get_es_client()
        resp = es.search(index=config.ES_INDEX_ASR, size=k, query=es_query)
    except Exception as e:
        return _EMPTY_ES, f"[ASR {label}] Elasticsearch not reachable at {config.ES_HOST} ({e}) — showing other legs only."

    rows = []
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        rows.append({"rank": rank, "score": float(hit["_score"]), "video_id": src["video_id"],
                      "segment_id": src["segment_id"], "start_sec": src["start_sec"], "text": src["text"]})
    result = pd.DataFrame(rows)
    cache[cache_key] = result
    return result, None


def search_asr_fuzzy(query, k: int = config.FETCH_K):
    return _es_leg(query, k, {"match": {"text": {"query": query, "fuzziness": "AUTO"}}}, _fuzzy_cache, "fuzzy")


def search_asr_exact(query, k: int = config.FETCH_K):
    """Phrase match: every query term present, contiguous and in order.
    Case-insensitive but diacritic-sensitive -- the index's standard
    analyzer lowercases and leaves Vietnamese diacritics intact, so
    "sut lun" finds nothing where "sụt lún" does. Runs against the same
    asr_segments index the fuzzy leg already uses -- no mapping change,
    no reindex."""
    return _es_leg(query, k, {"match_phrase": {"text": {"query": query}}}, _exact_cache, "exact")


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
