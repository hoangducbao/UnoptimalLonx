"""
ui/app.py — combined Streamlit UI for the three routing101 retrieval
layers, replacing the three standalone routing101_*_app.py scripts (see
routing101.ipynb / routing101_asr.ipynb / routing101_caption.ipynb for the
annotated per-layer walkthroughs this is ported from).

Running three separate `streamlit run` processes meant three separate
copies of every model (SigLIP2 text tower, Multilingual-CLIP text tower)
in RAM at once -- enough to blow past the Windows page file limit. This
app loads each text tower exactly once (st.cache_resource is keyed per
Python process, shared across all three modes below) and only ever runs
as a single process.

Five signals, picked with the segmented control (icon-only):
  Keyframe  SigLIP2 + CLIP ViT-B/32 (Multilingual-CLIP text tower) frame
            embeddings + RRF of the two.
  ASR       SigLIP2-ASR-segment embeddings + Elasticsearch fuzzy search
            over the transcript text + RRF of the two.
  Caption   SigLIP2-caption embeddings + Elasticsearch fuzzy search over
            frame captions + RRF of the two.
  OCR       Elasticsearch fuzzy search over per-frame OCR text only --
            no embedding leg, no RRF (single leg by design).
  Summary   SigLIP2-summary embeddings + Elasticsearch fuzzy search over
            each video's one-paragraph summary + RRF of the two. Video-
            level, not frame-level: one result per video (thumbnail is
            that video's frame 1), and "group by" groups by collection
            (lot) instead of by video, since every result is already a
            distinct video.

Every leg, from any mode, is normalized to the same result shape before
rendering -- {video_id, n, rank, score_label, score_val, text} where `n`
is always the 1-indexed keyframe number on disk (map-keyframes.n) -- so
one render_grid() and one "show more" neighbor popup serve all five
modes. The CLIP ViT-B/32 leg is Keyframe-only (dropped from the other
embedding legs per the current scope -- SigLIP2 + fuzzy + RRF only there).

Query input isn't limited to text: pasting an image directly into the
query box (outside TRAKE) runs a picture query instead -- SigLIP2-only
(every SigLIP2-embedded leg: frame/ASR/caption/summary), embedded with its
image tower instead of its text tower and searched the same way. The CLIP
ViT-B/32 leg (Keyframe only) and every fuzzy (Elasticsearch) leg have no
image counterpart, so they contribute nothing to a picture query, and OCR
(fuzzy-only) is unavailable for one entirely.
"""

import base64
import io
import json
import os
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

st.set_page_config(page_title="Routing101 by MiLF", layout="wide")

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import clip_encoder  # noqa: E402  (pipeline/clip_encoder.py) -- Multilingual-CLIP text tower, Keyframe mode only

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FETCH_K = 100      # candidates pulled per leg, gives RRF a real pool to fuse
DISPLAY_N = 30
RRF_K = 60
NEIGHBOR_WINDOW = 7  # "show more" popup: +/- this many frames by frame id
TOP_G_DEFAULT = 5   # Hierarchy Search: frames kept per video after drill-down (Top-G)

SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-384"

FRAME_SIGLIP2_GLOB = "D:/University/Summ26/AICDataExtracted/siglib_embed/*.npy"
FRAME_CLIP_GLOB = "D:/University/Summ26/AICData/clip-features-32/*.npy"

ASR_EMBED_DIR = Path("D:/University/Summ26/AICDataExtracted/asr_embed")
TRANSCRIPTS_DIR = Path("D:/University/Summ26/AICDataExtracted/transcripts")

CAPTIONING_DIR = Path("D:/University/Summ26/AICDataExtracted/captioning")
SIGLIP_CAPTION_DIR = Path("D:/University/Summ26/AICDataExtracted/siglip_caption")

OCR_DIR = Path("D:/University/Summ26/AICDataExtracted/ocr")

SUMMARY_DIR = Path("D:/University/Summ26/AICDataExtracted/summaries")
SUMMARY_EMBED_DIR = Path("D:/University/Summ26/AICDataExtracted/summary_embed")
SUMMARY_EMBED_DIR.mkdir(parents=True, exist_ok=True)

MAP_KEYFRAMES_DIR = Path("D:/University/Summ26/AICData/map-keyframes")
THUMBNAIL_ROOT = Path("D:/University/Summ26/AICData/keyframes")
VIDEO_DIR = Path("D:/University/Summ26/AICData/video")  # TRAKE playback dialog

INDEX_DIR = REPO_ROOT / "index"
ASR_INDEX_DIR = INDEX_DIR / "routing101_asr"
CAPTION_INDEX_DIR = INDEX_DIR / "routing101_caption"
SUMMARY_INDEX_DIR = INDEX_DIR / "routing101_summary"
ASR_INDEX_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_INDEX_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_INDEX_DIR.mkdir(parents=True, exist_ok=True)
SIGLIP_ASR_FAISS = ASR_INDEX_DIR / "siglip_asr_flat_ip.index"
SIGLIP_ASR_META = ASR_INDEX_DIR / "meta_siglip_asr.csv"
SIGLIP_CAPTION_FAISS = CAPTION_INDEX_DIR / "siglip_caption_flat_ip.index"
SIGLIP_CAPTION_META = CAPTION_INDEX_DIR / "meta_siglip_caption.csv"
SIGLIP_SUMMARY_FAISS = SUMMARY_INDEX_DIR / "siglip_summary_flat_ip.index"
SIGLIP_SUMMARY_META = SUMMARY_INDEX_DIR / "meta_siglip_summary.csv"

ES_HOST = "http://localhost:9200"
ES_INDEX_ASR = "asr_segments"
ES_INDEX_CAPTION = "caption_frames"
ES_INDEX_OCR = "ocr_frames"
ES_INDEX_SUMMARY = "summary_videos"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Thread-pool tuning -- CPU-only torch defaults to num-cores intraop threads
# AND num-cores interop threads (the latter unused here: this app never runs
# independent torch op graphs in parallel, just one encode call at a time),
# and FAISS's own OpenMP pool defaults to num-cores on top of that. Left
# uncapped, the pools compound into far more live threads than the box has
# cores (observed: 130 threads on a 22-logical-core machine), adding
# context-switch overhead on top of an already CPU-bound SigLIP2 +
# Multilingual-CLIP (XLM-RoBERTa-Large) encode per query. Streamlit re-execs
# this whole script on every rerun, so torch.set_num_interop_threads (unlike
# set_num_threads/omp_set_num_threads) can only legally be called once per
# process -- guard it, or the second rerun raises.
_CPU_BUDGET = max(1, (os.cpu_count() or 4) - 2)  # leave headroom for Streamlit/tornado/OS


@st.cache_resource(show_spinner=False)
def _tune_thread_pools():
    """Runs exactly once per process (st.cache_resource, not a bare
    module-level call) -- app.py's top-level code re-execs on EVERY
    Streamlit rerun, and re-resizing torch's intraop pool (or faiss's OpenMP
    pool) on every rerun risked racing an in-flight encode/search call from
    a rerun still winding down, which was observed to wedge the whole
    session (script "running" indicator stuck, near-zero CPU, no results --
    not just slow). set_num_interop_threads is even stricter: it's only
    legal to call once per process at all, ever."""
    if DEVICE == "cpu":
        torch.set_num_threads(_CPU_BUDGET)
    torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(_CPU_BUDGET)
    return True


_tune_thread_pools()

# Every search_* function below is st.cache_data'd on (query, k) -- Streamlit
# reruns the ENTIRE script on any widget interaction (a "Show more"/"Copy"
# button included), and without this every one of those clicks was silently
# re-running the full embed-plus-FAISS-or-ES search for whatever signal was
# on screen, not just an actual new query. hash_funcs is needed because a
# picture query's `query` argument is a PIL.Image, which Streamlit's default
# hasher doesn't know how to fingerprint -- hash it by raw pixel bytes.
_QUERY_HASH_FUNCS = {Image.Image: lambda img: img.tobytes()}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def video_id_from_filename(path_str: str, suffixes: tuple) -> str:
    stem = Path(path_str).stem
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype("float32", copy=True)
    faiss.normalize_L2(mat)
    return mat


def parse_lot_range(text: str):
    """'L21-L30' / 'L21' / '21-30' -> (lo, hi) lot numbers, or None if blank/unparsable."""
    text = (text or "").strip().upper()
    if not text:
        return None
    m = re.match(r"^L?(\d+)\s*-\s*L?(\d+)$", text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lo, hi) if lo <= hi else (hi, lo)
    m = re.match(r"^L?(\d+)$", text)
    if m:
        return int(m.group(1)), int(m.group(1))
    return None


def video_lot_num(video_id: str):
    m = re.match(r"^L(\d+)", str(video_id).upper())
    return int(m.group(1)) if m else None


def video_lot_str(video_id: str) -> str:
    lot = video_lot_num(video_id)
    return f"L{lot}" if lot is not None else str(video_id)


def apply_filters(df: pd.DataFrame, video_filter: str, lot_range) -> pd.DataFrame:
    """Restrict a leg's result df to a single video_id and/or a lot range,
    applied right after search (before RRF/head truncation) so both single-leg
    and RRF views respect the same filters."""
    if df is None or df.empty:
        return df
    out = df
    video_filter = (video_filter or "").strip().upper()
    if video_filter:
        out = out[out["video_id"].astype(str).str.upper() == video_filter]
    if lot_range:
        lo, hi = lot_range
        lots = out["video_id"].map(video_lot_num)
        out = out[lots.notna() & (lots >= lo) & (lots <= hi)]
    return out.reset_index(drop=True)


def thumbnail_path(video_id: str, n) -> str:
    if n is None or pd.isna(n):
        return ""
    return str(THUMBNAIL_ROOT / video_id / f"{int(n):03d}.jpg")


@st.cache_resource(show_spinner=False)
def get_map_keyframes_cache():
    return {}


def load_map_keyframes(video_id: str):
    cache = get_map_keyframes_cache()
    if video_id not in cache:
        path = MAP_KEYFRAMES_DIR / f"{video_id}.csv"
        cache[video_id] = pd.read_csv(path) if path.exists() else None
    return cache[video_id]


def nearest_keyframe_n_by_time(video_id: str, t: float):
    """Nearest map-keyframes row (by pts_time) to timestamp t -- used for
    the ASR fuzzy leg, which is segment-level only (no direct frame_id)."""
    mk = load_map_keyframes(video_id)
    if mk is None or mk.empty or pd.isna(t):
        return None
    idx = (mk["pts_time"] - t).abs().idxmin()
    return int(mk.loc[idx, "n"])


def keyframe_timestamp(video_id: str, n):
    """Symmetric counterpart to nearest_keyframe_n_by_time: direct n ->
    (pts_time, fps) lookup, used by TRAKE to place a matched frame on the
    video timeline. Returns (None, None) if unresolvable."""
    mk = load_map_keyframes(video_id)
    if mk is None or mk.empty or n is None or pd.isna(n):
        return None, None
    hit = mk.loc[mk["n"] == int(n)]
    if hit.empty:
        return None, None
    row = hit.iloc[0]
    return float(row["pts_time"]), float(row["fps"])


def video_path(video_id: str) -> str:
    return str(VIDEO_DIR / f"{video_id}.mp4")


@st.cache_data(show_spinner=False)
def image_b64(path: str) -> str:
    import base64
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Shared model loaders -- st.cache_resource is keyed per Python process, so
# every mode below (Keyframe/ASR/Caption) reuses the SAME SigLIP2 tower
# instance instead of loading its own copy.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading SigLIP2 text tower…")
def load_siglip2():
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(SIGLIP2_MODEL_ID).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)
    return model, processor


def encode_text_siglip2(texts: list) -> np.ndarray:
    model, processor = load_siglip2()
    inputs = processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return feats.float().cpu().numpy().astype("float32")


def encode_image_siglip2(images: list) -> np.ndarray:
    """SigLIP2 image tower -- same joint text/image embedding space as
    encode_text_siglip2(), so an image query is directly comparable to
    every SigLIP2-embedded leg (frame, ASR, caption, summary), not just
    the frame index."""
    model, processor = load_siglip2()
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return feats.float().cpu().numpy().astype("float32")


def is_image_query(query) -> bool:
    return isinstance(query, Image.Image)


def siglip2_query_vec(query) -> np.ndarray:
    """Picture-query support: a pasted image is embedded with the SigLIP2
    image tower instead of the text tower -- everywhere else the caller
    already treats the result as a plain query vector, so this is the only
    branch point needed for every SigLIP2-backed leg."""
    if is_image_query(query):
        return encode_image_siglip2([query])[0]
    return encode_text_siglip2([query])[0]


@st.cache_resource(show_spinner=False)
def get_es_client():
    from elasticsearch import Elasticsearch

    return Elasticsearch(ES_HOST)


# ---------------------------------------------------------------------------
# Mode 1 — Keyframe: SigLIP2 + CLIP ViT-B/32 frame embeddings, RRF.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building frame FAISS index…")
def build_frame_index(glob_pattern: str):
    import glob as glob_mod

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
    return index, pd.DataFrame(lookup_rows)


def _search_frame(index, lookup_df, qvec: np.ndarray, k: int) -> pd.DataFrame:
    q = l2_normalize(qvec.reshape(1, -1))
    n = min(k, index.ntotal)
    scores, ids = index.search(q, n)
    results = lookup_df.iloc[ids[0]].copy().reset_index(drop=True)
    results["score"] = scores[0]
    results["rank"] = np.arange(1, len(results) + 1)
    results["n"] = results["frame_id"] + 1
    return results[["rank", "score", "video_id", "frame_id", "n"]]


@st.cache_data(show_spinner=False, hash_funcs=_QUERY_HASH_FUNCS)
def search_siglip2_frame(query, k: int = FETCH_K) -> pd.DataFrame:
    index, lookup_df = build_frame_index(FRAME_SIGLIP2_GLOB)
    qvec = siglip2_query_vec(query)
    return _search_frame(index, lookup_df, qvec, k)


@st.cache_data(show_spinner=False, hash_funcs=_QUERY_HASH_FUNCS)
def search_clip_frame(query, k: int = FETCH_K) -> pd.DataFrame:
    if is_image_query(query):
        # Picture queries are SigLIP2-only -- CLIP ViT-B/32 here is the
        # Multilingual-CLIP *text* tower's paired space, no image tower
        # wired up for it, so this leg contributes nothing to an image query.
        return pd.DataFrame(columns=["rank", "score", "video_id", "frame_id", "n"])
    index, lookup_df = build_frame_index(FRAME_CLIP_GLOB)
    qvec = clip_encoder.encode_text([query])[0]
    return _search_frame(index, lookup_df, qvec, k)


def rrf_fuse_frame(dfs: list, k: int = RRF_K, top_n: int = DISPLAY_N) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Mode 2 — ASR: SigLIP2-ASR embeddings + Elasticsearch fuzzy, RRF.
# Segment-level (ES leg has no direct frame_id), so RRF keys on segment_id;
# `n` is resolved per-leg (direct for SigLIP2-ASR, nearest-time for fuzzy).
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building SigLIP2-ASR FAISS index…")
def build_siglip_asr_index():
    if not (SIGLIP_ASR_FAISS.exists() and SIGLIP_ASR_META.exists()):
        npy_paths = sorted(ASR_EMBED_DIR.glob("*_asr_siglip768.npy"))
        index = faiss.IndexFlatIP(768)
        rows = []
        gid = 0
        for npy_path in npy_paths:
            video_id = video_id_from_filename(str(npy_path), ("_asr_siglip768",))
            frames_path = ASR_EMBED_DIR / f"{video_id}_asr_siglip768_frames.csv"
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
        faiss.write_index(index, str(SIGLIP_ASR_FAISS))
        pd.DataFrame(rows, columns=["global_id", "video_id", "frame_id", "segment_id", "start_sec", "text"]).to_csv(SIGLIP_ASR_META, index=False)

    return faiss.read_index(str(SIGLIP_ASR_FAISS)), pd.read_csv(SIGLIP_ASR_META)


@st.cache_data(show_spinner=False, hash_funcs=_QUERY_HASH_FUNCS)
def search_siglip_asr(query, k: int = FETCH_K) -> pd.DataFrame:
    index, meta = build_siglip_asr_index()
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
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Indexing transcripts into Elasticsearch…")
def ensure_asr_fuzzy_index():
    from elasticsearch import helpers

    es = get_es_client()
    if es.indices.exists(index=ES_INDEX_ASR):
        return True  # already indexed (persists across restarts if ES's data dir is a volume) -- skip re-bulking every launch
    es.indices.create(index=ES_INDEX_ASR, mappings={"properties": {
        "video_id": {"type": "keyword"},
        "segment_id": {"type": "integer"},
        "start_sec": {"type": "float"},
        "text": {"type": "text"},
    }})

    def _docs():
        for csv_path in sorted(TRANSCRIPTS_DIR.glob("*.csv")):
            if csv_path.name == "manifest.csv":
                continue
            df = pd.read_csv(csv_path)
            video_id = csv_path.stem
            for _, r in df.iterrows():
                yield {
                    "_index": ES_INDEX_ASR,
                    "_id": f"{video_id}_{int(r['segment_id'])}",
                    "_source": {"video_id": video_id, "segment_id": int(r["segment_id"]),
                                 "start_sec": float(r["start_sec"]), "text": r["text"]},
                }

    helpers.bulk(es, _docs(), stats_only=True, raise_on_error=False)
    return True


@st.cache_data(show_spinner=False, hash_funcs=_QUERY_HASH_FUNCS, ttl=300)
def search_asr_fuzzy(query, k: int = FETCH_K) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["rank", "score", "video_id", "segment_id", "start_sec", "text"])
    if is_image_query(query):
        return empty  # fuzzy legs need text; picture queries skip them
    try:
        ensure_asr_fuzzy_index()
        es = get_es_client()
        resp = es.search(index=ES_INDEX_ASR, size=k, query={
            "match": {"text": {"query": query, "fuzziness": "AUTO"}}
        })
    except Exception as e:
        st.warning(f"[ASR fuzzy] Elasticsearch not reachable at {ES_HOST} ({e}) — showing other legs only.")
        return empty

    rows = []
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        rows.append({"rank": rank, "score": float(hit["_score"]), "video_id": src["video_id"],
                      "segment_id": src["segment_id"], "start_sec": src["start_sec"], "text": src["text"]})
    return pd.DataFrame(rows)


def rrf_fuse_asr(named_dfs: dict, k: int = RRF_K, top_n: int = DISPLAY_N) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Mode 3 — Caption: SigLIP2-caption embeddings + Elasticsearch fuzzy, RRF.
# Frame-level on both legs (frame_id == map-keyframes.n directly), so RRF
# keys on frame_id and `n` needs no time-based lookup at all.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building SigLIP2-caption FAISS index…")
def build_siglip_caption_index():
    if not (SIGLIP_CAPTION_FAISS.exists() and SIGLIP_CAPTION_META.exists()):
        npy_paths = sorted(SIGLIP_CAPTION_DIR.glob("*_caption_siglip768.npy"))
        index = faiss.IndexFlatIP(768)
        rows = []
        gid = 0
        for npy_path in npy_paths:
            video_id = video_id_from_filename(str(npy_path), ("_caption_siglip768",))
            frames_path = SIGLIP_CAPTION_DIR / f"{video_id}_caption_siglip768_frames.csv"
            if not frames_path.exists():
                continue
            vecs = l2_normalize(np.load(npy_path))
            frames = pd.read_csv(frames_path)
            if len(frames) != vecs.shape[0]:
                continue
            index.add(vecs)
            for _, r in frames.iterrows():
                rows.append((gid, video_id, int(r["frame_id"]), r["text"]))
                gid += 1
        faiss.write_index(index, str(SIGLIP_CAPTION_FAISS))
        pd.DataFrame(rows, columns=["global_id", "video_id", "frame_id", "text"]).to_csv(SIGLIP_CAPTION_META, index=False)

    return faiss.read_index(str(SIGLIP_CAPTION_FAISS)), pd.read_csv(SIGLIP_CAPTION_META)


@st.cache_data(show_spinner=False, hash_funcs=_QUERY_HASH_FUNCS)
def search_siglip_caption(query, k: int = FETCH_K) -> pd.DataFrame:
    index, meta = build_siglip_caption_index()
    qvec = l2_normalize(siglip2_query_vec(query).reshape(1, -1))
    n = min(k, index.ntotal)
    scores, ids = index.search(qvec, n)
    rows = []
    for rank, (gid, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if gid == -1:
            continue
        row = meta.iloc[int(gid)]
        rows.append({"rank": rank, "score": float(score), "video_id": row["video_id"],
                      "frame_id": int(row["frame_id"]), "text": row["text"]})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Indexing captions into Elasticsearch…")
def ensure_caption_fuzzy_index():
    from elasticsearch import helpers

    es = get_es_client()
    if es.indices.exists(index=ES_INDEX_CAPTION):
        return True  # already indexed -- skip re-bulking every launch
    es.indices.create(index=ES_INDEX_CAPTION, mappings={"properties": {
        "video_id": {"type": "keyword"},
        "frame_id": {"type": "integer"},
        "text": {"type": "text"},
    }})

    def _docs():
        for csv_path in sorted(CAPTIONING_DIR.glob("*_captions.csv")):
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                yield {
                    "_index": ES_INDEX_CAPTION,
                    "_id": f"{r['video_id']}_{int(r['frame_id'])}",
                    "_source": {"video_id": r["video_id"], "frame_id": int(r["frame_id"]), "text": r["caption_text"]},
                }

    helpers.bulk(es, _docs(), stats_only=True, raise_on_error=False)
    return True


@st.cache_data(show_spinner=False, hash_funcs=_QUERY_HASH_FUNCS, ttl=300)
def search_caption_fuzzy(query, k: int = FETCH_K) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["rank", "score", "video_id", "frame_id", "text"])
    if is_image_query(query):
        return empty  # fuzzy legs need text; picture queries skip them
    try:
        ensure_caption_fuzzy_index()
        es = get_es_client()
        resp = es.search(index=ES_INDEX_CAPTION, size=k, query={
            "match": {"text": {"query": query, "fuzziness": "AUTO"}}
        })
    except Exception as e:
        st.warning(f"[Caption fuzzy] Elasticsearch not reachable at {ES_HOST} ({e}) — showing other legs only.")
        return empty

    rows = []
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        rows.append({"rank": rank, "score": float(hit["_score"]), "video_id": src["video_id"],
                      "frame_id": src["frame_id"], "text": src["text"]})
    return pd.DataFrame(rows)


def rrf_fuse_caption(named_dfs: dict, k: int = RRF_K, top_n: int = DISPLAY_N) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Mode 4 — OCR: Elasticsearch fuzzy over per-frame OCR text only. Single
# leg by design (no embedding leg, no RRF) -- each video's CSV has one row
# per detected text box, grouped into one blob per frame before indexing.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Indexing OCR text into Elasticsearch…")
def ensure_ocr_fuzzy_index():
    from elasticsearch import helpers

    es = get_es_client()
    if es.indices.exists(index=ES_INDEX_OCR):
        return True  # already indexed -- skip re-bulking every launch
    es.indices.create(index=ES_INDEX_OCR, mappings={"properties": {
        "video_id": {"type": "keyword"},
        "frame_id": {"type": "integer"},
        "text": {"type": "text"},
    }})

    def _docs():
        for csv_path in sorted(OCR_DIR.glob("*.csv")):
            if csv_path.name.startswith("run_manifest"):
                continue  # per-run metadata sidecar, not a per-video OCR dump
            video_id = csv_path.stem
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            grouped = df.groupby("frame_id")["text"].apply(
                lambda s: " ".join(str(t) for t in s if pd.notna(t))
            )
            for frame_id, text in grouped.items():
                if not text.strip():
                    continue
                yield {
                    "_index": ES_INDEX_OCR,
                    "_id": f"{video_id}_{int(frame_id)}",
                    "_source": {"video_id": video_id, "frame_id": int(frame_id), "text": text},
                }

    helpers.bulk(es, _docs(), stats_only=True, raise_on_error=False)
    return True


@st.cache_data(show_spinner=False, hash_funcs=_QUERY_HASH_FUNCS, ttl=300)
def search_ocr_fuzzy(query, k: int = FETCH_K) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["rank", "score", "video_id", "frame_id", "text"])
    if is_image_query(query):
        return empty  # OCR is fuzzy-text-only; picture queries have no leg here
    try:
        ensure_ocr_fuzzy_index()
        es = get_es_client()
        resp = es.search(index=ES_INDEX_OCR, size=k, query={
            "match": {"text": {"query": query, "fuzziness": "AUTO"}}
        })
    except Exception as e:
        st.warning(f"[OCR fuzzy] Elasticsearch not reachable at {ES_HOST} ({e}) — showing no results.")
        return empty

    rows = []
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        rows.append({"rank": rank, "score": float(hit["_score"]), "video_id": src["video_id"],
                      "frame_id": src["frame_id"], "text": src["text"]})
    return pd.DataFrame(rows)


def attach_keyframe_ocr(df: pd.DataFrame) -> pd.DataFrame:
    """OCR is frame-level: frame_id == map-keyframes.n directly, no lookup needed."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["n"] = out["frame_id"].astype(int)
    return out


# ---------------------------------------------------------------------------
# Mode 5 — Summary: SigLIP2-summary embeddings + Elasticsearch fuzzy, RRF.
# Video-level, not frame-level: one row per video (its whole summary), so
# RRF keys on video_id alone, and the displayed thumbnail is always that
# video's frame 1 (no per-summary frame to point at).
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Embedding video summaries…")
def ensure_summary_embeddings():
    """One-time SigLIP2 text-tower embed of every summaries/*.txt, saved to
    SUMMARY_EMBED_DIR so later runs don't re-embed (resumable: skips any
    video that already has a .npy on disk)."""
    for txt_path in sorted(SUMMARY_DIR.glob("*.txt")):
        video_id = txt_path.stem
        out_path = SUMMARY_EMBED_DIR / f"{video_id}_summary_siglip768.npy"
        if out_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        vec = encode_text_siglip2([text])
        np.save(out_path, vec.astype("float32"))
    return True


@st.cache_resource(show_spinner="Building SigLIP2-summary FAISS index…")
def build_siglip_summary_index():
    ensure_summary_embeddings()
    if not (SIGLIP_SUMMARY_FAISS.exists() and SIGLIP_SUMMARY_META.exists()):
        npy_paths = sorted(SUMMARY_EMBED_DIR.glob("*_summary_siglip768.npy"))
        index = faiss.IndexFlatIP(768)
        rows = []
        gid = 0
        for npy_path in npy_paths:
            video_id = video_id_from_filename(str(npy_path), ("_summary_siglip768",))
            txt_path = SUMMARY_DIR / f"{video_id}.txt"
            if not txt_path.exists():
                continue
            vec = l2_normalize(np.load(npy_path))
            index.add(vec)
            rows.append((gid, video_id, txt_path.read_text(encoding="utf-8").strip()))
            gid += 1
        faiss.write_index(index, str(SIGLIP_SUMMARY_FAISS))
        pd.DataFrame(rows, columns=["global_id", "video_id", "text"]).to_csv(SIGLIP_SUMMARY_META, index=False)

    return faiss.read_index(str(SIGLIP_SUMMARY_FAISS)), pd.read_csv(SIGLIP_SUMMARY_META)


@st.cache_data(show_spinner=False, hash_funcs=_QUERY_HASH_FUNCS)
def search_siglip_summary(query, k: int = FETCH_K) -> pd.DataFrame:
    index, meta = build_siglip_summary_index()
    qvec = l2_normalize(siglip2_query_vec(query).reshape(1, -1))
    n = min(k, index.ntotal)
    scores, ids = index.search(qvec, n)
    rows = []
    for rank, (gid, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if gid == -1:
            continue
        row = meta.iloc[int(gid)]
        rows.append({"rank": rank, "score": float(score), "video_id": row["video_id"], "text": row["text"]})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Indexing summaries into Elasticsearch…")
def ensure_summary_fuzzy_index():
    from elasticsearch import helpers

    es = get_es_client()
    if es.indices.exists(index=ES_INDEX_SUMMARY):
        return True  # already indexed -- skip re-bulking every launch
    es.indices.create(index=ES_INDEX_SUMMARY, mappings={"properties": {
        "video_id": {"type": "keyword"},
        "text": {"type": "text"},
    }})

    def _docs():
        for txt_path in sorted(SUMMARY_DIR.glob("*.txt")):
            video_id = txt_path.stem
            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            yield {"_index": ES_INDEX_SUMMARY, "_id": video_id, "_source": {"video_id": video_id, "text": text}}

    helpers.bulk(es, _docs(), stats_only=True, raise_on_error=False)
    return True


@st.cache_data(show_spinner=False, hash_funcs=_QUERY_HASH_FUNCS, ttl=300)
def search_summary_fuzzy(query, k: int = FETCH_K) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["rank", "score", "video_id", "text"])
    if is_image_query(query):
        return empty  # fuzzy legs need text; picture queries skip them
    try:
        ensure_summary_fuzzy_index()
        es = get_es_client()
        resp = es.search(index=ES_INDEX_SUMMARY, size=k, query={
            "match": {"text": {"query": query, "fuzziness": "AUTO"}}
        })
    except Exception as e:
        st.warning(f"[Summary fuzzy] Elasticsearch not reachable at {ES_HOST} ({e}) — showing other legs only.")
        return empty

    rows = []
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        rows.append({"rank": rank, "score": float(hit["_score"]), "video_id": src["video_id"], "text": src["text"]})
    return pd.DataFrame(rows)


def rrf_fuse_summary(named_dfs: dict, k: int = RRF_K, top_n: int = DISPLAY_N) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Mode 6 — Mixed: a user-weighted RRF across Keyframe/ASR/Caption/OCR (not
# Summary -- video-level, kept out of this signal). For each signal, only
# its *checked* legs (chosen in the "Change weights" popup)
# contribute -- 2 checked legs are fused with that signal's own existing
# rrf_fuse_* first (so Mixed reuses the exact same per-signal RRF already
# used standalone), 1 checked leg is used at its own raw rank, 0 checked
# legs (or a 0 weight) drop the signal entirely. The resulting per-signal
# rank lists are then combined with a weighted RRF, keyed on (video_id, n)
# since every signal is normalized to that shape already.
# ---------------------------------------------------------------------------

def _mixed_keyframe_df(query, fetch_k, video_filter, lot_filter, legs) -> pd.DataFrame:
    named = {}
    if legs.get("kf_siglip2"):
        named["siglip2"] = apply_filters(search_siglip2_frame(query, k=fetch_k), video_filter, lot_filter)
    if legs.get("kf_clip"):
        named["clip"] = apply_filters(search_clip_frame(query, k=fetch_k), video_filter, lot_filter)
    if not named:
        return None
    if len(named) == 1:
        df = next(iter(named.values())).copy()
        df["n"] = df["frame_id"] + 1
    else:
        df = rrf_fuse_frame(list(named.values()), top_n=fetch_k)
    return df[["video_id", "n", "rank"]] if df is not None and not df.empty else None


def _mixed_asr_df(query, fetch_k, video_filter, lot_filter, legs) -> pd.DataFrame:
    named = {}
    if legs.get("asr_siglip"):
        named["siglip_asr"] = apply_filters(search_siglip_asr(query, k=fetch_k), video_filter, lot_filter)
    if legs.get("asr_fuzzy"):
        named["fuzzy"] = apply_filters(search_asr_fuzzy(query, k=fetch_k), video_filter, lot_filter)
    if not named:
        return None
    if len(named) == 1:
        df = attach_keyframe_asr(next(iter(named.values())))
    else:
        df = attach_keyframe_asr(rrf_fuse_asr(named, top_n=fetch_k))
    return df[["video_id", "n", "rank"]] if df is not None and not df.empty else None


def _mixed_caption_df(query, fetch_k, video_filter, lot_filter, legs) -> pd.DataFrame:
    named = {}
    if legs.get("cap_siglip"):
        named["siglip_caption"] = apply_filters(search_siglip_caption(query, k=fetch_k), video_filter, lot_filter)
    if legs.get("cap_fuzzy"):
        named["fuzzy"] = apply_filters(search_caption_fuzzy(query, k=fetch_k), video_filter, lot_filter)
    if not named:
        return None
    if len(named) == 1:
        df = attach_keyframe_caption(next(iter(named.values())))
    else:
        df = attach_keyframe_caption(rrf_fuse_caption(named, top_n=fetch_k))
    return df[["video_id", "n", "rank"]] if df is not None and not df.empty else None


def _mixed_ocr_df(query, fetch_k, video_filter, lot_filter) -> pd.DataFrame:
    """OCR has no leg choice -- its single Fuzzy OCR leg is used whenever
    its weight is > 0."""
    df = attach_keyframe_ocr(apply_filters(search_ocr_fuzzy(query, k=fetch_k), video_filter, lot_filter))
    return df[["video_id", "n", "rank"]] if df is not None and not df.empty else None


def rrf_fuse_weighted(signal_dfs: dict, weights: dict, k: int = RRF_K, top_n: int = DISPLAY_N) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Mode 7 — TRAKE: an ordered list of event sub-queries (each with its own
# text + one of the other six signals), find videos where every event's
# best-matching frame occurs in the declared order. Reuses each signal's
# existing search + RRF pipeline UNSCOPED (no video/lot filter -- TRAKE
# searches the whole corpus per event, on purpose), then adds a video-level
# coverage/order/score layer on top. No new embedding models, no new
# per-signal fusion logic -- see TRAKE_SPEC.md.
# ---------------------------------------------------------------------------

def trake_search_event(query: str, signal: str, fetch_k: int, video_filter: str = "", lot_filter=None) -> pd.DataFrame:
    """One TRAKE event's candidate frames for `signal`, normalized to
    [video_id, n, rank, score] (+text where the signal carries one).
    Dispatches to the exact same search + RRF calls the standalone signal
    modes already make -- see the `elif mode == ...` blocks below. Scoping
    (video_filter/lot_filter) is optional -- TRAKE_SPEC.md's default is an
    unscoped whole-corpus search per event, but the UI lets a user narrow
    every event to one video/collection same as the other signals."""
    empty = pd.DataFrame(columns=["video_id", "n", "rank", "score", "text"])
    if not query:
        return empty

    if signal == "Keyframe":
        siglip2_df = apply_filters(search_siglip2_frame(query, k=fetch_k), video_filter, lot_filter)
        clip_df = apply_filters(search_clip_frame(query, k=fetch_k), video_filter, lot_filter)
        df = rrf_fuse_frame([siglip2_df, clip_df], top_n=fetch_k)
        return empty if df.empty else df.rename(columns={"rrf_score": "score"})[["video_id", "n", "rank", "score"]]

    if signal == "ASR":
        siglip_df = apply_filters(search_siglip_asr(query, k=fetch_k), video_filter, lot_filter)
        fuzzy_df = apply_filters(search_asr_fuzzy(query, k=fetch_k), video_filter, lot_filter)
        fused = attach_keyframe_asr(rrf_fuse_asr({"siglip_asr": siglip_df, "fuzzy": fuzzy_df}, top_n=fetch_k))
        return empty if fused is None or fused.empty else fused.rename(columns={"rrf_score": "score"})[["video_id", "n", "rank", "score", "text"]]

    if signal == "Caption":
        siglip_df = apply_filters(search_siglip_caption(query, k=fetch_k), video_filter, lot_filter)
        fuzzy_df = apply_filters(search_caption_fuzzy(query, k=fetch_k), video_filter, lot_filter)
        fused = attach_keyframe_caption(rrf_fuse_caption({"siglip_caption": siglip_df, "fuzzy": fuzzy_df}, top_n=fetch_k))
        return empty if fused is None or fused.empty else fused.rename(columns={"rrf_score": "score"})[["video_id", "n", "rank", "score", "text"]]

    if signal == "OCR":
        df = attach_keyframe_ocr(apply_filters(search_ocr_fuzzy(query, k=fetch_k), video_filter, lot_filter))
        return empty if df is None or df.empty else df[["video_id", "n", "rank", "score", "text"]]

    if signal == "Summary":
        siglip_df = apply_filters(search_siglip_summary(query, k=fetch_k), video_filter, lot_filter)
        fuzzy_df = apply_filters(search_summary_fuzzy(query, k=fetch_k), video_filter, lot_filter)
        fused = attach_keyframe_summary(rrf_fuse_summary({"siglip_summary": siglip_df, "fuzzy": fuzzy_df}, top_n=fetch_k))
        return empty if fused is None or fused.empty else fused.rename(columns={"rrf_score": "score"})[["video_id", "n", "rank", "score", "text"]]

    if signal == "Mixed":
        # rrf_fuse_weighted only ever reads "rank" off its inputs (never a
        # score column), so the existing _mixed_*_df helpers -- already
        # trimmed to [video_id, n, rank] -- are safe to reuse unchanged;
        # the per-event "score" below is the weighted-RRF score, same as
        # standalone Mixed mode's own rrf_score.
        weights = st.session_state.mixed_weights
        legs = st.session_state.mixed_legs
        signal_dfs = {}
        if weights.get("Keyframe", 0):
            signal_dfs["Keyframe"] = _mixed_keyframe_df(query, fetch_k, video_filter, lot_filter, legs)
        if weights.get("ASR", 0):
            signal_dfs["ASR"] = _mixed_asr_df(query, fetch_k, video_filter, lot_filter, legs)
        if weights.get("Caption", 0):
            signal_dfs["Caption"] = _mixed_caption_df(query, fetch_k, video_filter, lot_filter, legs)
        if weights.get("OCR", 0):
            signal_dfs["OCR"] = _mixed_ocr_df(query, fetch_k, video_filter, lot_filter)
        if not signal_dfs:
            return empty
        fused = rrf_fuse_weighted(signal_dfs, weights, top_n=fetch_k)
        return empty if fused.empty else fused.rename(columns={"rrf_score": "score"})[["video_id", "n", "rank", "score"]]

    raise ValueError(f"unknown TRAKE event signal: {signal!r}")


def trake_rank_videos(event_dfs: list, labels: list, top_n: int,
                       bonus_video_ids: set = None, bonus_multiplier: float = 1.10) -> list:
    """TRAKE_SPEC.md Steps 2-4: group each event's candidates by video
    (keeping only the best-ranked hit per video per event), hard-drop any
    video whose matched-event timestamps aren't strictly increasing in the
    declared order, then rank survivors by coverage * mean(score across
    matched events). `labels` is the display label per event_dfs slot
    (e.g. "E0" for an included context query, "E1"/"E2"/... for the
    required events) -- kept separate from the 0-indexed event_index so
    the context row's presence/absence doesn't shift the required events'
    numbering. `bonus_video_ids` (from the optional context query's own
    top-K/2 results, computed by the caller) gets a flat score multiplier
    -- applied before the final sort/top_n cut so it can actually affect
    which videos make the cut, not just their displayed order within it."""
    n_events = len(event_dfs)
    best_by_video: dict = {}  # video_id -> {event_index: row}
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
        # (events list above is already built in that order) -- hard rule
        # per spec: any inversion drops the video outright.
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


# ---------------------------------------------------------------------------
# Mode 8 — Hierarchy Search, three steps:
#   1. A SigLIP2 frame search (text or picture query), grouped by video like
#      Keyframe's "group by video".
#   2. Per video, a seed-frame picker -- which of that group's own frames
#      becomes the NEW picture query for step 3. Defaults to the group's
#      top-1 frame; changing it only affects that one video.
#   3. Drill-down: the chosen seed frame is embedded and searched scoped to
#      that one video, pulling in up to Top-G frames total per video
#      (default G=5, "Expand" bumps one video's own G by +10).
# This only ever uses the SigLIP2 frame leg -- CLIP/fuzzy legs have no
# picture-query counterpart (see is_image_query()), and the drill-down step
# is a picture query by construction (the seed is a frame's own thumbnail),
# so there's no meaningful text/RRF path to offer here at all.
# ---------------------------------------------------------------------------

def hierarchy_expand_group(video_id: str, frames: list, top_g: int, fetch_k: int, seed_n: int = None) -> list:
    """`frames` is one video's results from the base search (Step 1),
    already in rank order (best first). If it's already at/over Top-G, just
    truncate. Otherwise, embed a seed frame's thumbnail as a new SigLIP2
    picture query, search scoped to this video only, and append
    not-yet-present frames (in that scoped search's own rank order) until
    Top-G is hit or the scoped search runs out of candidates.

    `seed_n` (Step 2): which frame number to use as that query -- defaults
    to the group's own top-1 frame when not given, but the caller can pass
    any frame number a user picked instead, applying only to this one
    video's drill-down."""
    if len(frames) >= top_g:
        return frames[:top_g]

    seed_n = seed_n if seed_n is not None else frames[0]["n"]
    thumb = thumbnail_path(video_id, seed_n)
    if not thumb or not Path(thumb).exists():
        return frames  # no seed image on disk -- nothing to drill down with

    try:
        seed_image = Image.open(thumb).convert("RGB")
    except Exception:
        return frames

    have_ns = {f["n"] for f in frames}
    needed = top_g - len(frames)
    # A plain video-id filter (not apply_filters' lot-range path) -- the
    # scoped search still runs over the whole FAISS index first, so k needs
    # enough headroom that this one video's frames actually surface in it.
    scoped_df = apply_filters(search_siglip2_frame(seed_image, k=max(fetch_k, top_g * 40)), video_id, None)
    if scoped_df is None or scoped_df.empty:
        return frames

    extra_rows = []
    for _, row in scoped_df.sort_values("rank").iterrows():
        if len(extra_rows) >= needed:
            break
        n = int(row["n"])
        if n in have_ns:
            continue
        extra_rows.append(row)
        have_ns.add(n)

    if not extra_rows:
        return frames
    extra_results = df_to_results(pd.DataFrame(extra_rows), "score")
    return frames + extra_results


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.thumb-wrap { position: relative; overflow: visible; z-index: 1; line-height: 0; }
.thumb-wrap img {
    width: 100%; display: block; border-radius: 4px;
    transition: transform 0.15s ease-out, box-shadow 0.15s ease-out;
    transform-origin: center center;
}
.thumb-wrap:hover { z-index: 100; }
.thumb-wrap:hover img {
    transform: scale(2.5);
    box-shadow: 0 12px 32px rgba(0,0,0,0.45);
    position: relative;
}
/* TRAKE result cards with only 1-2 events: the zoom-on-hover above is more
   distracting than useful with so few thumbnails already at a decent size,
   so it's suppressed for those (kept for 3+ event cards). */
.thumb-wrap-static:hover { z-index: 1; }
.thumb-wrap-static:hover img { transform: none; box-shadow: none; position: static; }
/* Enter already submits the query (see the JS below) -- hide the native
   "Press Ctrl+Enter to apply" hint so it doesn't contradict that. */
.st-key-query_text [data-testid="InputInstructions"] { display: none; }
/* Hidden carrier widget for a pasted picture query -- see the paste-capture
   script by the query box below. Never shown to the user; it only exists
   so JS has a Streamlit-tracked element to write the base64 payload into.
   NOT display:none / visibility:hidden -- Streamlit's text_area only sends
   a programmatically-set value to the backend on blur, and browsers refuse
   .focus()/.blur() on an element that isn't actually rendered, so it has to
   stay a real (if invisible, zero-size, unclickable) element in the layout. */
.st-key-query_image_data {
    position: absolute; width: 1px; height: 1px; overflow: hidden;
    opacity: 0; pointer-events: none;
}
</style>
""", unsafe_allow_html=True)


def round_top_k(k) -> int:
    """Snap K to a multiple of 5, except 1-4 which pass through unchanged."""
    k = int(k)
    if k < 1:
        return 1
    if k <= 4:
        return k
    return max(5, round(k / 5) * 5)


def copy_to_scope(video_id: str):
    """Copy button callback: fills both scope boxes from one frame's video_id."""
    st.session_state.video_filter_text = video_id
    st.session_state.lot_filter_text = video_lot_str(video_id)


def copy_collection_only(video_id: str):
    """Copy button callback for a collection-grouped result: the group spans
    many videos, so only the collection box is meaningful to fill."""
    st.session_state.lot_filter_text = video_lot_str(video_id)


# ---------------------------------------------------------------------------
# Mixed signal: "Change weights" popup -- 4 sliders (0-3) for Keyframe/ASR/
# Caption/OCR, plus per-signal leg checkboxes (OCR has none -- single leg).
# Edits are staged in mw_* widget keys and only committed to the applied
# mixed_weights/mixed_legs dicts on Save; Cancel discards them.
# ---------------------------------------------------------------------------

MIXED_SIGNAL_NAMES = ["Keyframe", "ASR", "Caption", "OCR"]
MIXED_LEG_DEFS = {
    "Keyframe": [("kf_siglip2", "SigLIP2"), ("kf_clip", "CLIP")],
    "ASR": [("asr_siglip", "SigLIP2 ASR"), ("asr_fuzzy", "Fuzzy ASR")],
    "Caption": [("cap_siglip", "SigLIP2 Caption"), ("cap_fuzzy", "Fuzzy Caption")],
}  # OCR intentionally omitted -- single fuzzy-only leg, nothing to choose

MIXED_DEFAULT_WEIGHTS = {name: 1 for name in MIXED_SIGNAL_NAMES}
MIXED_DEFAULT_LEGS = {
    "kf_siglip2": True, "kf_clip": True,
    "asr_siglip": False, "asr_fuzzy": True,
    "cap_siglip": False, "cap_fuzzy": True,
}

st.session_state.setdefault("mixed_weights", dict(MIXED_DEFAULT_WEIGHTS))
st.session_state.setdefault("mixed_legs", dict(MIXED_DEFAULT_LEGS))


def _stage_weights_dialog():
    if not st.session_state.get("_mw_staged", False):
        for name in MIXED_SIGNAL_NAMES:
            st.session_state[f"mw_weight_{name}"] = st.session_state.mixed_weights[name]
        for leg_key, val in st.session_state.mixed_legs.items():
            st.session_state[f"mw_leg_{leg_key}"] = val
        st.session_state._mw_staged = True


def _apply_default_weights():
    for name in MIXED_SIGNAL_NAMES:
        st.session_state[f"mw_weight_{name}"] = 1
    for leg_key, default_val in MIXED_DEFAULT_LEGS.items():
        st.session_state[f"mw_leg_{leg_key}"] = default_val


@st.dialog("Change weights")
def change_weights_dialog():
    _stage_weights_dialog()

    st.caption("Weight per signal (0 = off) with that signal's legs alongside")
    for name in MIXED_SIGNAL_NAMES:
        col_slider, col_legs = st.columns([2, 1])
        with col_slider:
            st.slider(name, min_value=0, max_value=3, key=f"mw_weight_{name}")
        with col_legs:
            if name in MIXED_LEG_DEFS:
                for leg_key, leg_label in MIXED_LEG_DEFS[name]:
                    st.checkbox(leg_label, key=f"mw_leg_{leg_key}")
            else:
                st.caption("Detailed legs")

    st.divider()
    with st.container(horizontal=True):
        st.button("Default", on_click=_apply_default_weights)
        if st.button("Cancel"):
            st.session_state._mw_staged = False
            st.rerun()
        if st.button("Save", type="primary"):
            st.session_state.mixed_weights = {name: st.session_state[f"mw_weight_{name}"] for name in MIXED_SIGNAL_NAMES}
            st.session_state.mixed_legs = {leg_key: st.session_state[f"mw_leg_{leg_key}"] for leg_key in st.session_state.mixed_legs}
            st.session_state._mw_staged = False
            st.rerun()


def df_to_results(df: pd.DataFrame, score_col: str, text_col: str = None) -> list:
    if df is None or df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        n = r.get("n")
        if n is None or pd.isna(n):
            continue
        n = int(n)
        text = r.get(text_col) if text_col else None
        out.append({
            "video_id": r["video_id"], "n": n, "rank": int(r["rank"]),
            "score_label": score_col, "score_val": float(r[score_col]),
            "text": text if isinstance(text, str) else None,
            "thumbnail_path": thumbnail_path(r["video_id"], n),
        })
    return out


@st.dialog("Nearby frames", width="large")
def show_neighbors(video_id: str, center_n: int):
    state_key = f"nbr_extra_{video_id}_{center_n}"
    extra = st.session_state.setdefault(state_key, {"before": 0, "after": 0})

    st.subheader(f"{video_id} — around frame {center_n}")
    if st.button("▲ 10 earlier", key=f"{state_key}_up", width="stretch"):
        extra["before"] += 10

    lo = NEIGHBOR_WINDOW + extra["before"]
    hi = NEIGHBOR_WINDOW + extra["after"]
    candidates = [center_n + d for d in range(-lo, hi + 1) if center_n + d >= 1]
    cols_per_row = 5
    for start in range(0, len(candidates), cols_per_row):
        chunk = candidates[start : start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, n in zip(cols, chunk):
            path = thumbnail_path(video_id, n)
            with col:
                if path and Path(path).exists():
                    st.image(path, width="stretch")
                else:
                    st.caption("(missing)")
                st.caption(f"**{n}**" if n == center_n else str(n))

    if st.button("▼ 10 later", key=f"{state_key}_down", width="stretch"):
        extra["after"] += 10


@st.dialog("Video playback", width="medium")
def frame_playback_dialog(video_id: str, n: int):
    """Play-icon action shared by render_actions (all non-TRAKE signals,
    Mixed, and Hierarchy): opens the source video seeked to this frame's
    own timestamp via map-keyframes -- no marker bar, since there's only
    ever the one frame here (TRAKE's multi-event marker bar/timer stays in
    trake_playback_dialog below, which this is a simpler sibling of)."""
    path = video_path(video_id)
    if not Path(path).exists():
        st.error(f"Video file not found: {path}")
        return
    ts, _fps = keyframe_timestamp(video_id, n)
    st.subheader(f"{video_id} — frame {n}")
    st.video(path, start_time=int(ts) if ts is not None else 0)


@st.dialog("Video playback", width="medium")
def trake_playback_dialog(video_id: str, events: list):
    """TRAKE's play-icon action: opens the source video seeked near the
    first matched event, with a click-to-seek marker row for every matched
    event and a live timestamp/frame readout (TRAKE_SPEC.md Step 5 -- this
    is purely a display aid for the human doing the final scrub; it never
    reads anything back into Streamlit/Python state)."""
    path = video_path(video_id)
    if not Path(path).exists():
        st.error(f"Video file not found: {path}")
        return

    matched = [e for e in events if e["matched"] and e["timestamp"] is not None]
    st.subheader(video_id)
    st.video(path, start_time=int(matched[0]["timestamp"]) if matched else 0)

    fps = next((keyframe_timestamp(video_id, e["n"])[1] for e in matched), None) or 25.0
    markers = [{"event_index": e["event_index"], "ts": e["timestamp"], "label": e["label"]} for e in matched]

    # Only static markup here (data-* attributes carrying the marker/fps
    # payload, base64'd to dodge HTML-attribute quoting entirely) -- no
    # <script> tag. st.html's embedded scripts don't run when the call
    # site is inside an @st.dialog function (confirmed empirically: the
    # identical script executes fine from the main page body but never
    # fires from here, even with unsafe_allow_javascript=True and no
    # console errors). The one script that actually does this element's
    # binding lives in the main page body instead -- see
    # trake_playback_binder_script() below, rendered once per TRAKE
    # results view -- and finds these elements via MutationObserver, the
    # same proven-working pattern as the sidebar's Enter-key script.
    markers_b64 = base64.b64encode(json.dumps(markers).encode()).decode()
    st.html(f"""
    <div id="trake-marker-bar" data-fps="{fps}" data-markers-b64="{markers_b64}" style="position:relative;
        height:18px;margin:4px 0 8px;background:rgba(127,127,127,0.15);border-radius:4px;"></div>
    <div id="trake-timer" style="font-family:monospace;font-size:0.9rem;">--:-- · frame --</div>
    """)

    gaps = [e for e in events if not e["matched"]]
    if gaps:
        st.divider()
        st.caption("Coverage gaps — scrub manually between the nearest matched anchors:")
        for e in gaps:
            idx = e["event_index"]
            before = [m for m in matched if m["event_index"] < idx]
            after = [m for m in matched if m["event_index"] > idx]
            lo = f"{before[-1]['timestamp']:.2f}s ({before[-1]['label']})" if before else "start"
            hi = f"{after[0]['timestamp']:.2f}s ({after[0]['label']})" if after else "end"
            st.write(f"{e['label']}: no direct match — between **{lo}** and **{hi}**")


def render_trake_playback_binder():
    """Companion to trake_playback_dialog's marker-bar markup: rendered
    once from the main page body (NOT from inside the dialog -- st.html's
    embedded <script> silently never executes when the call site is
    inside an @st.dialog function, confirmed empirically) so it actually
    runs, then watches for the dialog's #trake-marker-bar div via the same
    MutationObserver pattern the sidebar's Enter-key script uses, and
    binds the click-to-seek markers + live timer to whichever <video> is
    on the page at the time."""
    st.html("""
    <script>
    (function() {
        function bindOne(bar) {
            if (bar.dataset.trakeBound) return;
            const video = document.querySelector('video');
            const timer = document.getElementById('trake-timer');
            if (!video || !timer) return;
            bar.dataset.trakeBound = "1";
            const fps = parseFloat(bar.dataset.fps) || 25.0;
            const markers = JSON.parse(atob(bar.dataset.markersB64));

            function layoutMarkers() {
                if (!video.duration || !isFinite(video.duration)) return;
                bar.innerHTML = "";
                markers.forEach(function(m) {
                    const pct = Math.max(0, Math.min(100, (m.ts / video.duration) * 100));
                    const tick = document.createElement('div');
                    tick.title = m.label + ' @ ' + m.ts.toFixed(2) + 's';
                    tick.textContent = m.label;
                    tick.style.cssText = 'position:absolute;left:' + pct + '%;top:0;bottom:0;' +
                        'transform:translateX(-50%);cursor:pointer;background:#e64980;color:white;' +
                        'font-size:10px;padding:0 4px;border-radius:3px;line-height:18px;';
                    tick.addEventListener('click', function() { video.currentTime = m.ts; });
                    bar.appendChild(tick);
                });
            }

            function fmtTime(t) {
                const mm = Math.floor(t / 60).toString().padStart(2, '0');
                const ss = (t % 60).toFixed(2).padStart(5, '0');
                return mm + ':' + ss;
            }

            video.addEventListener('loadedmetadata', layoutMarkers);
            if (video.readyState >= 1) layoutMarkers();
            video.addEventListener('timeupdate', function() {
                timer.textContent = fmtTime(video.currentTime) + ' · frame ' + Math.round(video.currentTime * fps);
            });
        }
        function scan() {
            const bar = document.getElementById('trake-marker-bar');
            if (bar) bindOne(bar);
        }
        scan();
        // Guarded singleton -- render_trake_playback_binder() runs on every
        // rerun a TRAKE results view is showing, same accumulation risk as
        // the other MutationObserver-based scripts here.
        if (!window.__hierTrakePlaybackObserverBound) {
            window.__hierTrakePlaybackObserverBound = true;
            new MutationObserver(scan).observe(document.body, {childList: true, subtree: true});
        }
    })();
    </script>
    """, unsafe_allow_javascript=True)


def render_thumb(col, r: dict):
    with col:
        thumb = r["thumbnail_path"]
        if thumb and Path(thumb).exists():
            st.markdown(
                f'<div class="thumb-wrap"><img src="data:image/jpeg;base64,{image_b64(thumb)}"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("(image not found)")
        st.caption(f"**{r['video_id']}** · frame {r['n']}")
        st.caption(f"rank {r['rank']} · {r['score_label']}={r['score_val']:.4f}")
        if r["text"]:
            if st.session_state.get("show_full_text", False):
                st.caption(r["text"])
            else:
                st.caption(r["text"][:140] + ("…" if len(r["text"]) > 140 else ""))


def render_actions(video_id: str, center_n: int, key_prefix: str, key_suffix: str, collection_only: bool = False):
    with st.container(horizontal=True):
        if st.button(":material/more_horiz:", key=f"{key_prefix}_more_{key_suffix}", help="Show more"):
            show_neighbors(video_id, center_n)
        if st.button(":material/play_circle:", key=f"{key_prefix}_play_{key_suffix}", help="Play video"):
            frame_playback_dialog(video_id, center_n)
        if collection_only:
            st.button(":material/content_copy:", key=f"{key_prefix}_copy_{key_suffix}", help="Copy collection id",
                      on_click=copy_collection_only, args=(video_id,))
        else:
            st.button(":material/content_copy:", key=f"{key_prefix}_copy_{key_suffix}", help="Copy video id",
                      on_click=copy_to_scope, args=(video_id,))


def render_grid(results: list, key_prefix: str, group_mode: str = None):
    """group_mode: None (ungrouped), "video" (group frames by video_id), or
    "collection" (group by lot -- Summary mode, where every result is
    already a distinct video so grouping by video would be a no-op)."""
    if not results:
        st.info("No results.")
        return
    cols_per_row = 5

    if not group_mode:
        for start in range(0, len(results), cols_per_row):
            chunk = results[start : start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, r in zip(cols, chunk):
                render_thumb(col, r)
                with col:
                    render_actions(r["video_id"], r["n"], key_prefix, f"{r['video_id']}_{r['n']}_{r['rank']}")
        return

    # Grouped: results already arrive rank-sorted, so a group's first
    # occurrence is its best (lowest-rank) member -- that also decides the
    # group's own display order.
    group_key_fn = video_lot_str if group_mode == "collection" else (lambda vid: vid)
    unit = "video(s)" if group_mode == "collection" else "frame(s)"
    groups: dict = {}
    for r in results:
        groups.setdefault(group_key_fn(r["video_id"]), []).append(r)
    ordered_groups = sorted(groups.items(), key=lambda kv: kv[1][0]["rank"])

    for group_key, frames in ordered_groups:
        best = frames[0]
        st.markdown(f"**{group_key}** · best rank {best['rank']} · {best['score_label']}={best['score_val']:.4f} · {len(frames)} {unit}")
        for start in range(0, len(frames), cols_per_row):
            chunk = frames[start : start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, r in zip(cols, chunk):
                render_thumb(col, r)
        render_actions(best["video_id"], best["n"], key_prefix, f"grp_{group_key}",
                       collection_only=(group_mode == "collection"))
        st.divider()


# ---------------------------------------------------------------------------
# Startup: eagerly build every FAISS index and Elasticsearch index once, up
# front -- st.cache_resource is shared process-wide, so whichever session
# hits this first pays the cost here, and every later session (and every
# later query, on any signal) is served straight from cache instead of
# triggering a build mid-search. Shown as a status bar at the very top of
# the page, above the title, while it runs.
# ---------------------------------------------------------------------------

with st.status("Loading signals…", expanded=False) as _startup_status:
    st.write("Keyframe — SigLIP2 frame index")
    build_frame_index(FRAME_SIGLIP2_GLOB)
    st.write("Keyframe — CLIP frame index")
    build_frame_index(FRAME_CLIP_GLOB)
    st.write("ASR — SigLIP2 index + Elasticsearch")
    build_siglip_asr_index()
    ensure_asr_fuzzy_index()
    st.write("Caption — SigLIP2 index + Elasticsearch")
    build_siglip_caption_index()
    ensure_caption_fuzzy_index()
    st.write("OCR — Elasticsearch")
    ensure_ocr_fuzzy_index()
    st.write("Summary — embeddings + SigLIP2 index + Elasticsearch")
    build_siglip_summary_index()
    ensure_summary_fuzzy_index()
    _startup_status.update(label="All signals ready", state="complete")

st.title("Routing101 by MiLF")

SIGNAL_ICONS = {
    "Keyframe": ":material/image:",
    "ASR": ":material/mic:",
    "Caption": ":material/closed_caption:",
    "OCR": ":material/photo_camera:",
    "Summary": ":material/edit:",
    "Mixed": ":material/call_merge:",
    "TRAKE": ":material/route:",
    "Hierarchy": ":material/account_tree:",
}

# Signal choices offered per TRAKE event row -- same as the segmented
# control above minus TRAKE itself (nested TRAKE makes no sense) and
# Hierarchy (a video-grouped, drilled-down result set, not the single
# ranked frame list trake_search_event/trake_rank_videos expect per
# event) -- computed once so it stays in sync if SIGNAL_ICONS changes.
TRAKE_EVENT_SIGNALS = [name for name in SIGNAL_ICONS if name not in ("TRAKE", "Hierarchy")]

with st.sidebar:
    st.header("Search")

    # Two rows: the five frame/text signals, then Mixed + TRAKE + Hierarchy
    # on their own row below. Two independent segmented_control widgets
    # (Streamlit has no multi-row option within one) kept mutually
    # exclusive via on_change -- picking one clears the other's key so
    # exactly one of the two is ever highlighted, and `mode` is just
    # whichever key is non-None.
    ROW1_SIGNALS = ["Keyframe", "ASR", "Caption", "OCR", "Summary"]
    ROW2_SIGNALS = ["Mixed", "TRAKE", "Hierarchy"]
    st.session_state.setdefault("mode_row1", "Keyframe")
    st.session_state.setdefault("mode_row2", None)

    def _pick_row1():
        st.session_state.mode_row2 = None

    def _pick_row2():
        st.session_state.mode_row1 = None

    st.segmented_control("Signal", options=ROW1_SIGNALS, format_func=lambda m: SIGNAL_ICONS[m],
                          key="mode_row1", label_visibility="collapsed", on_change=_pick_row1)
    st.segmented_control("Signal (row 2)", options=ROW2_SIGNALS, format_func=lambda m: SIGNAL_ICONS[m],
                          key="mode_row2", label_visibility="collapsed", on_change=_pick_row2)
    mode = st.session_state.mode_row2 or st.session_state.mode_row1 or "Keyframe"

    def _weights_button(key: str):
        """Shared by Mixed mode and any TRAKE row (context or event) whose
        signal is set to Mixed -- Mixed has no per-row weight config of its
        own, they all share the one st.session_state.mixed_weights/legs
        dict, same as standalone Mixed mode."""
        if st.button(":material/tune:", key=key, help="Change weights"):
            change_weights_dialog()

    query = None
    if mode != "TRAKE":
        # A plain keyed text_area (not st.chat_input) so the query stays
        # visible in the box after searching -- st.chat_input always clears
        # on submit and has no way to pre-fill a value. To still get
        # Enter-submits-no-manual-newline (text_area's native Enter always
        # inserts a newline), a small JS snippet intercepts Enter on the
        # underlying <textarea> and blurs it instead, which is what
        # actually commits a text_area's value and triggers the rerun --
        # the box still soft-wraps long text on its own, it just never
        # accepts an explicit newline from the user.
        query = st.text_area("Query", placeholder="e.g. một người đàn ông đang lái xe máy (or paste an image)",
                              key="query_text", label_visibility="collapsed")
        st.html("""
        <script>
        (function() {
            function bind() {
                const ta = document.querySelector('.st-key-query_text textarea');
                if (!ta || ta.dataset.enterSubmitBound) return;
                ta.dataset.enterSubmitBound = "1";
                ta.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.isComposing) {
                        e.preventDefault();
                        e.stopPropagation();
                        ta.blur();
                    }
                });
            }
            bind();
            // Guarded singleton: this whole <script> block re-runs on EVERY
            // Streamlit rerun (any button/dialog/widget interaction, not
            // just a new query), since it's emitted unconditionally each
            // time the sidebar renders. Without this guard, each rerun left
            // behind one more permanent MutationObserver on document.body
            // (bind()'s own dataset check only stops double-binding the
            // *listener*, not the observer) -- they compound over a
            // session until the accumulated observers make every DOM
            // mutation (i.e. every future rerun) progressively slower,
            // which is exactly what made unrelated buttons like "Show
            // more"/"Copy" feel like they hung.
            if (!window.__hierQueryEnterObserverBound) {
                window.__hierQueryEnterObserverBound = true;
                new MutationObserver(bind).observe(document.body, {childList: true, subtree: true});
            }
        })();
        </script>
        """, unsafe_allow_javascript=True)

        # Picture-query input: paste an image directly into the query box
        # above, no separate upload control. A 'paste' listener on that
        # textarea intercepts any image in the clipboard, base64-encodes it
        # client-side, and writes it into this hidden text_area (the only
        # way to get binary-ish data into Streamlit's session_state without
        # a real file uploader) via focus + native-setter + dispatched-
        # 'input' + blur (see commitValue() below) -- React controlled
        # inputs ignore a plain `.value =` assignment, so it has to go
        # through the underlying property setter for 'input' to stick, and
        # Streamlit's text_area itself only ships a value to the backend on
        # blur/Ctrl+Enter, not on every 'input' event.
        def _consume_pasted_image():
            """on_change callback for the hidden carrier widget -- runs
            BEFORE the rerun re-instantiates it, so it's safe to clear its
            backing value here (Streamlit forbids writing to a widget's
            session_state key after instantiation, same convention as
            copy_to_scope's on_click). This is what keeps a pasted picture
            query cheap long-term: a widget's session_state value is part
            of what Streamlit re-transmits over the websocket on EVERY
            future rerun -- ANY rerun, not just ones touching the image --
            so leaving a multi-MB base64 string sitting in one there made
            every later click (Show more, Copy, an unrelated text search,
            ...) progressively heavier for the rest of the session. Once
            decoded here, only the raw bytes survive, in a plain
            (non-widget) session_state entry that never leaves the server."""
            raw = st.session_state.get("query_image_data", "")
            if raw.startswith("data:image"):
                try:
                    _, b64_payload = raw.split(",", 1)
                    st.session_state.query_image_bytes = base64.b64decode(b64_payload)
                except Exception:
                    st.session_state.query_image_bytes = None
            st.session_state.query_image_data = ""

        st.text_area("Pasted image (hidden)", key="query_image_data", label_visibility="collapsed",
                     on_change=_consume_pasted_image)
        st.html("""
        <script>
        (function() {
            function commitValue(el, value) {
                // Streamlit's text_area only ships a programmatically-set
                // value back to the backend on blur (or Ctrl+Enter) -- so
                // this has to focus the element, set the value through
                // React's tracked setter, fire 'input' so React notices,
                // THEN blur to actually trigger the commit + rerun.
                const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                el.focus();
                setter.call(el, value);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.blur();
            }
            function bind() {
                const ta = document.querySelector('.st-key-query_text textarea');
                const hidden = document.querySelector('.st-key-query_image_data textarea');
                if (!ta || !hidden || ta.dataset.pasteImageBound) return;
                ta.dataset.pasteImageBound = "1";
                ta.addEventListener('paste', function(e) {
                    const items = (e.clipboardData || window.clipboardData).items;
                    for (let i = 0; i < items.length; i++) {
                        if (items[i].type.indexOf('image') === 0) {
                            e.preventDefault();
                            const file = items[i].getAsFile();
                            const reader = new FileReader();
                            reader.onload = function(ev) { commitValue(hidden, ev.target.result); };
                            reader.readAsDataURL(file);
                            break;
                        }
                    }
                });
            }
            bind();
            // Same guarded-singleton reasoning as the Enter-submit script
            // above -- this block also re-runs on every rerun.
            if (!window.__hierQueryPasteObserverBound) {
                window.__hierQueryPasteObserverBound = true;
                new MutationObserver(bind).observe(document.body, {childList: true, subtree: true});
            }
        })();
        </script>
        """, unsafe_allow_javascript=True)

        image_query = None
        img_bytes = st.session_state.get("query_image_bytes")
        if img_bytes:
            try:
                image_query = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                st.warning(f"Couldn't decode the pasted image ({e}).")
                st.session_state.query_image_bytes = None

        if image_query is not None:
            query = image_query  # an image query takes priority over any typed text
            col_prev, col_clear = st.columns([3, 1])
            col_prev.image(image_query, caption="Image query", width=120)
            if col_clear.button("Clear", key="clear_query_image"):
                st.session_state.query_image_bytes = None
                st.rerun()

        if mode == "Mixed":
            _weights_button("mixed_weights_btn")
            w = st.session_state.mixed_weights
            st.caption(f"Weights — Keyframe {w['Keyframe']} · ASR {w['ASR']} · Caption {w['Caption']} · OCR {w['OCR']}")
    else:
        # TRAKE: an optional context row (E0 -- defaults to Summary, can be
        # left blank, contributes a top-K/2 ranking bonus -- see
        # trake_rank_videos) plus a dynamic list of required event rows
        # (E1, E2, ... -- minimum 1). Row order among E1+ IS the required
        # temporal order (TRAKE_SPEC.md); E0 slots into that same ordered
        # chain when filled in. Each row is 2 lines: the query text (an
        # auto-wrapping text_area, same as the single-signal Query box)
        # then a second line of signal + (Mixed only) weights + remove.
        # Rows carry a stable "id" used as their widget key (NOT list
        # index) -- widget state is keyed by session key, so keying by
        # position would show stale text/signal on rows after a mid-list
        # removal shifts them into a lower-numbered slot.
        st.session_state.setdefault("trake_context", {"text": "", "signal": "Summary"})
        st.session_state.setdefault("trake_events", [{"id": 0, "text": "", "signal": "Keyframe"}])
        st.session_state.setdefault("trake_next_id", 1)

        st.caption("Context — E0 (optional, boosts matching videos)")
        ctx = st.session_state.trake_context
        ctx["text"] = st.text_area("Context text", value=ctx["text"], placeholder="optional context query",
                                    key="trake_ctx_text", label_visibility="collapsed", height=68)
        col_ctx_sig, col_ctx_w = st.columns([4, 1])
        ctx["signal"] = col_ctx_sig.selectbox("Context signal", TRAKE_EVENT_SIGNALS,
                                               index=TRAKE_EVENT_SIGNALS.index(ctx["signal"]),
                                               key="trake_ctx_signal", label_visibility="collapsed")
        if ctx["signal"] == "Mixed":
            with col_ctx_w:
                _weights_button("trake_ctx_weights_btn")

        st.divider()
        st.caption("Events, in required order")
        for i, ev in enumerate(st.session_state.trake_events):
            key_id = ev["id"]
            ev["text"] = st.text_area(f"Event {i + 1} text", value=ev["text"], placeholder=f"E{i + 1} query text",
                                       key=f"trake_ev_text_{key_id}", label_visibility="collapsed", height=68)
            # Read the widget's OWN current session_state value (not ev
            # ["signal"], which still holds last rerun's value until the
            # selectbox below re-assigns it) so the weights button appears
            # the same rerun a user switches this row to Mixed, not one
            # rerun later.
            if st.session_state.get(f"trake_ev_signal_{key_id}", ev["signal"]) == "Mixed":
                col_sig, col_w, col_rm = st.columns([3, 1, 1])
                with col_w:
                    _weights_button(f"trake_ev_weights_{key_id}")
            else:
                col_sig, col_rm = st.columns([4, 1])
            ev["signal"] = col_sig.selectbox(f"Event {i + 1} signal", TRAKE_EVENT_SIGNALS,
                                              index=TRAKE_EVENT_SIGNALS.index(ev["signal"]),
                                              key=f"trake_ev_signal_{key_id}", label_visibility="collapsed")
            if col_rm.button(":material/close:", key=f"trake_ev_rm_{key_id}", disabled=len(st.session_state.trake_events) <= 1,
                              help="Remove event"):
                st.session_state.trake_events.pop(i)
                st.rerun()
        if st.button("Add event", icon=":material/add:"):
            new_id = st.session_state.trake_next_id
            st.session_state.trake_next_id += 1
            st.session_state.trake_events.append({"id": new_id, "text": "", "signal": "Keyframe"})
            st.rerun()

        # Same Enter-submits-no-manual-newline trick as the single-signal
        # Query box above, generalized to every context/event text_area at
        # once (class-prefix selector, not a fixed id) since rows are added
        # and removed dynamically -- MutationObserver re-scans and binds
        # any newly-added row's textarea the same way.
        st.html("""
        <script>
        (function() {
            function bindAll() {
                const areas = document.querySelectorAll(
                    '[class*="st-key-trake_ev_text_"] textarea, [class*="st-key-trake_ctx_text"] textarea');
                areas.forEach(function(ta) {
                    if (ta.dataset.enterSubmitBound) return;
                    ta.dataset.enterSubmitBound = "1";
                    ta.addEventListener('keydown', function(e) {
                        if (e.key === 'Enter' && !e.isComposing) {
                            e.preventDefault();
                            e.stopPropagation();
                            ta.blur();
                        }
                    });
                });
            }
            bindAll();
            // Guarded singleton -- see the same note on the single-signal
            // Query box's Enter-submit script; this block re-runs on every
            // TRAKE-mode rerun too.
            if (!window.__hierTrakeEnterObserverBound) {
                window.__hierTrakeEnterObserverBound = true;
                new MutationObserver(bindAll).observe(document.body, {childList: true, subtree: true});
            }
        })();
        </script>
        """, unsafe_allow_javascript=True)

    st.session_state.setdefault("use_video_scope", False)
    st.session_state.setdefault("use_collection_scope", False)

    def _exclusive_scope(changed_key: str):
        other_key = "use_collection_scope" if changed_key == "use_video_scope" else "use_video_scope"
        if st.session_state[changed_key]:
            st.session_state[other_key] = False

    col_v, col_c = st.columns(2)
    video_filter_text = col_v.text_input("Search in video", placeholder="e.g. L21_V001", key="video_filter_text")
    lot_filter_text = col_c.text_input("Search in collection", placeholder="e.g. L21-L30", key="lot_filter_text")
    use_video_scope = col_v.checkbox("Use video", key="use_video_scope",
                                      on_change=_exclusive_scope, args=("use_video_scope",))
    use_collection_scope = col_c.checkbox("Use collection", key="use_collection_scope",
                                           on_change=_exclusive_scope, args=("use_collection_scope",))

    video_filter = video_filter_text if use_video_scope else ""
    lot_filter = parse_lot_range(lot_filter_text) if use_collection_scope else None

    group_mode, trake_top_v, hier_top_g = None, DISPLAY_N, TOP_G_DEFAULT

    def _snap_top_k():
        st.session_state.top_k = round_top_k(st.session_state.top_k)

    if mode == "TRAKE":
        col_k, col_v2 = st.columns(2)
        col_k.number_input("Top-K", min_value=1, max_value=200, value=DISPLAY_N, step=5,
                            key="top_k", on_change=_snap_top_k)
        trake_top_v = col_v2.number_input("Top-V", min_value=1, max_value=50, value=10, step=1, key="trake_top_v")
    elif mode == "Hierarchy":
        col_k, col_g = st.columns(2)
        col_k.number_input("Top-K", min_value=1, max_value=200, value=DISPLAY_N, step=5,
                            key="top_k", on_change=_snap_top_k)
        hier_top_g = col_g.number_input("Top-G", min_value=1, max_value=50, value=TOP_G_DEFAULT, step=1, key="hier_top_g",
                                         help="Frames kept per video after per-video drill-down.")
    else:
        st.number_input("Top-K", min_value=1, max_value=200, value=DISPLAY_N, step=5,
                         key="top_k", on_change=_snap_top_k)
    top_k = round_top_k(st.session_state.top_k)
    fetch_k = max(FETCH_K, top_k)

    if mode not in ("TRAKE", "Hierarchy"):
        # Summary results are already one-per-video, so "group by video"
        # would be a no-op there -- group by collection (lot) instead.
        # Hierarchy is always grouped by video by construction, so it skips
        # this toggle entirely (see the Hierarchy branch below).
        group_toggle_label = "Group by collection" if mode == "Summary" else "Group by video"
        group_toggled = st.toggle(group_toggle_label, key="group_by_video")
        if group_toggled:
            group_mode = "collection" if mode == "Summary" else "video"

    st.toggle("Show full text", key="show_full_text")

    st.divider()

    if mode == "Keyframe":
        use_siglip2 = st.checkbox("SigLIP2", value=True, key="kf_use_siglip2")
        use_clip = st.checkbox("CLIP", value=True, key="kf_use_clip")
        use_rrf = st.checkbox("RRF", value=True, key="kf_use_rrf")
    elif mode == "ASR":
        use_siglip_asr = st.checkbox("SigLIP2 ASR", value=True, key="asr_use_siglip")
        use_fuzzy = st.checkbox("Fuzzy ASR", value=True, key="asr_use_fuzzy")
        use_rrf = st.checkbox("RRF ASR", value=True, key="asr_use_rrf")
    elif mode == "Caption":
        use_siglip_cap = st.checkbox("SigLIP2 Caption", value=True, key="cap_use_siglip")
        use_fuzzy = st.checkbox("Fuzzy Caption", value=True, key="cap_use_fuzzy")
        use_rrf = st.checkbox("RRF Caption", value=True, key="cap_use_rrf")
    elif mode == "OCR":
        st.caption("Single leg: fuzzy text search only, no embedding leg.")
    elif mode == "Summary":
        use_siglip_sum = st.checkbox("SigLIP2 Summary", value=True, key="sum_use_siglip")
        use_fuzzy = st.checkbox("Fuzzy Summary", value=True, key="sum_use_fuzzy")
        use_rrf = st.checkbox("RRF Summary", value=True, key="sum_use_rrf")
    elif mode == "Hierarchy":
        st.caption("SigLIP2 frame search, grouped by video, drilled down to Top-G frames/video. "
                   "No leg choice -- text or picture query, SigLIP2 only.")

if mode == "Keyframe":
    if query:
        siglip2_df = clip_df = None
        if use_siglip2 or use_rrf:
            siglip2_df = apply_filters(search_siglip2_frame(query, k=fetch_k), video_filter, lot_filter)
        if use_clip or use_rrf:
            clip_df = apply_filters(search_clip_frame(query, k=fetch_k), video_filter, lot_filter)

        if use_siglip2:
            st.subheader("SigLIP2")
            render_grid(df_to_results(siglip2_df.head(top_k), "score"), "kf_sig", group_mode)
        if use_clip:
            st.subheader("CLIP")
            if is_image_query(query):
                st.caption("Skipped — picture queries are SigLIP2-only.")
            else:
                render_grid(df_to_results(clip_df.head(top_k), "score"), "kf_clip", group_mode)
        if use_rrf:
            st.subheader("RRF")
            if is_image_query(query):
                st.caption("Skipped — picture queries only ever have one active leg (SigLIP2), nothing to fuse.")
            else:
                render_grid(df_to_results(rrf_fuse_frame([siglip2_df, clip_df], top_n=top_k), "rrf_score"), "kf_rrf", group_mode)
        if not (use_siglip2 or use_clip or use_rrf):
            st.info("Check at least one search option in the sidebar.")
    else:
        st.info("Type a query to search.")

elif mode == "ASR":
    if query:
        siglip_df = fuzzy_df = None
        if use_siglip_asr or use_rrf:
            siglip_df = apply_filters(search_siglip_asr(query, k=fetch_k), video_filter, lot_filter)
        if use_fuzzy or use_rrf:
            fuzzy_df = apply_filters(search_asr_fuzzy(query, k=fetch_k), video_filter, lot_filter)

        if use_siglip_asr:
            st.subheader("SigLIP2 ASR")
            render_grid(df_to_results(attach_keyframe_asr(siglip_df).head(top_k), "score", "text"), "asr_sig", group_mode)
        if use_fuzzy:
            st.subheader("Fuzzy ASR")
            render_grid(df_to_results(attach_keyframe_asr(fuzzy_df).head(top_k), "score", "text"), "asr_fuzzy", group_mode)
        if use_rrf:
            st.subheader("RRF ASR")
            if is_image_query(query):
                st.caption("Skipped — picture queries only ever have one active leg (SigLIP2), nothing to fuse.")
            else:
                fused = attach_keyframe_asr(rrf_fuse_asr({"siglip_asr": siglip_df, "fuzzy": fuzzy_df}, top_n=top_k))
                render_grid(df_to_results(fused, "rrf_score", "text"), "asr_rrf", group_mode)
        if not (use_siglip_asr or use_fuzzy or use_rrf):
            st.info("Check at least one search option in the sidebar.")
    else:
        st.info("Type a query to search.")

elif mode == "Caption":
    if query:
        siglip_df = fuzzy_df = None
        if use_siglip_cap or use_rrf:
            siglip_df = apply_filters(search_siglip_caption(query, k=fetch_k), video_filter, lot_filter)
        if use_fuzzy or use_rrf:
            fuzzy_df = apply_filters(search_caption_fuzzy(query, k=fetch_k), video_filter, lot_filter)

        if use_siglip_cap:
            st.subheader("SigLIP2 Caption")
            render_grid(df_to_results(attach_keyframe_caption(siglip_df).head(top_k), "score", "text"), "cap_sig", group_mode)
        if use_fuzzy:
            st.subheader("Fuzzy Caption")
            render_grid(df_to_results(attach_keyframe_caption(fuzzy_df).head(top_k), "score", "text"), "cap_fuzzy", group_mode)
        if use_rrf:
            st.subheader("RRF Caption")
            if is_image_query(query):
                st.caption("Skipped — picture queries only ever have one active leg (SigLIP2), nothing to fuse.")
            else:
                fused = attach_keyframe_caption(rrf_fuse_caption({"siglip_caption": siglip_df, "fuzzy": fuzzy_df}, top_n=top_k))
                render_grid(df_to_results(fused, "rrf_score", "text"), "cap_rrf", group_mode)
        if not (use_siglip_cap or use_fuzzy or use_rrf):
            st.info("Check at least one search option in the sidebar.")
    else:
        st.info("Type a query to search.")

elif mode == "OCR":
    if query:
        if is_image_query(query):
            st.info("OCR is fuzzy text search only — not available for picture queries.")
        else:
            df = apply_filters(search_ocr_fuzzy(query, k=fetch_k), video_filter, lot_filter)
            st.subheader("Fuzzy OCR")
            render_grid(df_to_results(attach_keyframe_ocr(df).head(top_k), "score", "text"), "ocr_fuzzy", group_mode)
    else:
        st.info("Type a query to search.")

elif mode == "Summary":
    if query:
        siglip_df = fuzzy_df = None
        if use_siglip_sum or use_rrf:
            siglip_df = apply_filters(search_siglip_summary(query, k=fetch_k), video_filter, lot_filter)
        if use_fuzzy or use_rrf:
            fuzzy_df = apply_filters(search_summary_fuzzy(query, k=fetch_k), video_filter, lot_filter)

        if use_siglip_sum:
            st.subheader("SigLIP2 Summary")
            render_grid(df_to_results(attach_keyframe_summary(siglip_df).head(top_k), "score", "text"), "sum_sig", group_mode)
        if use_fuzzy:
            st.subheader("Fuzzy Summary")
            render_grid(df_to_results(attach_keyframe_summary(fuzzy_df).head(top_k), "score", "text"), "sum_fuzzy", group_mode)
        if use_rrf:
            st.subheader("RRF Summary")
            if is_image_query(query):
                st.caption("Skipped — picture queries only ever have one active leg (SigLIP2), nothing to fuse.")
            else:
                fused = attach_keyframe_summary(rrf_fuse_summary({"siglip_summary": siglip_df, "fuzzy": fuzzy_df}, top_n=top_k))
                render_grid(df_to_results(fused, "rrf_score", "text"), "sum_rrf", group_mode)
        if not (use_siglip_sum or use_fuzzy or use_rrf):
            st.info("Check at least one search option in the sidebar.")
    else:
        st.info("Type a query to search.")

elif mode == "Mixed":
    if query:
        weights = st.session_state.mixed_weights
        legs = st.session_state.mixed_legs
        signal_dfs = {}
        if weights.get("Keyframe", 0):
            signal_dfs["Keyframe"] = _mixed_keyframe_df(query, fetch_k, video_filter, lot_filter, legs)
        if weights.get("ASR", 0):
            signal_dfs["ASR"] = _mixed_asr_df(query, fetch_k, video_filter, lot_filter, legs)
        if weights.get("Caption", 0):
            signal_dfs["Caption"] = _mixed_caption_df(query, fetch_k, video_filter, lot_filter, legs)
        if weights.get("OCR", 0):
            signal_dfs["OCR"] = _mixed_ocr_df(query, fetch_k, video_filter, lot_filter)

        if not signal_dfs:
            st.info("Every signal weight is 0 — open **Change weights** and enable at least one.")
        else:
            st.subheader("Mixed (weighted RRF)")
            fused = rrf_fuse_weighted(signal_dfs, weights, top_n=top_k)
            render_grid(df_to_results(fused, "rrf_score"), "mixed", group_mode)
    else:
        st.info("Type a query to search.")

elif mode == "Hierarchy":
    if query:
        base_df = apply_filters(search_siglip2_frame(query, k=fetch_k), video_filter, lot_filter)
        if base_df is None or base_df.empty:
            st.info("No results.")
        else:
            # Same grouping render_grid's group_mode="video" does -- results
            # already arrive rank-sorted, so each video's first occurrence
            # is its best (lowest-rank) frame and fixes that group's own
            # display order.
            base_results = df_to_results(base_df.head(top_k), "score")
            groups: dict = {}
            order = []
            for r in base_results:
                vid = r["video_id"]
                if vid not in groups:
                    groups[vid] = []
                    order.append(vid)
                groups[vid].append(r)

            # Per-video Top-G override: the "Expand" button below bumps just
            # that one video's effective G by 10, independent of every other
            # group and of the sidebar's Top-G control -- keyed by video_id
            # so it survives reruns until the page is reset.
            st.session_state.setdefault("hier_extra_g", {})
            extra_g = st.session_state.hier_extra_g

            st.subheader("Hierarchy Search")
            for vid in order:
                best = groups[vid][0]
                st.markdown(f"**{vid}** · best rank {best['rank']} · {best['score_label']}={best['score_val']:.4f} · "
                            f"{len(groups[vid])} frame(s) from Step 1")

                # Step 2: seed-frame picker, scoped to this video only --
                # options are this video's own Step 1 frames, default
                # (index 0) is the top-1 frame. The widget's own
                # session_state (keyed per video_id) IS the seed store, so
                # picking a different frame here and nothing else already
                # persists it across reruns.
                seed_options = [f["n"] for f in groups[vid]]
                top1_n = seed_options[0]
                seed_n = st.selectbox(
                    f"Seed frame for {vid}", seed_options, key=f"hier_seed_{vid}",
                    label_visibility="collapsed",
                    format_func=lambda n, top1=top1_n: f"Seed: frame {n}" + (" (top-1)" if n == top1 else ""),
                )

                # Step 3: drill-down using that seed, up to this video's
                # effective Top-G (sidebar Top-G + any "Expand" bumps).
                effective_g = hier_top_g + extra_g.get(vid, 0)
                frames = hierarchy_expand_group(vid, groups[vid], effective_g, fetch_k, seed_n=seed_n)
                cols_per_row = 5
                for start in range(0, len(frames), cols_per_row):
                    chunk = frames[start : start + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, r in zip(cols, chunk):
                        render_thumb(col, r)
                col_actions, col_expand = st.columns([4, 1])
                with col_actions:
                    render_actions(vid, best["n"], "hier", f"grp_{vid}")
                with col_expand:
                    if st.button("Expand", icon=":material/arrow_downward:", key=f"hier_expand_{vid}",
                                 help="Pull in 10 more frames from this video", width="stretch"):
                        extra_g[vid] = extra_g.get(vid, 0) + 10
                        st.rerun()
                st.divider()
    else:
        st.info("Type a query, or paste an image, to search.")

elif mode == "TRAKE":
    events_cfg = st.session_state.get("trake_events", [])
    texts = [ev["text"].strip() for ev in events_cfg]
    if len(events_cfg) < 1 or not all(texts):
        st.info("Fill in every event's query text to search (minimum 1 event).")
    else:
        ctx = st.session_state.get("trake_context", {"text": "", "signal": "Summary"})
        ctx_text = ctx["text"].strip()

        # Context query (E0) is optional: when filled in it slots into the
        # SAME ordered chain as E1+ (must precede them, contributes to
        # coverage) -- when left blank it's dropped entirely rather than
        # counted as an always-missing event, so it never drags down every
        # other video's coverage fraction.
        all_dfs, labels = [], []
        if ctx_text:
            all_dfs.append(trake_search_event(ctx_text, ctx["signal"], fetch_k, video_filter, lot_filter))
            labels.append("E0")
        for i, ev in enumerate(events_cfg):
            all_dfs.append(trake_search_event(ev["text"], ev["signal"], fetch_k, video_filter, lot_filter))
            labels.append(f"E{i + 1}")

        # Context bonus: any video in the context query's own top-(Top-K/2)
        # candidates gets a flat score bump, independent of whether that
        # video also satisfies the ordered-chain match above.
        bonus_video_ids = None
        if ctx_text:
            ctx_df = all_dfs[0]
            if ctx_df is not None and not ctx_df.empty:
                half = max(1, top_k // 2)
                bonus_video_ids = set(ctx_df.sort_values("rank").head(half)["video_id"])

        candidates = trake_rank_videos(all_dfs, labels, trake_top_v, bonus_video_ids=bonus_video_ids)
        render_trake_playback_binder()
        st.subheader("TRAKE")
        if not candidates:
            st.info("No video matches every event in the required order. Try broader event text, "
                     "fewer events, or a different signal per event.")
        for c in candidates:
            n_matched = sum(1 for e in c["events"] if e["matched"])
            st.markdown(f"**{c['video_id']}** · video_score={c['video_score']:.4f} · "
                        f"coverage {n_matched}/{len(c['events'])}")
            # Force at least a 2-wide layout so a single-event result's
            # thumbnail renders at the same size as a 2-event one, instead
            # of stretching to the full row width.
            n_display = len(c["events"])
            cols = st.columns(max(2, n_display))
            # Hover zoom (thumb-wrap's CSS :hover rule) is only useful once
            # there are enough thumbnails that a closer look actually helps
            # -- for 1-2 events it just makes the single/pair of frames
            # jump around, so it's suppressed via the thumb-wrap-static
            # modifier class below.
            thumb_class = "thumb-wrap" if n_display >= 3 else "thumb-wrap thumb-wrap-static"
            for col, e in zip(cols, c["events"]):
                with col:
                    if e["matched"]:
                        thumb = thumbnail_path(e["video_id"], e["n"])
                        if thumb and Path(thumb).exists():
                            st.markdown(
                                f'<div class="{thumb_class}"><img src="data:image/jpeg;base64,{image_b64(thumb)}"></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("(image not found)")
                        st.caption(f"**{e['label']}** · frame {e['n']}")
                        st.caption(f"{e['score_label']}={e['score_val']:.4f}")
                    else:
                        st.caption(f"**{e['label']}**")
                        st.caption("no match")
            # Single play-icon action per video: acts as both "show more"
            # (opens playback) and "copy" (sets video/collection scope) --
            # per-event thumbnails above are display-only, no own actions.
            # copy_to_scope runs via on_click (not called directly here) --
            # the scope text_input widgets above have already been
            # instantiated by the time this line runs, and Streamlit
            # forbids writing to a widget's session_state key after that;
            # on_click callbacks run before the rerun that re-instantiates
            # them, same convention render_actions' copy button uses.
            if st.button(":material/play_circle:", key=f"trake_play_{c['video_id']}", help="Video playback",
                          on_click=copy_to_scope, args=(c["video_id"],)):
                trake_playback_dialog(c["video_id"], c["events"])
            st.divider()
