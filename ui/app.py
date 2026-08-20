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
"""

import base64
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import torch

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


def search_siglip2_frame(query: str, k: int = FETCH_K) -> pd.DataFrame:
    index, lookup_df = build_frame_index(FRAME_SIGLIP2_GLOB)
    qvec = encode_text_siglip2([query])[0]
    return _search_frame(index, lookup_df, qvec, k)


def search_clip_frame(query: str, k: int = FETCH_K) -> pd.DataFrame:
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


def search_siglip_asr(query: str, k: int = FETCH_K) -> pd.DataFrame:
    index, meta = build_siglip_asr_index()
    qvec = l2_normalize(encode_text_siglip2([query]))
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
    if not es.indices.exists(index=ES_INDEX_ASR):
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


def search_asr_fuzzy(query: str, k: int = FETCH_K) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["rank", "score", "video_id", "segment_id", "start_sec", "text"])
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


def search_siglip_caption(query: str, k: int = FETCH_K) -> pd.DataFrame:
    index, meta = build_siglip_caption_index()
    qvec = l2_normalize(encode_text_siglip2([query]))
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
    if not es.indices.exists(index=ES_INDEX_CAPTION):
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


def search_caption_fuzzy(query: str, k: int = FETCH_K) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["rank", "score", "video_id", "frame_id", "text"])
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
    if not es.indices.exists(index=ES_INDEX_OCR):
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


def search_ocr_fuzzy(query: str, k: int = FETCH_K) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["rank", "score", "video_id", "frame_id", "text"])
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


def search_siglip_summary(query: str, k: int = FETCH_K) -> pd.DataFrame:
    index, meta = build_siglip_summary_index()
    qvec = l2_normalize(encode_text_siglip2([query]))
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
    if not es.indices.exists(index=ES_INDEX_SUMMARY):
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


def search_summary_fuzzy(query: str, k: int = FETCH_K) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["rank", "score", "video_id", "text"])
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
        new MutationObserver(scan).observe(document.body, {childList: true, subtree: true});
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
}

# Signal choices offered per TRAKE event row -- same six as the segmented
# control above, computed once so it stays in sync if SIGNAL_ICONS changes.
TRAKE_EVENT_SIGNALS = [name for name in SIGNAL_ICONS if name != "TRAKE"]

with st.sidebar:
    st.header("Search")

    # Two rows: the five frame/text signals, then Mixed + TRAKE on their own
    # row below. Two independent segmented_control widgets (Streamlit has no
    # multi-row option within one) kept mutually exclusive via on_change --
    # picking one clears the other's key so exactly one of the two is ever
    # highlighted, and `mode` is just whichever key is non-None.
    ROW1_SIGNALS = ["Keyframe", "ASR", "Caption", "OCR", "Summary"]
    ROW2_SIGNALS = ["Mixed", "TRAKE"]
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
        query = st.text_area("Query", placeholder="e.g. một người đàn ông đang lái xe máy",
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
            new MutationObserver(bind).observe(document.body, {childList: true, subtree: true});
        })();
        </script>
        """, unsafe_allow_javascript=True)

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
            new MutationObserver(bindAll).observe(document.body, {childList: true, subtree: true});
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

    group_mode, trake_top_v = None, DISPLAY_N

    def _snap_top_k():
        st.session_state.top_k = round_top_k(st.session_state.top_k)

    if mode == "TRAKE":
        col_k, col_v2 = st.columns(2)
        col_k.number_input("Top-K", min_value=1, max_value=200, value=DISPLAY_N, step=5,
                            key="top_k", on_change=_snap_top_k)
        trake_top_v = col_v2.number_input("Top-V", min_value=1, max_value=50, value=10, step=1, key="trake_top_v")
    else:
        st.number_input("Top-K", min_value=1, max_value=200, value=DISPLAY_N, step=5,
                         key="top_k", on_change=_snap_top_k)
    top_k = round_top_k(st.session_state.top_k)
    fetch_k = max(FETCH_K, top_k)

    if mode != "TRAKE":
        # Summary results are already one-per-video, so "group by video"
        # would be a no-op there -- group by collection (lot) instead.
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
            render_grid(df_to_results(clip_df.head(top_k), "score"), "kf_clip", group_mode)
        if use_rrf:
            st.subheader("RRF")
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
            fused = attach_keyframe_caption(rrf_fuse_caption({"siglip_caption": siglip_df, "fuzzy": fuzzy_df}, top_n=top_k))
            render_grid(df_to_results(fused, "rrf_score", "text"), "cap_rrf", group_mode)
        if not (use_siglip_cap or use_fuzzy or use_rrf):
            st.info("Check at least one search option in the sidebar.")
    else:
        st.info("Type a query to search.")

elif mode == "OCR":
    if query:
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
