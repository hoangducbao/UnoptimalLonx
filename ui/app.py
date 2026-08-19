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

Three modes, picked with the segmented control:
  Keyframe  SigLIP2 + CLIP ViT-B/32 (Multilingual-CLIP text tower) frame
            embeddings + RRF of the two.
  ASR       SigLIP2-ASR-segment embeddings + Elasticsearch fuzzy search
            over the transcript text + RRF of the two.
  Caption   SigLIP2-caption embeddings + Elasticsearch fuzzy search over
            frame captions + RRF of the two.

Every leg, from any mode, is normalized to the same result shape before
rendering -- {video_id, n, rank, score_label, score_val, text} where `n`
is always the 1-indexed keyframe number on disk (map-keyframes.n) -- so
one render_grid() and one "show more" neighbor popup serve all three
modes. The CLIP ViT-B/32 leg is Keyframe-only (dropped from ASR/Caption
per the current scope -- SigLIP2 + fuzzy + RRF only there).
"""

import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import torch

st.set_page_config(page_title="Routing101 — retrieval preview", layout="wide")

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import clip_encoder  # noqa: E402  (pipeline/clip_encoder.py) -- Multilingual-CLIP text tower, Keyframe mode only

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FETCH_K = 100      # candidates pulled per leg, gives RRF a real pool to fuse
DISPLAY_N = 20
RRF_K = 60
NEIGHBOR_WINDOW = 7  # "show more" popup: +/- this many frames by frame id

SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-384"

FRAME_SIGLIP2_GLOB = "D:/University/Summ26/AICDataExtracted/siglib_embed/*.npy"
FRAME_CLIP_GLOB = "D:/University/Summ26/AICData/clip-features-32/*.npy"

ASR_EMBED_DIR = Path("D:/University/Summ26/AICDataExtracted/asr_embed")
TRANSCRIPTS_DIR = Path("D:/University/Summ26/AICDataExtracted/transcripts")

CAPTIONING_DIR = Path("D:/University/Summ26/AICDataExtracted/captioning")
SIGLIP_CAPTION_DIR = Path("D:/University/Summ26/AICDataExtracted/siglip_caption")

MAP_KEYFRAMES_DIR = Path("D:/University/Summ26/AICData/map-keyframes")
THUMBNAIL_ROOT = Path("D:/University/Summ26/AICData/keyframes")

INDEX_DIR = REPO_ROOT / "index"
ASR_INDEX_DIR = INDEX_DIR / "routing101_asr"
CAPTION_INDEX_DIR = INDEX_DIR / "routing101_caption"
ASR_INDEX_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_INDEX_DIR.mkdir(parents=True, exist_ok=True)
SIGLIP_ASR_FAISS = ASR_INDEX_DIR / "siglip_asr_flat_ip.index"
SIGLIP_ASR_META = ASR_INDEX_DIR / "meta_siglip_asr.csv"
SIGLIP_CAPTION_FAISS = CAPTION_INDEX_DIR / "siglip_caption_flat_ip.index"
SIGLIP_CAPTION_META = CAPTION_INDEX_DIR / "meta_siglip_caption.csv"

ES_HOST = "http://localhost:9200"
ES_INDEX_ASR = "asr_segments"
ES_INDEX_CAPTION = "caption_frames"

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
    transform: scale(2);
    box-shadow: 0 12px 32px rgba(0,0,0,0.45);
    position: relative;
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
    lot = video_lot_num(video_id)
    st.session_state.lot_filter_text = f"L{lot}" if lot is not None else ""


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
            st.caption(r["text"][:140] + ("…" if len(r["text"]) > 140 else ""))


def render_actions(video_id: str, center_n: int, key_prefix: str, key_suffix: str):
    with st.container(horizontal=True):
        if st.button(":material/more_horiz:", key=f"{key_prefix}_more_{key_suffix}", help="Show more"):
            show_neighbors(video_id, center_n)
        st.button(":material/content_copy:", key=f"{key_prefix}_copy_{key_suffix}", help="Copy video id",
                  on_click=copy_to_scope, args=(video_id,))


def render_grid(results: list, key_prefix: str, group_by_video: bool = False):
    if not results:
        st.info("No results.")
        return
    cols_per_row = 5

    if not group_by_video:
        for start in range(0, len(results), cols_per_row):
            chunk = results[start : start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, r in zip(cols, chunk):
                render_thumb(col, r)
                with col:
                    render_actions(r["video_id"], r["n"], key_prefix, f"{r['video_id']}_{r['n']}_{r['rank']}")
        return

    # Grouped by video: results already arrive rank-sorted, so a video's
    # first occurrence is its best (lowest-rank) frame -- that also
    # decides the group's own display order.
    groups: dict = {}
    for r in results:
        groups.setdefault(r["video_id"], []).append(r)
    ordered_groups = sorted(groups.items(), key=lambda kv: kv[1][0]["rank"])

    for video_id, frames in ordered_groups:
        best = frames[0]
        st.markdown(f"**{video_id}** · best rank {best['rank']} · {best['score_label']}={best['score_val']:.4f} · {len(frames)} frame(s)")
        for start in range(0, len(frames), cols_per_row):
            chunk = frames[start : start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, r in zip(cols, chunk):
                render_thumb(col, r)
        render_actions(video_id, best["n"], key_prefix, f"grp_{video_id}")
        st.divider()


st.title("Routing101 — retrieval preview")

with st.sidebar:
    st.header("Search")

    mode = st.segmented_control("Signal", options=["Keyframe", "ASR", "Caption"], default="Keyframe", key="mode")

    submitted_query = st.chat_input("e.g. một người đàn ông đang lái xe máy", key="query_input")
    if submitted_query:
        st.session_state.query_text = submitted_query
    query = st.session_state.get("query_text", "")
    if query:
        st.caption(f"Query: {query}")

    st.session_state.setdefault("use_video_scope", False)
    st.session_state.setdefault("use_collection_scope", False)

    def _exclusive_scope(changed_key: str):
        other_key = "use_collection_scope" if changed_key == "use_video_scope" else "use_video_scope"
        if st.session_state[changed_key]:
            st.session_state[other_key] = False

    col_v, col_c = st.columns(2)
    video_filter_text = col_v.text_input("Search in video", placeholder="e.g. L21_V001", key="video_filter_text")
    lot_filter_text = col_c.text_input("Search in collection", placeholder="e.g. L21-L30", key="lot_filter_text")
    use_video_scope = col_v.checkbox("Use video filter", key="use_video_scope",
                                      on_change=_exclusive_scope, args=("use_video_scope",))
    use_collection_scope = col_c.checkbox("Use collection filter", key="use_collection_scope",
                                           on_change=_exclusive_scope, args=("use_collection_scope",))

    video_filter = video_filter_text if use_video_scope else ""
    lot_filter = parse_lot_range(lot_filter_text) if use_collection_scope else None

    def _snap_top_k():
        st.session_state.top_k = round_top_k(st.session_state.top_k)

    col_k, col_g = st.columns(2)
    col_k.number_input("Top-K", min_value=1, max_value=200, value=DISPLAY_N, step=5,
                        key="top_k", on_change=_snap_top_k)
    top_k = round_top_k(st.session_state.top_k)
    fetch_k = max(FETCH_K, top_k)

    group_by_video = col_g.toggle("Group by video", key="group_by_video")

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

if mode == "Keyframe":
    if query:
        siglip2_df = clip_df = None
        if use_siglip2 or use_rrf:
            siglip2_df = apply_filters(search_siglip2_frame(query, k=fetch_k), video_filter, lot_filter)
        if use_clip or use_rrf:
            clip_df = apply_filters(search_clip_frame(query, k=fetch_k), video_filter, lot_filter)

        if use_siglip2:
            st.subheader("SigLIP2")
            render_grid(df_to_results(siglip2_df.head(top_k), "score"), "kf_sig", group_by_video)
        if use_clip:
            st.subheader("CLIP")
            render_grid(df_to_results(clip_df.head(top_k), "score"), "kf_clip", group_by_video)
        if use_rrf:
            st.subheader("RRF")
            render_grid(df_to_results(rrf_fuse_frame([siglip2_df, clip_df], top_n=top_k), "rrf_score"), "kf_rrf", group_by_video)
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
            render_grid(df_to_results(attach_keyframe_asr(siglip_df).head(top_k), "score", "text"), "asr_sig", group_by_video)
        if use_fuzzy:
            st.subheader("Fuzzy ASR")
            render_grid(df_to_results(attach_keyframe_asr(fuzzy_df).head(top_k), "score", "text"), "asr_fuzzy", group_by_video)
        if use_rrf:
            st.subheader("RRF ASR")
            fused = attach_keyframe_asr(rrf_fuse_asr({"siglip_asr": siglip_df, "fuzzy": fuzzy_df}, top_n=top_k))
            render_grid(df_to_results(fused, "rrf_score", "text"), "asr_rrf", group_by_video)
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
            render_grid(df_to_results(attach_keyframe_caption(siglip_df).head(top_k), "score", "text"), "cap_sig", group_by_video)
        if use_fuzzy:
            st.subheader("Fuzzy Caption")
            render_grid(df_to_results(attach_keyframe_caption(fuzzy_df).head(top_k), "score", "text"), "cap_fuzzy", group_by_video)
        if use_rrf:
            st.subheader("RRF Caption")
            fused = attach_keyframe_caption(rrf_fuse_caption({"siglip_caption": siglip_df, "fuzzy": fuzzy_df}, top_n=top_k))
            render_grid(df_to_results(fused, "rrf_score", "text"), "cap_rrf", group_by_video)
        if not (use_siglip_cap or use_fuzzy or use_rrf):
            st.info("Check at least one search option in the sidebar.")
    else:
        st.info("Type a query to search.")
