"""
backend/config.py -- paths + constants shared by every backend module.

Ported verbatim from ui/app.py's Config block (ui/app.py:70-134) -- see
CLAUDE.md and the plan at the top of this rewrite for why these values are
hardcoded here rather than in an env file (single-developer local scaffold,
data lives outside the repo under absolute D:/University/Summ26/AICData*
paths). Update these constants, not a config file, if the data moves.
"""

import os
from pathlib import Path

FETCH_K = 100      # candidates pulled per leg, gives RRF a real pool to fuse
DISPLAY_N = 200
RRF_K = 60
NEIGHBOR_WINDOW = 7  # "show more" popup: +/- this many frames by frame id when the caller names no window of its own (the frontend always does -- it sizes the popup per its tile-size setting, frontend/js/settings.js)
TOP_G_DEFAULT = 10   # Hierarchy Search: frames kept per video after drill-down (Top-G)

# ---------------------------------------------------------------------------
# Embedding profiles -- which SigLIP2 checkpoint (and so which vector
# dimension, and which set of precomputed .npy files) this process runs on.
# Chosen once from the R101_EMBED environment variable before anything
# loads; there is deliberately no in-app switch. A profile costs ~2.9GB
# (768) / ~5.5GB (1152) resident in model weights plus FAISS indices, so
# holding both in one process would roughly double a footprint this project
# has already trimmed once on purpose (see ARCHITECTURE.md's Signals table
# on the removed M-CLIP text tower). Run two processes on two ports instead
# -- run_768.bat / run_1152.bat. They share Elasticsearch, /media, the
# OD vocabulary and the metadata facets; only the embedding legs differ.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
_EXTRACTED = Path("D:/University/Summ26/AICDataExtracted")

EMBED_PROFILE = os.getenv("R101_EMBED", "768")

_PROFILES = {
    "768": dict(
        dim=768,
        model_id="google/siglip2-base-patch16-384",
        frame_glob=str(_EXTRACTED / "siglib_embed" / "*.npy"),
        asr_dir=_EXTRACTED / "transcript_embed",   # was asr_embed
        caption_dir=_EXTRACTED / "caption_embed",  # was siglip_caption
        summary_embed_dir=_EXTRACTED / "summary_embed",
        summary_chunked=False,
        index_sub=".",  # index/routing101_* -- the original on-disk layout
    ),
    "1152": dict(
        dim=1152,
        model_id="google/siglip2-so400m-patch14-384",
        frame_glob=str(_EXTRACTED / "1152embed" / "1152keyframe" / "*.npy"),
        asr_dir=_EXTRACTED / "1152embed" / "1152transcript",
        caption_dir=_EXTRACTED / "1152embed" / "1152caption",
        summary_embed_dir=_EXTRACTED / "1152embed" / "1152summary",
        # This checkpoint's summaries were embedded chunk-by-chunk (the
        # dir's manifest.csv says strategy=chunks_separate: 2501 chunks
        # over 785 videos) rather than one vector per summary, because
        # SigLIP2's text tower only ever sees 64 tokens and a summary runs
        # far longer than that -- the 768 profile silently truncated most
        # of every summary it embedded. The summary index therefore holds
        # every chunk and collapses to the best one per video at query
        # time; see backend/search/summary.py.
        summary_chunked=True,
        index_sub="1152",  # index/1152/routing101_*
    ),
}
if EMBED_PROFILE not in _PROFILES:
    raise SystemExit(f"R101_EMBED={EMBED_PROFILE!r} -- expected one of {sorted(_PROFILES)}")
_P = _PROFILES[EMBED_PROFILE]

EMBED_DIM = _P["dim"]
SIGLIP2_MODEL_ID = _P["model_id"]
FRAME_SIGLIP2_GLOB = _P["frame_glob"]
ASR_EMBED_DIR = _P["asr_dir"]
SIGLIP_CAPTION_DIR = _P["caption_dir"]
SUMMARY_EMBED_DIR = _P["summary_embed_dir"]
SUMMARY_CHUNKED = _P["summary_chunked"]

# Profile-independent source data -- raw text, thumbnails, video, metadata.
# Shared by every profile, never duplicated per dimension.
TRANSCRIPTS_DIR = _EXTRACTED / "transcripts"
CAPTIONING_DIR = _EXTRACTED / "captions"  # was captioning
OCR_DIR = _EXTRACTED / "ocr"
SUMMARY_DIR = _EXTRACTED / "summaries"

# OD (object-detection) text filter (backend/od_filter.py) -- per-video
# filtered-detections CSVs produced upstream by AICPreprocess/filter_apply.py
# (outside this repo) plus the offline class vocabulary built from them by
# pipeline/build_class_vocab.py.
FILTERED_OBJECT_DIR = Path("D:/University/Summ26/AICDataExtracted/filtered_object")
CLASS_VOCAB_CSV = FILTERED_OBJECT_DIR / "class_vocab.csv"

MAP_KEYFRAMES_DIR = Path("D:/University/Summ26/AICData/map-keyframes")
THUMBNAIL_ROOT = Path("D:/University/Summ26/AICData/keyframes")
VIDEO_DIR = Path("D:/University/Summ26/AICData/video")  # TRAKE playback dialog

INDEX_DIR = REPO_ROOT / "index" / _P["index_sub"]
PIPELINE_DIR = REPO_ROOT / "pipeline"  # rule/LLM-extracted per-lot metadata CSVs (backend/metadata_filter.py)
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

# Thread-pool tuning -- CPU-only torch defaults to num-cores intraop threads
# AND num-cores interop threads, and FAISS's own OpenMP pool defaults to
# num-cores on top of that; left uncapped the pools compound into far more
# live threads than the box has cores. Unlike ui/app.py (Streamlit re-execs
# the whole module on every rerun, so this needed a cache_resource + guard
# dance to only ever run once per process), a FastAPI process has a real
# single startup -- see backend/main.py's lifespan, which calls
# tune_thread_pools() exactly once.
CPU_BUDGET = max(1, (os.cpu_count() or 4) - 2)  # leave headroom for uvicorn/OS


def tune_thread_pools(device: str) -> None:
    import faiss
    import torch

    if device == "cpu":
        torch.set_num_threads(CPU_BUDGET)
    torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(CPU_BUDGET)
