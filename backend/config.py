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
DISPLAY_N = 100
RRF_K = 60
NEIGHBOR_WINDOW = 7  # "show more" popup: +/- this many frames by frame id
TOP_G_DEFAULT = 5   # Hierarchy Search: frames kept per video after drill-down (Top-G)

SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-384"

FRAME_SIGLIP2_GLOB = "D:/University/Summ26/AICDataExtracted/siglib_embed/*.npy"

ASR_EMBED_DIR = Path("D:/University/Summ26/AICDataExtracted/transcript_embed")  # was asr_embed
TRANSCRIPTS_DIR = Path("D:/University/Summ26/AICDataExtracted/transcripts")

CAPTIONING_DIR = Path("D:/University/Summ26/AICDataExtracted/captions")  # was captioning
SIGLIP_CAPTION_DIR = Path("D:/University/Summ26/AICDataExtracted/caption_embed")  # was siglip_caption

OCR_DIR = Path("D:/University/Summ26/AICDataExtracted/ocr")

# OD (object-detection) text filter (backend/od_filter.py) -- per-video
# filtered-detections CSVs produced upstream by AICPreprocess/filter_apply.py
# (outside this repo) plus the offline class vocabulary built from them by
# pipeline/build_class_vocab.py.
FILTERED_OBJECT_DIR = Path("D:/University/Summ26/AICDataExtracted/filtered_object")
CLASS_VOCAB_CSV = FILTERED_OBJECT_DIR / "class_vocab.csv"

SUMMARY_DIR = Path("D:/University/Summ26/AICDataExtracted/summaries")
SUMMARY_EMBED_DIR = Path("D:/University/Summ26/AICDataExtracted/summary_embed")
SUMMARY_EMBED_DIR.mkdir(parents=True, exist_ok=True)

MAP_KEYFRAMES_DIR = Path("D:/University/Summ26/AICData/map-keyframes")
THUMBNAIL_ROOT = Path("D:/University/Summ26/AICData/keyframes")
VIDEO_DIR = Path("D:/University/Summ26/AICData/video")  # TRAKE playback dialog

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO_ROOT / "index"
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
