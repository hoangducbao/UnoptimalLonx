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

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

FETCH_K = 100      # candidates pulled per leg, gives RRF a real pool to fuse
DISPLAY_N = 100
RRF_K = 60
NEIGHBOR_WINDOW = 7  # "show more" popup: +/- this many frames by frame id when the caller names no window of its own (the frontend always does -- it sizes the popup per its tile-size setting, frontend/js/settings.js)
TOP_G_DEFAULT = 5   # Hierarchy Search: frames kept per video after drill-down (Top-G)

DATASET_MODE = os.getenv("DATASET_MODE", "AIC").upper()  # "AIC" or "ADL"

if Path("D:/hf_cache").exists():
    os.environ.setdefault("HF_HOME", "D:/hf_cache")
    os.environ.setdefault("TORCH_HOME", "D:/hf_cache/torch")

IS_KAGGLE = Path("/kaggle").exists()

def _find_dir(names: list, fallback: str) -> Path:
    if IS_KAGGLE:
        input_root = Path("/kaggle/input")
        if input_root.exists():
            for root, dirs, _ in os.walk(input_root):
                rpath = Path(root)
                if rpath.name.lower() in [n.lower() for n in names]:
                    return rpath
                dirs[:] = [d for d in dirs if not d.startswith(("L0", "L1", "L2", "v_"))]
    return Path(fallback)

# ---------------------------------------------------------------------------
# Embedding profiles -- which SigLIP2 checkpoint (768 or 1152)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
_EXTRACTED = Path("D:/University/Summ26/AICDataExtracted")

EMBED_PROFILE = os.getenv("R101_EMBED", "768")

_PROFILES = {
    "768": dict(
        dim=768,
        model_id="google/siglip2-base-patch16-384",
        frame_glob=str(_EXTRACTED / "siglib_embed" / "*.npy"),
        asr_dir=_EXTRACTED / "transcript_embed",
        caption_dir=_EXTRACTED / "caption_embed",
        summary_embed_dir=_EXTRACTED / "summary_embed",
        summary_chunked=False,
        index_sub=".",
    ),
    "1152": dict(
        dim=1152,
        model_id="google/siglip2-so400m-patch14-384",
        frame_glob=str(_EXTRACTED / "1152embed" / "1152keyframe" / "*.npy"),
        asr_dir=_EXTRACTED / "1152embed" / "1152transcript",
        caption_dir=_EXTRACTED / "1152embed" / "1152caption",
        summary_embed_dir=_EXTRACTED / "1152embed" / "1152summary",
        summary_chunked=True,
        index_sub="1152",
    ),
}
if EMBED_PROFILE not in _PROFILES:
    raise SystemExit(f"R101_EMBED={EMBED_PROFILE!r} -- expected one of {sorted(_PROFILES)}")
_P = _PROFILES[EMBED_PROFILE]

EMBED_DIM = _P["dim"]
SIGLIP2_MODEL_ID = _P["model_id"]
SUMMARY_CHUNKED = _P["summary_chunked"]

# OD (object-detection) text filter (backend/od_filter.py)
od_dir = _find_dir(["filtered_object", "filtered_objects", "objects", "object_detection"], "D:/University/Summ26/AICDataExtracted/filtered_object")
FILTERED_OBJECT_DIR = Path(os.getenv("FILTERED_OBJECT_DIR", str(od_dir)))
CLASS_VOCAB_CSV = Path(os.getenv("CLASS_VOCAB_CSV", str(FILTERED_OBJECT_DIR / "class_vocab.csv")))

if DATASET_MODE == "ADL":
    ADL_RAW_DIR = Path(os.getenv("ADL_RAW_DIR", "D:/ADLData"))
    ADL_EXTRACTED_DIR = Path(os.getenv("ADL_EXTRACTED_DIR", "D:/ADLDataExtracted"))

    FRAME_SIGLIP2_GLOB = str(ADL_EXTRACTED_DIR / "siglib_embed" / "*.npy")

    ASR_EMBED_DIR = ADL_EXTRACTED_DIR / "transcript_embed"
    TRANSCRIPTS_DIR = ADL_EXTRACTED_DIR / "transcripts"

    CAPTIONING_DIR = ADL_EXTRACTED_DIR / "captions"
    SIGLIP_CAPTION_DIR = ADL_EXTRACTED_DIR / "caption_embed"

    OCR_DIR = ADL_EXTRACTED_DIR / "ocr"

    SUMMARY_DIR = ADL_EXTRACTED_DIR / "summaries"
    SUMMARY_EMBED_DIR = ADL_EXTRACTED_DIR / "summary_embed"

    MAP_KEYFRAMES_DIR = ADL_RAW_DIR / "map-keyframes"
    THUMBNAIL_ROOT = ADL_RAW_DIR / "keyframes"
    VIDEO_DIR = ADL_RAW_DIR / "video"

    INDEX_PREFIX = "adl"
else:
    # Default AIC dataset with dynamic fallback
    if IS_KAGGLE:
        siglip_dir = _find_dir(["siglib_embed", "siglip_embed", "siglip2_embed", "1152keyframe"], str(_P["frame_glob"]))
        FRAME_SIGLIP2_GLOB = os.getenv("FRAME_SIGLIP2_GLOB", str(siglip_dir / "*.npy"))
        ASR_EMBED_DIR = Path(os.getenv("ASR_EMBED_DIR", str(_find_dir(["transcript_embed", "transcripts_embed", "1152transcript", "asr_embed"], str(_P["asr_dir"])))))
        TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", str(_find_dir(["transcripts", "transcript"], "D:/University/Summ26/AICDataExtracted/transcripts"))))
        CAPTIONING_DIR = Path(os.getenv("CAPTIONING_DIR", str(_find_dir(["captions", "caption", "captioning"], "D:/University/Summ26/AICDataExtracted/captions"))))
        SIGLIP_CAPTION_DIR = Path(os.getenv("SIGLIP_CAPTION_DIR", str(_find_dir(["caption_embed", "captions_embed", "1152caption", "siglip_caption"], str(_P["caption_dir"])))))
        OCR_DIR = Path(os.getenv("OCR_DIR", str(_find_dir(["ocr"], "D:/University/Summ26/AICDataExtracted/ocr"))))
        SUMMARY_DIR = Path(os.getenv("SUMMARY_DIR", str(_find_dir(["summaries", "summary"], "D:/University/Summ26/AICDataExtracted/summaries"))))
        SUMMARY_EMBED_DIR = Path(os.getenv("SUMMARY_EMBED_DIR", str(_find_dir(["summary_embed", "summaries_embed", "1152summary"], "/kaggle/working/summary_embed"))))
        MAP_KEYFRAMES_DIR = Path(os.getenv("MAP_KEYFRAMES_DIR", str(_find_dir(["map-keyframes", "map_keyframes"], "D:/University/Summ26/AICData/map-keyframes"))))
        THUMBNAIL_ROOT = Path(os.getenv("THUMBNAIL_ROOT", str(_find_dir(["keyframes"], "D:/University/Summ26/AICData/keyframes"))))
        VIDEO_DIR = Path(os.getenv("VIDEO_DIR", str(_find_dir(["degarr", "videos", "video"], "D:/University/Summ26/AICData/video"))))
    else:
        FRAME_SIGLIP2_GLOB = os.getenv("FRAME_SIGLIP2_GLOB", _P["frame_glob"])
        ASR_EMBED_DIR = Path(os.getenv("ASR_EMBED_DIR", str(_P["asr_dir"])))
        TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", str(_EXTRACTED / "transcripts")))
        CAPTIONING_DIR = Path(os.getenv("CAPTIONING_DIR", str(_EXTRACTED / "captions")))
        SIGLIP_CAPTION_DIR = Path(os.getenv("SIGLIP_CAPTION_DIR", str(_P["caption_dir"])))
        OCR_DIR = Path(os.getenv("OCR_DIR", str(_EXTRACTED / "ocr")))
        SUMMARY_DIR = Path(os.getenv("SUMMARY_DIR", str(_EXTRACTED / "summaries")))
        SUMMARY_EMBED_DIR = Path(os.getenv("SUMMARY_EMBED_DIR", str(_P["summary_embed_dir"])))
        MAP_KEYFRAMES_DIR = Path(os.getenv("MAP_KEYFRAMES_DIR", "D:/University/Summ26/AICData/map-keyframes"))
        THUMBNAIL_ROOT = Path(os.getenv("THUMBNAIL_ROOT", "D:/University/Summ26/AICData/keyframes"))
        VIDEO_DIR = Path(os.getenv("VIDEO_DIR", "D:/University/Summ26/AICData/video"))

    INDEX_PREFIX = "routing101"

try:
    SUMMARY_EMBED_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

INDEX_DIR = Path(os.getenv("INDEX_DIR", str(REPO_ROOT / "index" / _P["index_sub"])))
PIPELINE_DIR = REPO_ROOT / "pipeline"  # rule/LLM-extracted per-lot metadata CSVs (backend/metadata_filter.py)
ASR_INDEX_DIR = INDEX_DIR / f"{INDEX_PREFIX}_asr"
CAPTION_INDEX_DIR = INDEX_DIR / f"{INDEX_PREFIX}_caption"
SUMMARY_INDEX_DIR = INDEX_DIR / f"{INDEX_PREFIX}_summary"
ASR_INDEX_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_INDEX_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_INDEX_DIR.mkdir(parents=True, exist_ok=True)
SIGLIP_ASR_FAISS = ASR_INDEX_DIR / "siglip_asr_flat_ip.index"
SIGLIP_ASR_META = ASR_INDEX_DIR / "meta_siglip_asr.csv"
SIGLIP_CAPTION_FAISS = CAPTION_INDEX_DIR / "siglip_caption_flat_ip.index"
SIGLIP_CAPTION_META = CAPTION_INDEX_DIR / "meta_siglip_caption.csv"
SIGLIP_SUMMARY_FAISS = SUMMARY_INDEX_DIR / "siglip_summary_flat_ip.index"
SIGLIP_SUMMARY_META = SUMMARY_INDEX_DIR / "meta_siglip_summary.csv"

ES_HOST = os.getenv("ES_HOST", "http://127.0.0.1:9200")
ES_INDEX_ASR = f"{INDEX_PREFIX}_asr_segments" if DATASET_MODE == "ADL" else "asr_segments"
ES_INDEX_CAPTION = f"{INDEX_PREFIX}_caption_frames" if DATASET_MODE == "ADL" else "caption_frames"
ES_INDEX_OCR = f"{INDEX_PREFIX}_ocr_frames" if DATASET_MODE == "ADL" else "ocr_frames"
ES_INDEX_SUMMARY = f"{INDEX_PREFIX}_summary_videos" if DATASET_MODE == "ADL" else "summary_videos"

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
