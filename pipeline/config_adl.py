"""
pipeline/config_adl.py -- Central configuration for ADL dataset processing & feature extraction.
"""

import os
from pathlib import Path

# Base directories
# Users can override via environment variables or pass CLI flags
ADL_RAW_DIR = Path(os.getenv("ADL_RAW_DIR", "D:/ADLData"))
ADL_EXTRACTED_DIR = Path(os.getenv("ADL_EXTRACTED_DIR", "D:/ADLDataExtracted"))

# Raw data subdirectories
VIDEO_DIR = ADL_RAW_DIR / "video"
KEYFRAME_DIR = ADL_RAW_DIR / "keyframes"
MAP_KEYFRAMES_DIR = ADL_RAW_DIR / "map-keyframes"

# Extracted features subdirectories
SIGLIP2_EMBED_DIR = ADL_EXTRACTED_DIR / "siglib_embed"
CLIP_EMBED_DIR = ADL_RAW_DIR / "clip-features-32"

CAPTION_DIR = ADL_EXTRACTED_DIR / "captions"
CAPTION_EMBED_DIR = ADL_EXTRACTED_DIR / "caption_embed"

ASR_TRANSCRIPT_DIR = ADL_EXTRACTED_DIR / "transcripts"
ASR_EMBED_DIR = ADL_EXTRACTED_DIR / "transcript_embed"

OCR_DIR = ADL_EXTRACTED_DIR / "ocr"

FILTERED_OBJECT_DIR = ADL_EXTRACTED_DIR / "filtered_object"
CLASS_VOCAB_CSV = FILTERED_OBJECT_DIR / "class_vocab.csv"

SUMMARY_DIR = ADL_EXTRACTED_DIR / "summaries"
SUMMARY_EMBED_DIR = ADL_EXTRACTED_DIR / "summary_embed"

# Model IDs
SIGLIP2_MODEL_ID = os.getenv("SIGLIP2_MODEL_ID", "google/siglip2-base-patch16-384")
CLIP_VISION_MODEL_ID = os.getenv("CLIP_VISION_MODEL_ID", "openai/clip-vit-base-patch32")
WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "large-v3")
VLM_CAPTION_MODEL_ID = os.getenv("VLM_CAPTION_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
YOLO_MODEL_ID = os.getenv("YOLO_MODEL_ID", "yolov8x.pt")

# Batching & Hardware defaults
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
DEVICE = "cuda" if os.getenv("FORCE_CPU", "0") != "1" else "cpu"


def ensure_directories():
    """Create all required output folders if they do not exist."""
    dirs = [
        VIDEO_DIR,
        KEYFRAME_DIR,
        MAP_KEYFRAMES_DIR,
        SIGLIP2_EMBED_DIR,
        CLIP_EMBED_DIR,
        CAPTION_DIR,
        CAPTION_EMBED_DIR,
        ASR_TRANSCRIPT_DIR,
        ASR_EMBED_DIR,
        OCR_DIR,
        FILTERED_OBJECT_DIR,
        SUMMARY_DIR,
        SUMMARY_EMBED_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

