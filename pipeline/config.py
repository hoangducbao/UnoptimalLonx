"""
config.py — central paths/constants shared by routing101.py (CLI) and the
query-time text encoders (clip_encoder.py, viclip_encoder.py). Everything
here is a plain module-level constant (no env vars, no config file) since
this is a single-developer local testing scaffold. Paths point at data
that lives outside the repo.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# --- Source data -------------------------------------------------------
# ViCLIP-OT 768-d frame embeddings, routing101.py's --backend viclip.
FRAME_DATA_ROOT = Path("D:/University/Summ26/AICDataExtracted")
EMBEDDINGS_DIR = FRAME_DATA_ROOT / "embeddings"

# Pre-existing CLIP ViT-B/32 frame embeddings (512-d, float16 on disk),
# routing101.py's --backend clip_vitb32. Row-for-row aligned with the
# ViCLIP-OT embeddings above for every one of the 873 videos (verified
# directly), so both share the same map-keyframes join.
CLIP_VITB32_DIR = Path("D:/University/Summ26/AICData/clip-features-32")

# Per-keyframe timestamps and thumbnail images, row-for-row aligned with
# both embedding sources above (verified during planning).
TIMESTAMP_ROOT = Path("D:/University/Summ26/AICData/map-keyframes")
THUMBNAIL_ROOT = Path("D:/University/Summ26/AICData/keyframes")

# --- Models ----------------------------------------------------------------
VICLIP_MODEL_ID = "minhnguyent546/ViCLIP-OT"
# Multilingual-CLIP's text tower is trained to land in the SAME embedding
# space as OpenAI's untouched ViT-B/32 image tower, so it's the query-side
# encoder for the clip_vitb32 backend (see clip_encoder.py). Understands
# Vietnamese natively, no re-extraction of the organizer-provided features
# needed.
CLIP_MCLIP_MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-B-32"

# --- Selectable embedding backends (routing101.py) --------------------------
BACKENDS = {
    "viclip": {"dim": 768},
    "clip_vitb32": {"dim": 512},
}
