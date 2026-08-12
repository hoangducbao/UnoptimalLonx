"""
config.py — central paths/constants for the C1 baseline pipeline.

Everything here is a plain module-level constant (no env vars, no config
file) since this is a single-developer local testing scaffold, per
docs/c1-baseline-spec.md. Paths point at data that lives outside the repo.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# --- Source data -------------------------------------------------------
# New dataset: ViCLIP-OT 768-d frame embeddings + filtered object detections.
FRAME_DATA_ROOT = Path("D:/University/Summ26/AICDataExtracted")
EMBEDDINGS_DIR = FRAME_DATA_ROOT / "embeddings"
DETECTIONS_CSV = FRAME_DATA_ROOT / "filtered_detections.csv"

# Old dataset: reused only for the two things not (yet) in AICDataExtracted —
# per-keyframe timestamps and, best-effort, thumbnail images. Row-for-row
# aligned with the new embeddings (verified during planning); not code, so
# reusing it doesn't conflict with "old files -> old_version".
TIMESTAMP_ROOT = Path("D:/University/Summ26/AICData/map-keyframes")
THUMBNAIL_ROOT = Path("D:/University/Summ26/AICData/keyframes")

# --- Generated index artifacts ------------------------------------------
INDEX_DIR = REPO_ROOT / "index"
FRAME_FAISS_PATH = INDEX_DIR / "frame_viclip768_flat_ip.index"
CLASS_FAISS_PATH = INDEX_DIR / "class_viclip768_flat_ip.index"
DB_PATH = INDEX_DIR / "metadata.db"

# --- Model ---------------------------------------------------------------
MODEL_ID = "minhnguyent546/ViCLIP-OT"
VECTOR_DIM = 768

# --- Fusion ---------------------------------------------------------------
RRF_K = 60

# Per-leg weight in the RRF sum (contribution = weight / (k + rank + 1)).
# Class-leg cosine similarities are compressed for compound/sentence queries
# (measured: e.g. "a person riding a motorbike" -> Motorcycle 0.63 vs Candle
# 0.61 vs Measuring cup 0.59 -- barely separated), so its top-N matched
# classes pull in a lot of weakly-related frames. Down-weighted by default
# so the frame leg (which does the real semantic matching) dominates;
# exposed as a live slider in ui/app.py since fusion strategy is explicitly
# meant to be tuned/swapped (see fusion.py).
DEFAULT_LEG_WEIGHTS = {"frame": 1.0, "class": 0.4, "text": 1.0}

# How many nearest classes the object-class leg considers a "match" before
# mapping back to frames. Kept small by default for the same reason above --
# a wide top-N pulls in classes with no real relevance to the query.
DEFAULT_CLASS_TOP_K = 5
