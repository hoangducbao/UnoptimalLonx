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
# Primary dataset: ViCLIP-OT 768-d frame embeddings + filtered object detections.
FRAME_DATA_ROOT = Path("D:/University/Summ26/AICDataExtracted")
EMBEDDINGS_DIR = FRAME_DATA_ROOT / "embeddings"
DETECTIONS_CSV = FRAME_DATA_ROOT / "filtered_detections.csv"

# Second backend: pre-existing CLIP ViT-B/32 frame embeddings (512-d,
# float16 on disk). Row-for-row aligned with the ViCLIP-OT embeddings above
# for every one of the 873 videos (verified directly) -- so both backends
# share one SQLite metadata store and one global_id numbering scheme; only
# the FAISS vector files and the text encoder differ per backend. See
# backends.py.
CLIP_VITB32_DIR = Path("D:/University/Summ26/AICData/clip-features-32")

# Old dataset: reused only for the two things not (yet) in AICDataExtracted —
# per-keyframe timestamps and, best-effort, thumbnail images. Row-for-row
# aligned with the new embeddings (verified during planning); not code, so
# reusing it doesn't conflict with "old files -> old_version".
TIMESTAMP_ROOT = Path("D:/University/Summ26/AICData/map-keyframes")
THUMBNAIL_ROOT = Path("D:/University/Summ26/AICData/keyframes")

# --- Generated index artifacts ------------------------------------------
INDEX_DIR = REPO_ROOT / "index"
DB_PATH = INDEX_DIR / "metadata.db"

# --- Models ----------------------------------------------------------------
VICLIP_MODEL_ID = "minhnguyent546/ViCLIP-OT"
# Multilingual-CLIP's text tower is trained to land in the SAME embedding
# space as OpenAI's untouched ViT-B/32 image tower, so it's the query-side
# encoder for the clip_vitb32 backend (see clip_encoder.py). Understands
# Vietnamese natively, no re-extraction of the organizer-provided features
# needed.
CLIP_MCLIP_MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-B-32"

# --- Selectable embedding backends -----------------------------------------
# Both index the exact same keyframes in the exact same per-video order, so
# they share metadata/global_id -- only the vector files + text encoder
# differ. See backends.py for the encode_text/load_video wiring.
BACKENDS = {
    "viclip": {
        "label": "ViCLIP-OT (768-d)",
        "dim": 768,
        "frame_faiss_path": INDEX_DIR / "frame_viclip768_flat_ip.index",
        "class_faiss_path": INDEX_DIR / "class_viclip768_flat_ip.index",
    },
    "clip_vitb32": {
        "label": "CLIP ViT-B/32 (512-d)",
        "dim": 512,
        "frame_faiss_path": INDEX_DIR / "frame_clipvitb32_flat_ip.index",
        "class_faiss_path": INDEX_DIR / "class_clipvitb32_flat_ip.index",
    },
}
# Measured (not assumed): on this dataset, clip_vitb32 discriminates true
# matches from noise far more sharply than viclip across the whole corpus
# (top-1 z-score ~5.5 vs ~2.3 for the same English query, ~5.4 vs ~3.5 for
# the same query in Vietnamese) -- so it's the default despite viclip's
# larger raw similarity numbers. Both stay selectable in the UI.
DEFAULT_BACKEND = "clip_vitb32"

# --- Fusion ---------------------------------------------------------------
RRF_K = 60

# Per-leg weight in the RRF sum (contribution = weight / (k + rank + 1)).
# Class-leg cosine similarities are compressed for compound/sentence queries
# (measured: e.g. "a person riding a motorbike" -> Motorcycle 0.63 vs Candle
# 0.61 vs Measuring cup 0.59 -- barely separated), so its top-N matched
# classes pull in a lot of weakly-related frames. Exposed as a live slider
# in ui/app.py since fusion strategy is explicitly meant to be tuned/swapped
# (see fusion.py).
DEFAULT_LEG_WEIGHTS = {"frame": 1.0, "class": 0.7, "text": 1.0}

# How many nearest classes the object-class leg considers a "match" before
# mapping back to frames. Kept small by default -- a wide top-N pulls in
# classes with no real relevance to the query.
DEFAULT_CLASS_TOP_K = 5

# How many frames on each side of a clicked result to show as "nearby
# frames" in the UI.
NEARBY_FRAMES_WINDOW = 5
