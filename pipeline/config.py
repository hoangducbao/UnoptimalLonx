"""
config.py — central constants shared by the query-time text encoders
(clip_encoder.py). Everything here is a plain module-level constant (no
env vars, no config file) since this is a single-developer local testing
scaffold.
"""

# Multilingual-CLIP's text tower is trained to land in the SAME embedding
# space as OpenAI's untouched ViT-B/32 image tower, so it's the query-side
# encoder for the CLIP ViT-B/32 leg (see clip_encoder.py). Understands
# Vietnamese natively, no re-extraction of the organizer-provided features
# needed.
CLIP_MCLIP_MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
