"""
backend/models.py -- shared model loaders. Ported from ui/app.py:121,
285-327. @st.cache_resource -> plain module-level singleton (`_siglip2`),
built once by backend/main.py's lifespan hook instead of lazily on first
call -- a FastAPI process has a real one-time startup, so there's no need
for Streamlit's per-process-cache-keyed-by-args dance.
"""

import numpy as np
import torch
from PIL import Image

from . import config

def _select_device() -> str:
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 7:
                # sm_70+ (T4, V100, RTX series, A100, etc.)
                torch.zeros(1, device="cuda")
                return "cuda"
            else:
                print(f"[Device Warning] GPU compute capability {cap[0]}.{cap[1]} < 7.0 (e.g. Tesla P100). Falling back to CPU.")
                return "cpu"
        except Exception as e:
            print(f"[Device Warning] CUDA initialization check failed ({e}). Falling back to CPU.")
            return "cpu"
    return "cpu"

DEVICE = _select_device()

_siglip2 = None  # (model, processor), set by load_siglip2()


def load_siglip2():
    global _siglip2
    if _siglip2 is None:
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(config.SIGLIP2_MODEL_ID).to(DEVICE).eval()
        processor = AutoProcessor.from_pretrained(config.SIGLIP2_MODEL_ID)
        _siglip2 = (model, processor)
    return _siglip2


def encode_text_siglip2(texts: list) -> np.ndarray:
    model, processor = load_siglip2()
    inputs = processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return feats.float().cpu().numpy().astype("float32")


def encode_image_siglip2(images: list) -> np.ndarray:
    """SigLIP2 image tower -- same joint text/image embedding space as
    encode_text_siglip2(), so an image query is directly comparable to
    every SigLIP2-embedded leg (frame, ASR, caption, summary), not just
    the frame index."""
    model, processor = load_siglip2()
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return feats.float().cpu().numpy().astype("float32")


def is_image_query(query) -> bool:
    return isinstance(query, Image.Image)


def siglip2_query_vec(query) -> np.ndarray:
    """Picture-query support: an uploaded/pasted image is embedded with the
    SigLIP2 image tower instead of the text tower -- everywhere else the
    caller already treats the result as a plain query vector, so this is
    the only branch point needed for every SigLIP2-backed leg."""
    if is_image_query(query):
        return encode_image_siglip2([query])[0]
    return encode_text_siglip2([query])[0]
